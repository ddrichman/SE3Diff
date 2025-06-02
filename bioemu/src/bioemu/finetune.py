import logging
import os
import typing
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import fire
import hydra
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch._prims_common import DeviceLikeType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm

from bioemu.denoiser import SDEs
from bioemu.models import DiGConditionalScoreModel
from bioemu.ppft import (
    compute_ev_loss_from_int_dws,
    compute_int_dws,
    compute_int_u_u_dt,
    compute_kl_loss_from_int_dws,
)
from bioemu.sample import (
    DEFAULT_MODEL_CHECKPOINT_DIR,
    SupportedModelNamesLiteral,
    generate_finetune_batch,
    maybe_download_checkpoint,
)
from bioemu.seq_io import check_protein_valid
from bioemu.utils import print_traceback_on_exception

logger = logging.getLogger(__name__)

# Finetune denoiser config
DEFAULT_FINETUNE_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/finetune_denoiser/"
SupportedFinetuneDenoisersLiteral = Literal[
    "sde_dpm_finetune", "heun_finetune", "euler_maruyama_finetune"
]
SUPPORTED_FINETUNE_DENOISERS = list(typing.get_args(SupportedFinetuneDenoisersLiteral))

# h-function config
DEFAULT_H_FUNC_CONFIG_DIR = Path(__file__).parent / "config/h_func/"
SupportedHFuncsLiteral = Literal["folding_stability"]
SUPPORTED_H_FUNCS = list(typing.get_args(SupportedHFuncsLiteral))

# Finetune config
DEFAULT_FINETUNE_CONFIG_DIR = Path(__file__).parent / "config/finetune/"


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning process."""

    # Data parameters
    data_batch_size: int
    shuffle: bool
    num_workers: int

    # Loss parameters
    lambda_: float
    tol: float

    # Training parameters
    batch_size: int
    epochs: int
    save_every_n_epochs: int
    val_every_n_epochs: int
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eta_min: float


class FinetuneBundle(NamedTuple):
    sdes: SDEs
    score_model: DiGConditionalScoreModel
    finetune_model: DiGConditionalScoreModel
    denoiser: Callable
    h_func: Callable
    finetune_config: FinetuneConfig


def load_finetune_bundle(
    model_name: SupportedModelNamesLiteral | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    finetune_ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedFinetuneDenoisersLiteral | None = "heun_finetune",
    denoiser_config_path: str | Path | None = None,
    h_func_type: SupportedHFuncsLiteral | None = "folding_stability",
    h_func_config_path: str | Path | None = None,
    finetune_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
) -> FinetuneBundle:

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    if cache_so3_dir is not None:
        model_config["sdes"]["node_orientations"]["cache_dir"] = cache_so3_dir

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    finetune_model: DiGConditionalScoreModel = hydra.utils.instantiate(
        model_config.get("finetune_model", model_config["score_model"])
    )
    if finetune_ckpt_path is not None:
        finetune_model.load_state_dict(
            torch.load(finetune_ckpt_path, map_location="cpu", weights_only=True)
        )
    sdes: SDEs = hydra.utils.instantiate(model_config["sdes"])

    if denoiser_config_path is None:
        if denoiser_type not in SUPPORTED_FINETUNE_DENOISERS:
            raise ValueError(f"denoiser_type must be one of {SUPPORTED_FINETUNE_DENOISERS}")
        denoiser_config_path = DEFAULT_FINETUNE_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
    with open(denoiser_config_path, "r") as f:
        denoiser_config = yaml.safe_load(f)
    denoiser: Callable = hydra.utils.instantiate(denoiser_config)

    if h_func_config_path is None:
        if h_func_type not in SUPPORTED_H_FUNCS:
            raise ValueError(f"h_func_type must be one of {SUPPORTED_H_FUNCS}")
        h_func_config_path = DEFAULT_H_FUNC_CONFIG_DIR / f"{h_func_type}.yaml"
    with open(h_func_config_path, "r") as f:
        h_func_config = yaml.safe_load(f)
    h_func: Callable = hydra.utils.instantiate(h_func_config)

    # Load finetune config
    if finetune_config_path is None:
        finetune_config_path = DEFAULT_FINETUNE_CONFIG_DIR / "finetune.yaml"
    with open(finetune_config_path, "r") as f:
        finetune_config_yaml = yaml.safe_load(f)
    finetune_config: FinetuneConfig = hydra.utils.instantiate(finetune_config_yaml)

    return FinetuneBundle(
        sdes=sdes,
        score_model=score_model,
        finetune_model=finetune_model,
        denoiser=denoiser,
        h_func=h_func,
        finetune_config=finetune_config,
    )


class SequenceHStarsDataset(Dataset):
    """Dataset for loading sequences and h_stars from CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        sequence_col: str,
        h_stars_cols: str | list[str],
        device: DeviceLikeType | None = None,
    ):
        """Initialize the dataset.

        Args:
            csv_path: Path to CSV file
            sequence_col: Name of the sequence column
            h_stars_cols: List of h_stars column names.
        """
        self.df = pd.read_csv(csv_path)
        self.sequence_col = sequence_col
        if isinstance(h_stars_cols, str):
            h_stars_cols = [h_stars_cols]
        self.h_stars_cols = h_stars_cols
        self.device = device

        # Validate columns exist
        if sequence_col not in self.df.columns:
            raise ValueError(f"Sequence column '{sequence_col}' not found in CSV")

        if missing_h_cols := [col for col in self.h_stars_cols if col not in self.df.columns]:
            raise ValueError(f"h_stars columns not found in CSV: {missing_h_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            sequence: Protein sequence string
            h_stars: Tensor of shape (K,) where K is number of h_stars columns
        """

        sequence = self.df.iloc[idx][self.sequence_col]
        h_stars = torch.from_numpy(self.df[self.h_stars_cols].iloc[idx].values).to(
            device=self.device
        )

        return sequence, h_stars


def create_dataloaders(
    csv_path: str | Path,
    csv_path_val: str | Path,
    sequence_col: str,
    h_stars_cols: str | list[str],
    data_batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    device: DeviceLikeType | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation data.

    Args:
        csv_path: Path to training CSV file
        csv_path_val: Path to validation CSV file
        sequence_col: Name of the sequence column
        h_stars_cols: List of h_stars column names
        data_batch_size: Batch size for training data
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    """

    train_dataset = SequenceHStarsDataset(
        csv_path=csv_path,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        device=device,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    val_dataset = SequenceHStarsDataset(
        csv_path=csv_path_val,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        device=device,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # validation does not depend on data batch size
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    return train_loader, val_loader


def compute_finetune_loss(
    sequence: str,
    h_stars: torch.Tensor,
    sdes: SDEs,
    score_model: nn.Module,
    finetune_model: nn.Module,
    denoiser: Callable,
    h_func: Callable,
    batch_size: int,
    device: DeviceLikeType | None = None,
    lambda_: float = 0.1,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    seed: int | None = None,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the loss for the fine-tuning process."""

    check_protein_valid(sequence)
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 for estimating variances.")

    batches, timesteps, dWs_batch = generate_finetune_batch(
        sequence=sequence,
        sdes=sdes,
        score_model=score_model,
        finetune_model=finetune_model,
        denoiser=denoiser,
        batch_size=batch_size,
        device=device,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        seed=seed,
    )
    dts = torch.diff(timesteps)

    # Pop the last batch (x_0) to use for computing h
    hs = h_func(batch=batches.pop(), sequence=sequence)  # (B, K)

    fields = list(sdes.keys())  # ["pos", "node_orientations"]

    # Track the gradients here
    us_list_batch = defaultdict(list)
    for i, batch in enumerate(tqdm(batches, desc="Reverse diffusion", leave=False)):
        timestep = timesteps[i].item()
        t = torch.full((batch_size,), timestep, device=device)
        u_t_batch = finetune_model(batch, t)
        for field in fields:
            u_t = u_t_batch[field]
            us_list_batch[field].append(to_dense_batch(u_t, batch.batch)[0])  # (B, L, 3)
    us_batch = {field: torch.stack(us_list_batch[field], dim=0) for field in fields}

    us_batch_flattened = {field: us_batch[field].flatten(-2, -1) for field in fields}
    dWs_batch_flattened = {field: dWs_batch[field].flatten(-2, -1) for field in fields}

    # Compute the stochastic integrals on riemannian manifolds
    int_dws = sum(
        compute_int_dws(us=us_batch_flattened[field], dWs=dWs_batch_flattened[field])
        for field in fields
    )  # (B,)
    assert isinstance(int_dws, torch.Tensor)
    int_u_u_dt = sum(
        compute_int_u_u_dt(us=us_batch_flattened[field], dts=dts) for field in fields
    )  # (B,)
    assert isinstance(int_u_u_dt, torch.Tensor)

    # Compute the expected value loss and KL divergence loss
    loss_ev = compute_ev_loss_from_int_dws(int_dws=int_dws, hs=hs, h_stars=h_stars, tol=tol)
    loss_kl = compute_kl_loss_from_int_dws(int_u_u_dt=int_u_u_dt, int_dws=int_dws)

    # Combine the losses
    loss = loss_ev + lambda_ * loss_kl

    return loss


def finetune(
    csv_path: str | Path,
    csv_path_val: str | Path,
    sequence_col: str,
    h_stars_cols: str | list[str],
    sdes: SDEs,
    score_model: nn.Module,
    finetune_model: nn.Module,
    denoiser: Callable,
    h_func: Callable,
    finetune_config: FinetuneConfig,
    output_dir: str | Path | None = None,
    device: DeviceLikeType | None = None,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    seed: int | None = None,
):
    """Fine-tune the model using the provided data loaders and configuration."""

    data_batch_size: int = finetune_config.data_batch_size
    shuffle: bool = finetune_config.shuffle
    num_workers: int = finetune_config.num_workers

    lambda_: float = finetune_config.lambda_
    tol: float = finetune_config.tol

    batch_size: int = finetune_config.batch_size
    epochs: int = finetune_config.epochs
    save_every_n_epochs: int = finetune_config.save_every_n_epochs
    val_every_n_epochs: int = finetune_config.val_every_n_epochs
    lr: float = finetune_config.lr
    betas: tuple[float, float] = finetune_config.betas
    weight_decay: float = finetune_config.weight_decay
    eta_min: float = finetune_config.eta_min

    # Load datasets and create dataloaders
    dataloader, dataloader_val = create_dataloaders(
        csv_path=csv_path,
        csv_path_val=csv_path_val,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        data_batch_size=data_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        device=device,
    )
    num_batches = len(dataloader)
    num_batches_val = len(dataloader_val)

    # Instantiate SDE and score model
    score_model.to(device).eval()
    finetune_model.to(device)
    sdes["node_orientations"].to(device)

    optimizer = AdamW(finetune_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    best_model_state = {}
    best_loss = float("inf")

    if output_dir is None:
        output_dir = DEFAULT_MODEL_CHECKPOINT_DIR / "finetune"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        finetune_model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for data_batch in pbar:
            loss = torch.tensor(0.0, device=device)
            for sequence, h_stars in data_batch:
                loss += compute_finetune_loss(
                    sequence=sequence,
                    h_stars=h_stars,
                    sdes=sdes,
                    score_model=score_model,
                    finetune_model=finetune_model,
                    denoiser=denoiser,
                    h_func=h_func,
                    batch_size=batch_size,
                    device=device,
                    lambda_=lambda_,
                    cache_embeds_dir=cache_embeds_dir,
                    msa_file=msa_file,
                    msa_host_url=msa_host_url,
                    seed=seed,
                    tol=tol,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.detach().item()
            epoch_loss += l
            pbar.set_postfix(loss=f"{l:.2f}")

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = finetune_model.state_dict()
            logger.info(f"Updated best model at epoch {epoch}")
        logger.info(f"Epoch {epoch}: Average training loss = {avg_loss:.4f}")

        if epoch % val_every_n_epochs == 0 or epoch == epochs:
            finetune_model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                val_loss = torch.tensor(0.0, device=device)
                pbar_val = tqdm(dataloader_val, desc=f"Validation Epoch {epoch}", leave=False)
                for val_data_batch in pbar_val:
                    for sequence, h_stars in val_data_batch:
                        val_loss += compute_finetune_loss(
                            sequence=sequence,
                            h_stars=h_stars,
                            sdes=sdes,
                            score_model=score_model,
                            finetune_model=finetune_model,
                            denoiser=denoiser,
                            h_func=h_func,
                            batch_size=batch_size,
                            device=device,
                            lambda_=lambda_,
                            cache_embeds_dir=cache_embeds_dir,
                            msa_file=msa_file,
                            msa_host_url=msa_host_url,
                            seed=seed,
                            tol=tol,
                        )

                    val_l = val_loss.detach().item()
                    epoch_val_loss += val_l
                    pbar_val.set_postfix(val_loss=f"{val_l:.2f}")

                avg_val_loss = epoch_val_loss / num_batches_val
                logger.info(f"Epoch {epoch}: Average validation loss = {avg_val_loss:.4f}")

        if epoch % save_every_n_epochs == 0 or epoch == epochs:
            torch.save(
                finetune_model.state_dict(),
                os.path.join(output_dir, f"finetune_model_{epoch}.pt"),
            )
            logger.info(f"Model saved to {os.path.join(output_dir, f'finetune_model_{epoch}.pt')}")

    finetune_model.load_state_dict(best_model_state)
    torch.save(finetune_model.state_dict(), os.path.join(output_dir, "finetune_model.pt"))


@print_traceback_on_exception
def main(
    csv_path: str | Path,
    csv_path_val: str | Path,
    sequence_col: str,
    h_stars_cols: list[str],
    output_dir: str | Path | None = None,
    finetune_config_path: str | Path | None = None,
    model_name: SupportedModelNamesLiteral | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    finetune_ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedFinetuneDenoisersLiteral | None = "heun_finetune",
    denoiser_config_path: str | Path | None = None,
    h_func_type: SupportedHFuncsLiteral | None = "folding_stability",
    h_func_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    seed: int | None = None,
):
    """Main function to run the fine-tuning process.

    Args:
        csv_path: Path to training CSV file
        csv_path_val: Path to validation CSV file
        sequence_col: Name of the sequence column in the CSV
        h_stars_cols: List of h_stars column names in the CSV
        output_dir: Directory to save the fine-tuned model
        finetune_config_path: Path to the finetune config YAML file
        model_name: Name of the model to fine-tune
        ckpt_path: Path to the model checkpoint file
        finetune_ckpt_path: Path to the fine-tune checkpoint file
        model_config_path: Path to the model config YAML file
        denoiser_type: Type of denoiser to use for fine-tuning
        denoiser_config_path: Path to the denoiser config YAML file
        h_func_type: Type of h_func to use for fine-tuning
        h_func_config_path: Path to the h_func config YAML file
        cache_embeds_dir: Directory to cache embeddings
        cache_so3_dir: Directory to cache SO3 embeddings
        msa_file: Path to MSA file (optional)
        msa_host_url: URL for MSA host (optional)
        seed: Random seed for reproducibility (optional)

    """

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and denoiser
    sdes, score_model, finetune_model, denoiser, h_func, finetune_config = load_finetune_bundle(
        model_name=model_name,
        ckpt_path=ckpt_path,
        finetune_ckpt_path=finetune_ckpt_path,
        model_config_path=model_config_path,
        denoiser_type=denoiser_type,
        denoiser_config_path=denoiser_config_path,
        h_func_type=h_func_type,
        h_func_config_path=h_func_config_path,
        finetune_config_path=finetune_config_path,
        cache_so3_dir=cache_so3_dir,
    )

    finetune(
        csv_path=csv_path,
        csv_path_val=csv_path_val,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        sdes=sdes,
        score_model=score_model,
        finetune_model=finetune_model,
        denoiser=denoiser,
        h_func=h_func,
        finetune_config=finetune_config,
        output_dir=output_dir,
        device=device,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        seed=seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)
