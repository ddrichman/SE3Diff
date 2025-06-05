import logging
import math
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
import numpy as np
from torch._prims_common import DeviceLikeType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from tqdm.auto import tqdm, trange

from bioemu.chemgraph import ChemGraph
from bioemu.denoiser import DenoisedSDEPath, SDEs
from bioemu.models import DiGConditionalScoreModel
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.ppft import (
    compute_ev_loss,
    compute_int_dws,
    compute_int_u_u_dt,
    compute_kl_loss,
)
from bioemu.sample import (
    DEFAULT_MODEL_CHECKPOINT_DIR,
    SupportedModelNamesLiteral,
    generate_chemgraph,
    maybe_download_checkpoint,
)
from bioemu.seq_io import check_protein_valid
from bioemu.utils import clean_gpu_cache, print_traceback_on_exception

from bioemu.utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)

logger = logging.getLogger(__name__)

# Finetune denoiser config
DEFAULT_FINETUNE_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser"
SupportedFinetuneDenoisersLiteral = Literal[
    "sde_dpm_finetune", "heun_finetune", "euler_maruyama_finetune"
]
SUPPORTED_FINETUNE_DENOISERS = list(typing.get_args(SupportedFinetuneDenoisersLiteral))

# h-function config
DEFAULT_H_FUNC_CONFIG_DIR = Path(__file__).parent / "config/h_func/"
SupportedHFuncsLiteral = Literal["folding_stability"]
SUPPORTED_H_FUNCS = list(typing.get_args(SupportedHFuncsLiteral))

# Finetune config
DEFAULT_FINETUNE_CONFIG_DIR = Path(__file__).parent / "config/finetune"


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
    micro_batch_size: int
    num_epochs: int
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


def initialize_weights_to_near_zero(module: nn.Module, scale: float = 0.1):
    """Initialize weights of a module to near zero."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        with torch.no_grad():
            module.weight.mul_(scale)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
        with torch.no_grad():
            module.weight.mul_(scale)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    else:
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            nn.init.zeros_(module.bias)


def load_finetune_bundle(
    model_name: SupportedModelNamesLiteral | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    finetune_ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedFinetuneDenoisersLiteral | None = "heun_finetune",
    denoiser_config_path: str | Path | None = None,
    h_func_type: SupportedHFuncsLiteral | None = "folding_stability",
    h_func_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
) -> FinetuneBundle:

    # Load model config
    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    if cache_so3_dir is not None:
        model_config["sdes"]["node_orientations"]["cache_dir"] = cache_so3_dir

    # Load score model
    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)

    # Load finetune model
    if "finetune_model" not in model_config:
        raise ValueError(
            "Model config must contain 'finetune_model' to use this function. "
            "If you want to use a pretrained model, please set `model_name`."
        )
    logger.info("Detected finetune model in config, will use controlled diffusion model.")
    finetune_model: DiGConditionalScoreModel = hydra.utils.instantiate(
        model_config["finetune_model"]
    )
    finetune_model.apply(initialize_weights_to_near_zero)
    if finetune_ckpt_path is not None:
        logger.info(f"Loading finetune model from {finetune_ckpt_path}.")
        finetune_model.load_state_dict(
            torch.load(finetune_ckpt_path, map_location="cpu", weights_only=True)
        )

    # Load denoiser
    if denoiser_config_path is None:
        if denoiser_type not in SUPPORTED_FINETUNE_DENOISERS:
            raise ValueError(f"denoiser_type must be one of {SUPPORTED_FINETUNE_DENOISERS}")
        denoiser_config_path = DEFAULT_FINETUNE_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
    with open(denoiser_config_path, "r") as f:
        denoiser_config = yaml.safe_load(f)
    denoiser: Callable = hydra.utils.instantiate(denoiser_config)

    # Load h function
    if h_func_config_path is None:
        if h_func_type not in SUPPORTED_H_FUNCS:
            raise ValueError(f"h_func_type must be one of {SUPPORTED_H_FUNCS}")
        h_func_config_path = DEFAULT_H_FUNC_CONFIG_DIR / f"{h_func_type}.yaml"
    with open(h_func_config_path, "r") as f:
        h_func_config = yaml.safe_load(f)
    h_func: Callable = hydra.utils.instantiate(h_func_config)

    sdes: SDEs = hydra.utils.instantiate(model_config["sdes"])

    return FinetuneBundle(
        sdes=sdes,
        score_model=score_model,
        finetune_model=finetune_model,
        denoiser=denoiser,
        h_func=h_func,
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
            device=self.device, dtype=torch.float32
        )

        return sequence, h_stars


def create_dataloader(
    csv_path: str | Path,
    sequence_col: str,
    h_stars_cols: str | list[str],
    data_batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    device: DeviceLikeType | None = None,
) -> DataLoader:
    """
    Create DataLoaders for training data.

    Args:
        csv_path: Path to CSV file
        sequence_col: Name of the sequence column
        h_stars_cols: List of h_stars column names
        data_batch_size: Batch size for data
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        device: Device to load data onto
    """

    dataset = SequenceHStarsDataset(
        csv_path=csv_path,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=data_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )

    return dataloader


@clean_gpu_cache
@torch.no_grad()
def generate_finetune_batch(
    sequence: str,
    finetune_bundle: FinetuneBundle,
    batch_size: int,
    device: DeviceLikeType | None = None,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    seed: int | None = None,
) -> DenoisedSDEPath:
    """Generate one batch of samples, using GPU if available.

    Args:
        sequence: Amino acid sequence.
        finetune_bundle: Bundle containing SDEs, score model, finetune model, denoiser, and h_func.
        batch_size: Batch size.
        device: Device to use for sampling. If not set, this defaults to the current device.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        msa_file: Optional path to an MSA A3M file.
        msa_host_url: MSA server URL for colabfold.
        seed: Random seed.
    """

    if seed is not None:
        torch.manual_seed(seed)

    chemgraph = generate_chemgraph(
        sequence=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    batch = Batch.from_data_list([chemgraph for _ in range(batch_size)])

    sdes, score_model, finetune_model, denoiser, h_func = finetune_bundle

    return denoiser(
        batch=batch,
        sdes=sdes,
        score_model=score_model,
        finetune_model=finetune_model,
        device=device,
    )


def _chunk_update(
    batches: list[ChemGraph],
    timesteps: torch.Tensor,
    dts: torch.Tensor,
    dWs_batch: dict[str, torch.Tensor],
    int_u_u_dt_sg: torch.Tensor,
    hs: torch.Tensor,
    h_stars: torch.Tensor,
    finetune_model: DiGConditionalScoreModel,
    fields: list[str],
    batch_size: int,
    device: DeviceLikeType | None = None,
    lambda_: float = 0.1,
    tol: float = 1e-7,
):

    us_list_batch = defaultdict(list)
    for i, batch in enumerate(batches):
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
    loss_ev = compute_ev_loss(
        ws=int_dws, hs=hs, h_stars=h_stars, from_int_dws=True, use_stab=True, tol=tol
    )
    loss_kl = compute_kl_loss(
        ws=int_dws,
        int_u_u_dt=int_u_u_dt,
        int_u_u_dt_sg=int_u_u_dt_sg,
        from_int_dws=True,
        use_rloo=True,
    )

    loss = loss_ev + lambda_ * loss_kl

    loss.backward()


def compute_finetune_loss(
    sequence: str,
    h_stars: torch.Tensor,
    finetune_bundle: FinetuneBundle,
    denoised_sde_path: DenoisedSDEPath,
    batch_size: int,
    device: DeviceLikeType | None = None,
    for_grad: bool = True,
    micro_batch_size: int = 1,
    lambda_: float = 0.1,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the loss for the fine-tuning process."""

    check_protein_valid(sequence)
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 for estimating variances.")

    sdes, score_model, finetune_model, denoiser, h_func = finetune_bundle
    batches, timesteps, us_batch_sg, dWs_batch = denoised_sde_path

    fields = list(sdes.keys())  # ["pos", "node_orientations"]

    ### DDR
    output_dir = Path('ddr_debug')
    os.makedirs(output_dir, exist_ok=True)
    npz_path = output_dir / format_npz_samples_filename(0, batch_size)

    data_cpu = {k: v.cpu().numpy() for k, v in batches[-1].items()}
    data_cpu['pos'] = data_cpu['pos'].reshape(batch_size, -1, 3)
    data_cpu['node_orientations'] = data_cpu['node_orientations'].reshape(batch_size, -1, 3, 3)

    np.savez(npz_path, **data_cpu, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    # we often have multiple distinct seqs
    #if set(sequences) != {sequence}:
        #raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")

    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    node_orientations = torch.tensor(
        np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
    )
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
        filter_samples=False,
    )
    ###

    hs = h_func(batch=batches[-1], sequence=sequence)  # (B, K)
    logger.debug(f"Current h_stars: {h_stars}")
    logger.debug(f"Computed expected value of h function: {hs.mean(dim=0)}")

    dts = torch.diff(timesteps)
    num_steps = len(dts)
    if micro_batch_size > num_steps:
        raise ValueError(
            f"micro_batch_size ({micro_batch_size}) must be less than or equal to num_steps ({num_steps})."
        )

    # Store the stop gradient integral of u for computing aggregated gradient
    # No randomness in the score model inference, so we can compute it once
    us_batch_sg_flattened = {field: us_batch_sg[field].flatten(-2, -1) for field in fields}
    int_u_u_dt_sg = sum(
        compute_int_u_u_dt(us=us_batch_sg_flattened[field], dts=dts) for field in fields
    )  # (B,)
    assert isinstance(int_u_u_dt_sg, torch.Tensor)

    if for_grad:
        num_micro_batches = math.ceil(num_steps / micro_batch_size)

        for i in trange(num_micro_batches, desc="Reverse diffusion", leave=False):
            # Compute the micro-batch indices
            start_idx = i * micro_batch_size
            end_idx = min((i + 1) * micro_batch_size, num_steps)

            _chunk_update(
                batches=batches[start_idx:end_idx],
                timesteps=timesteps[start_idx:end_idx],
                dts=dts[start_idx:end_idx],
                dWs_batch={field: dWs_batch[field][start_idx:end_idx] for field in fields},
                int_u_u_dt_sg=int_u_u_dt_sg,
                hs=hs,
                h_stars=h_stars,
                finetune_model=finetune_model,
                fields=fields,
                batch_size=batch_size,
                device=device,
                lambda_=lambda_,
                tol=tol,
            )

    # Return the real loss (not for gradients) used for validation
    ws = torch.ones_like(int_u_u_dt_sg)  # (B,)
    loss_ev = compute_ev_loss(
        ws=ws, hs=hs, h_stars=h_stars, from_int_dws=False, use_stab=False, tol=tol
    )
    loss_kl = compute_kl_loss(
        ws=ws,
        int_u_u_dt=int_u_u_dt_sg,
        int_u_u_dt_sg=int_u_u_dt_sg,
        from_int_dws=False,
        use_rloo=False,
    )

    logger.info(f"The expected value loss: {loss_ev.item():.4f}")
    logger.info(f"The KL divergence loss: {loss_kl.item():.4f}")

    loss = loss_ev + lambda_ * loss_kl

    logger.info(f"Total loss for sequence '{sequence}': {loss.item():.4f}")

    return loss


def finetune(
    csv_path: str | Path,
    csv_path_val: str | Path,
    sequence_col: str,
    h_stars_cols: str | list[str],
    finetune_bundle: FinetuneBundle,
    finetune_config: FinetuneConfig,
    device: DeviceLikeType | None = None,
    output_dir: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
):
    """Fine-tune the model using the provided data loaders and configuration."""

    # Data parameters
    data_batch_size: int = finetune_config.data_batch_size
    shuffle: bool = finetune_config.shuffle
    num_workers: int = finetune_config.num_workers

    # Loss parameters
    lambda_: float = finetune_config.lambda_
    tol: float = finetune_config.tol

    # Training parameters
    batch_size: int = finetune_config.batch_size
    micro_batch_size: int = finetune_config.micro_batch_size
    num_epochs: int = finetune_config.num_epochs
    save_every_n_epochs: int = finetune_config.save_every_n_epochs
    val_every_n_epochs: int = finetune_config.val_every_n_epochs
    lr: float = finetune_config.lr
    betas: tuple[float, float] = finetune_config.betas
    weight_decay: float = finetune_config.weight_decay
    eta_min: float = finetune_config.eta_min

    # Load datasets and create dataloaders
    dataloader = create_dataloader(
        csv_path=csv_path,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        data_batch_size=data_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        device=device,
    )
    dataloader_val = create_dataloader(
        csv_path=csv_path_val,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        data_batch_size=1,  # validation does not depend on data batch size
        shuffle=False,  # validation should not shuffle
        num_workers=num_workers,
        device=device,
    )
    num_batches = len(dataloader)

    sdes, score_model, finetune_model, denoiser, h_func = finetune_bundle

    # Instantiate SDE and score model
    score_model.to(device).eval()
    finetune_model.to(device)
    sdes["node_orientations"].to(device)

    optimizer = AdamW(finetune_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * num_batches, eta_min=eta_min)

    best_model_state = {}
    best_val_loss = float("inf")

    if output_dir is None:
        output_dir = DEFAULT_MODEL_CHECKPOINT_DIR / "finetune"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs + 1):
        # Train the model
        if epoch > 0:
            finetune_model.train()
            epoch_loss = 0.0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for n_batch, data_batch in enumerate(pbar, start=1):
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=device)
                for sequence, h_stars in data_batch:
                    denoised_sde_path = generate_finetune_batch(
                        sequence=sequence,
                        finetune_bundle=finetune_bundle,
                        batch_size=batch_size,
                        device=device,
                        cache_embeds_dir=cache_embeds_dir,
                        msa_file=msa_file,
                        msa_host_url=msa_host_url,
                        seed=None,  # Use random seed for each batch
                    )
                    loss += compute_finetune_loss(
                        sequence=sequence,
                        h_stars=h_stars,
                        finetune_bundle=finetune_bundle,
                        denoised_sde_path=denoised_sde_path,
                        batch_size=batch_size,
                        device=device,
                        for_grad=True,  # used for training
                        micro_batch_size=micro_batch_size,
                        lambda_=lambda_,
                        tol=tol,
                    )

                optimizer.step()
                scheduler.step()

                l = loss.detach().item()
                epoch_loss += l
                pbar.set_postfix(loss=f"{l:.2f}")

                avg_loss = epoch_loss / n_batch
                logger.info(f"Epoch {epoch}: Running average training loss = {avg_loss:.4f}")

        # Validation step
        if epoch % val_every_n_epochs == 0 or epoch == num_epochs:
            finetune_model.eval()
            epoch_val_loss = 0.0
            avg_val_loss = 0.0

            pbar_val = tqdm(dataloader_val, desc=f"Validation Epoch {epoch}")
            with torch.no_grad():
                for n_batch_val, val_data_batch in enumerate(pbar_val, start=1):
                    val_loss = torch.tensor(0.0, device=device)
                    for sequence, h_stars in val_data_batch:
                        denoised_sde_path = generate_finetune_batch(
                            sequence=sequence,
                            finetune_bundle=finetune_bundle,
                            batch_size=batch_size,
                            device=device,
                            cache_embeds_dir=cache_embeds_dir,
                            msa_file=msa_file,
                            msa_host_url=msa_host_url,
                            seed=None,  # Use random seed for each batch
                        )
                        val_loss += compute_finetune_loss(
                            sequence=sequence,
                            h_stars=h_stars,
                            finetune_bundle=finetune_bundle,
                            denoised_sde_path=denoised_sde_path,
                            batch_size=batch_size,
                            device=device,
                            for_grad=False,  # Used for validation
                            micro_batch_size=1,  # validation does not depend on micro_batch_size
                            lambda_=lambda_,
                            tol=tol,
                        )

                    val_l = val_loss.detach().item()
                    epoch_val_loss += val_l
                    pbar_val.set_postfix(val_loss=f"{val_l:.2f}")

                    avg_val_loss = epoch_val_loss / n_batch_val
                    logger.info(
                        f"Epoch {epoch}: Running average validation loss = {avg_val_loss:.4f}"
                    )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = finetune_model.state_dict()
                logger.info(f"Updated best model at epoch {epoch}")

        # Save the model periodically
        if epoch % save_every_n_epochs == 0 or epoch == num_epochs:
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
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and denoiser
    finetune_bundle = load_finetune_bundle(
        model_name=model_name,
        ckpt_path=ckpt_path,
        finetune_ckpt_path=finetune_ckpt_path,
        model_config_path=model_config_path,
        denoiser_type=denoiser_type,
        denoiser_config_path=denoiser_config_path,
        h_func_type=h_func_type,
        h_func_config_path=h_func_config_path,
        cache_so3_dir=cache_so3_dir,
    )

    # Load finetune config
    if finetune_config_path is None:
        finetune_config_path = DEFAULT_FINETUNE_CONFIG_DIR / "finetune.yaml"
    with open(finetune_config_path, "r") as f:
        finetune_config_yaml = yaml.safe_load(f)
    finetune_config: FinetuneConfig = hydra.utils.instantiate(finetune_config_yaml)

    finetune(
        csv_path=csv_path,
        csv_path_val=csv_path_val,
        sequence_col=sequence_col,
        h_stars_cols=h_stars_cols,
        finetune_bundle=finetune_bundle,
        finetune_config=finetune_config,
        device=device,
        output_dir=output_dir,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)
