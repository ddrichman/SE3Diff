# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import shutil
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple

import fire
import hydra
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch._prims_common import DeviceLikeType
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.denoiser import SDEs
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.models import DiGConditionalScoreModel
from bioemu.seq_io import check_protein_valid, parse_sequence, write_fasta
from bioemu.utils import (
    clean_gpu_cache,
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)

logger = logging.getLogger(__name__)

# Denoiser config
DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser"
SupportedDenoisersLiteral = Literal["dpm", "heun", "euler_maruyama"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

# Model checkpoint directory
DEFAULT_MODEL_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
SupportedModelNamesLiteral = Literal["bioemu-v1.0", "bioemu-rev"]
SUPPORTED_MODEL_NAMES = list(typing.get_args(SupportedModelNamesLiteral))


class Bundle(NamedTuple):
    sdes: SDEs
    score_model: DiGConditionalScoreModel
    denoiser: Callable


def maybe_download_checkpoint(
    *,
    model_name: SupportedModelNamesLiteral | None,
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
) -> tuple[str, str]:
    """If ckpt_path and model config_path are specified, return them, else download named model from huggingface.
    Returns:
        tuple[str, str]: path to checkpoint, path to model config
    """

    if model_name is None:
        logger.info("No model_name specified, using ckpt_path and model_config_path.")
        if ckpt_path is None:
            raise ValueError("If model_name is not specified, you must provide ckpt_path.")
        if model_config_path is None:
            raise ValueError("If model_name is not specified, you must provide model_config_path.")

        ckpt_path = Path(ckpt_path).expanduser().resolve()
        model_config_path = Path(model_config_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise ValueError(f"Checkpoint {ckpt_path} does not exist.")
        if not model_config_path.is_file():
            raise ValueError(f"Model config {model_config_path} does not exist.")

        return str(ckpt_path), str(model_config_path)

    ckpt_path_default = DEFAULT_MODEL_CHECKPOINT_DIR / model_name / "checkpoint.ckpt"
    ckpt_path_download = hf_hub_download(
        repo_id="microsoft/bioemu",
        filename=f"checkpoints/{model_name}/checkpoint.ckpt",
    )
    model_config_path_default = DEFAULT_MODEL_CHECKPOINT_DIR / model_name / "config.yaml"
    model_config_path_download = hf_hub_download(
        repo_id="microsoft/bioemu",
        filename=f"checkpoints/{model_name}/config.yaml",
    )

    ckpt_path = Path(ckpt_path_default) if ckpt_path is None else Path(ckpt_path)
    if not ckpt_path.is_file():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt_path_download, ckpt_path)
        logger.info(f"Copied checkpoint to {ckpt_path}")
    model_config_path = (
        Path(model_config_path_default) if model_config_path is None else Path(model_config_path)
    )
    if not model_config_path.is_file():
        model_config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_config_path_download, model_config_path)
        logger.info(f"Copied model config to {model_config_path}")

    return str(ckpt_path), str(model_config_path)


def load_bundle(
    model_name: SupportedModelNamesLiteral | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
) -> Bundle:
    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    if cache_so3_dir is not None:
        model_config["sdes"]["node_orientations"]["cache_dir"] = cache_so3_dir

    model_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    score_model: DiGConditionalScoreModel = hydra.utils.instantiate(model_config["score_model"])
    score_model.load_state_dict(model_state)
    sdes: SDEs = hydra.utils.instantiate(model_config["sdes"])

    if denoiser_config_path is None:
        if denoiser_type not in SUPPORTED_DENOISERS:
            raise ValueError(f"denoiser_type must be one of {SUPPORTED_DENOISERS}")
        denoiser_config_path = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"

    with open(denoiser_config_path) as f:
        denoiser_config = yaml.safe_load(f)
    denoiser: Callable = hydra.utils.instantiate(denoiser_config)

    return Bundle(sdes=sdes, score_model=score_model, denoiser=denoiser)


def generate_chemgraph(
    *,
    sequence: str,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
) -> ChemGraph:
    seq_len = len(sequence)

    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
    )
    single_embeds = np.load(single_embeds_file)
    pair_embeds = np.load(pair_embeds_file)
    _, _, n_pair_feats = pair_embeds.shape  # [seq_len, seq_len, n_pair_feats]

    single_embeds, pair_embeds = torch.from_numpy(single_embeds), torch.from_numpy(pair_embeds)
    pair_embeds = pair_embeds.view(seq_len**2, n_pair_feats)

    edge_index = torch.cat(
        [
            torch.arange(seq_len).repeat_interleave(seq_len).view(1, seq_len**2),
            torch.arange(seq_len).repeat(seq_len).view(1, seq_len**2),
        ],
        dim=0,
    )
    pos = torch.full((seq_len, 3), float("nan"))
    node_orientations = torch.full((seq_len, 3, 3), float("nan"))

    chemgraph = ChemGraph(
        pos=pos,
        node_orientations=node_orientations,
        edge_index=edge_index,
        single_embeds=single_embeds,
        pair_embeds=pair_embeds,
    )

    return chemgraph


@clean_gpu_cache
@torch.no_grad()
def generate_batch(
    sequence: str,
    bundle: Bundle,
    batch_size: int,
    device: DeviceLikeType | None = None,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate one batch of samples, using GPU if available.

    Args:
        sequence: Amino acid sequence.
        sdes: SDEs defining corruption process. Keys should be 'node_orientations' and 'pos'.
        score_model: Score model.
        finetune_model: Finetune model.
        denoiser: Denoiser function.
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

    sdes, score_model, denoiser = bundle

    sampled_chemgraph_batch = denoiser(
        batch=batch,
        sdes=sdes,
        score_model=score_model,
        device=device,
    )

    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu")
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

    return {"pos": pos, "node_orientations": node_orientations}


def sample(
    sequence: str | Path,
    output_dir: str | Path,
    num_samples: int,
    bundle: Bundle,
    device: DeviceLikeType | None = None,
    batch_size_100: int = 10,
    cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
):

    # Parse FASTA or A3M file if sequence is a file path. Extract the actual sequence.
    sequence = parse_sequence(sequence)

    # Check input sequence is valid
    check_protein_valid(sequence)

    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # Fail fast if output_dir is non-writeable

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(
                f"{fasta_path} already exists, but contains a sequence different from {sequence}!"
            )
    else:
        # Save FASTA file in output_dir
        write_fasta([sequence], fasta_path)

    logger.info(
        f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues..."
    )
    batch_size = int(batch_size_100 * (100 / len(sequence)) ** 2)
    if batch_size == 0:
        logger.warning(f"Sequence {sequence} may be too long. Attempting with batch_size = 1.")
        batch_size = 1
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")

    for seed in tqdm(
        range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."
    ):
        n = min(batch_size, num_samples - seed)
        npz_path = output_dir / format_npz_samples_filename(seed, n)
        if npz_path.exists():
            raise ValueError(
                f"Not sure why {npz_path} already exists when so far only {existing_num_samples} samples have been generated."
            )
        logger.info(f"Sampling {seed=}")
        batch = generate_batch(
            sequence=sequence,
            bundle=bundle,
            batch_size=min(batch_size, n),
            device=device,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            seed=seed,
        )
        np.savez(npz_path, **{k: v.cpu().numpy() for k, v in batch.items()}, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    if set(sequences) != {sequence}:
        raise ValueError(f"Expected all sequences to be {sequence}, but got {set(sequences)}")
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    node_orientations = torch.tensor(
        np.concatenate([np.load(f)["node_orientations"] for f in samples_files])
    )
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        sequence=sequence,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        filter_samples=filter_samples,
    )
    logger.info(f"Completed. Your samples are in {output_dir}.")


@print_traceback_on_exception
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    model_name: SupportedModelNamesLiteral | None = "bioemu-v1.0",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config_path: str | Path | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
) -> None:
    """
    Generate samples for a specified sequence, using a trained model.

    Args:
        sequence: Amino acid sequence for which to generate samples or path to a FASTA or A3M file containing the sequence.
        num_samples: Number of samples to generate. If `output_dir` already contains samples, this function will only generate additional samples necessary to reach the specified `num_samples`.
        output_dir: Directory to save the samples. Each batch of samples will initially be dumped as .npz files. Once all batches are sampled, they will be converted to .xtc and .pdb.
        batch_size_100: Batch size you'd use for a sequence of length 100. The batch size will be calculated from this, assuming
           that the memory requirement to compute each sample scales quadratically with the sequence length.
        model_name: Name of pretrained model to use. The model will be retrieved from huggingface. If not set,
           this defaults to `bioemu-v1.0`. If this is set, you do not need to provide `ckpt_path` or `model_config_path`.
        ckpt_path: Path to the model checkpoint. If this is set, `model_name` will be ignored.
        model_config_path: Path to the model config, defining score model architecture and the corruption process the model was trained with.
           Only required if `ckpt_path` is set.
        denoiser_type: Denoiser to use for sampling, if `denoiser_config_path` not specified. Comes in with default parameter configuration. Must be one of ['dpm', 'heun']
        denoiser_config_path: Path to the denoiser config, defining the denoising process.
        cache_embeds_dir: Directory to store MSA embeddings. If not set, this defaults to `COLABFOLD_DIR/embeds_cache`.
        cache_so3_dir: Directory to store SO3 precomputations. If not set, this defaults to `~/sampling_so3_cache`.
        msa_file: Optional path to an MSA A3M file. If this is set, the MSA embeddings will be used for sampling.
        msa_host_url: MSA server URL. If not set, this defaults to colabfold's remote server. If sequence is an a3m file, this is ignored.
        filter_samples: Filter out unphysical samples with e.g. long bond distances or steric clashes.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_bundle(
        model_name=model_name,
        ckpt_path=ckpt_path,
        model_config_path=model_config_path,
        denoiser_type=denoiser_type,
        denoiser_config_path=denoiser_config_path,
        cache_so3_dir=cache_so3_dir,
    )

    sample(
        sequence=sequence,
        output_dir=output_dir,
        num_samples=num_samples,
        bundle=bundle,
        device=device,
        batch_size_100=batch_size_100,
        cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file,
        msa_host_url=msa_host_url,
        filter_samples=filter_samples,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fire.Fire(main)
