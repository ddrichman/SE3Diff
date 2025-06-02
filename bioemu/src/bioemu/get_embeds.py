# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import glob
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

StrPath = str | os.PathLike


COLABFOLD_INSTALL_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "colabfold_setup", "setup.sh"
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def shahexencode(s: str) -> str:
    """Simple sha256 string encoding"""
    return hashlib.sha256(s.encode()).hexdigest()


def write_fasta(seqs: list[str], fasta_file: StrPath, ids: list[str] | None = None) -> None:
    """Writes sequences in `seqs` in FASTA format

    Args:
        seqs (list[str]): Sequences in 1-letter amino-acid code
        fasta_file (StrPath): Destination FASTA file
    """
    Path(fasta_file).parent.mkdir(parents=True, exist_ok=True)
    if ids is None:
        ids = list(range(len(seqs)))  # type: ignore

    seq_records = [SeqRecord(seq=Seq(seq), id=str(_id)) for _id, seq in zip(ids, seqs)]  # type: ignore
    with open(fasta_file, "w") as fasta_handle:
        SeqIO.write(seq_records, fasta_handle, format="fasta")


def merge_a3ms(input_paths: list[StrPath], output_path: StrPath) -> None:
    """
    Merge multiple A3M files for the same query into a single A3M.

    Steps:
      1) Write the first file in its entirety (query header + sequence + hits).
      2) For each subsequent file, skip its first two lines (query header + sequence)
         and append the remainder (all hit-alignment lines) to `output_path`.
    """

    with open(output_path, "w") as out_handle:
        # 1) Copy the first A3M file completely:
        for i, a3m_file in enumerate(input_paths):
            with open(a3m_file, "r") as handle:
                if i > 0:
                    # Skip the first two lines (query header + query sequence) for non-first files
                    next(handle)
                    next(handle)
                # Write/append the rest of the file
                for line in handle:
                    out_handle.write(line)


def replace_query_in_a3m(a3m_file: StrPath, new_seq: str) -> None:
    """Replace the query sequence in an A3M file with a new sequence.
    Args:
        a3m_file (StrPath): Path to the original A3M file.
        new_seq (str): The new query sequence to replace the old one.
    """

    with open(a3m_file, "r") as a3m_handle:
        lines = a3m_handle.readlines()

    if len(lines) < 2:
        raise ValueError(f"{a3m_file} appears too short to be a valid A3M.")

    # Replace the query sequence (line 2) with the new sequence
    lines[1] = f"{new_seq}\n"

    with open(a3m_file, "w") as a3m_handle:
        a3m_handle.writelines(lines)


def _get_colabfold_dir() -> StrPath:
    """
    Get colabfold environment folder
    """
    return os.environ.get(
        "BIOEMU_COLABFOLD_DIR", os.path.join(os.path.expanduser("~"), ".bioemu_colabfold")
    )


def ensure_colabfold_install() -> str:
    """
    Ensures colabfold is installed under the `colabfold_envname` environment
    """
    colabfold_dir = _get_colabfold_dir()
    colabfold_batch_exec = os.path.join(colabfold_dir, "bin", "colabfold_batch")
    colabfold_patched_file = os.path.join(colabfold_dir, ".COLABFOLD_PATCHED")
    colabfold_bin_dir = os.path.dirname(colabfold_batch_exec)
    if os.path.exists(colabfold_batch_exec):
        # Colabfold present
        # Check whether it's been patched
        assert os.path.exists(colabfold_patched_file), "Colabfold not patched!"
    else:
        logger.info(f"Colabfold not present under {colabfold_dir}. Installing...")
        result = subprocess.run(
            ["bash", COLABFOLD_INSTALL_SCRIPT, sys.executable, colabfold_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # dump log into a log file
        log_file_path = f"{colabfold_dir}/install_log.txt"
        with open(log_file_path, mode="wb") as log_file:
            log_file.write(result.stdout)
        logger.info(f"Install log saved in {log_file_path}.")
        assert result.returncode == 0, (
            f"Something went wrong during colabfold install:\n{result.stdout.decode()}\n"
            f"Please check the colabfold install log saved in {log_file_path}."
        )
    return colabfold_bin_dir


def _get_default_embeds_dir() -> StrPath:
    """Returns the directory where precomputed embeddings are stored"""
    return os.path.join(os.path.expanduser("~"), ".bioemu_embeds_cache")


def run_colabfold(
    input_file: StrPath,
    res_dir: StrPath,
    colabfold_env: dict[str, str],
    msa_host_url: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Runs colabfold.
    Args:
        input_file: Input file path. It can be either a fasta or a3m file. If it is a .fasta file,
            colabfold will retrieve an MSA from an MSA server. If it is a .a3m file,
            the MSA in the .a3m file will be used instead.
        res_dir: Directory to store results.
        colabfold_env: Environment variables for colabfold.
        msa_host_url: MSA host URL. If None, defaults to the colabfold default, which is a remote server.
    """

    assert str(input_file).endswith(".fasta") or str(input_file).endswith(".a3m")

    cmd = [
        "colabfold_batch",
        input_file,
        res_dir,
        "--num-models",
        "1",
        "--model-order",
        "3",
        "--model-type",
        "alphafold2",
        "--num-recycle",
        "0",
        "--save-single-representations",
        "--save-pair-representations",
    ]
    if msa_host_url is not None:
        cmd.extend(["--host-url", msa_host_url])
    return subprocess.run(cmd, env=colabfold_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def get_colabfold_embeds(
    seq: str,
    cache_embeds_dir: StrPath | None,
    msa_file: StrPath | None = None,
    msa_host_url: str | None = None,
) -> tuple[StrPath, StrPath]:
    """
    Uses colabfold to retrieve embeddings for a given sequence. If the embeddings are already stored under `cache_embeds_dir`,
    the cached embeddings are returned. Otherwise, colabfold is used to compute the embeddings and they are saved under `cache_embeds_dir`.
    Optionally uses an MSA A3M file if provided.

    Args:
        seq: Protein sequence to query
        cache_embeds_dir: Cache directory where embeddings will be stored. If None, defaults to a child of the colabfold install directory.
        msa_file: MSA A3M file to use as input. If None, the sequence is used as input.
        msa_host_url: MSA host URL. If None, defaults to the colabfold default, which is a remote server.

    Returns:
        Tuple of paths to single and pair embeddings.
    """
    seqsha = shahexencode(seq)

    # Setup embedding cache
    cache_embeds_dir = cache_embeds_dir or _get_default_embeds_dir()
    cache_embeds_dir = os.path.expanduser(cache_embeds_dir)
    os.makedirs(cache_embeds_dir, exist_ok=True)

    # Check whether embeds have already been computed
    single_rep_file = os.path.join(cache_embeds_dir, f"{seqsha}_single.npy")
    pair_rep_file = os.path.join(cache_embeds_dir, f"{seqsha}_pair.npy")

    if os.path.exists(single_rep_file) and os.path.exists(pair_rep_file):
        logger.info(f"Using cached embeddings in {cache_embeds_dir}.")
        return single_rep_file, pair_rep_file

    # If we don't already have embeds, run colabfold
    colabfold_bin_dir = ensure_colabfold_install()

    colabfold_env = os.environ.copy()
    colabfold_env["PATH"] = f'{colabfold_bin_dir}:{colabfold_env["PATH"]}'
    # Delete MPLBACKEND to avoid matplotlib issues when running in jupyter notebook
    colabfold_env.pop("MPLBACKEND", None)

    with tempfile.TemporaryDirectory() as tempdir:
        fasta_file = os.path.join(tempdir, f"{seqsha}.fasta")
        res_dir = os.path.join(tempdir, "results")
        os.makedirs(res_dir, exist_ok=True)
        write_fasta(seqs=[seq], fasta_file=fasta_file, ids=[seqsha])
        if msa_file is not None:
            logger.info(
                "Using user provided MSAs. This might result in suboptimal performance of model generated distributions!\n"
                "BioEmu has been using MSAs from the ColabFold MSA server, or following:\n"
                "https://github.com/sokrypton/ColabFold?tab=readme-ov-file#generating-msas-for-large-scale-structurecomplex-predictions.\n"
                "If your MSA is generated differently, the generated results could be different."
            )
            msa_file = Path(msa_file).expanduser().resolve()
            replace_query_in_a3m(a3m_file=msa_file, new_seq=seq)
            res = run_colabfold(msa_file, res_dir, colabfold_env)
            embed_prefix = msa_file.stem
        else:
            res = run_colabfold(fasta_file, res_dir, colabfold_env, msa_host_url)
            embed_prefix = f"{seqsha}__unknown_description_"
            msa_dir = os.path.join(res_dir, f"{embed_prefix}_env")
            msa_file = os.path.join(res_dir, f"{seqsha}.a3m")
            a3m_list = glob.glob(os.path.join(msa_dir, "*.a3m"))
            merge_a3ms(input_paths=a3m_list, output_path=msa_file)  # type: ignore
        if res.returncode != 0:
            raise RuntimeError(
                f"{res.stdout.decode()}\nFailed to run colabfold_batch due to the above error."
            )

        single_rep_tempfile = os.path.join(
            res_dir,
            f"{embed_prefix}_single_repr_evo_rank_001_alphafold2_model_3_seed_000.npy",
        )
        pair_rep_tempfile = os.path.join(
            res_dir,
            f"{embed_prefix}_pair_repr_evo_rank_001_alphafold2_model_3_seed_000.npy",
        )
        pdb_tempfile = os.path.join(
            res_dir, f"{embed_prefix}_unrelaxed_rank_001_alphafold2_model_3_seed_000.pdb"
        )
        shutil.copy(single_rep_tempfile, single_rep_file)
        shutil.copy(pair_rep_tempfile, pair_rep_file)
        # Just to be friendly, we also keep a .fasta as a human-readable record of what sequence this is for.
        shutil.copy(fasta_file, os.path.join(cache_embeds_dir, f"{seqsha}.fasta"))
        shutil.copy(msa_file, os.path.join(cache_embeds_dir, f"{seqsha}.a3m"))
        shutil.copy(pdb_tempfile, os.path.join(cache_embeds_dir, f"{seqsha}.pdb"))

    return single_rep_file, pair_rep_file
