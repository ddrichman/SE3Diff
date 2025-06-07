import numpy as np
import torch
import pandas as pd
from torch.distributions import Normal
from torch_geometric.data.batch import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from functools import lru_cache

from Bio.PDB import PDBParser, MMCIFParser

from .chemgraph import ChemGraph
from .sde_lib import SDE, CosineVPSDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat, rotmat_to_rotvec

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, List

import torch
from Bio import pairwise2


@dataclass
class FNCSettings:
    """
    Data class for collecting fraction of native contact settings.

    Attributes:
        sequence_separation: Minimum separation of residues for which contacts are computed.
        contact_cutoff: Maximum cutoff distance (in Angstrom) used for contact computation.
        contact_beta: Scaling factor for contact score computations (see `_compute_contact_score`).
        contact_delta: Offset for contact score computations (see `_compute_contact_score`).
        contact_lambda: Scaling off offset in contact score computation (see
          `_compute_contact_score`).
    """

    sequence_separation: int = 3
    contact_cutoff: float = 10.0
    contact_beta: float = 5.0
    contact_delta: float = 0.0
    contact_lambda: float = 1.2


def _compute_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between all atoms.
    
    Args:
        coords: Tensor of shape [n_atoms, 3] containing atomic coordinates.
        
    Returns:
        Tensor of shape [n_atoms, n_atoms] containing pairwise distances.
    """
    # Compute squared distances using broadcasting
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [n_atoms, n_atoms, 3]
    distances = torch.sqrt(torch.sum(diff**2, dim=-1))  # [n_atoms, n_atoms]
    return distances

@lru_cache(maxsize=10)
def _compute_reference_contacts(
    reference_coords: torch.Tensor,
    sequence_separation: int,
    contact_cutoff: float,
    device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute native contacts in reference conformation based on given parameters.

    Args:
        reference_coords: Reference coordinates tensor of shape [n_atoms, 3].
        sequence_separation: Minimum separation between residues in sequence to be considered for
          contact computation.
        contact_cutoff: Distance cutoff. Only contacts between atoms below this cutoff are
          considered.
        device: PyTorch device to use for computations.

    Returns:
        Tuple of tensors containing the contact indices and contact distances.
    """
    if device is None:
        device = reference_coords.device
    
    n_atoms = reference_coords.shape[0]
    
    # Generate all possible pairs with sufficient sequence separation
    valid_pairs = []
    for i in range(n_atoms):
        for j in range(i + sequence_separation + 1, n_atoms):
            valid_pairs.append((i, j))
            # Make symmetric for per-residue resolution of contacts
            valid_pairs.append((j, i))
    
    if not valid_pairs:
        # Return empty tensors if no valid pairs
        return (torch.empty((0, 2), dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device))
    
    # Convert to tensor
    pair_indices = torch.tensor(valid_pairs, dtype=torch.long, device=device)
    
    # Compute distances for all pairs
    distances = torch.norm(
        reference_coords[pair_indices[:, 0]] - reference_coords[pair_indices[:, 1]], 
        dim=1
    )
    
    # Filter according to cutoff
    mask = distances <= contact_cutoff
    filtered_pairs = pair_indices[mask]
    filtered_distances = distances[mask]
    
    return filtered_pairs, filtered_distances


def _get_aligned_indices(seq_alignment_1: str, seq_alignment_2: str) -> list[int]:
    """
    Compute the indices of the aligned residues in sequence 1 without gaps in the alignment.

    E.g. seq1=ABCDE, seq2=GABDF, then seq_alignment_1='-ABCDE-', seq_alignment_2='GAB-D-F'.
    The dashes are gaps and not counted when incrementing the index.

    Args:
        seq_alignment_1: First of the aligned sequences.
        seq_alignment_2: Second sequence.

    Returns:
        List of indices.
    """
    aligned_indices = []
    n = 0
    for i, s in enumerate(seq_alignment_1):
        if s != "-":
            if seq_alignment_2[i] != "-":
                aligned_indices.append(n)
            n += 1
    return aligned_indices

@lru_cache(maxsize=10)
def _get_sequence_index_map(
    samples_sequence: str, 
    reference_sequence: str, 
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Perform pairwise alignment of reference and sample sequence to get the indices mapping reference
    atoms to sample atoms.

    Args:
        samples_sequence: Sequence of sample structures.
        reference_sequence: Reference sequence.
        device: PyTorch device to use for computations.

    Returns:
        Tensor mapping reference atom indices to sample atom indices. Atoms with no mapping are
        assigned -1.
    """
    if device is None:
        device = torch.device('cpu')
        
    alignments = pairwise2.align.globalxx(samples_sequence, reference_sequence)
    aligned_indices_sample = _get_aligned_indices(alignments[0].seqA, alignments[0].seqB)
    aligned_indices_ref = _get_aligned_indices(alignments[0].seqB, alignments[0].seqA)
    assert len(aligned_indices_sample) == len(aligned_indices_ref)

    ref_to_samples_map = torch.full((max(aligned_indices_ref) + 1,), -1, 
                                   dtype=torch.long, device=device)
    
    for ref, smp in zip(aligned_indices_ref, aligned_indices_sample):
        ref_to_samples_map[ref] = smp

    return ref_to_samples_map


def _compute_contact_distances_batch(
    samples_coords: torch.Tensor,
    contact_pairs: torch.Tensor
) -> torch.Tensor:
    """
    Compute contact distances for all samples and contact pairs.
    
    Args:
        samples_coords: Tensor of shape [n_samples, n_atoms, 3].
        contact_pairs: Tensor of shape [n_contacts, 2] containing atom pair indices.
        
    Returns:
        Tensor of shape [n_samples, n_contacts] containing contact distances.
    """
    # Extract coordinates for each pair
    coords1 = samples_coords[:, contact_pairs[:, 0], :]  # [n_samples, n_contacts, 3]
    coords2 = samples_coords[:, contact_pairs[:, 1], :]  # [n_samples, n_contacts, 3]
    
    # Compute distances
    distances = torch.norm(coords1 - coords2, dim=-1)  # [n_samples, n_contacts]
    
    return distances


def _compute_contact_score(
    samples_contact_distances: torch.Tensor,
    reference_contact_distances: torch.Tensor,
    contact_delta: float,
    contact_beta: float,
    contact_lambda: float,
) -> torch.Tensor:
    """
    Compute contact scores for all pairs of contacts using the equation:

    .. math::

        q = \frac{1}{N_\mathrm{contacts}} \sum_{c}^{N_\mathrm{contacts}} \frac{1}{1 + \exp(-\beta(d_c - \lambda (d^\mathrm{ref}_c + \delta)))}

    Args:
        samples_contact_distances: Tensor of contact distances in samples with shape
          [num_samples x num_contacts].
        reference_contact_distances: Tensor of reference contact distances with shape [num_contacts].
        contact_delta: Offset for contact score determination.
        contact_beta: Scaling of exponential term.
        contact_lambda: Scaling for reference contact to account for fluctuations.

    Returns:
        Contact scores for all pairwise interactions.
    """
    # Compute the sigmoid (expit equivalent in PyTorch)
    q_ij = torch.sigmoid(
        -contact_beta
        * (
            samples_contact_distances
            - contact_lambda * (reference_contact_distances.unsqueeze(0) + contact_delta)
        )
    )
    return torch.mean(q_ij, dim=-1)


def get_fnc_from_coords(
    samples_coords: torch.Tensor,
    reference_coords: torch.Tensor,
    samples_sequence: Optional[str] = None,
    reference_sequence: Optional[str] = None,
    sequence_separation: int = 3,
    contact_cutoff: float = 10.0,
    contact_beta: float = 5.0,
    contact_lambda: float = 0.0,
    contact_delta: float = 1.2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute fraction of native contact scores for samples using coordinate tensors.

    Args:
        samples_coords: Sample coordinates tensor of shape [n_samples, n_atoms, 3].
        reference_coords: Reference coordinates tensor of shape [n_atoms, 3].
        samples_sequence: Optional sequence string for sample structures (for alignment).
        reference_sequence: Optional sequence string for reference structure (for alignment).
        sequence_separation: Minimum separation of residues for which contacts are computed.
        contact_cutoff: Maximum cutoff distance (in Angstrom) used for contact computation.
        contact_beta: Scaling factor for contact score computations.
        contact_delta: Offset for contact score computations.
        contact_lambda: Scaling off offset in contact score computation.
        device: PyTorch device to use for computations.

    Returns:
        Tensor containing fraction of native contacts score for each sample.
    """
    if device is None:
        device = samples_coords.device
    
    # Move tensors to device if needed
    samples_coords = samples_coords.to(device)
    reference_coords = reference_coords.to(device)

    # Compute reference contacts
    reference_contact_pairs, reference_contact_distances = _compute_reference_contacts(
        reference_coords=reference_coords,
        sequence_separation=sequence_separation,
        contact_cutoff=contact_cutoff,
        device=device,
    )
    
    # If sequences are provided, perform alignment
    if samples_sequence is not None and reference_sequence is not None:
        ref_to_samples_map = _get_sequence_index_map(
            samples_sequence, reference_sequence, device=device
        )
        
        # Map contact indices to current samples
        aligned_contact_pairs = ref_to_samples_map[reference_contact_pairs]
        
        # Filter out pairs with unmapped atoms (-1)
        valid_mask = (aligned_contact_pairs >= 0).all(dim=1)
        aligned_contact_pairs = aligned_contact_pairs[valid_mask]
        reference_contact_distances = reference_contact_distances[valid_mask]
        
        contact_pairs = aligned_contact_pairs
    else:
        # Assume same atom ordering
        contact_pairs = reference_contact_pairs
    
    # Handle case where no valid contacts exist
    if contact_pairs.shape[0] == 0:
        return torch.zeros(samples_coords.shape[0], device=device, dtype=torch.float32)
    
    # Compute sample contact distances
    samples_contact_distances = _compute_contact_distances_batch(
        samples_coords, contact_pairs
    )
    
    # Compute contact score
    contact_score = _compute_contact_score(
        samples_contact_distances=samples_contact_distances,
        reference_contact_distances=reference_contact_distances,
        contact_beta=contact_beta,
        contact_lambda=contact_lambda,
        contact_delta=contact_delta,
    )

    return contact_score


def weighted_rigid_align(
    coords,
    ref_coords,
    weights=None,
):
    """Compute weighted alignment without masking.  ADAPTED from Boltz-1 (Wohlwend et al. 2024)
    
    Parameters
    ----------
    coords: torch.Tensor
        The ground truth atom coordinates, shape [N, 3] or [batch_size, N, 3]
    ref_coords: torch.Tensor
        The predicted atom coordinates, shape [N, 3] or [batch_size, N, 3]
    weights: torch.Tensor, optional
        The weights for alignment, shape [N] or [batch_size, N]
        If None, equal weights will be used.
        
    Returns
    -------
    torch.Tensor
        Aligned coordinates with same shape as true_coords
    """
    # Add batch dimension if inputs are not batched
    original_shape = coords.shape
    batched = len(original_shape) > 2
    
    if not batched:
        coords = coords.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        ref_coords = ref_coords.unsqueeze(0)  # [N, 3] -> [1, N, 3]
    
    batch_size, num_points, dim = coords.shape
    
    # Handle weights - use uniform weights if not provided
    if weights is None:
        weights = torch.ones(batch_size, num_points, device=coords.device)
    else:
        if not batched and len(weights.shape) == 1:
            weights = weights.unsqueeze(0)  # [N] -> [1, N]
    
    weights = weights.unsqueeze(-1)  # [batch_size, N] -> [batch_size, N, 1]
    
    # Compute weighted centroids
    true_centroid = (coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (ref_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    
    # Center the coordinates
    true_coords_centered = coords - true_centroid
    pred_coords_centered = ref_coords - pred_centroid
    
    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )
    
    # Compute the weighted covariance matrix
    cov_matrix = torch.einsum(
        "b n i, b n j -> b i j", 
        weights * pred_coords_centered,
        true_coords_centered
    )
    
    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH
    
    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)
    
    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = torch.einsum("b i j, b j k, b l k -> b i l", U, F, V)
    rot_matrix = rot_matrix.to(dtype=original_dtype)
    
    # Apply the rotation and translation
    aligned_coords = (
        torch.einsum("b n i, b j i -> b n j", true_coords_centered, rot_matrix)
        + pred_centroid
    )
    
    # Remove batch dimension if input wasn't batched
    if not batched:
        aligned_coords = aligned_coords.squeeze(0)
    
    return aligned_coords

@lru_cache(maxsize=10)
def load_ref(structure_file):
    """
    Parameters:
    structure_file: Path to CIF/PDB file
    
    Returns:
    residue_coords: Tensor of alpha carbon coordinates (one per residue)
    """
    # Determine file type and parse accordingly
    if structure_file.endswith(".cif"):
        parser = MMCIFParser()
    elif structure_file.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Unsupported file format. Please provide a .cif or .pdb file.")
    structure = parser.get_structure("structure", structure_file)
    
    # Build residue info list (using alpha carbons)
    residue_coords = []
    residue_info = []  # (model_id, chain_id, res_id, res_name)
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ':  # Skip hetero-atoms and waters
                    continue
                res_id = res.id[1]
                # Look for alpha carbon in this residue
                if 'CA' in res:
                    ca_atom = res['CA']
                    residue_coords.append(ca_atom.coord)
                    residue_info.append((model.id, chain.id, res_id, res.resname))
    
    residue_coords = torch.tensor(np.array(residue_coords))

    # Convert to nanometers for BioEmu
    residue_coords = residue_coords / 10.0

    return residue_coords

def h_star_for_grb2_sh3(
    info_path: str
) -> torch.Tensor:
    """
    Return (sequences, h*)
    where sequences is list of sequences
    and h* is (target param) values for the GRB2 SH3 domain.
    h* is of shape (nseq, 2)

    Parameter: info_path is path to CSV. 
    """

    mutants_data = pd.read_csv(info_path)
    
    seq = mutants_data.seq
    h_star = torch.zeros((len(seq), 2))

    # probability of folded state (see Faure et al (2022) Fig 2)
    h_star[:, 0] = 1/(1 + torch.exp(torch.tensor(mutants_data.f_dg_pred)))

    # probability of loop in bound state
    h_star[:, 1] = 1/(1 + torch.exp(torch.tensor(mutants_data.b_dg_pred)))

    return seq, h_star

    

def compute_h_for_grb2_sh3(
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    ref_path: str
) -> torch.Tensor:
    """Compute the h function for the GRB2 SH3 domain.
    """

    assert pos.ndim == 3
    assert node_orientations.ndim == 4

    B = pos.shape[0]

    h = torch.zeros((B, 2)) # 2 observables for GRB2 SH3

    ref_coords = load_ref(ref_path)


    ## For folding
    # Compute native contacts
    reference_contact_pairs, reference_contact_distances = _compute_reference_contacts(
        reference_coords=ref_coords * 10, # nm to A
        sequence_separation=FNCSettings.sequence_separation,
        contact_cutoff=FNCSettings.contact_cutoff,
    )

    samples_contact_distances = _compute_contact_distances_batch(
        pos * 10, # nm to A
        reference_contact_pairs
    )

    contact_score = _compute_contact_score(
        samples_contact_distances=samples_contact_distances,
        reference_contact_distances=reference_contact_distances,
        contact_beta=FNCSettings.contact_beta,
        contact_lambda=FNCSettings.contact_lambda,
        contact_delta=FNCSettings.contact_delta,
    )

    protein_folded_q_threshold = 0.7
    h[:, 0] = contact_score > protein_folded_q_threshold

    ## For binding
    # from definition in the GRB2_SH3 mutants csv
    residues_interface = [6, 8, 11, 12, 15, 31, 33, 34, 36, 45, 47, 49, 50]

    aligned_to_ref = weighted_rigid_align(
        coords=pos[:, residues_interface],
        ref_coords=ref_coords[residues_interface].unsqueeze(0),
    )

    # Compute RMSD of the loop region
    #loop_region = aligned_to_ref[:, 6:21, :]
    #loop_rmsd = torch.sqrt(
    #    ((loop_region - ref_coords[6:21, :]) ** 2).sum(dim=-1).mean(dim=-1)
    #)

    print(aligned_to_ref[0], ref_coords[residues_interface])

    loop_rmsd = torch.sqrt(
        ((aligned_to_ref - ref_coords[residues_interface])**2).sum(dim=-1).mean(dim=-1)
    )
    
    # Classify as folded or unfolded based on RMSD
    loop_folded_threshold = 0.2 # 2A = 0.2 nm threshold for folded state, based on MD sim
    h[:, 1] = (loop_rmsd < loop_folded_threshold).float()
    #h[:, 1] = loop_rmsd
    
    return h

def compute_h_for_grb2_sh3_raw(
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    ref_path: str
) -> torch.Tensor:
    """Compute the h function for the GRB2 SH3 domain.
    """

    assert pos.ndim == 3
    assert node_orientations.ndim == 4

    B = pos.shape[0]

    h = torch.zeros((B, 2)) # 2 observables for GRB2 SH3

    ref_coords = load_ref(ref_path)


    ## For folding
    # Compute native contacts
    reference_contact_pairs, reference_contact_distances = _compute_reference_contacts(
        reference_coords=ref_coords * 10, # nm to A
        sequence_separation=FNCSettings.sequence_separation,
        contact_cutoff=FNCSettings.contact_cutoff,
    )

    samples_contact_distances = _compute_contact_distances_batch(
        pos * 10, # nm to A
        reference_contact_pairs
    )

    contact_score = _compute_contact_score(
        samples_contact_distances=samples_contact_distances,
        reference_contact_distances=reference_contact_distances,
        contact_beta=FNCSettings.contact_beta,
        contact_lambda=FNCSettings.contact_lambda,
        contact_delta=FNCSettings.contact_delta,
    )

    protein_folded_q_threshold = 0.7
    h[:, 0] = contact_score

    ## For binding
    # from definition in the GRB2_SH3 mutants csv
    residues_interface = [6, 8, 11, 12, 15, 31, 33, 34, 36, 45, 47, 49, 50]

    aligned_to_ref = weighted_rigid_align(
        coords=pos[:, residues_interface],
        ref_coords=ref_coords[residues_interface].unsqueeze(0),
    )

    # Compute RMSD of the loop region
    #loop_region = aligned_to_ref[:, 6:21, :]
    #loop_rmsd = torch.sqrt(
    #    ((loop_region - ref_coords[6:21, :]) ** 2).sum(dim=-1).mean(dim=-1)
    #)

    print(aligned_to_ref[0], ref_coords[residues_interface])

    loop_rmsd = torch.sqrt(
        ((aligned_to_ref - ref_coords[residues_interface])**2).sum(dim=-1).mean(dim=-1)
    )
    
    # Classify as folded or unfolded based on RMSD
    loop_folded_threshold = 0.2 # 2A = 0.2 nm threshold for folded state, based on MD sim
    #h[:, 1] = (loop_rmsd < loop_folded_threshold).float()
    h[:, 1] = loop_rmsd
    
    return h


def compute_h_for_grb2_sh3_from_batch(
    chemgraph_batch: Batch,
    ref_path: str
) -> torch.Tensor:
    assert isinstance(chemgraph_batch, Batch)
    pos = to_dense_batch(chemgraph_batch.pos, chemgraph_batch.batch_idx)
    node_orientations = to_dense_batch(chemgraph_batch.node_orientations, chemgraph_batch.batch_idx)

    return compute_h_for_grb2_sh3(pos,
                                node_orientations,
                                ref_path=Path(__file__).parent / '../../../structures/2vwf_trimmed_SH3.pdb')

def h_star_for_grb2_sh3_from_batch(
    sequence: List[str]
) -> torch.Tensor:
    ref_seqs, ref_h_star = h_star_for_grb2_sh3(Path(__file__).parent / '../../../reference_h/GRB2_SH3_high_confidence.csv')

    selected_rows = [ref_seqs.index(s) for s in sequence]
    return ref_h_star[selected_rows]


def compute_h_for_psd95_pdz3(
    pos: torch.Tensor,
    node_orientations: torch.Tensor,
    ref_path: str
) -> torch.Tensor:
    """Compute the h function for the PSD95 PDZ3 domain.
    """

    assert pos.ndim == 3
    assert node_orientations.ndim == 4

    B = pos.shape[0]

    h = torch.zeros((B, 2)) # 2 observables

    ref_coords = load_ref(ref_path)

    aligned_to_ref = weighted_rigid_align(
        coords=pos,
        ref_coords=ref_coords,
    )

    ## For folding
    # Compute native contacts
    reference_contact_pairs, reference_contact_distances = _compute_reference_contacts(
        reference_coords=ref_coords * 10, # nm to A
        sequence_separation=FNCSettings.sequence_separation,
        contact_cutoff=FNCSettings.contact_cutoff,
    )

    samples_contact_distances = _compute_contact_distances_batch(
        pos * 10, # nm to A
        reference_contact_pairs
    )

    contact_score = _compute_contact_score(
        samples_contact_distances=samples_contact_distances,
        reference_contact_distances=reference_contact_distances,
        contact_beta=FNCSettings.contact_beta,
        contact_lambda=FNCSettings.contact_lambda,
        contact_delta=FNCSettings.contact_delta,
    )

    protein_folded_q_threshold = 0.7
    h[:, 0] = contact_score > protein_folded_q_threshold

    ## For binding
    # Compute RMSD of the loop region
    loop_region = aligned_to_ref[:, 6:21, :]
    loop_rmsd = torch.sqrt(
        ((loop_region - ref_coords[6:21, :]) ** 2).sum(dim=-1).mean(dim=-1)
    )
    
    # Classify as folded or unfolded based on RMSD
    loop_folded_threshold = 0.2 # 2A = 0.2 nm threshold for folded state, based on MD sim
    h[:, 1] = (loop_rmsd < loop_folded_threshold).float()
    
    return h