from __future__ import annotations

from typing import Iterable

import numpy as np


def morgan_bit_matrix(
    smiles: Iterable[str],
    *,
    radius: int = 2,
    fp_size: int = 2048,
    max_n: int = 128,
) -> np.ndarray:
    """
    Build a dense Morgan fingerprint bit matrix from SMILES.

    Returns a uint8 matrix shaped (n_molecules, fp_size). Invalid SMILES are skipped.
    """
    smiles_list = [s for s in smiles if isinstance(s, str) and s]
    if not smiles_list:
        return np.zeros((0, int(fp_size)), dtype=np.uint8)
    if len(smiles_list) > max_n:
        smiles_list = smiles_list[: int(max_n)]

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return np.zeros((0, int(fp_size)), dtype=np.uint8)

    morgan_gen = None
    try:
        morgan_gen = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))
    except Exception:
        morgan_gen = None

    rows: list[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if morgan_gen is not None:
            fp = morgan_gen.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(fp_size))
        arr = np.zeros((int(fp_size),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)

    if not rows:
        return np.zeros((0, int(fp_size)), dtype=np.uint8)
    return np.stack(rows, axis=0).astype(np.uint8, copy=False)


def pairwise_tanimoto_diversity_mean(
    bit_matrix: np.ndarray,
    *,
    use_gpu: bool = True,
) -> float:
    """
    Mean pairwise Tanimoto distance (1 - similarity) across all unique pairs.

    Uses CuPy on GPU when requested and available, otherwise NumPy on CPU.
    """
    if bit_matrix is None:
        return 0.0

    x = np.asarray(bit_matrix, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] <= 1:
        return 0.0

    n = int(x.shape[0])

    if use_gpu:
        try:
            import cupy as cp

            gx = cp.asarray(x, dtype=cp.float32)
            inter = gx @ gx.T
            ones = cp.sum(gx, axis=1)
            denom = ones[:, None] + ones[None, :] - inter
            sim = cp.where(denom > 0.0, inter / denom, 0.0)
            iu = cp.triu_indices(n, k=1)
            if iu[0].size == 0:
                return 0.0
            dist_mean = cp.mean(1.0 - sim[iu])
            return float(cp.asnumpy(dist_mean))
        except Exception:
            pass

    inter = x @ x.T
    ones = np.sum(x, axis=1)
    denom = ones[:, None] + ones[None, :] - inter
    sim = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0.0)
    iu = np.triu_indices(n, k=1)
    if iu[0].size == 0:
        return 0.0
    return float(np.mean(1.0 - sim[iu]))


def pairwise_tanimoto_diversity_from_smiles(
    smiles: Iterable[str],
    *,
    max_n: int = 128,
    radius: int = 2,
    fp_size: int = 2048,
    prefer_gpu: bool = True,
) -> float:
    """
    End-to-end helper: SMILES -> Morgan bits -> mean pairwise Tanimoto distance.
    """
    bits = morgan_bit_matrix(
        smiles,
        radius=int(radius),
        fp_size=int(fp_size),
        max_n=int(max_n),
    )
    if bits.shape[0] <= 1:
        return 0.0
    return pairwise_tanimoto_diversity_mean(bits, use_gpu=bool(prefer_gpu))
