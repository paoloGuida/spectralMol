"""Loaders for per-molecule, per-generation GuacaMol ablation data (real run outputs)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT

RUN_ROOT = ROOT / "molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container/compare_GuacaMol_20260515_224644"
MODELS = {"full": "local_evolution", "high": "local_evolution_high_only",
          "low": "local_evolution_low_only", "random": "local_evolution_random_matrix"}
SEEDS = [7, 8, 9, 10, 11, 12]


def task_dir(model: str, seed: int, task: str) -> Path:
    g = glob.glob(str(RUN_ROOT / "runs" / model / f"seed_{seed}" / "case_*" / "benchmark_runs" / "*" / f"*_{task}"))
    if not g:
        raise FileNotFoundError(f"{model} seed {seed} {task}")
    return Path(g[0])


def load_run(model: str, seed: int, task: str, comps: list[str]) -> pd.DataFrame:
    """Rows: generation, smiles, score, <comps...>. Generation from molecule_scores_by_generation.tsv
    (row-aligned with scores.csv; verified: same length, gmean == score)."""
    d = task_dir(model, seed, task)
    sc = pd.read_csv(d / "scores.csv")
    mg = pd.read_csv(d / "molecule_scores_by_generation.tsv", sep="\t")
    if len(sc) != len(mg):
        raise ValueError(f"row mismatch {len(sc)} vs {len(mg)} in {d}")
    out = mg[["generation", "smiles", "score"]].copy()
    for c in comps:
        out[c] = sc[c].values
    return out


def front_idx(P: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated set (maximise both columns). O(n log n)."""
    order = np.lexsort((-P[:, 1], -P[:, 0]))
    best, keep = -np.inf, []
    for i in order:
        if P[i, 1] > best:
            keep.append(i); best = P[i, 1]
    return np.array(keep, dtype=int)


def hypervolume(P: np.ndarray) -> float:
    f = P[front_idx(P)]
    f = f[np.argsort(-f[:, 0])]
    hv, prev_y = 0.0, 0.0
    for x, y in f:
        hv += x * (y - prev_y) if y > prev_y else 0.0
        prev_y = max(prev_y, y)
    return float(hv)
