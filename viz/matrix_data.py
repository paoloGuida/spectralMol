"""Real wall-clock data from the round-2 acceleration matrix (baseline CPU / Dask CPU / RAPIDS V100 / Dask+RAPIDS V100)."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from common import ROOT

M = ROOT / "molscore/outputs/comparison_matrix"
CFG_DIRS = {
    "baseline_cpu": "baseline_cpu/compare_GuacaMol_20260506_120038",
    "dask_cpu": "dask_cpu/compare_GuacaMol_20260506_120224",
    "rapids_v100": "rapids_v100/compare_GuacaMol_20260506_120813",
    "dask_rapids_v100": "dask_rapids_v100/compare_GuacaMol_20260506_121020",
}
SEEDS = [7, 8, 9, 10, 11, 12]


def _task_dir(cfg: str, seed: int, task: str) -> str:
    g = glob.glob(str(M / CFG_DIRS[cfg] / "runs/local_evolution" / f"seed_{seed}" / "case_*" / "benchmark_runs" / "*" / f"*_{task}"))
    if not g:
        raise FileNotFoundError(f"{cfg} seed {seed} {task}")
    return g[0]


def per_generation(cfg: str, seed: int, task: str):
    """(t_end[g], best_so_far[g]) : wall-clock seconds at which generation g finished, and best score so far."""
    d = _task_dir(cfg, seed, task)
    t = pd.read_csv(f"{d}/scores.csv", usecols=["absolute_time"]).absolute_time.values
    mg = pd.read_csv(f"{d}/molecule_scores_by_generation.tsv", sep="\t")
    if len(t) != len(mg):
        raise ValueError("row mismatch")
    df = pd.DataFrame({"g": mg.generation.values, "t": t, "s": mg.score.values})
    grp = df.groupby("g")
    t_end = grp.t.max().values
    best = np.maximum.accumulate(grp.s.max().values)
    return np.maximum.accumulate(t_end), best


def race_curves(task: str):
    """Seed-averaged {cfg: (t_end[g], best[g])}. best-so-far is identical across cfgs for a given seed
    (verified separately), so it is averaged over seeds only."""
    out = {}
    for cfg in CFG_DIRS:
        ts, bs = zip(*[per_generation(cfg, s, task) for s in SEEDS])
        out[cfg] = (np.mean(ts, axis=0), np.mean(bs, axis=0))
    return out
