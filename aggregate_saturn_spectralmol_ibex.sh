#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/ibex/scratch/${USER:-colleoe}/spectralMol/saturn_spectralmol_theta}"
SEED_LIST_RAW="${SEED_LIST:-${SEED_LIST_CSV:-7}}"

/ibex/user/colleoe/conda-environments/molscore/bin/python - <<'PY'
from __future__ import annotations

from pathlib import Path
import csv
import math
import os

output_root = Path(os.environ.get("OUTPUT_ROOT", "/ibex/scratch/colleoe/spectralMol/saturn_spectralmol_theta"))
seed_raw = os.environ.get("SEED_LIST") or os.environ.get("SEED_LIST_CSV", "7")
seed_raw = seed_raw.replace(":", ",").replace(";", ",")
seeds = [s.strip() for s in seed_raw.split(",") if s.strip()]

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))

rows = []
missing = []
for seed in seeds:
    paths = sorted(
        output_root.glob(f"seed_{seed}/case_saturn_objective_*/benchmark_runs/SATURN/evolution_ga/strategy_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not paths:
        missing.append(seed)
        continue
    with paths[-1].open() as f:
        data = list(csv.DictReader(f))
    if not data:
        missing.append(seed)
        continue
    r = data[0]
    rows.append(
        {
            "seed": seed,
            "task": r.get("task", ""),
            "best_score": float(r.get("best_score", "nan")),
            "mean_top10": float(r.get("mean_top10", "nan")),
            "unique_molecules": int(float(r.get("unique_molecules", "0") or 0)),
            "evaluated": int(float(r.get("evaluated", "0") or 0)),
            "generations": int(float(r.get("generations", "0") or 0)),
            "run_dir": r.get("run_dir", str(paths[-1].parent)),
        }
    )

print("Using output root:", output_root)
print("Seeds:", ",".join(seeds))
print("\nSATURN SpectralMol theta-only results:")
for r in rows:
    print(
        f"  seed={r['seed']:>4s} task={r['task']} best={r['best_score']:.6f} "
        f"top10={r['mean_top10']:.6f} unique={r['unique_molecules']} "
        f"evaluated={r['evaluated']} generations={r['generations']}"
    )

best = [r["best_score"] for r in rows]
top10 = [r["mean_top10"] for r in rows]
print("\nSummary:")
print(f"  n={len(rows)}")
print(f"  best_score_mean={mean(best):.6f} std={std(best):.6f} min={min(best) if best else float('nan'):.6f} max={max(best) if best else float('nan'):.6f}")
print(f"  mean_top10_mean={mean(top10):.6f} std={std(top10):.6f}")
print("Missing seeds:", ",".join(missing) if missing else "none")
PY

