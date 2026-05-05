#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from core.gpu_utils import pairwise_tanimoto_diversity_from_smiles


def _canonical_smiles(s: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(s).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _load_smiles_pool(repo_root: Path, limit: int = 5000) -> list[str]:
    candidates = [
        repo_root / "guacamol" / "guacamol_dataset" / "chembl24_canon_train.smiles",
        repo_root / "env" / "lib" / "python3.12" / "site-packages" / "guacamol" / "data" / "holdout_set_gcm_v1.smiles",
    ]

    pool: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                smi = _canonical_smiles(line)
                if smi:
                    pool.append(smi)
                    if len(pool) >= limit:
                        break
        if len(pool) >= limit:
            break

    if not pool:
        from core import config as cfg

        for s in cfg.FALLBACK_SEED_SMILES:
            smi = _canonical_smiles(s)
            if smi:
                pool.append(smi)

    pool = sorted(set(pool))
    if not pool:
        raise RuntimeError("No valid SMILES available for benchmark sweep.")
    return pool


def _old_pairwise_tanimoto_diversity(smiles: list[str], max_n: int = 128) -> float:
    smiles_list = [s for s in smiles if isinstance(s, str) and s]
    if len(smiles_list) <= 1:
        return 0.0
    if len(smiles_list) > max_n:
        smiles_list = smiles_list[:max_n]

    morgan_gen = None
    try:
        morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    except Exception:
        morgan_gen = None

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if morgan_gen is not None:
            fps.append(morgan_gen.GetFingerprint(mol))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    if len(fps) <= 1:
        return 0.0

    dsum = 0.0
    count = 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for sim in sims:
            dsum += 1.0 - float(sim)
            count += 1
    return float(dsum / count) if count else 0.0


def _time_fn(fn, smiles: list[str], repeats: int, warmups: int) -> tuple[float, float, float, float]:
    vals = []
    score = float("nan")
    for i in range(warmups + repeats):
        t0 = time.perf_counter()
        out = fn(smiles)
        dt = time.perf_counter() - t0
        if i >= warmups:
            vals.append(dt)
            score = float(out)
    return (
        float(min(vals)),
        float(statistics.median(vals)),
        float(np.mean(vals)),
        score,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Speed sweep: old vs new pairwise diversity.")
    p.add_argument("--sizes", default="32,64,128", help="Comma-separated set sizes.")
    p.add_argument("--repeats", type=int, default=20, help="Timed repetitions per size.")
    p.add_argument("--warmups", type=int, default=3, help="Warmup iterations per size.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    p.add_argument("--output-dir", default="molscore/outputs/phase2_benchmarks", help="Output directory.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    sizes = [int(x.strip()) for x in str(args.sizes).split(",") if x.strip()]
    if not sizes:
        raise RuntimeError("No benchmark sizes supplied.")

    pool = _load_smiles_pool(repo_root)

    gpu_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cupy_ok = False
    cupy_devices = 0
    try:
        import cupy as cp

        cupy_devices = int(cp.cuda.runtime.getDeviceCount())
        cupy_ok = cupy_devices > 0
    except Exception:
        cupy_ok = False

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = (repo_root / args.output_dir / f"diversity_speed_sweep_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for n in sizes:
        if len(pool) >= n:
            smiles = rng.sample(pool, n)
        else:
            smiles = [pool[i % len(pool)] for i in range(n)]

        old_min, old_med, old_mean, old_score = _time_fn(
            lambda x: _old_pairwise_tanimoto_diversity(x, max_n=max(sizes)),
            smiles,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        new_cpu_min, new_cpu_med, new_cpu_mean, new_cpu_score = _time_fn(
            lambda x: pairwise_tanimoto_diversity_from_smiles(x, max_n=max(sizes), prefer_gpu=False),
            smiles,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        new_auto_min, new_auto_med, new_auto_mean, new_auto_score = _time_fn(
            lambda x: pairwise_tanimoto_diversity_from_smiles(x, max_n=max(sizes), prefer_gpu=True),
            smiles,
            repeats=args.repeats,
            warmups=args.warmups,
        )

        rows.append(
            {
                "n_smiles": int(n),
                "old_median_sec": old_med,
                "new_cpu_median_sec": new_cpu_med,
                "new_auto_median_sec": new_auto_med,
                "speedup_new_cpu_vs_old": (old_med / new_cpu_med) if new_cpu_med > 0 else float("nan"),
                "speedup_new_auto_vs_old": (old_med / new_auto_med) if new_auto_med > 0 else float("nan"),
                "old_mean_sec": old_mean,
                "new_cpu_mean_sec": new_cpu_mean,
                "new_auto_mean_sec": new_auto_mean,
                "old_min_sec": old_min,
                "new_cpu_min_sec": new_cpu_min,
                "new_auto_min_sec": new_auto_min,
                "old_score": old_score,
                "new_cpu_score": new_cpu_score,
                "new_auto_score": new_auto_score,
                "abs_diff_old_vs_new_cpu": abs(old_score - new_cpu_score),
                "abs_diff_old_vs_new_auto": abs(old_score - new_auto_score),
            }
        )

    tsv_path = out_dir / "diversity_speed_sweep.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(str(r[h]) for h in headers) + "\n")

    summary = {
        "output_dir": str(out_dir),
        "tsv": str(tsv_path),
        "sizes": sizes,
        "repeats": int(args.repeats),
        "warmups": int(args.warmups),
        "gpu_visible_devices": gpu_visible,
        "cupy_available": bool(cupy_ok),
        "cupy_device_count": int(cupy_devices),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Output directory: {out_dir}", flush=True)
    print(f"TSV results: {tsv_path}", flush=True)
    print("", flush=True)
    print("n_smiles\told_med_s\tnew_cpu_med_s\tnew_auto_med_s\tspeedup_cpu\tspeedup_auto", flush=True)
    for r in rows:
        print(
            f"{r['n_smiles']}\t{r['old_median_sec']:.6f}\t{r['new_cpu_median_sec']:.6f}\t"
            f"{r['new_auto_median_sec']:.6f}\t{r['speedup_new_cpu_vs_old']:.3f}x\t{r['speedup_new_auto_vs_old']:.3f}x",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
