#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from core.gpu_utils import pairwise_tanimoto_diversity_from_smiles
except Exception:
    pairwise_tanimoto_diversity_from_smiles = None


PREFERRED_SCORE_COLUMNS = (
    "filtered_single",
    "single",
    "Score",
    "score",
    "total_score",
)

_DIVERSITY_GPU_ENABLED = os.environ.get("MOLEVO_DIVERSITY_GPU_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_DIVERSITY_GPU_MIN_N = max(1, int(os.environ.get("MOLEVO_SATURN_DIVERSITY_GPU_MIN_N", "64")))


def _parse_bool_like(values: pd.Series, default: bool = True) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
            "t": True,
            "f": False,
        }
    )
    return mapped.fillna(default).astype(bool)


def _detect_score_column(df: pd.DataFrame) -> str | None:
    for col in PREFERRED_SCORE_COLUMNS:
        if col in df.columns:
            return col
    numeric_cols: list[str] = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        except Exception:
            continue
    skip = {"step", "batch_idx", "absolute_time", "total_time", "batch_time", "valid_score", "filter", "occurrences"}
    numeric_cols = [c for c in numeric_cols if c not in skip and not str(c).startswith("raw_")]
    return numeric_cols[-1] if numeric_cols else None


def _pairwise_tanimoto_diversity(smiles: Iterable[str], max_n: int = 128) -> float:
    smiles_list = [s for s in smiles if isinstance(s, str) and s]
    if len(smiles_list) <= 1:
        return 0.0
    if len(smiles_list) > max_n:
        smiles_list = smiles_list[:max_n]

    # Prefer the shared GPU utility when available; keep the CPU fallback for portability.
    if (
        pairwise_tanimoto_diversity_from_smiles is not None
        and _DIVERSITY_GPU_ENABLED
        and len(smiles_list) >= _DIVERSITY_GPU_MIN_N
    ):
        try:
            return pairwise_tanimoto_diversity_from_smiles(
                smiles_list,
                max_n=int(max_n),
                radius=2,
                fp_size=2048,
                prefer_gpu=True,
            )
        except Exception:
            pass

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except Exception:
        return float("nan")

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


def write_guacamol_like_reports(task_dir: Path) -> bool:
    """
    Write GuacaMol-style generation reports into one task output folder.
    Creates:
      - evolution_log.tsv
      - gen_metrics.tsv
      - valid_molecules_by_generation.tsv
      - invalid_reasons_by_generation.tsv
    """
    task_dir = Path(task_dir)
    scores_path = task_dir / "scores.csv"
    if not scores_path.exists():
        return False

    try:
        df = pd.read_csv(scores_path)
    except Exception:
        return False
    if df.empty:
        return False

    if "step" in df.columns:
        gen = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)
    elif "generation" in df.columns:
        gen = pd.to_numeric(df["generation"], errors="coerce").fillna(0).astype(int)
    else:
        gen = pd.Series(np.zeros(len(df), dtype=int))
    df = df.copy()
    df["gen"] = gen
    df = df.sort_values(["gen"] + ([c for c in ["batch_idx"] if c in df.columns]))

    score_col = _detect_score_column(df)
    score_vals = pd.to_numeric(df[score_col], errors="coerce") if score_col else pd.Series(np.nan, index=df.index)
    valid_vals = _parse_bool_like(df["valid"], default=True) if "valid" in df.columns else pd.Series(True, index=df.index)
    filter_vals = pd.to_numeric(df["filter"], errors="coerce").fillna(1.0) if "filter" in df.columns else pd.Series(1.0, index=df.index)
    valid_score_vals = pd.to_numeric(df["valid_score"], errors="coerce") if "valid_score" in df.columns else pd.Series(np.nan, index=df.index)
    smiles_vals = df["smiles"].astype(str) if "smiles" in df.columns else pd.Series([""] * len(df), index=df.index)

    # valid_molecules_by_generation.tsv
    valid_df = df.loc[valid_vals].copy()
    if "batch_idx" in valid_df.columns:
        valid_idx = pd.to_numeric(valid_df["batch_idx"], errors="coerce").fillna(0).astype(int)
    else:
        valid_idx = valid_df.groupby("gen").cumcount()
    valid_rows = pd.DataFrame(
        {
            "gen": valid_df["gen"].astype(int),
            "idx": valid_idx.astype(int),
            "score": pd.to_numeric(score_vals.loc[valid_df.index], errors="coerce"),
            "f1": pd.to_numeric(valid_score_vals.loc[valid_df.index], errors="coerce"),
            "f2": pd.to_numeric(filter_vals.loc[valid_df.index], errors="coerce"),
            "smiles": smiles_vals.loc[valid_df.index].astype(str),
        }
    )
    valid_rows.to_csv(task_dir / "valid_molecules_by_generation.tsv", sep="\t", index=False)

    # evolution_log.tsv and gen_metrics.tsv
    evo_rows: list[dict] = []
    gen_rows: list[dict] = []
    best_so_far = float("-inf")
    best_smiles_so_far = ""
    seen_valid_smiles: set[str] = set()
    generations = sorted(df["gen"].dropna().astype(int).unique().tolist())

    for g in generations:
        gmask = df["gen"] == g
        gdf = df.loc[gmask]
        g_valid_mask = valid_vals.loc[gdf.index]
        g_scores = pd.to_numeric(score_vals.loc[gdf.index], errors="coerce")
        g_valid_scores = g_scores.loc[g_valid_mask]

        pop = int(len(gdf))
        valid_count = int(g_valid_mask.sum())
        valid_rate = 100.0 * valid_count / pop if pop else 0.0
        mean_score = float(g_valid_scores.mean()) if len(g_valid_scores.dropna()) else float(g_scores.mean())

        if len(g_valid_scores.dropna()):
            best_idx = g_valid_scores.idxmax()
            best_score_gen = float(g_valid_scores.max())
            best_smiles_gen = str(smiles_vals.loc[best_idx])
        elif len(g_scores.dropna()):
            best_idx = g_scores.idxmax()
            best_score_gen = float(g_scores.max())
            best_smiles_gen = str(smiles_vals.loc[best_idx])
        else:
            best_score_gen = float("nan")
            best_smiles_gen = ""

        if not np.isnan(best_score_gen) and best_score_gen >= best_so_far:
            best_so_far = best_score_gen
            best_smiles_so_far = best_smiles_gen

        g_valid_smiles = [s for s in smiles_vals.loc[gdf.index[g_valid_mask]].astype(str).tolist() if s]
        g_unique_valid = set(g_valid_smiles)
        unique_valid_count = len(g_unique_valid)
        unique_valid_rate = 100.0 * unique_valid_count / valid_count if valid_count else 0.0
        new_unique_count = len(g_unique_valid - seen_valid_smiles)
        seen_valid_smiles |= g_unique_valid
        diversity_mean = _pairwise_tanimoto_diversity(sorted(g_unique_valid))

        evo_rows.append(
            {
                "gen": int(g),
                "valid_rate": f"{valid_rate:.3f}",
                "mean_score": f"{mean_score:.6f}" if not np.isnan(mean_score) else "",
                "best_score_so_far": f"{best_so_far:.6f}" if best_so_far != float("-inf") else "",
                "best_smiles": best_smiles_so_far,
                "valid": int(valid_count),
                "pop": int(pop),
            }
        )
        gen_rows.append(
            {
                "gen": int(g),
                "valid_count": int(valid_count),
                "valid_rate": f"{valid_rate:.4f}",
                "unique_valid_count": int(unique_valid_count),
                "unique_valid_rate": f"{unique_valid_rate:.4f}",
                "new_unique_count": int(new_unique_count),
                "diversity_mean": f"{diversity_mean:.6f}" if not np.isnan(diversity_mean) else "",
                "mean_score": f"{mean_score:.6f}" if not np.isnan(mean_score) else "",
                "best_score_gen": f"{best_score_gen:.6f}" if not np.isnan(best_score_gen) else "",
                "best_score_so_far": f"{best_so_far:.6f}" if best_so_far != float("-inf") else "",
                # Not directly available in MolScore outputs; use a neutral proxy.
                "pareto_unique_count": int(unique_valid_count),
                "hypervolume": f"{best_so_far:.6f}" if best_so_far != float("-inf") else "",
            }
        )

    pd.DataFrame(evo_rows).to_csv(task_dir / "evolution_log.tsv", sep="\t", index=False)
    pd.DataFrame(gen_rows).to_csv(task_dir / "gen_metrics.tsv", sep="\t", index=False)

    # invalid_reasons_by_generation.tsv
    reasons_rows: list[dict] = []
    for g in generations:
        gmask = df["gen"] == g
        idxs = df.index[gmask]
        g_valid = valid_vals.loc[idxs]
        g_filter = filter_vals.loc[idxs]
        g_smiles = smiles_vals.loc[idxs].astype(str)

        reasons: list[str] = []
        for is_valid, filt_val, smi in zip(g_valid.tolist(), g_filter.tolist(), g_smiles.tolist()):
            if not smi:
                reasons.append("EMPTY_SMILES")
            elif not bool(is_valid):
                reasons.append("INVALID_SMILES")
            elif float(filt_val) < 1.0:
                reasons.append("FILTER_FAIL")
            else:
                reasons.append("OK")

        if not reasons:
            continue
        reason_counts = pd.Series(reasons).value_counts()
        for reason, count in reason_counts.items():
            reasons_rows.append({"gen": int(g), "reason": str(reason), "count": int(count)})

    pd.DataFrame(reasons_rows).to_csv(task_dir / "invalid_reasons_by_generation.tsv", sep="\t", index=False)
    return True


def write_guacamol_like_reports_under(root: Path) -> int:
    """Write reports for every task folder under root that has a scores.csv."""
    root = Path(root)
    if not root.exists():
        return 0
    count = 0
    for scores_path in root.rglob("scores.csv"):
        if write_guacamol_like_reports(scores_path.parent):
            count += 1
    return count
