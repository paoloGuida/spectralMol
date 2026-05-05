#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .gpu_utils import pairwise_tanimoto_diversity_from_smiles


PREFERRED_SCORE_COLUMNS = (
    "filtered_score",
    "score",
    "Score",
    "total_score",
    "filtered_single",
    "single",
    "filtered_gmean",
    "gmean",
    "filtered_amean",
    "amean",
    "filtered_wmean",
    "wmean",
    "filtered_wsum",
    "wsum",
    "filtered_sum",
    "sum",
)


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


SCORE_NUMERIC_BLACKLIST = {
    "step",
    "gen",
    "generation",
    "batch_idx",
    "absolute_time",
    "total_time",
    "batch_time",
    "valid_score",
    "filter",
    "occurrences",
    "score_time",
    "idx",
    "index",
    "Unnamed: 0",
}


def _extract_task_scoring_method(task_dir: Path) -> str | None:
    config_candidates = sorted(task_dir.glob("*_config.json"))
    for cfg in config_candidates:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        scoring = data.get("scoring") if isinstance(data, dict) else None
        if not isinstance(scoring, dict):
            continue
        method = scoring.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return None


def _detect_score_column(df: pd.DataFrame, preferred_method: str | None = None) -> str | None:
    preferred_cols: list[str] = []
    if preferred_method:
        method = preferred_method.strip()
        if method:
            preferred_cols.extend((f"filtered_{method}", method))
    preferred_cols.extend(PREFERRED_SCORE_COLUMNS)
    preferred_cols = list(dict.fromkeys(preferred_cols))

    for col in preferred_cols:
        if col in df.columns:
            return col

    filtered_candidates = [
        str(c)
        for c in df.columns
        if str(c).startswith("filtered_") and str(c) not in {"filter", "filtered_valid_score"}
    ]
    if len(filtered_candidates) == 1:
        return filtered_candidates[0]

    numeric_cols: list[str] = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
        except Exception:
            continue
    numeric_cols = [
        c
        for c in numeric_cols
        if c not in SCORE_NUMERIC_BLACKLIST
        and not str(c).startswith("raw_")
        and not str(c).startswith("time")
        and not str(c).endswith("_time")
    ]
    return numeric_cols[-1] if numeric_cols else None


def _pairwise_tanimoto_diversity(smiles: Iterable[str], max_n: int = 128) -> float:
    try:
        return pairwise_tanimoto_diversity_from_smiles(
            smiles,
            max_n=int(max_n),
            radius=2,
            fp_size=2048,
            prefer_gpu=True,
        )
    except Exception:
        return float("nan")


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

    preferred_method = _extract_task_scoring_method(task_dir)
    score_col = _detect_score_column(df, preferred_method=preferred_method)
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
