#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make the repo root importable so the 'core' package can be found
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from core import config as cfg
from core.reports import write_guacamol_like_reports_under

COMPARISON_GENERATIONS_DEFAULT = int(getattr(cfg, "COMPARISON_GENERATIONS_DEFAULT", 50))
EQUAL_INITIAL_POPULATION_DEFAULT = bool(getattr(cfg, "EQUAL_INITIAL_POPULATION_DEFAULT", True))


META_NUMERIC_BLACKLIST = {
    "step",
    "gen",
    "generation",
    "batch_idx",
    "absolute_time",
    "total_time",
    "batch_time",
    "batch_idx",
    "idx",
    "index",
    "Unnamed: 0",
    "valid_score",
    "filter",
    "score_time",
}

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


@dataclass
class ModelSpec:
    name: str
    type: str
    description: str
    enabled: bool
    command: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run multiple generators under identical MolScore benchmark settings and "
            "create unified comparison tables across seeds/tasks."
        )
    )
    p.add_argument("--benchmark", default=cfg.BENCHMARK_DEFAULT, help="MolScore preset benchmark name.")
    p.add_argument("--custom-benchmark", default=cfg.CUSTOM_BENCHMARK_DEFAULT, help="Custom benchmark config directory. Overrides --benchmark.")
    p.add_argument("--include", default=cfg.INCLUDE_CSV_DEFAULT, help="Comma-separated task names to include.")
    p.add_argument("--exclude", default=cfg.EXCLUDE_CSV_DEFAULT, help="Comma-separated task names to exclude.")
    p.add_argument("--budget", type=int, default=cfg.BUDGET_DEFAULT, help="Budget per task.")
    p.add_argument(
        "--generations",
        type=int,
        default=COMPARISON_GENERATIONS_DEFAULT,
        help="Target generations per task for models that support generation control.",
    )
    p.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE_DEFAULT, help="Population size for local_evolution model.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE_DEFAULT, help="Batch size for local_evolution model.")
    p.add_argument(
        "--local-evo-tournament-k",
        type=int,
        default=int(getattr(cfg, "TOURNAMENT_K_DEFAULT", 8)),
        help="Tournament size for builtin local_evolution parent selection.",
    )
    p.add_argument(
        "--local-evo-elite-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_ELITE_FRACTION", 0.15)),
        help="Elite fraction for builtin local_evolution.",
    )
    p.add_argument(
        "--local-evo-immigrant-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_IMMIGRANT_FRACTION", 0.10)),
        help="Immigrant fraction for builtin local_evolution.",
    )
    p.add_argument(
        "--local-evo-parent-pool-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_PARENT_POOL_FRACTION", 0.50)),
        help="Parent pool fraction for builtin local_evolution.",
    )
    p.add_argument(
        "--local-evo-stagnation-patience",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_PATIENCE", 12)),
        help="Stagnation patience for builtin local_evolution.",
    )
    p.add_argument(
        "--local-evo-stagnation-mutation-boost",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_MUTATION_BOOST", 2)),
        help="Extra mutation depth when local_evolution stagnates.",
    )
    p.add_argument(
        "--seed-smiles-file",
        default=cfg.SEED_SMILES_FILE_DEFAULT_STR,
        help="Seed SMILES file used by local_evolution model.",
    )
    p.add_argument("--seed-pool-size", type=int, default=cfg.SEED_POOL_SIZE_DEFAULT, help="Seed pool size used by local_evolution model.")
    p.add_argument("--seeds", default=cfg.SEEDS_CSV_DEFAULT, help="Comma-separated seeds for repeated runs.")
    p.add_argument("--models", default=cfg.MODELS_CSV_DEFAULT, help="Comma-separated model keys from model-spec file.")
    p.add_argument("--model-spec-file", default=cfg.MODEL_SPEC_FILE_DEFAULT_STR, help="JSON file describing runnable models.")
    p.add_argument("--examples-root", default=cfg.EXAMPLES_ROOT_DEFAULT, help="Path to local MolScore_examples clone for command templates.")
    p.add_argument("--python-bin", default=cfg.PYTHON_BIN_DEFAULT, help="Python interpreter used for local evolution and templates.")
    p.add_argument("--output-dir", default=cfg.OUTPUT_COMPARISONS_DIR_DEFAULT_STR, help="Root output directory for comparison runs.")
    p.add_argument(
        "--equal-initial-population",
        action=argparse.BooleanOptionalAction,
        default=EQUAL_INITIAL_POPULATION_DEFAULT,
        help="Use same per-seed initial population file across models when supported by command templates.",
    )
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue with other models/seeds when one fails.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    p.add_argument(
        "--graphga-n-jobs",
        type=int,
        default=int(os.environ.get("MOLSCORE_GRAPHGA_N_JOBS", "1")),
        help="Parallel workers passed to GraphGA --n_jobs.",
    )
    p.add_argument(
        "--crem-ncpu",
        type=int,
        default=int(os.environ.get("MOLSCORE_CREM_NCPU", "1")),
        help="CPU workers passed to CReM --ncpu.",
    )
    p.add_argument(
        "--crem-replacements",
        type=int,
        default=int(os.environ.get("MOLSCORE_CREM_REPLACEMENTS", "1000")),
        help="Max replacements passed to CReM --replacements.",
    )
    p.add_argument(
        "--smiles-rnn-device",
        default=os.environ.get("MOLSCORE_SMILES_RNN_DEVICE", "auto"),
        help="Device for SMILES-RNN wrapper: cpu, cuda, or auto.",
    )
    p.add_argument(
        "--bootstrap-scoring-envs",
        action=argparse.BooleanOptionalAction,
        default=bool(getattr(cfg, "BOOTSTRAP_SCORING_ENVS_DEFAULT", True)),
        help="Create PIDGIN/ms_molopt scoring envs before launching model jobs.",
    )
    p.add_argument(
        "--scoring-env-clone-from",
        default=os.environ.get("MOLSCORE_SCORING_ENV_CLONE_FROM", getattr(cfg, "SCORING_ENV_CLONE_FROM_DEFAULT", "")),
        help="Fallback env name used for cloning if scoring-env YAML solve fails.",
    )
    return p.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        seeds.append(int(t))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Model spec file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Model spec file must be a JSON object mapping model keys to specs.")
    specs: dict[str, ModelSpec] = {}
    for key, spec_cfg in data.items():
        if not isinstance(spec_cfg, dict):
            raise ValueError(f"Spec for {key!r} must be an object.")
        specs[key] = ModelSpec(
            name=key,
            type=str(spec_cfg.get("type", "")).strip(),
            description=str(spec_cfg.get("description", "")).strip(),
            enabled=bool(spec_cfg.get("enabled", True)),
            command=None if spec_cfg.get("command") is None else str(spec_cfg.get("command")),
        )
    return specs


def choose_python_bin(cli_python: str) -> str:
    if cli_python:
        return cli_python
    candidate = Path("/opt/anaconda3/envs/molscore/bin/python")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_builtin_local_evolution_command(
    python_bin: str,
    repo_root: Path,
    benchmark: str,
    custom_benchmark: str,
    include_csv: str,
    exclude_csv: str,
    budget: int,
    generations: int,
    population_size: int,
    batch_size: int,
    tournament_k: int,
    elite_fraction: float,
    immigrant_fraction: float,
    parent_pool_fraction: float,
    stagnation_patience: int,
    stagnation_mutation_boost: int,
    seed_smiles_file: str,
    seed_pool_size: int,
    seed: int,
    model_output_dir: Path,
) -> list[str]:
    script = repo_root / "molscore" / "software" / "evolve_vs_molscore_benchmark.py"
    cmd = [
        python_bin,
        str(script),
        "--budget",
        str(budget),
        "--population-size",
        str(population_size),
        "--batch-size",
        str(batch_size),
        "--tournament-k",
        str(tournament_k),
        "--elite-fraction",
        str(elite_fraction),
        "--immigrant-fraction",
        str(immigrant_fraction),
        "--parent-pool-fraction",
        str(parent_pool_fraction),
        "--stagnation-patience",
        str(stagnation_patience),
        "--stagnation-mutation-boost",
        str(stagnation_mutation_boost),
        "--seed-smiles-file",
        seed_smiles_file,
        "--seed-pool-size",
        str(seed_pool_size),
        "--seed",
        str(seed),
        "--skip-random-baseline",
        "--output-dir",
        str(model_output_dir),
    ]
    if custom_benchmark:
        cmd += ["--custom-benchmark", custom_benchmark]
    else:
        cmd += ["--benchmark", benchmark]
    if include_csv:
        cmd += ["--include", include_csv]
    if exclude_csv:
        cmd += ["--exclude", exclude_csv]
    if generations > 0:
        cmd += ["--max-generations", str(generations)]
    return cmd


def render_command_template(
    template: str,
    *,
    python_bin: str,
    repo_root: Path,
    examples_root: str,
    benchmark: str,
    custom_benchmark: str,
    include_csv: str,
    exclude_csv: str,
    budget: int,
    generations: int,
    population_size: int,
    batch_size: int,
    seed_smiles_file: str,
    seed_pool_size: int,
    seed: int,
    shared_init_file: str,
    model_output_dir: Path,
    graphga_n_jobs: int,
    crem_ncpu: int,
    crem_replacements: int,
    smiles_rnn_device: str,
) -> list[str]:
    molscore_target = custom_benchmark if custom_benchmark else benchmark
    include_args = f"--include {include_csv}" if include_csv else ""
    exclude_args = f"--exclude {exclude_csv}" if exclude_csv else ""
    formatted = template.format(
        python=python_bin,
        repo_root=str(repo_root),
        molscore_root=str(repo_root / "molscore"),
        examples_root=examples_root,
        benchmark=benchmark,
        molscore_target=molscore_target,
        custom_benchmark=custom_benchmark,
        include_csv=include_csv,
        exclude_csv=exclude_csv,
        include_args=include_args,
        exclude_args=exclude_args,
        budget=budget,
        generations=generations,
        population_size=population_size,
        batch_size=batch_size,
        seed_smiles_file=seed_smiles_file,
        seed_pool_size=seed_pool_size,
        seed=seed,
        shared_init_file=shared_init_file,
        model_output_dir=str(model_output_dir),
        graphga_n_jobs=graphga_n_jobs,
        crem_ncpu=crem_ncpu,
        crem_replacements=crem_replacements,
        smiles_rnn_device=smiles_rnn_device,
    )
    return shlex.split(formatted)


def run_command(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    ensure_dir(log_path.parent)
    line = " ".join(shlex.quote(x) for x in cmd)
    if dry_run:
        log_path.write_text(f"[DRY RUN] {line}\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[COMMAND] {line}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
        )
        return proc.wait()


def _extract_task_name(task_dir: Path) -> str:
    config_candidates = sorted(task_dir.glob("*_config.json"))
    for cfg in config_candidates:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            task = data.get("task")
            if isinstance(task, str) and task.strip():
                return task.strip()
        except Exception:
            continue
    name = task_dir.name
    return name


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


def _detect_score_column(df: pd.DataFrame, preferred_method: str | None = None) -> str:
    preferred_cols: list[str] = []
    if preferred_method:
        method = preferred_method.strip()
        if method:
            preferred_cols.extend((f"filtered_{method}", method))
    preferred_cols.extend(PREFERRED_SCORE_COLUMNS)
    # Keep insertion order while removing duplicates.
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

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in META_NUMERIC_BLACKLIST
        and not str(c).startswith("raw_")
        and not str(c).startswith("time")
        and not str(c).endswith("_time")
    ]
    if not numeric_cols:
        raise ValueError("No numeric score-like column found in scores.csv.")
    return numeric_cols[-1]


def collect_task_scores(model_output_dir: Path, model: str, seed: int) -> list[dict[str, Any]]:
    score_files = sorted(model_output_dir.rglob("scores.csv"))
    rows: list[dict[str, Any]] = []
    for scores_path in score_files:
        task_dir = scores_path.parent
        preferred_method = _extract_task_scoring_method(task_dir)
        try:
            df = pd.read_csv(scores_path)
        except Exception:
            continue
        if df.empty:
            continue
        try:
            score_col = _detect_score_column(df, preferred_method=preferred_method)
        except Exception:
            continue
        values = pd.to_numeric(df[score_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        task = _extract_task_name(task_dir)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "task": task,
                "score_column": score_col,
                "best_score": float(np.max(values)),
                "avg_score": float(np.mean(values)),
                "n_scored": int(values.size),
                "scores_csv": str(scores_path.resolve()),
            }
        )
    return rows


def _read_seed_smiles_pool(seed_smiles_file: Path, seed_pool_size: int) -> list[str]:
    if not seed_smiles_file.exists():
        raise FileNotFoundError(f"Seed SMILES file not found: {seed_smiles_file}")
    smiles: list[str] = []
    seen: set[str] = set()
    with seed_smiles_file.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            smi = token.split()[0].strip()
            if not smi or smi in seen:
                continue
            seen.add(smi)
            smiles.append(smi)
    if not smiles:
        raise ValueError(f"No valid SMILES found in {seed_smiles_file}")
    if seed_pool_size > 0 and len(smiles) > seed_pool_size:
        smiles = smiles[:seed_pool_size]
    return smiles


def _write_shared_init_file(pool: list[str], population_size: int, seed: int, path: Path) -> Path:
    if not pool:
        raise ValueError("Cannot build shared initial population from empty pool.")
    rng = random.Random(seed)
    if len(pool) >= population_size:
        init = rng.sample(pool, population_size)
    else:
        init = list(pool)
        while len(init) < population_size:
            init.append(rng.choice(pool))
    ensure_dir(path.parent)
    text = "\n".join(init) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def collect_generation_metrics(model_output_dir: Path, model: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_files = sorted(model_output_dir.rglob("gen_metrics.tsv"))
    if not candidate_files:
        candidate_files = sorted(model_output_dir.rglob("evolution_log.tsv"))

    for metrics_path in candidate_files:
        task_dir = metrics_path.parent
        task = _extract_task_name(task_dir)
        try:
            df = pd.read_csv(metrics_path, sep="\t")
        except Exception:
            continue
        if df.empty:
            continue

        gen_col = "gen" if "gen" in df.columns else ("generation" if "generation" in df.columns else None)
        if gen_col is None:
            continue

        for _, r in df.iterrows():
            gen_val = pd.to_numeric(pd.Series([r.get(gen_col)]), errors="coerce").iloc[0]
            if pd.isna(gen_val):
                continue
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "task": task,
                    "generation": int(gen_val),
                    "mean_score": pd.to_numeric(pd.Series([r.get("mean_score")]), errors="coerce").iloc[0],
                    "best_score_so_far": pd.to_numeric(pd.Series([r.get("best_score_so_far")]), errors="coerce").iloc[0],
                    "best_score_gen": pd.to_numeric(pd.Series([r.get("best_score_gen")]), errors="coerce").iloc[0],
                    "valid_rate": pd.to_numeric(pd.Series([r.get("valid_rate")]), errors="coerce").iloc[0],
                    "valid_count": pd.to_numeric(pd.Series([r.get("valid_count")]), errors="coerce").iloc[0],
                    "unique_valid_count": pd.to_numeric(pd.Series([r.get("unique_valid_count")]), errors="coerce").iloc[0],
                    "new_unique_count": pd.to_numeric(pd.Series([r.get("new_unique_count")]), errors="coerce").iloc[0],
                    "diversity_mean": pd.to_numeric(pd.Series([r.get("diversity_mean")]), errors="coerce").iloc[0],
                    "source_tsv": str(metrics_path.resolve()),
                }
            )
    return rows


def _parse_valid_like(values: pd.Series, default: bool = True) -> pd.Series:
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


def _infer_generation(df: pd.DataFrame) -> pd.Series:
    for col in ("generation", "gen", "step"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int))


def write_requested_generation_exports(
    *,
    model_output_dir: Path,
    model: str,
    seed: int,
    export_root: Path,
) -> list[Path]:
    """
    Write the four requested-style outputs (2 files per model/task):
      - molecules_scores_by_generation.tsv
      - best_molecule_score_valid_per_generation.tsv
    """
    written: list[Path] = []
    score_files = sorted(model_output_dir.rglob("scores.csv"))
    for scores_path in score_files:
        task_dir = scores_path.parent
        task_name = _extract_task_name(task_dir)
        preferred_method = _extract_task_scoring_method(task_dir)
        out_dir = export_root / model / f"seed_{seed}" / task_name
        ensure_dir(out_dir)

        try:
            sdf = pd.read_csv(scores_path)
        except Exception:
            continue
        if sdf.empty:
            continue

        if "smiles" not in sdf.columns:
            continue
        try:
            score_col = _detect_score_column(sdf, preferred_method=preferred_method)
        except Exception:
            continue

        gen = _infer_generation(sdf)
        mol_rows = pd.DataFrame(
            {
                "generation": gen.astype(int),
                "smiles": sdf["smiles"].astype(str),
                "score": pd.to_numeric(sdf[score_col], errors="coerce"),
            }
        ).dropna(subset=["score"])
        mol_rows = mol_rows.sort_values(["generation"]).reset_index(drop=True)
        molecules_out = out_dir / "molecules_scores_by_generation.tsv"
        mol_rows.to_csv(molecules_out, sep="\t", index=False)
        written.append(molecules_out)

        # Best molecule per generation
        best_gen_df: pd.DataFrame
        valid_mol_path = task_dir / "valid_molecules_by_generation.tsv"
        if valid_mol_path.exists():
            try:
                vdf = pd.read_csv(valid_mol_path, sep="\t")
            except Exception:
                vdf = pd.DataFrame()
            if not vdf.empty and "smiles" in vdf.columns:
                gcol = "gen" if "gen" in vdf.columns else ("generation" if "generation" in vdf.columns else None)
                scol = "score" if "score" in vdf.columns else None
                if gcol is not None and scol is not None:
                    vdf[gcol] = pd.to_numeric(vdf[gcol], errors="coerce")
                    vdf[scol] = pd.to_numeric(vdf[scol], errors="coerce")
                    vdf = vdf.dropna(subset=[gcol, scol])
                    if not vdf.empty:
                        idx = vdf.groupby(gcol)[scol].idxmax()
                        best_gen_df = (
                            vdf.loc[idx, [gcol, "smiles", scol]]
                            .rename(
                                columns={
                                    gcol: "generation",
                                    "smiles": "best_smiles_generation",
                                    scol: "best_score_generation",
                                }
                            )
                            .copy()
                        )
                    else:
                        best_gen_df = pd.DataFrame()
                else:
                    best_gen_df = pd.DataFrame()
            else:
                best_gen_df = pd.DataFrame()
        else:
            best_gen_df = pd.DataFrame()

        if best_gen_df.empty and not mol_rows.empty:
            idx = mol_rows.groupby("generation")["score"].idxmax()
            best_gen_df = (
                mol_rows.loc[idx, ["generation", "smiles", "score"]]
                .rename(
                    columns={
                        "smiles": "best_smiles_generation",
                        "score": "best_score_generation",
                    }
                )
                .copy()
            )

        # valid rate + best_score_so_far from evolution reports if present
        evolution_log = task_dir / "evolution_log.tsv"
        gen_metrics = task_dir / "gen_metrics.tsv"

        aux = pd.DataFrame(columns=["generation", "best_score_so_far", "valid_molecule_percent"])
        if evolution_log.exists():
            try:
                edf = pd.read_csv(evolution_log, sep="\t")
            except Exception:
                edf = pd.DataFrame()
            if not edf.empty:
                gcol = "gen" if "gen" in edf.columns else ("generation" if "generation" in edf.columns else None)
                if gcol is not None:
                    tmp = pd.DataFrame()
                    tmp["generation"] = pd.to_numeric(edf[gcol], errors="coerce")
                    if "best_score_so_far" in edf.columns:
                        tmp["best_score_so_far"] = pd.to_numeric(edf["best_score_so_far"], errors="coerce")
                    if "valid_rate" in edf.columns:
                        tmp["valid_molecule_percent"] = pd.to_numeric(edf["valid_rate"], errors="coerce")
                    aux = tmp

        if aux.empty and gen_metrics.exists():
            try:
                gdf = pd.read_csv(gen_metrics, sep="\t")
            except Exception:
                gdf = pd.DataFrame()
            if not gdf.empty:
                gcol = "gen" if "gen" in gdf.columns else ("generation" if "generation" in gdf.columns else None)
                if gcol is not None:
                    tmp = pd.DataFrame()
                    tmp["generation"] = pd.to_numeric(gdf[gcol], errors="coerce")
                    if "best_score_so_far" in gdf.columns:
                        tmp["best_score_so_far"] = pd.to_numeric(gdf["best_score_so_far"], errors="coerce")
                    if "valid_rate" in gdf.columns:
                        tmp["valid_molecule_percent"] = pd.to_numeric(gdf["valid_rate"], errors="coerce")
                    aux = tmp

        if aux.empty and not sdf.empty:
            tmp = pd.DataFrame()
            tmp["generation"] = _infer_generation(sdf)
            if "valid" in sdf.columns:
                valid_mask = _parse_valid_like(sdf["valid"], default=True)
            else:
                valid_mask = pd.Series(True, index=sdf.index)
            tmp["valid_mask"] = valid_mask.astype(int)
            valid_rate_df = (
                tmp.groupby("generation", as_index=False)["valid_mask"]
                .mean()
                .rename(columns={"valid_mask": "valid_molecule_percent"})
            )
            valid_rate_df["valid_molecule_percent"] = 100.0 * valid_rate_df["valid_molecule_percent"]
            aux = valid_rate_df

        if not aux.empty:
            aux = aux.dropna(subset=["generation"]).copy()
            aux["generation"] = aux["generation"].astype(int)
            aux = aux.drop_duplicates(subset=["generation"], keep="last")

        if best_gen_df.empty:
            best_export = pd.DataFrame(
                columns=[
                    "generation",
                    "best_smiles_generation",
                    "best_score_generation",
                    "best_score_so_far",
                    "valid_molecule_percent",
                ]
            )
        else:
            best_export = best_gen_df.copy()
            best_export["generation"] = pd.to_numeric(best_export["generation"], errors="coerce")
            best_export = best_export.dropna(subset=["generation"])
            best_export["generation"] = best_export["generation"].astype(int)
            if not aux.empty:
                best_export = best_export.merge(aux, on="generation", how="left")
            if "best_score_so_far" not in best_export.columns:
                best_export["best_score_so_far"] = np.nan
            if best_export["best_score_so_far"].isna().all():
                best_export = best_export.sort_values("generation")
                best_export["best_score_so_far"] = pd.to_numeric(
                    best_export["best_score_generation"], errors="coerce"
                ).cummax()
            if "valid_molecule_percent" not in best_export.columns:
                best_export["valid_molecule_percent"] = np.nan
            best_export = best_export[
                [
                    "generation",
                    "best_smiles_generation",
                    "best_score_generation",
                    "best_score_so_far",
                    "valid_molecule_percent",
                ]
            ].sort_values("generation")

        best_out = out_dir / "best_molecule_score_valid_per_generation.tsv"
        best_export.to_csv(best_out, sep="\t", index=False)
        written.append(best_out)

    return written


def build_summary_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    task_summary = (
        raw_df.groupby(["model", "task"], as_index=False)
        .agg(
            mean_best_score=("best_score", "mean"),
            std_best_score=("best_score", "std"),
            mean_avg_score=("avg_score", "mean"),
            std_avg_score=("avg_score", "std"),
            mean_n_scored=("n_scored", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(["task", "mean_best_score"], ascending=[True, False])
    )

    overall = (
        task_summary.groupby("model", as_index=False)
        .agg(
            sum_task_best_score=("mean_best_score", "sum"),
            sum_task_avg_score=("mean_avg_score", "sum"),
            mean_task_best_score=("mean_best_score", "mean"),
            mean_task_avg_score=("mean_avg_score", "mean"),
            mean_task_std_best=("std_best_score", "mean"),
            n_tasks=("task", "count"),
            total_runs=("n_runs", "sum"),
        )
        .sort_values("sum_task_best_score", ascending=False)
    )
    overall["rank"] = np.arange(1, len(overall) + 1)
    return task_summary, overall


def resolve_parallel_jobs(n_jobs: int) -> int:
    if n_jobs <= 0:
        return 0
    env_raw = os.environ.get("MOLSCORE_PARALLEL_JOBS", "").strip()
    if env_raw:
        try:
            requested = int(env_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid MOLSCORE_PARALLEL_JOBS={env_raw!r}; expected integer."
            ) from exc
    else:
        requested = int(getattr(cfg, "MODEL_RUN_PARALLEL_JOBS", 0))

    if requested <= 0:
        return n_jobs
    return max(1, min(requested, n_jobs))


def bootstrap_scoring_envs(
    *,
    python_bin: str,
    repo_root: Path,
    clone_from: str,
    out_root: Path,
    dry_run: bool,
) -> tuple[int, Path]:
    script = repo_root / "molscore" / "software" / "bootstrap_scoring_envs.py"
    if not script.exists():
        raise FileNotFoundError(f"bootstrap script not found: {script}")
    log_path = out_root / "bootstrap_scoring_envs.log"
    cmd = [
        python_bin,
        str(script),
        "--targets",
        "pidgin,ms_molopt",
    ]
    if clone_from:
        cmd += ["--clone-from", clone_from]
    rc = run_command(cmd=cmd, log_path=log_path, dry_run=dry_run)
    return rc, log_path


def main() -> int:
    args = parse_args()
    repo_root = cfg.REPO_ROOT
    include = parse_csv_list(args.include)
    exclude = parse_csv_list(args.exclude)
    include_csv = ",".join(include)
    exclude_csv = ",".join(exclude)
    seeds = parse_seeds(args.seeds)
    python_bin = choose_python_bin(args.python_bin)
    examples_root = args.examples_root.strip()
    custom_benchmark = args.custom_benchmark.strip()
    smiles_rnn_device = args.smiles_rnn_device.strip().lower()
    if args.graphga_n_jobs < 1:
        raise ValueError(f"--graphga-n-jobs must be >= 1, got {args.graphga_n_jobs}")
    if args.crem_ncpu < 1:
        raise ValueError(f"--crem-ncpu must be >= 1, got {args.crem_ncpu}")
    if args.crem_replacements < 1:
        raise ValueError(f"--crem-replacements must be >= 1, got {args.crem_replacements}")
    if smiles_rnn_device not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            f"--smiles-rnn-device must be one of cpu|cuda|auto, got {args.smiles_rnn_device!r}"
        )
    if args.local_evo_tournament_k < 1:
        raise ValueError(f"--local-evo-tournament-k must be >= 1, got {args.local_evo_tournament_k}")
    if not 0.0 <= float(args.local_evo_elite_fraction) <= 0.9:
        raise ValueError("--local-evo-elite-fraction must be in [0.0, 0.9]")
    if not 0.0 <= float(args.local_evo_immigrant_fraction) <= 0.9:
        raise ValueError("--local-evo-immigrant-fraction must be in [0.0, 0.9]")
    if not 0.1 <= float(args.local_evo_parent_pool_fraction) <= 1.0:
        raise ValueError("--local-evo-parent-pool-fraction must be in [0.1, 1.0]")
    if int(args.local_evo_stagnation_patience) < 0:
        raise ValueError("--local-evo-stagnation-patience must be >= 0")
    if int(args.local_evo_stagnation_mutation_boost) < 0:
        raise ValueError("--local-evo-stagnation-mutation-boost must be >= 0")
    model_spec_path = cfg.resolve_from_repo(args.model_spec_file)
    seed_smiles_file = cfg.resolve_from_repo(args.seed_smiles_file)

    model_specs = load_model_specs(model_spec_path)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not selected_models:
        raise ValueError("No models selected.")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_base = cfg.resolve_from_repo(args.output_dir)
    out_root = (output_base / f"compare_{args.benchmark}_{ts}").resolve()
    ensure_dir(out_root)
    logs_dir = out_root / "logs"
    ensure_dir(logs_dir)

    required_budget = args.budget
    if args.generations > 0:
        required_budget = max(required_budget, args.population_size + (args.batch_size * args.generations))
    effective_budget = int(required_budget)
    if effective_budget > int(args.budget):
        print(
            f"[budget] increased from {args.budget} to {effective_budget} to allow {args.generations} generations "
            f"(population_size={args.population_size}, batch_size={args.batch_size})",
            flush=True,
        )

    shared_init_files: dict[int, Path] = {}
    shared_init_dir = out_root / "shared_initial_population"
    if args.equal_initial_population:
        pool = _read_seed_smiles_pool(seed_smiles_file=seed_smiles_file, seed_pool_size=args.seed_pool_size)
        for seed in seeds:
            shared_init_files[int(seed)] = _write_shared_init_file(
                pool=pool,
                population_size=args.population_size,
                seed=int(seed),
                path=shared_init_dir / f"seed_{seed}.smi",
            )

    all_rows: list[dict[str, Any]] = []
    all_gen_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []

    for model in selected_models:
        if model not in model_specs:
            msg = f"Model {model!r} not found in spec file."
            if args.continue_on_error:
                failures.append({"model": model, "seed": None, "error": msg})
                continue
            raise ValueError(msg)

        spec = model_specs[model]
        if not spec.enabled:
            msg = f"Model {model!r} is disabled in spec file."
            if args.continue_on_error:
                failures.append({"model": model, "seed": None, "error": msg})
                continue
            raise ValueError(msg)

        for seed in seeds:
            model_seed_dir = out_root / "runs" / model / f"seed_{seed}"
            ensure_dir(model_seed_dir)
            log_path = logs_dir / f"{model}_seed_{seed}.log"
            shared_init_path = shared_init_files.get(int(seed), seed_smiles_file)

            if spec.type == "builtin_local_evolution":
                cmd = render_builtin_local_evolution_command(
                    python_bin=python_bin,
                    repo_root=repo_root,
                    benchmark=args.benchmark,
                    custom_benchmark=custom_benchmark,
                    include_csv=include_csv,
                    exclude_csv=exclude_csv,
                    budget=effective_budget,
                    generations=args.generations,
                    population_size=args.population_size,
                    batch_size=args.batch_size,
                    tournament_k=args.local_evo_tournament_k,
                    elite_fraction=args.local_evo_elite_fraction,
                    immigrant_fraction=args.local_evo_immigrant_fraction,
                    parent_pool_fraction=args.local_evo_parent_pool_fraction,
                    stagnation_patience=args.local_evo_stagnation_patience,
                    stagnation_mutation_boost=args.local_evo_stagnation_mutation_boost,
                    seed_smiles_file=str(shared_init_path) if args.equal_initial_population else str(seed_smiles_file),
                    seed_pool_size=(args.population_size if args.equal_initial_population else args.seed_pool_size),
                    seed=seed,
                    model_output_dir=model_seed_dir,
                )
            elif spec.type == "command":
                if not spec.command:
                    msg = f"Model {model!r} has type=command but no command template."
                    if args.continue_on_error:
                        failures.append({"model": model, "seed": seed, "error": msg})
                        continue
                    raise ValueError(msg)
                if not examples_root:
                    msg = (
                        f"Model {model!r} requires --examples-root, but it was not provided. "
                        f"Set --examples-root to your local MolScore_examples path."
                    )
                    if args.continue_on_error:
                        failures.append({"model": model, "seed": seed, "error": msg})
                        continue
                    raise ValueError(msg)
                if not Path(examples_root).exists():
                    msg = (
                        f"--examples-root path does not exist: {examples_root}. "
                        f"Provide a valid local MolScore_examples checkout path."
                    )
                    if args.continue_on_error:
                        failures.append({"model": model, "seed": seed, "error": msg})
                        continue
                    raise ValueError(msg)
                cmd = render_command_template(
                    template=spec.command,
                    python_bin=python_bin,
                    repo_root=repo_root,
                    examples_root=examples_root,
                    benchmark=args.benchmark,
                    custom_benchmark=custom_benchmark,
                    include_csv=include_csv,
                    exclude_csv=exclude_csv,
                    budget=effective_budget,
                    generations=args.generations,
                    population_size=args.population_size,
                    batch_size=args.batch_size,
                    seed_smiles_file=str(seed_smiles_file),
                    seed_pool_size=args.seed_pool_size,
                    seed=seed,
                    shared_init_file=str(shared_init_path),
                    model_output_dir=model_seed_dir,
                    graphga_n_jobs=args.graphga_n_jobs,
                    crem_ncpu=args.crem_ncpu,
                    crem_replacements=args.crem_replacements,
                    smiles_rnn_device=smiles_rnn_device,
                )
            else:
                msg = f"Unsupported model type {spec.type!r} for model {model!r}."
                if args.continue_on_error:
                    failures.append({"model": model, "seed": seed, "error": msg})
                    continue
                raise ValueError(msg)

            jobs.append(
                {
                    "model": model,
                    "seed": seed,
                    "model_seed_dir": model_seed_dir,
                    "log_path": log_path,
                    "cmd": cmd,
                }
            )

    parallel_jobs = resolve_parallel_jobs(len(jobs))
    run_meta = {
        "benchmark": args.benchmark,
        "custom_benchmark": custom_benchmark,
        "include": include,
        "exclude": exclude,
        "budget_input": int(args.budget),
        "budget_effective": int(effective_budget),
        "generations": int(args.generations),
        "equal_initial_population": bool(args.equal_initial_population),
        "population_size": args.population_size,
        "batch_size": args.batch_size,
        "seed_smiles_file": str(seed_smiles_file),
        "seed_pool_size": args.seed_pool_size,
        "shared_initial_population_dir": str(shared_init_dir) if args.equal_initial_population else "",
        "requested_exports_dir": str((out_root / "exports").resolve()),
        "seeds": seeds,
        "models": selected_models,
        "model_spec_file": str(model_spec_path),
        "examples_root": examples_root,
        "python_bin": python_bin,
        "dry_run": bool(args.dry_run),
        "continue_on_error": bool(args.continue_on_error),
        "parallel_jobs": int(parallel_jobs),
        "n_jobs_total": int(len(jobs)),
        "graphga_n_jobs": int(args.graphga_n_jobs),
        "crem_ncpu": int(args.crem_ncpu),
        "crem_replacements": int(args.crem_replacements),
        "smiles_rnn_device": smiles_rnn_device,
        "local_evo_tournament_k": int(args.local_evo_tournament_k),
        "local_evo_elite_fraction": float(args.local_evo_elite_fraction),
        "local_evo_immigrant_fraction": float(args.local_evo_immigrant_fraction),
        "local_evo_parent_pool_fraction": float(args.local_evo_parent_pool_fraction),
        "local_evo_stagnation_patience": int(args.local_evo_stagnation_patience),
        "local_evo_stagnation_mutation_boost": int(args.local_evo_stagnation_mutation_boost),
        "bootstrap_scoring_envs": bool(args.bootstrap_scoring_envs),
        "scoring_env_clone_from": str(args.scoring_env_clone_from),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if args.bootstrap_scoring_envs and jobs:
        rc, bootstrap_log = bootstrap_scoring_envs(
            python_bin=python_bin,
            repo_root=repo_root,
            clone_from=str(args.scoring_env_clone_from).strip(),
            out_root=out_root,
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            msg = (
                f"Scoring env bootstrap failed with rc={rc}. "
                f"See log: {bootstrap_log}"
            )
            if args.continue_on_error:
                failures.append({"model": "bootstrap_scoring_envs", "seed": None, "error": msg})
            else:
                raise RuntimeError(msg)

    def execute_job(
        job: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any] | None, dict[str, Any]]:
        model = str(job["model"])
        seed = int(job["seed"])
        model_seed_dir = Path(job["model_seed_dir"])
        log_path = Path(job["log_path"])
        cmd = list(job["cmd"])

        started = datetime.now(timezone.utc)
        t0 = time.time()
        rc = run_command(cmd=cmd, log_path=log_path, dry_run=bool(args.dry_run))
        elapsed = float(max(0.0, time.time() - t0))
        finished = datetime.now(timezone.utc)
        runtime_row = {
            "model": model,
            "seed": seed,
            "status": "ok" if rc == 0 else "failed",
            "return_code": int(rc),
            "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": elapsed,
            "log_path": str(log_path.resolve()),
            "model_seed_dir": str(model_seed_dir.resolve()),
            "dry_run": bool(args.dry_run),
            "n_task_rows": 0,
            "n_generation_rows": 0,
        }
        if rc != 0:
            return (
                [],
                [],
                0,
                {
                    "model": model,
                    "seed": seed,
                    "error": f"Command failed with return code {rc}. See log: {log_path}",
                },
                runtime_row,
            )

        if not args.dry_run:
            write_guacamol_like_reports_under(model_seed_dir)
            written_exports = write_requested_generation_exports(
                model_output_dir=model_seed_dir,
                model=model,
                seed=seed,
                export_root=out_root / "exports",
            )
        else:
            written_exports = []

        rows = collect_task_scores(model_output_dir=model_seed_dir, model=model, seed=seed)
        gen_rows = collect_generation_metrics(model_output_dir=model_seed_dir, model=model, seed=seed)
        runtime_row["n_task_rows"] = int(len(rows))
        runtime_row["n_generation_rows"] = int(len(gen_rows))
        if not rows and not args.dry_run:
            return (
                [],
                gen_rows,
                len(written_exports),
                {
                    "model": model,
                    "seed": seed,
                    "error": f"No task scores found under {model_seed_dir}. Ensure command writes MolScore outputs there.",
                },
                runtime_row,
            )
        return rows, gen_rows, len(written_exports), None, runtime_row

    if jobs:
        print(f"[run] scheduling {len(jobs)} jobs with max_workers={parallel_jobs}", flush=True)
        with cf.ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_job = {executor.submit(execute_job, job): job for job in jobs}
            for future in cf.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    rows, gen_rows, n_exports, failure, runtime_row = future.result()
                except Exception as exc:
                    rows = []
                    gen_rows = []
                    n_exports = 0
                    failure = {
                        "model": str(job["model"]),
                        "seed": int(job["seed"]),
                        "error": f"Unhandled exception: {exc}",
                    }
                    runtime_row = {
                        "model": str(job["model"]),
                        "seed": int(job["seed"]),
                        "status": "failed",
                        "return_code": -1,
                        "started_at_utc": "",
                        "finished_at_utc": "",
                        "elapsed_seconds": float("nan"),
                        "log_path": str(Path(job["log_path"]).resolve()),
                        "model_seed_dir": str(Path(job["model_seed_dir"]).resolve()),
                        "dry_run": bool(args.dry_run),
                        "n_task_rows": 0,
                        "n_generation_rows": 0,
                    }

                if failure:
                    if args.continue_on_error:
                        failures.append(failure)
                    else:
                        for pending in future_to_job:
                            if pending is not future:
                                pending.cancel()
                        raise RuntimeError(str(failure["error"]))

                all_rows.extend(rows)
                all_gen_rows.extend(gen_rows)
                runtime_rows.append(runtime_row)
                print(
                    f"[done] model={job['model']} seed={job['seed']} rows={len(rows)} gen_rows={len(gen_rows)} exports={n_exports}"
                    + (" failed=1" if failure else ""),
                    flush=True,
                )

    raw_df = pd.DataFrame(all_rows)
    raw_path = out_root / "comparison_raw.tsv"
    raw_df.to_csv(raw_path, sep="\t", index=False)

    task_summary, overall = build_summary_tables(raw_df)
    task_path = out_root / "comparison_task_summary.tsv"
    overall_path = out_root / "comparison_overall.tsv"
    task_summary.to_csv(task_path, sep="\t", index=False)
    overall.to_csv(overall_path, sep="\t", index=False)

    gen_raw_df = pd.DataFrame(all_gen_rows)
    gen_raw_path = out_root / "comparison_generation_raw.tsv"
    gen_raw_df.to_csv(gen_raw_path, sep="\t", index=False)

    if gen_raw_df.empty:
        gen_summary_df = pd.DataFrame()
        gen_model_df = pd.DataFrame()
    else:
        gen_summary_df = (
            gen_raw_df.groupby(["model", "task", "generation"], as_index=False)
            .agg(
                mean_score=("mean_score", "mean"),
                mean_best_score_so_far=("best_score_so_far", "mean"),
                mean_best_score_gen=("best_score_gen", "mean"),
                mean_valid_rate=("valid_rate", "mean"),
                mean_valid_count=("valid_count", "mean"),
                mean_unique_valid_count=("unique_valid_count", "mean"),
                mean_new_unique_count=("new_unique_count", "mean"),
                mean_diversity=("diversity_mean", "mean"),
                n_runs=("seed", "nunique"),
            )
            .sort_values(["task", "generation", "model"], ascending=[True, True, True])
        )
        gen_model_df = (
            gen_summary_df.groupby(["model", "generation"], as_index=False)
            .agg(
                mean_score=("mean_score", "mean"),
                mean_best_score_so_far=("mean_best_score_so_far", "mean"),
                mean_best_score_gen=("mean_best_score_gen", "mean"),
                mean_valid_rate=("mean_valid_rate", "mean"),
                mean_diversity=("mean_diversity", "mean"),
                n_tasks=("task", "nunique"),
                n_runs=("n_runs", "mean"),
            )
            .sort_values(["generation", "model"], ascending=[True, True])
        )

    gen_summary_path = out_root / "comparison_generation_summary.tsv"
    gen_model_path = out_root / "comparison_generation_model_summary.tsv"
    gen_summary_df.to_csv(gen_summary_path, sep="\t", index=False)
    gen_model_df.to_csv(gen_model_path, sep="\t", index=False)

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_path = out_root / "model_runtime.tsv"
    runtime_df.to_csv(runtime_path, sep="\t", index=False)
    if runtime_df.empty:
        runtime_summary_df = pd.DataFrame()
    else:
        runtime_summary_df = (
            runtime_df.groupby(["model"], as_index=False)
            .agg(
                mean_elapsed_seconds=("elapsed_seconds", "mean"),
                min_elapsed_seconds=("elapsed_seconds", "min"),
                max_elapsed_seconds=("elapsed_seconds", "max"),
                n_runs=("seed", "count"),
                n_failed=("status", lambda s: int((s != "ok").sum())),
            )
            .sort_values(["mean_elapsed_seconds", "model"], ascending=[True, True])
        )
    runtime_summary_path = out_root / "model_runtime_summary.tsv"
    runtime_summary_df.to_csv(runtime_summary_path, sep="\t", index=False)

    failures_df = pd.DataFrame(failures)
    failures_path = out_root / "failures.tsv"
    failures_df.to_csv(failures_path, sep="\t", index=False)

    print(f"Output directory: {out_root}")
    print(f"Raw comparison:   {raw_path}")
    print(f"Task summary:     {task_path}")
    print(f"Overall summary:  {overall_path}")
    print(f"Generation raw:   {gen_raw_path}")
    print(f"Generation task:  {gen_summary_path}")
    print(f"Generation model: {gen_model_path}")
    print(f"Model runtime:    {runtime_path}")
    print(f"Runtime summary:  {runtime_summary_path}")
    print(f"Failures log:     {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
