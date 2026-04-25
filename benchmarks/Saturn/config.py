#!/usr/bin/env python3
from __future__ import annotations

"""
Central configuration for MolScore runs.

Usage model:
- CLI arguments override these defaults.
- `run_molscore_case.sh` can also override via environment variables.
- If neither is provided, these constants are used.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return int(default)


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return float(default)


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _env_str_any(names: tuple[str, ...], default: str) -> str:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return str(default)

# -----------------------------------------------------------------------------
# Repository layout (normally no need to change)
# -----------------------------------------------------------------------------
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MOLSCORE_ROOT: Final[Path] = REPO_ROOT / "molscore"
SOFTWARE_ROOT: Final[Path] = MOLSCORE_ROOT / "software"
OUTPUT_ROOT: Final[Path] = MOLSCORE_ROOT / "outputs"

# -----------------------------------------------------------------------------
# Shared run defaults (applies to evolve + comparison runners)
# -----------------------------------------------------------------------------
# Preset benchmark name recognized by MolScore (e.g., "GuacaMol", "MolOpt").
BENCHMARK_DEFAULT: Final[str] = "GuacaMol"
# Optional path to custom benchmark directory (empty = use preset above).
CUSTOM_BENCHMARK_DEFAULT: Final[str] = ""
# Comma-separated include/exclude task filters (empty = run all tasks).
INCLUDE_CSV_DEFAULT: Final[str] = ""
EXCLUDE_CSV_DEFAULT: Final[str] = ""

# Molecule evaluations budget per task.
BUDGET_DEFAULT: Final[int] = 100000000

# Evolution population size kept between generations.
POPULATION_SIZE_DEFAULT: Final[int] = 256
# New molecules scored per generation.
BATCH_SIZE_DEFAULT: Final[int] = 256

# 0 means no generation cap (run until budget/task completion).
MAX_GENERATIONS_DEFAULT: Final[int] = 50

# Tournament size for parent selection (higher = more exploitative).
TOURNAMENT_K_DEFAULT: Final[int] = 8

# Global random seed.
SEED_DEFAULT: Final[int] = 7

# Maximum number of seed SMILES loaded from seed file.
SEED_POOL_SIZE_DEFAULT: Final[int] = 2000

# If True, only evolution is run; random baseline is skipped.
SKIP_RANDOM_BASELINE_DEFAULT: Final[bool] = False

# Default input/output/config file locations.
SEED_SMILES_FILE_DEFAULT: Final[Path] = REPO_ROOT / "guacamol" / "guacamol_dataset" / "chembl24_canon_train.smiles"
# Initialization policy:
# - "current": use SEED_SMILES_FILE_DEFAULT / --seed-smiles-file (current behavior)
# - "graphga_zinc250k": use ZINC-250k seed file and sample initial population from it
INIT_POPULATION_MODE_DEFAULT: Final[str] = _env_str_any(
    ("SATURN_INIT_POPULATION_MODE", "MOLSCORE_INIT_POPULATION_MODE"),
    "current",
)
GRAPHGA_ZINC250K_SEED_SMILES_FILE_DEFAULT: Final[str] = _env_str_any(
    (
        "SATURN_GRAPHGA_ZINC250K_SEED_SMILES_FILE",
        "MOLSCORE_GRAPHGA_ZINC250K_SEED_SMILES_FILE",
    ),
    "",
)
# GraphGA paper protocol samples initial population from ZINC-250k.
GRAPHGA_ZINC250K_MIN_POOL_SIZE: Final[int] = 250000
MODEL_SPEC_FILE_DEFAULT: Final[Path] = SOFTWARE_ROOT / "model_specs.json"
OUTPUT_RUNS_DIR_DEFAULT: Final[Path] = OUTPUT_ROOT / "runs"
OUTPUT_COMPARISONS_DIR_DEFAULT: Final[Path] = OUTPUT_ROOT / "comparisons"

# -----------------------------------------------------------------------------
# Comparison runner defaults
# -----------------------------------------------------------------------------
# Comma-separated seeds used when no CLI --seeds is provided.
SEEDS_CSV_DEFAULT: Final[str] = "7,8,9"
# Comma-separated models from model_specs.json.
MODELS_CSV_DEFAULT: Final[str] = "local_evolution"
# Local path to MolScore_examples (needed for command-based models).
EXAMPLES_ROOT_DEFAULT: Final[str] = ""
# Python interpreter override (empty = auto-detect).
PYTHON_BIN_DEFAULT: Final[str] = ""
# Max concurrent (model, seed) jobs in benchmark_compare_models.
# 0 means auto: run all prepared jobs in parallel.
MODEL_RUN_PARALLEL_JOBS: Final[int] = 0
# Target generations for comparison runs (used by benchmark_compare_models).
COMPARISON_GENERATIONS_DEFAULT: Final[int] = 50
# If True, benchmark_compare_models builds a per-seed shared init population used by all models.
EQUAL_INITIAL_POPULATION_DEFAULT: Final[bool] = True

# -----------------------------------------------------------------------------
# Evolution strategy behavior
# -----------------------------------------------------------------------------
# Elements allowed in mutation operators: C, N, O, F, P, S, Cl, Br.
ALLOWED_ATOMIC_NUMBERS: Final[tuple[int, ...]] = (6, 7, 8, 9, 15, 16, 17, 35)
# Fallback initial seed set if seed file is missing or empty.
FALLBACK_SEED_SMILES: Final[tuple[str, ...]] = (
    "CCO",
    "CCN",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccncc1",
    "CCOc1ccccc1",
    "CCN(CC)CC",
    "CC(C)NCC(O)CO",
    "CC1=CC=CC=C1",
    "CCOC(=O)N1CCCC1",
    "COc1ccc(cc1)N",
    "CC1CCCCC1",
    "CC(C)OC(=O)N",
    "CN1CCN(CC1)C",
    "CC(C)C(=O)O",
    "COc1ccccc1O",
    "CCN1CCCC1",
    "CCOC(=O)c1ccccc1",
    "CC(C)N",
    "CCC(=O)N",
)

# Mutation depth while creating initial population from seeds.
INIT_MUTATION_MAX_STEPS: Final[int] = 2
# Mutation depth for offspring generation in evolution loop.
OFFSPRING_MUTATION_MAX_STEPS: Final[int] = 3
# Retry factor for generating unique offspring per batch.
OFFSPRING_ATTEMPT_FACTOR: Final[int] = 30
# K used for mean-top-K score reporting.
TOPK_MEAN_K: Final[int] = 10
# Number of top molecules persisted to generator_top_molecules.csv.
TOP_MOLECULES_TO_SAVE: Final[int] = 200
# Refresh GuacaMol-style TSV summaries every N generations during local evolution.
REPORT_WRITE_EVERY_N_GEN: Final[int] = 5

# Local evolution selection/exploration defaults.
LOCAL_EVO_ELITE_FRACTION: Final[float] = 0.15
LOCAL_EVO_IMMIGRANT_FRACTION: Final[float] = 0.10
LOCAL_EVO_PARENT_POOL_FRACTION: Final[float] = 0.50
LOCAL_EVO_STAGNATION_PATIENCE: Final[int] = 12
LOCAL_EVO_STAGNATION_MUTATION_BOOST: Final[int] = 2
LOCAL_EVO_MUTATION_STEP_CAP: Final[int] = _env_int_any(
    ("SATURN_LOCAL_EVO_MUTATION_STEP_CAP", "MOLSCORE_LOCAL_EVO_MUTATION_STEP_CAP"),
    8,
)
# BRICS is explicitly disabled by default in this SATURN workflow.
LOCAL_EVO_DISABLE_BRICS: Final[bool] = _env_bool_any(
    ("SATURN_LOCAL_EVO_DISABLE_BRICS", "MOLSCORE_LOCAL_EVO_DISABLE_BRICS"),
    True,
)
# Fraction of non-immigrant offspring generated through BRICS crossover.
# Kept at zero by default to avoid BRICS exploration.
LOCAL_EVO_CROSSOVER_FRACTION: Final[float] = _env_float_any(
    ("SATURN_LOCAL_EVO_CROSSOVER_FRACTION", "MOLSCORE_LOCAL_EVO_CROSSOVER_FRACTION"),
    0.00,
)
# Fraction of non-immigrant offspring generated through BRICS fragment replacement.
# Kept at zero by default to avoid BRICS exploration.
LOCAL_EVO_FRAGMENT_REPLACE_FRACTION: Final[float] = _env_float_any(
    ("SATURN_LOCAL_EVO_FRAGMENT_REPLACE_FRACTION", "MOLSCORE_LOCAL_EVO_FRAGMENT_REPLACE_FRACTION"),
    0.00,
)
# Minimum fragment size (heavy atoms) when extracting BRICS fragments.
LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE: Final[int] = _env_int_any(
    ("SATURN_LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", "MOLSCORE_LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE"),
    2,
)
# BRICS assembly search depth.
LOCAL_EVO_BRICS_MAX_DEPTH: Final[int] = _env_int_any(
    ("SATURN_LOCAL_EVO_BRICS_MAX_DEPTH", "MOLSCORE_LOCAL_EVO_BRICS_MAX_DEPTH"),
    4,
)
# Number of seed molecules sampled to build BRICS fragment pool.
LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE: Final[int] = _env_int_any(
    ("SATURN_LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", "MOLSCORE_LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE"),
    512,
)
# If True, disable explicitly size-reducing single-molecule mutations
# (leaf-atom deletion and bond deletion) to reduce bias toward smaller molecules.
LOCAL_EVO_DISABLE_SIZE_REDUCING_MUTATIONS: Final[bool] = _env_bool_any(
    (
        "SATURN_LOCAL_EVO_DISABLE_SIZE_REDUCING_MUTATIONS",
        "MOLSCORE_LOCAL_EVO_DISABLE_SIZE_REDUCING_MUTATIONS",
    ),
    False,
)

# Bootstrap PIDGIN/ms_molopt scoring envs before benchmark launches.
BOOTSTRAP_SCORING_ENVS_DEFAULT: Final[bool] = True
SCORING_ENV_CLONE_FROM_DEFAULT: Final[str] = ""


def as_repo_relative(path: Path) -> str:
    """Represent path relative to repo root when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_from_repo(path_like: str | Path) -> Path:
    """Resolve relative paths from repo root; keep absolute paths unchanged."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


# -----------------------------------------------------------------------------
# String defaults used directly by argparse
# -----------------------------------------------------------------------------
SEED_SMILES_FILE_DEFAULT_STR: Final[str] = as_repo_relative(SEED_SMILES_FILE_DEFAULT)
MODEL_SPEC_FILE_DEFAULT_STR: Final[str] = as_repo_relative(MODEL_SPEC_FILE_DEFAULT)
OUTPUT_RUNS_DIR_DEFAULT_STR: Final[str] = as_repo_relative(OUTPUT_RUNS_DIR_DEFAULT)
OUTPUT_COMPARISONS_DIR_DEFAULT_STR: Final[str] = as_repo_relative(OUTPUT_COMPARISONS_DIR_DEFAULT)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime knobs used by the latent encoder/decoder/evolution utility modules."""

    THETA_INIT_STD: float = 0.80
    GAUSS_STD_THETA: float = 0.38
    P_PARAM_NOISE: float = 0.80
    P_ROW_RESET: float = 0.09
    CLIP_THETA_NORM: float = 7.0
    GENERATIONS: int = MAX_GENERATIONS_DEFAULT

    DECODE_TOPK: int = 15
    DECODE_TEMP: float = 1.5
    P_SAMPLE: float = 0.95

    MIN_ATOMS_BEFORE_PAD: int = 28
    MIN_ATOMS_WARMUP_START: int = 24
    MIN_MW_WARMUP_GENS: int = 0
    MIN_TOKENS_BEFORE_PAD: int = 24
    EARLY_PAD_CUTOFF: int = 45
    PAD_PENALTY: float = 0.1

    P_START_RING_PLAN: float = 0.60
    RING_PLAN_LEN_BIAS_6: float = 0.78
    RING_PLAN_AROM_BIAS: float = 0.92
    RING_PLAN_MIN_REMAIN: int = 14
    RING_PLAN_ALLOW_MACROS: bool = True

    STRUCT_FORBID_LAST_N: int = 5
    STRUCT_FORBID_CONSECUTIVE: bool = False
    STRUCT_SUPPRESS_BASE: float = 0.10

    FP_RADIUS: int = 2
    FP_NBITS: int = 2048


CFG: Final[RuntimeConfig] = RuntimeConfig()
