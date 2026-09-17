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


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else str(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(default)
    values = tuple(token.strip() for token in raw.replace(";", ",").split(",") if token.strip())
    return values or tuple(default)

# -----------------------------------------------------------------------------
# Repository layout (normally no need to change)
# -----------------------------------------------------------------------------
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
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
LOCAL_EVO_GENERATOR_DEFAULT: Final[str] = os.environ.get("MOLSCORE_LOCAL_EVO_GENERATOR", "spectral").strip() or "spectral"
FREQUENCY_MODE_DEFAULT: Final[str] = os.environ.get("MOLSCORE_FREQUENCY_MODE", "full-spectrum").strip() or "full-spectrum"
LOCAL_EVO_ELITE_FRACTION: Final[float] = 0.15
LOCAL_EVO_IMMIGRANT_FRACTION: Final[float] = 0.10
LOCAL_EVO_PARENT_POOL_FRACTION: Final[float] = 0.50
LOCAL_EVO_STAGNATION_PATIENCE: Final[int] = 12
LOCAL_EVO_STAGNATION_MUTATION_BOOST: Final[int] = 2
LOCAL_EVO_MUTATION_STEP_CAP: Final[int] = _env_int("MOLSCORE_LOCAL_EVO_MUTATION_STEP_CAP", 8)
# Fraction of non-immigrant offspring generated via BRICS crossover.
# Kept off by default for faster local-evolution runs.
LOCAL_EVO_CROSSOVER_FRACTION: Final[float] = _env_float("MOLSCORE_LOCAL_EVO_CROSSOVER_FRACTION", 0.00)
# Fraction of non-immigrant offspring generated via BRICS fragment replacement.
# Kept off by default for faster local-evolution runs.
LOCAL_EVO_FRAGMENT_REPLACE_FRACTION: Final[float] = _env_float("MOLSCORE_LOCAL_EVO_FRAGMENT_REPLACE_FRACTION", 0.00)
# Minimum BRICS fragment atom count during decomposition.
LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE: Final[int] = _env_int("MOLSCORE_LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", 2)
# BRICS assembly search depth.
LOCAL_EVO_BRICS_MAX_DEPTH: Final[int] = _env_int("MOLSCORE_LOCAL_EVO_BRICS_MAX_DEPTH", 4)
# Number of seed molecules sampled to build BRICS fragment pool.
LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE: Final[int] = _env_int("MOLSCORE_LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", 512)

# SpectralMol keeps Fourier Theta as the stored genotype and evolves it by
# default. Phenotype-space proposals are opt-in compatibility helpers: when
# enabled, SMILES proposals are encoded back into Theta before scoring.
SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION: Final[float] = _env_float("MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION", 0.0)
SPECTRAL_BRICS_CROSSOVER_FRACTION: Final[float] = _env_float("MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION", 0.0)
SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION: Final[float] = _env_float("MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION", 0.0)
SPECTRAL_IMMIGRANT_MUTATION_STEPS: Final[int] = _env_int("MOLSCORE_SPECTRAL_IMMIGRANT_MUTATION_STEPS", 2)
SPECTRAL_PARENT_MUTATION_STEPS: Final[int] = _env_int("MOLSCORE_SPECTRAL_PARENT_MUTATION_STEPS", 2)
SPECTRAL_THETA_CROSSOVER_PROBABILITY: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_CROSSOVER_PROBABILITY", 0.45)
SPECTRAL_THETA_LOCAL_SEARCH_FRACTION: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_LOCAL_SEARCH_FRACTION", 0.50)
SPECTRAL_THETA_LOCAL_TOP_FRACTION: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_LOCAL_TOP_FRACTION", 0.10)
SPECTRAL_THETA_LOCAL_MUTATION_STEPS: Final[int] = _env_int("MOLSCORE_SPECTRAL_THETA_LOCAL_MUTATION_STEPS", 1)
SPECTRAL_THETA_LOCAL_SIGMA_SCALE: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_LOCAL_SIGMA_SCALE", 0.30)
SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALE: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALE", 0.35)
SPECTRAL_THETA_LOCAL_ROW_RESET_SCALE: Final[float] = _env_float("MOLSCORE_SPECTRAL_THETA_LOCAL_ROW_RESET_SCALE", 0.0)
SPECTRAL_THETA_LOCAL_SIGMA_SCALES: Final[str] = _env_str("MOLSCORE_SPECTRAL_THETA_LOCAL_SIGMA_SCALES", "0.15,0.35,0.80")
SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALES: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALES",
    "0.25,0.55,1.00",
)
SPECTRAL_THETA_LOCAL_ROW_RESET_SCALES: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_THETA_LOCAL_ROW_RESET_SCALES",
    "0.00,0.00,0.20",
)
SPECTRAL_THETA_LOCAL_MUTATION_STEP_SCHEDULE: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_THETA_LOCAL_MUTATION_STEP_SCHEDULE",
    "1,1,2",
)
SPECTRAL_THETA_TOKEN_MUTATION_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION",
    0.35,
)
SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION",
    0.10,
)
SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS",
    2,
)
SPECTRAL_THETA_TOKEN_INSERT_PROB: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB",
    0.20,
)
SPECTRAL_THETA_TOKEN_DELETE_PROB: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB",
    0.05,
)
SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB",
    0.25,
)
SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION",
    0.70,
)
SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION",
    0.03,
)
SPECTRAL_TASK_TARGET_SAMPLE_BIAS: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_TASK_TARGET_SAMPLE_BIAS",
    0.0,
)
SPECTRAL_THETA_TARGET_ANALOG_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_FRACTION",
    0.0,
)
SPECTRAL_THETA_TARGET_ANALOG_MUTATION_STEPS: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_MUTATION_STEPS",
    2,
)
SPECTRAL_THETA_TARGET_ANALOG_SIGMA_SCALE: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_SIGMA_SCALE",
    1.0,
)
SPECTRAL_MPO_DECODE_STYLE: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_MPO_DECODE_STYLE",
    "druglike",
)
SPECTRAL_THETA_TOKEN_MUTATION_BLEND: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND",
    0.80,
)
SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION",
    0.25,
)
SPECTRAL_THETA_DIFFERENTIAL_FRACTION: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION",
    0.15,
)
SPECTRAL_THETA_DIFFERENTIAL_SCALE: Final[float] = _env_float(
    "MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE",
    0.45,
)
SPECTRAL_ELITE_MACRO_WEIGHT: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT",
    4,
)
SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH: Final[bool] = _env_bool(
    "MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH",
    True,
)
SPECTRAL_ENABLE_TASK_TARGET_MACROS: Final[bool] = _env_bool("MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS", True)
SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS: Final[bool] = _env_bool(
    "MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS",
    True,
)
SPECTRAL_TASK_TARGET_FULL_MACRO_MAX: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX", 16)
SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX",
    "TASKT",
)
SPECTRAL_TASK_TARGET_MACRO_MAX: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX", 192)
SPECTRAL_TASK_TARGET_MACRO_MIN_N: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N", 2)
SPECTRAL_TASK_TARGET_MACRO_MAX_N: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N", 16)
SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS", 2)
SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY",
    1,
)
SPECTRAL_TASK_TARGET_MACRO_PREFIX: Final[str] = _env_str("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_PREFIX", "TASKM")
SPECTRAL_TASK_TARGET_MACRO_WEIGHT: Final[int] = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT", 8)
SPECTRAL_ENABLE_TASK_TARGET_WINDOW_MACROS: Final[bool] = _env_bool(
    "MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_WINDOW_MACROS",
    True,
)
SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX",
    128,
)
SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_N: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_N",
    2,
)
SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N",
    12,
)
SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_ATOMS: Final[int] = _env_int(
    "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_ATOMS",
    2,
)
SPECTRAL_TASK_TARGET_WINDOW_MACRO_PREFIX: Final[str] = _env_str(
    "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_PREFIX",
    "TASKW",
)
SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS: Final[bool] = _env_bool(
    "MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS",
    True,
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

# -----------------------------------------------------------------------------
# Runtime knobs for the latent encoder / decoder / evolution utility modules.
# fourier_theta.py, decoder.py and embedding.py import `CFG` from this module.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeConfig:
    L: int = _env_int("MOLSCORE_SPECTRAL_L", 32)
    K: int = _env_int("MOLSCORE_SPECTRAL_K", 16)
    D: int = _env_int("MOLSCORE_SPECTRAL_D", 32)
    GENERATIONS: int = _env_int("MOLSCORE_SPECTRAL_GENERATIONS", MAX_GENERATIONS_DEFAULT)

    PAD_TOKEN: str = "[PAD]"
    EMBED_SEED: int = _env_int("MOLSCORE_EMBED_SEED", 13)
    EMBED_TARGET_STD: float = _env_float("MOLSCORE_EMBED_TARGET_STD", 1.0)
    SPECTRAL_EMBED_MEDCHEM_BIAS: float = _env_float("MOLSCORE_SPECTRAL_EMBED_MEDCHEM_BIAS", 0.0)
    SPECTRAL_EMBED_MACRO_EXPANSION_BLEND: float = _env_float(
        "MOLSCORE_SPECTRAL_EMBED_MACRO_EXPANSION_BLEND",
        0.0,
    )
    ALLOWED_ELEMENTS: tuple[str, ...] = _env_tuple(
        "MOLSCORE_SPECTRAL_ALLOWED_ELEMENTS",
        ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H"),
    )
    ALLOW_CHARGED_TOKENS: bool = _env_bool("MOLSCORE_SPECTRAL_ALLOW_CHARGED_TOKENS", True)

    THETA_INIT_STD: float = _env_float("MOLSCORE_THETA_INIT_STD", 0.75)
    GAUSS_STD_THETA: float = _env_float("MOLSCORE_GAUSS_STD_THETA", 0.18)
    P_PARAM_NOISE: float = _env_float("MOLSCORE_P_PARAM_NOISE", 0.10)
    P_ROW_RESET: float = _env_float("MOLSCORE_P_ROW_RESET", 0.015)
    CLIP_THETA_NORM: float = _env_float("MOLSCORE_CLIP_THETA_NORM", 4.0)
    FREQUENCY_LOW_CUTOFF_FRACTION: float = _env_float("MOLSCORE_FREQUENCY_LOW_CUTOFF_FRACTION", 0.5)
    FREQUENCY_ZERO_INACTIVE_ROWS: bool = _env_bool("MOLSCORE_FREQUENCY_ZERO_INACTIVE_ROWS", True)

    DECODE_ATTEMPTS: int = _env_int("MOLSCORE_DECODE_ATTEMPTS", 8)
    DECODE_TOPK: int = _env_int("MOLSCORE_DECODE_TOPK", 16)
    DECODE_TEMP: float = _env_float("MOLSCORE_DECODE_TEMP", 0.75)
    DECODE_FP32: bool = _env_bool("MOLSCORE_DECODE_FP32", True)
    DECODE_INCLUDE_GREEDY_NEAREST: bool = _env_bool("MOLSCORE_DECODE_INCLUDE_GREEDY_NEAREST", False)
    P_SAMPLE: float = _env_float("MOLSCORE_P_SAMPLE", 0.85)
    ENABLE_MACROS: bool = _env_bool("MOLSCORE_ENABLE_MACROS", True)
    SPECTRAL_ENABLE_SEED_MACROS: bool = _env_bool("MOLSCORE_SPECTRAL_ENABLE_SEED_MACROS", True)
    SPECTRAL_SEED_MACRO_MAX: int = _env_int("MOLSCORE_SPECTRAL_SEED_MACRO_MAX", 256)
    SPECTRAL_SEED_MACRO_MIN_N: int = _env_int("MOLSCORE_SPECTRAL_SEED_MACRO_MIN_N", 3)
    SPECTRAL_SEED_MACRO_MAX_N: int = _env_int("MOLSCORE_SPECTRAL_SEED_MACRO_MAX_N", 10)
    SPECTRAL_SEED_MACRO_MIN_ATOMS: int = _env_int("MOLSCORE_SPECTRAL_SEED_MACRO_MIN_ATOMS", 3)
    SPECTRAL_SEED_MACRO_MIN_FREQUENCY: int = _env_int("MOLSCORE_SPECTRAL_SEED_MACRO_MIN_FREQUENCY", 1)
    SPECTRAL_SEED_MACRO_PREFIX: str = _env_str("MOLSCORE_SPECTRAL_SEED_MACRO_PREFIX", "SEEDM")
    SPECTRAL_TASK_AWARE_DECODE: bool = _env_bool("MOLSCORE_SPECTRAL_TASK_AWARE_DECODE", True)
    SPECTRAL_TASK_AWARE_DECODE_CANDIDATES: int = _env_int("MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES", 1)
    SPECTRAL_ELITE_MACRO_REFRESH_EVERY: int = _env_int("MOLSCORE_SPECTRAL_ELITE_MACRO_REFRESH_EVERY", 25)
    SPECTRAL_ELITE_MACRO_TOP_N: int = _env_int("MOLSCORE_SPECTRAL_ELITE_MACRO_TOP_N", 96)
    SPECTRAL_ELITE_MACRO_SEED_KEEP: int = _env_int("MOLSCORE_SPECTRAL_ELITE_MACRO_SEED_KEEP", 256)
    SPECTRAL_THETA_TOKEN_MUTATION_FRACTION: float = _env_float("MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION", 0.35)
    SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION: float = _env_float(
        "MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION",
        0.10,
    )
    SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS: int = _env_int("MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS", 2)
    SPECTRAL_THETA_TOKEN_INSERT_PROB: float = _env_float("MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB", 0.20)
    SPECTRAL_THETA_TOKEN_DELETE_PROB: float = _env_float("MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB", 0.05)
    SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB: float = _env_float("MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB", 0.25)
    SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION: float = _env_float(
        "MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION",
        0.70,
    )
    SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION: float = _env_float(
        "MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION",
        0.03,
    )
    SPECTRAL_TASK_TARGET_SAMPLE_BIAS: float = _env_float(
        "MOLSCORE_SPECTRAL_TASK_TARGET_SAMPLE_BIAS",
        0.0,
    )
    SPECTRAL_THETA_TARGET_ANALOG_FRACTION: float = _env_float(
        "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_FRACTION",
        0.0,
    )
    SPECTRAL_THETA_TARGET_ANALOG_MUTATION_STEPS: int = _env_int(
        "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_MUTATION_STEPS",
        2,
    )
    SPECTRAL_THETA_TARGET_ANALOG_SIGMA_SCALE: float = _env_float(
        "MOLSCORE_SPECTRAL_THETA_TARGET_ANALOG_SIGMA_SCALE",
        1.0,
    )
    SPECTRAL_MPO_DECODE_STYLE: str = _env_str(
        "MOLSCORE_SPECTRAL_MPO_DECODE_STYLE",
        "druglike",
    )
    SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE: float = _env_float("MOLSCORE_SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE", 0.0)
    SPECTRAL_THETA_TOKEN_MUTATION_BLEND: float = _env_float("MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND", 0.80)
    SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION: float = _env_float("MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION", 0.25)
    SPECTRAL_THETA_DIFFERENTIAL_FRACTION: float = _env_float("MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION", 0.15)
    SPECTRAL_THETA_DIFFERENTIAL_SCALE: float = _env_float("MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE", 0.45)
    SPECTRAL_ELITE_MACRO_WEIGHT: int = _env_int("MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT", 4)
    SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH: bool = _env_bool(
        "MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH",
        True,
    )
    SPECTRAL_ENABLE_TASK_TARGET_MACROS: bool = _env_bool("MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS", True)
    SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS: bool = _env_bool(
        "MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS",
        True,
    )
    SPECTRAL_TASK_TARGET_FULL_MACRO_MAX: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX", 16)
    SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX: str = _env_str(
        "MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX",
        "TASKT",
    )
    SPECTRAL_TASK_TARGET_MACRO_MAX: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX", 192)
    SPECTRAL_TASK_TARGET_MACRO_MIN_N: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N", 2)
    SPECTRAL_TASK_TARGET_MACRO_MAX_N: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N", 16)
    SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS", 2)
    SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY: int = _env_int(
        "MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY",
        1,
    )
    SPECTRAL_TASK_TARGET_MACRO_PREFIX: str = _env_str("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_PREFIX", "TASKM")
    SPECTRAL_TASK_TARGET_MACRO_WEIGHT: int = _env_int("MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT", 8)
    SPECTRAL_ENABLE_TASK_TARGET_WINDOW_MACROS: bool = _env_bool(
        "MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_WINDOW_MACROS",
        True,
    )
    SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX: int = _env_int(
        "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX",
        128,
    )
    SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_N: int = _env_int(
        "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_N",
        2,
    )
    SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N: int = _env_int(
        "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N",
        12,
    )
    SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_ATOMS: int = _env_int(
        "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_ATOMS",
        2,
    )
    SPECTRAL_TASK_TARGET_WINDOW_MACRO_PREFIX: str = _env_str(
        "MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_PREFIX",
        "TASKW",
    )
    SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS: bool = _env_bool(
        "MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS",
        True,
    )

    MIN_ATOMS_BEFORE_PAD: int = _env_int("MOLSCORE_MIN_ATOMS_BEFORE_PAD", 8)
    MIN_ATOMS_WARMUP_START: int = _env_int("MOLSCORE_MIN_ATOMS_WARMUP_START", 3)
    MIN_MW_WARMUP_GENS: int = _env_int("MOLSCORE_MIN_MW_WARMUP_GENS", 12)
    MIN_TOKENS_BEFORE_PAD: int = _env_int("MOLSCORE_MIN_TOKENS_BEFORE_PAD", 4)
    EARLY_PAD_CUTOFF: int = _env_int("MOLSCORE_EARLY_PAD_CUTOFF", 8)
    PAD_PENALTY: float = _env_float("MOLSCORE_PAD_PENALTY", 0.02)

    P_START_RING_PLAN: float = _env_float("MOLSCORE_P_START_RING_PLAN", 0.06)
    RING_PLAN_LEN_BIAS_6: float = _env_float("MOLSCORE_RING_PLAN_LEN_BIAS_6", 0.75)
    RING_PLAN_AROM_BIAS: float = _env_float("MOLSCORE_RING_PLAN_AROM_BIAS", 0.75)
    RING_PLAN_MIN_REMAIN: int = _env_int("MOLSCORE_RING_PLAN_MIN_REMAIN", 7)
    RING_PLAN_ALLOW_MACROS: bool = _env_bool("MOLSCORE_RING_PLAN_ALLOW_MACROS", False)

    STRUCT_FORBID_LAST_N: int = _env_int("MOLSCORE_STRUCT_FORBID_LAST_N", 2)
    STRUCT_FORBID_CONSECUTIVE: bool = _env_bool("MOLSCORE_STRUCT_FORBID_CONSECUTIVE", True)
    STRUCT_SUPPRESS_BASE: float = _env_float("MOLSCORE_STRUCT_SUPPRESS_BASE", 0.25)

    MACRO_FORBID_CONSECUTIVE: bool = _env_bool("MOLSCORE_MACRO_FORBID_CONSECUTIVE", True)
    MACRO_FORBID_RING_AFTER_MACRO: bool = _env_bool("MOLSCORE_MACRO_FORBID_RING_AFTER_MACRO", True)
    MAX_CONSEC_HETERO_TOKENS: int = _env_int("MOLSCORE_MAX_CONSEC_HETERO_TOKENS", 3)
    HETERO_STREAK_PENALTY: float = _env_float("MOLSCORE_HETERO_STREAK_PENALTY", 0.15)
    USE_CUDA: bool = _env_bool("MOLSCORE_USE_CUDA", False)
    USE_MPS: bool = _env_bool("MOLSCORE_USE_MPS", False)
    CUDA_DEVICE: int = _env_int("MOLSCORE_CUDA_DEVICE", 0)

    MACRO_WHITELIST: tuple = ()
    MACRO_BLACKLIST: tuple = ()


# Singleton instance imported by fourier_theta, decoder, embedding.
CFG: Final[RuntimeConfig] = RuntimeConfig()
