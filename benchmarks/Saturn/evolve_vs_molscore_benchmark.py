#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, rdchem

# Ensure repo root and Core/ are on sys.path so 'core' and oracle imports resolve
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
_CORE_ROOT = _REPO_ROOT / "core"
for _p in [str(_REPO_ROOT), str(_CORE_ROOT)]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import config as cfg  # local Benchmarks/Saturn/config.py
from guacamol_reports import write_guacamol_like_reports  # local

DEFAULT_SEEDS: list[str] = list(cfg.FALLBACK_SEED_SMILES)
ALLOWED_ATOMIC_NUMBERS: list[int] = list(cfg.ALLOWED_ATOMIC_NUMBERS)
BOND_TYPES: list[rdchem.BondType] = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
]
DEFAULT_SATURN_REPO_ROOT: Path = _CORE_ROOT.resolve()
DEFAULT_SATURN_ORACLE_CONFIG: Path = (Path(__file__).resolve().parent / "saturn_oracle_config.example.json").resolve()
_SCORE_ADAPTER_CACHE: dict[int, Callable[[list[str], int], object]] = {}
_BRICS_FRAGMENT_CACHE: dict[tuple[str, int], tuple[str, ...]] = {}
INIT_MODE_CURRENT = "current"
INIT_MODE_GRAPHGA_ZINC250K = "graphga_zinc250k"
INIT_MODE_CHOICES: tuple[str, ...] = (INIT_MODE_CURRENT, INIT_MODE_GRAPHGA_ZINC250K)
INIT_MODE_CURRENT = "current"
INIT_MODE_GRAPHGA_ZINC250K = "graphga_zinc250k"
INIT_MODE_CHOICES: tuple[str, ...] = (INIT_MODE_CURRENT, INIT_MODE_GRAPHGA_ZINC250K)


@dataclass
class StrategyResult:
    strategy: str
    benchmark: str
    task: str
    run_dir: str
    score_column: str
    budget: int
    evaluated: int
    generations: int
    best_score: float
    mean_top10: float
    unique_molecules: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an evolutionary molecule generator against MolScoreBenchmark and "
            "compare it to a non-evolutionary baseline on the same MolScore preset tasks."
        )
    )
    p.add_argument("--benchmark", default=cfg.BENCHMARK_DEFAULT, help="MolScore benchmark preset (e.g., GuacaMol, MolOpt).")
    p.add_argument(
        "--custom-benchmark",
        default=cfg.CUSTOM_BENCHMARK_DEFAULT,
        help="Directory with task JSON files. Overrides --benchmark when set.",
    )
    p.add_argument("--include", default=cfg.INCLUDE_CSV_DEFAULT, help="Comma-separated task names to include.")
    p.add_argument("--exclude", default=cfg.EXCLUDE_CSV_DEFAULT, help="Comma-separated task names to exclude.")
    p.add_argument("--output-dir", default=cfg.OUTPUT_RUNS_DIR_DEFAULT_STR, help="Root output directory.")
    p.add_argument("--budget", type=int, default=cfg.BUDGET_DEFAULT, help="Molecule budget per task.")
    p.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE_DEFAULT, help="Population size for evolutionary strategy.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE_DEFAULT, help="Scoring batch size per generation.")
    p.add_argument("--max-generations", type=int, default=cfg.MAX_GENERATIONS_DEFAULT, help="Optional hard stop; 0 disables generation cap.")
    p.add_argument("--tournament-k", type=int, default=cfg.TOURNAMENT_K_DEFAULT, help="Tournament size for parent selection.")
    p.add_argument(
        "--elite-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_ELITE_FRACTION", 0.15)),
        help="Top fraction preserved each generation (local evolution only).",
    )
    p.add_argument(
        "--immigrant-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_IMMIGRANT_FRACTION", 0.10)),
        help="Fraction of each generation sampled from seed-pool mutations (local evolution only).",
    )
    p.add_argument(
        "--parent-pool-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_PARENT_POOL_FRACTION", 0.50)),
        help="Fraction of current population eligible for parent selection (local evolution only).",
    )
    p.add_argument(
        "--stagnation-patience",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_PATIENCE", 12)),
        help="If no best-score improvement for this many generations, increase exploration.",
    )
    p.add_argument(
        "--stagnation-mutation-boost",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_MUTATION_BOOST", 2)),
        help="Extra mutation depth when stagnating.",
    )
    p.add_argument("--seed", type=int, default=cfg.SEED_DEFAULT, help="Random seed.")
    p.add_argument(
        "--seed-smiles-file",
        default=cfg.SEED_SMILES_FILE_DEFAULT_STR,
        help="Optional seed SMILES file for initial population.",
    )
    p.add_argument("--seed-pool-size", type=int, default=cfg.SEED_POOL_SIZE_DEFAULT, help="Max seed SMILES loaded from file.")
    p.add_argument(
        "--init-population-mode",
        choices=INIT_MODE_CHOICES,
        default=str(getattr(cfg, "INIT_POPULATION_MODE_DEFAULT", INIT_MODE_CURRENT)),
        help=(
            "Initialization policy. 'current' uses --seed-smiles-file; "
            "'graphga_zinc250k' uses --graphga-zinc250k-seed-smiles-file and enforces "
            "a ZINC-250k-sized seed pool for GraphGA-style initialization."
        ),
    )
    p.add_argument(
        "--graphga-zinc250k-seed-smiles-file",
        default=str(getattr(cfg, "GRAPHGA_ZINC250K_SEED_SMILES_FILE_DEFAULT", "")),
        help=(
            "Path to ZINC-250k SMILES used when --init-population-mode graphga_zinc250k. "
            "Can also be provided with SATURN_GRAPHGA_ZINC250K_SEED_SMILES_FILE."
        ),
    )
    p.add_argument(
        "--objective-backend",
        choices=("molscore", "saturn"),
        default="molscore",
        help="Scoring backend: MolScore benchmark tasks or SATURN oracle objective.",
    )
    p.add_argument(
        "--saturn-repo-root",
        default=str(DEFAULT_SATURN_REPO_ROOT),
        help="Path to SATURN repository root (used when --objective-backend saturn).",
    )
    p.add_argument(
        "--saturn-oracle-config",
        default=str(DEFAULT_SATURN_ORACLE_CONFIG),
        help="Path to SATURN oracle JSON config (full config or object containing top-level 'oracle').",
    )
    p.add_argument(
        "--saturn-oracle-config-key",
        default="oracle",
        help="Top-level key containing oracle payload when SATURN config is a full SATURN run config.",
    )
    p.add_argument(
        "--saturn-task-name",
        default="saturn_objective",
        help="Task label used in outputs for SATURN objective mode.",
    )
    p.add_argument(
        "--saturn-apply-diversity-penalty",
        action="store_true",
        default=False,
        help="Apply SATURN diversity filter penalties across calls.",
    )
    p.add_argument(
        "--saturn-diversity-bucket-size",
        type=int,
        default=10,
        help="SATURN diversity filter bucket size.",
    )
    p.add_argument(
        "--saturn-invalid-score",
        type=float,
        default=0.0,
        help="Score assigned to invalid/unscorable molecules in SATURN mode.",
    )
    saturn_repeat_group = p.add_mutually_exclusive_group()
    saturn_repeat_group.add_argument(
        "--saturn-allow-oracle-repeats",
        dest="saturn_allow_oracle_repeats",
        action="store_true",
        help="Override SATURN oracle config and allow repeat calls to identical molecules.",
    )
    saturn_repeat_group.add_argument(
        "--saturn-disallow-oracle-repeats",
        dest="saturn_allow_oracle_repeats",
        action="store_false",
        help="Override SATURN oracle config and disallow repeat calls to identical molecules.",
    )
    p.set_defaults(saturn_allow_oracle_repeats=None)
    p.add_argument(
        "--skip-random-baseline",
        action="store_true",
        default=cfg.SKIP_RANDOM_BASELINE_DEFAULT,
        help="Only run evolutionary strategy (skip random-mutation baseline).",
    )
    return p.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_initialization_inputs(
    *,
    init_population_mode: str,
    seed_smiles_file: str,
    seed_pool_size: int,
    graphga_zinc250k_seed_smiles_file: str,
) -> tuple[str, int]:
    mode = str(init_population_mode or INIT_MODE_CURRENT).strip().lower()
    if mode == INIT_MODE_CURRENT:
        return str(seed_smiles_file), int(seed_pool_size)
    if mode == INIT_MODE_GRAPHGA_ZINC250K:
        zinc_path = str(graphga_zinc250k_seed_smiles_file or "").strip()
        if not zinc_path:
            raise ValueError(
                "init-population-mode=graphga_zinc250k requires --graphga-zinc250k-seed-smiles-file "
                "(or SATURN_GRAPHGA_ZINC250K_SEED_SMILES_FILE)."
            )
        min_pool_size = int(getattr(cfg, "GRAPHGA_ZINC250K_MIN_POOL_SIZE", 250000))
        return zinc_path, max(int(seed_pool_size), min_pool_size)
    raise ValueError(f"Unsupported init-population-mode: {init_population_mode!r}")


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_seed_pool(seed_smiles_file: str, seed_pool_size: int, rng: random.Random) -> list[str]:
    pool: list[str] = []
    p = Path(seed_smiles_file)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                smi = canonical_smiles(line.strip())
                if smi:
                    pool.append(smi)
    if not pool:
        pool = [s for s in (canonical_smiles(s) for s in DEFAULT_SEEDS) if s]
    pool = sorted(set(pool))
    if len(pool) > seed_pool_size:
        pool = rng.sample(pool, seed_pool_size)
    return pool


def sanitize_rw_mol(rwmol: Chem.RWMol) -> str | None:
    try:
        mol = rwmol.GetMol()
        Chem.SanitizeMol(mol)
        parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if parts:
            mol = max(parts, key=lambda m: m.GetNumHeavyAtoms())
        smi = Chem.MolToSmiles(mol, canonical=True)
        if "." in smi:
            frags = smi.split(".")
            smi = max(frags, key=len)
        return smi
    except Exception:
        return None


def mutate_substitute_atom(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not atoms:
        return None
    atom = rng.choice(atoms)
    cur = atom.GetAtomicNum()
    choices = [z for z in ALLOWED_ATOMIC_NUMBERS if z != cur]
    if not choices:
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(rng.choice(choices))
    return sanitize_rw_mol(rw)


def mutate_add_leaf_atom(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    rw = Chem.RWMol(mol)
    attach_idx = rng.randrange(rw.GetNumAtoms())
    new_atom = Chem.Atom(rng.choice(ALLOWED_ATOMIC_NUMBERS))
    new_idx = rw.AddAtom(new_atom)
    rw.AddBond(int(attach_idx), int(new_idx), rdchem.BondType.SINGLE)
    return sanitize_rw_mol(rw)


def mutate_remove_leaf_atom(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    removable = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetDegree() == 1 and not a.IsInRing() and a.GetAtomicNum() > 1
    ]
    if not removable:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(int(rng.choice(removable)))
    return sanitize_rw_mol(rw)


def mutate_bond_order(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    bonds = [b for b in mol.GetBonds() if not b.GetIsAromatic() and not b.IsInRing()]
    if not bonds:
        return None
    bond = rng.choice(bonds)
    rw = Chem.RWMol(mol)
    rb = rw.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    if rb is None:
        return None
    current = rb.GetBondType()
    options = [b for b in BOND_TYPES if b != current]
    if not options:
        return None
    rb.SetBondType(rng.choice(options))
    return sanitize_rw_mol(rw)


def mutate_insert_atom_in_bond(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    bonds = [b for b in mol.GetBonds() if not b.GetIsAromatic()]
    if not bonds:
        return None
    bond = rng.choice(bonds)
    a_idx = int(bond.GetBeginAtomIdx())
    b_idx = int(bond.GetEndAtomIdx())
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(a_idx, b_idx) is None:
        return None
    rw.RemoveBond(a_idx, b_idx)
    new_atom = Chem.Atom(rng.choice((6, 7, 8, 16)))
    new_idx = int(rw.AddAtom(new_atom))
    rw.AddBond(a_idx, new_idx, rdchem.BondType.SINGLE)
    rw.AddBond(new_idx, b_idx, rdchem.BondType.SINGLE)
    return sanitize_rw_mol(rw)


def mutate_add_ring_bond(mol: Chem.Mol, rng: random.Random, max_trials: int = 12) -> str | None:
    if mol is None:
        return None
    atom_ids = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(atom_ids) < 3:
        return None
    for _ in range(max(1, int(max_trials))):
        a_idx, b_idx = rng.sample(atom_ids, 2)
        if mol.GetBondBetweenAtoms(int(a_idx), int(b_idx)) is not None:
            continue
        rw = Chem.RWMol(mol)
        try:
            rw.AddBond(int(a_idx), int(b_idx), rdchem.BondType.SINGLE)
        except Exception:
            continue
        smi = sanitize_rw_mol(rw)
        if smi:
            return smi
    return None


def mutate_remove_bond(mol: Chem.Mol, rng: random.Random) -> str | None:
    if mol is None:
        return None
    preferred = [
        b
        for b in mol.GetBonds()
        if b.IsInRing() and not b.GetIsAromatic() and b.GetBondType() == rdchem.BondType.SINGLE
    ]
    fallback = [
        b for b in mol.GetBonds() if not b.GetIsAromatic() and b.GetBondType() == rdchem.BondType.SINGLE
    ]
    candidates = preferred if preferred else fallback
    if not candidates:
        return None
    bond = rng.choice(candidates)
    rw = Chem.RWMol(mol)
    rw.RemoveBond(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
    return sanitize_rw_mol(rw)


def sanitize_mol(mol: Chem.Mol) -> str | None:
    try:
        clean = Chem.Mol(mol)
        Chem.SanitizeMol(clean)
        parts = Chem.GetMolFrags(clean, asMols=True, sanitizeFrags=True)
        if parts:
            clean = max(parts, key=lambda m: m.GetNumHeavyAtoms())
        smi = Chem.MolToSmiles(clean, canonical=True)
        if "." in smi:
            frags = smi.split(".")
            smi = max(frags, key=len)
        return smi
    except Exception:
        return None


def brics_fragment_smiles(smiles: str, min_fragment_size: int) -> tuple[str, ...]:
    key = (smiles, int(min_fragment_size))
    cached = _BRICS_FRAGMENT_CACHE.get(key)
    if cached is not None:
        return cached

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _BRICS_FRAGMENT_CACHE[key] = tuple()
        return tuple()

    frags: set[str] = set()
    for keep_non_leaf in (True, False):
        try:
            parts = BRICS.BRICSDecompose(
                mol,
                minFragmentSize=max(1, int(min_fragment_size)),
                keepNonLeafNodes=keep_non_leaf,
                returnMols=False,
            )
        except Exception:
            continue
        for frag in parts:
            fs = str(frag).strip()
            if fs:
                frags.add(fs)

    out = tuple(sorted(frags))
    _BRICS_FRAGMENT_CACHE[key] = out
    return out


def _build_brics_product(
    fragment_smiles: Sequence[str],
    *,
    max_depth: int,
    max_products: int,
) -> str | None:
    if len(fragment_smiles) < 2:
        return None
    reagents: list[Chem.Mol] = []
    for fs in fragment_smiles:
        mol = Chem.MolFromSmiles(fs)
        if mol is not None:
            reagents.append(mol)
    if len(reagents) < 2:
        return None

    try:
        gen = BRICS.BRICSBuild(
            reagents,
            onlyCompleteMols=True,
            uniquify=True,
            scrambleReagents=True,
            maxDepth=max(1, int(max_depth)),
        )
    except Exception:
        return None

    for idx, cand in enumerate(gen):
        smi = sanitize_mol(cand)
        if smi and "*" not in smi:
            return smi
        if idx + 1 >= max(1, int(max_products)):
            break
    return None


def build_seed_fragment_pool(
    seed_pool: Sequence[str],
    rng: random.Random,
    *,
    sample_size: int,
    min_fragment_size: int,
) -> list[str]:
    if not seed_pool:
        return []
    sample_n = min(len(seed_pool), max(1, int(sample_size)))
    sampled = rng.sample(list(seed_pool), sample_n) if sample_n < len(seed_pool) else list(seed_pool)

    pool: set[str] = set()
    for smi in sampled:
        for frag in brics_fragment_smiles(smi, min_fragment_size=min_fragment_size):
            if frag:
                pool.add(frag)
    return sorted(pool)


def crossover_brics_smiles(
    parent_a: str,
    parent_b: str,
    rng: random.Random,
    *,
    min_fragment_size: int,
    max_depth: int,
    max_trials: int = 3,
    extra_fragment_pool: Sequence[str],
) -> str | None:
    fa = brics_fragment_smiles(parent_a, min_fragment_size=min_fragment_size)
    fb = brics_fragment_smiles(parent_b, min_fragment_size=min_fragment_size)
    if not fa or not fb:
        return None

    combined = list(dict.fromkeys(list(fa) + list(fb) + list(extra_fragment_pool)))
    for _ in range(max(1, int(max_trials))):
        selected = [rng.choice(fa), rng.choice(fb)]
        max_extra = min(2, max(0, len(combined) - 2))
        n_extra = rng.randint(0, max_extra) if max_extra > 0 else 0
        if n_extra > 0:
            selected.extend(rng.sample(combined, n_extra))
        child = _build_brics_product(selected, max_depth=max_depth, max_products=8)
        if child and child != parent_a and child != parent_b:
            return child
    return None


def mutate_brics_fragment_replace(
    base_smiles: str,
    rng: random.Random,
    *,
    seed_fragment_pool: Sequence[str],
    min_fragment_size: int,
    max_depth: int,
    max_trials: int = 3,
) -> str | None:
    if not seed_fragment_pool:
        return None
    base_frags = brics_fragment_smiles(base_smiles, min_fragment_size=min_fragment_size)
    if not base_frags:
        return None

    for _ in range(max(1, int(max_trials))):
        selected = [rng.choice(base_frags), rng.choice(seed_fragment_pool)]
        if len(base_frags) > 1 and rng.random() < 0.5:
            selected.append(rng.choice(base_frags))
        if len(seed_fragment_pool) > 1 and rng.random() < 0.35:
            selected.append(rng.choice(seed_fragment_pool))
        child = _build_brics_product(selected, max_depth=max_depth, max_products=8)
        if child and child != base_smiles:
            return child
    return None


def mutate_smiles(base_smiles: str, rng: random.Random, max_steps: int = 2) -> str | None:
    mol = Chem.MolFromSmiles(base_smiles)
    if mol is None:
        return None
    candidate = base_smiles
    disable_size_reducing = bool(getattr(cfg, "LOCAL_EVO_DISABLE_SIZE_REDUCING_MUTATIONS", False))
    n_steps = rng.randint(1, max(1, max_steps))
    for _ in range(n_steps):
        ops: list[Callable[[Chem.Mol, random.Random], str | None]] = [
            mutate_substitute_atom,
            mutate_add_leaf_atom,
            mutate_bond_order,
            mutate_insert_atom_in_bond,
            mutate_add_ring_bond,
        ]
        if not disable_size_reducing:
            ops.extend([mutate_remove_leaf_atom, mutate_remove_bond])
        rng.shuffle(ops)
        updated = None
        candidate_mol = Chem.MolFromSmiles(candidate)
        if candidate_mol is None:
            return None
        for op in ops:
            try:
                updated = op(candidate_mol, rng)
            except Exception:
                updated = None
            if not updated:
                continue
            canonical = canonical_smiles(updated)
            if canonical:
                updated = canonical
                break
        if not updated:
            return None
        candidate = updated
    if candidate == base_smiles:
        return None
    return candidate


def build_initial_population(seed_pool: Sequence[str], pop_size: int, rng: random.Random) -> list[str]:
    if not seed_pool:
        raise ValueError("seed_pool cannot be empty")
    if len(seed_pool) >= pop_size:
        return rng.sample(list(seed_pool), pop_size)
    pop = list(seed_pool)
    seen = set(pop)
    while len(pop) < pop_size:
        parent = rng.choice(seed_pool)
        child = mutate_smiles(parent, rng, max_steps=cfg.INIT_MUTATION_MAX_STEPS)
        if child and child not in seen:
            pop.append(child)
            seen.add(child)
        elif parent not in seen:
            pop.append(parent)
            seen.add(parent)
        else:
            pop.append(rng.choice(seed_pool))
    return pop[:pop_size]


def tournament_select(pop: Sequence[str], scores: Sequence[float], k: int, rng: random.Random) -> str:
    if len(pop) == 1:
        return pop[0]
    k_eff = max(1, min(k, len(pop)))
    ids = rng.sample(range(len(pop)), k_eff)
    best_idx = max(ids, key=lambda i: float(scores[i]))
    return pop[best_idx]


def as_float_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return arr


def topk_mean(values: Sequence[float], k: int = cfg.TOPK_MEAN_K) -> float:
    arr = as_float_array(values)
    if arr.size == 0:
        return float("nan")
    k_eff = min(k, arr.size)
    top = np.partition(arr, arr.size - k_eff)[arr.size - k_eff :]
    return float(np.mean(top))


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _numeric_score_column(df: pd.DataFrame, preferred: Sequence[str]) -> str | None:
    for col in preferred:
        if col and col in df.columns:
            return str(col)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return str(numeric_cols[-1])
    return None


def _scores_from_dataframe(df: pd.DataFrame, smiles: Sequence[str], preferred_cols: Sequence[str]) -> np.ndarray:
    if df is None or df.empty:
        return np.zeros((len(smiles),), dtype=np.float64)

    score_col = _numeric_score_column(df, preferred_cols)
    if not score_col:
        return np.zeros((len(smiles),), dtype=np.float64)

    scores = pd.to_numeric(df[score_col], errors="coerce")
    if "smiles" in df.columns and len(smiles) > 0:
        keyed = pd.DataFrame({"smiles": df["smiles"].astype(str), "score": scores})
        keyed = keyed.dropna(subset=["score"])
        if not keyed.empty:
            best_by_smiles = keyed.groupby("smiles", as_index=True)["score"].max().to_dict()
            arr = np.asarray([float(best_by_smiles.get(str(s), 0.0)) for s in smiles], dtype=np.float64)
            return arr

    clean = scores.dropna().to_numpy(dtype=np.float64)
    if clean.size == len(smiles):
        return clean
    if clean.size == 0:
        return np.zeros((len(smiles),), dtype=np.float64)
    out = np.zeros((len(smiles),), dtype=np.float64)
    n = min(len(out), clean.size)
    out[:n] = clean[:n]
    return out


def _coerce_scores(raw, smiles: Sequence[str], preferred_cols: Sequence[str]) -> np.ndarray:
    if isinstance(raw, pd.DataFrame):
        return _scores_from_dataframe(raw, smiles=smiles, preferred_cols=preferred_cols)
    if isinstance(raw, pd.Series):
        vals = pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        return vals if vals.size == len(smiles) else _scores_from_dataframe(pd.DataFrame({"score": vals}), smiles, preferred_cols)
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        return _scores_from_dataframe(pd.DataFrame(list(raw)), smiles=smiles, preferred_cols=preferred_cols)

    arr = as_float_array(raw if isinstance(raw, Iterable) else [raw])
    if arr.size == len(smiles):
        return arr
    if arr.size == 0:
        return np.zeros((len(smiles),), dtype=np.float64)
    out = np.zeros((len(smiles),), dtype=np.float64)
    n = min(len(out), arr.size)
    out[:n] = arr[:n]
    return out


def _looks_like_representation_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "must be lists" in msg or "molecular representations" in msg


def _looks_like_call_signature_error(exc: Exception) -> bool:
    if not isinstance(exc, TypeError):
        return False
    msg = str(exc).lower()
    markers = [
        "unexpected keyword",
        "required positional",
        "positional argument",
        "got multiple values",
        "takes",
        "arguments but",
    ]
    return any(m in msg for m in markers)


def _score_payload_variants(batch: list[str]) -> list[tuple[str, list]]:
    return [
        ("flat_smiles", batch),
        ("repr_channel_major", [batch]),
        ("repr_molecule_major", [[s] for s in batch]),
    ]


def _score_kwarg_variants(step: int) -> list[dict]:
    s = int(step)
    return [
        {"step": s, "canonicalize": True, "recalculate": False, "flt": False},
        {"step": s, "canonicalize": True, "recalculate": False},
        {"step": s, "canonicalize": True, "flt": False},
        {"step": s, "canonicalize": True},
        {"step": s, "flt": False},
        {"step": s},
        {},
    ]


def _build_score_adapters(ms_task) -> list[Callable[[list[str], int], object]]:
    targets: list[object] = []
    # Prefer the task callable first: this matches the original fast path.
    if callable(ms_task):
        targets.append(ms_task)
    score_fn = getattr(ms_task, "score", None)
    if callable(score_fn):
        targets.append(score_fn)
    score_only = getattr(ms_task, "score_only", None)
    if callable(score_only):
        # Keep score_only as fallback only; it may skip writing per-task scores.csv.
        targets.append(score_only)

    adapters: list[Callable[[list[str], int], object]] = []
    seen: set[tuple[int, int, str, int]] = set()

    for target in targets:
        if not callable(target):
            continue
        for payload_idx in range(3):
            for keyword_name in ("", "smiles", "molecular_representations", "mols"):
                for kwargs_idx in range(7):
                    key = (id(target), payload_idx, keyword_name, kwargs_idx)
                    if key in seen:
                        continue
                    seen.add(key)

                    def make_adapter(
                        tgt=target,
                        p_idx=payload_idx,
                        k_name=keyword_name,
                        k_idx=kwargs_idx,
                    ) -> Callable[[list[str], int], object]:
                        def _adapter(batch: list[str], step: int):
                            payload = _score_payload_variants(batch)[p_idx][1]
                            kwargs = _score_kwarg_variants(step)[k_idx]
                            if k_name:
                                call_kwargs = dict(kwargs)
                                call_kwargs[k_name] = payload
                                return tgt(**call_kwargs)
                            return tgt(payload, **kwargs)

                        return _adapter

                    adapters.append(make_adapter())

    return adapters


def score_task_batch(ms_task, smiles: Sequence[str], step: int, score_method: str) -> np.ndarray:
    batch = list(smiles)
    preferred_cols = [f"filtered_{score_method}", score_method]
    task_id = id(ms_task)
    cached = _SCORE_ADAPTER_CACHE.get(task_id)
    if callable(cached):
        try:
            raw = cached(batch, int(step))
            return _coerce_scores(raw, smiles=batch, preferred_cols=preferred_cols)
        except Exception as exc:
            if not (_looks_like_representation_error(exc) or _looks_like_call_signature_error(exc)):
                raise
            _SCORE_ADAPTER_CACHE.pop(task_id, None)

    last_rep_exc: Exception | None = None
    last_sig_exc: Exception | None = None
    for adapter in _build_score_adapters(ms_task):
        try:
            raw = adapter(batch, int(step))
            _SCORE_ADAPTER_CACHE[task_id] = adapter
            return _coerce_scores(raw, smiles=batch, preferred_cols=preferred_cols)
        except Exception as exc:
            if _looks_like_representation_error(exc):
                last_rep_exc = exc
                continue
            if _looks_like_call_signature_error(exc):
                last_sig_exc = exc
                continue
            raise

    if last_rep_exc is not None:
        raise last_rep_exc
    if last_sig_exc is not None:
        raise last_sig_exc
    raise RuntimeError("Failed to score batch: no compatible MolScore scoring API variant worked.")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def ensure_scores_csv(
    run_dir: Path,
    *,
    strategy: str,
    task_name: str,
    molecule_score_rows: list[dict],
) -> bool:
    """
    Ensure a per-task scores.csv exists.

    Normally MolScore writes this when using full scoring APIs. As a fallback for
    score-only execution paths, synthesize a minimally compatible scores.csv from
    tracked molecule scores so downstream comparison/reporting still works.
    """
    scores_path = run_dir / "scores.csv"
    if scores_path.exists():
        return False
    if not molecule_score_rows:
        return False

    by_generation_counts: dict[int, int] = {}
    seen_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for row in molecule_score_rows:
        generation = int(row.get("generation", 0))
        smiles = str(row.get("smiles", ""))
        score = float(row.get("score", 0.0))
        batch_idx = by_generation_counts.get(generation, 0)
        by_generation_counts[generation] = batch_idx + 1
        occurrences = seen_counts.get(smiles, 0)
        seen_counts[smiles] = occurrences + 1
        rows.append(
            {
                "model": f"{strategy}_ga",
                "task": task_name,
                "step": generation,
                "batch_idx": batch_idx,
                "absolute_time": float("nan"),
                "smiles": smiles,
                "valid": True,
                "valid_score": 1.0,
                "unique": occurrences == 0,
                "occurrences": occurrences,
                "single": score,
                "filter": 1.0,
                "score_time": float("nan"),
            }
        )

    pd.DataFrame(rows).to_csv(scores_path, index=True)
    return True


def evolve_on_task(
    ms_task,
    strategy: str,
    seed_pool: Sequence[str],
    budget: int,
    pop_size: int,
    batch_size: int,
    max_generations: int,
    tournament_k: int,
    elite_fraction: float,
    immigrant_fraction: float,
    parent_pool_fraction: float,
    stagnation_patience: int,
    stagnation_mutation_boost: int,
    rng: random.Random,
) -> StrategyResult:
    start = time.time()
    score_method = str(ms_task.cfg["scoring"]["method"])
    score_column = f"filtered_{score_method}"
    task_name = str(ms_task.cfg["task"])
    run_dir = Path(ms_task.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    archive: dict[str, float] = {}
    progress_rows: list[dict] = []
    molecule_score_rows: list[dict] = []
    best_score_rows: list[dict] = []
    avg_score_rows: list[dict] = []
    elite_fraction = clamp_float(elite_fraction, 0.0, 0.9)
    immigrant_fraction = clamp_float(immigrant_fraction, 0.0, 0.9)
    parent_pool_fraction = clamp_float(parent_pool_fraction, 0.1, 1.0)
    stagnation_patience = max(0, int(stagnation_patience))
    stagnation_mutation_boost = max(0, int(stagnation_mutation_boost))
    mutation_step_cap = max(
        int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
        int(getattr(cfg, "LOCAL_EVO_MUTATION_STEP_CAP", cfg.OFFSPRING_MUTATION_MAX_STEPS)),
    )
    crossover_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_CROSSOVER_FRACTION", 0.0)), 0.0, 0.95)
    fragment_replace_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_FRAGMENT_REPLACE_FRACTION", 0.0)), 0.0, 0.95)
    brics_disabled = bool(getattr(cfg, "LOCAL_EVO_DISABLE_BRICS", False))
    use_brics_operators = (
        strategy == "evolution"
        and not brics_disabled
        and (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
    )
    brics_min_fragment_size = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", 2)))
    brics_max_depth = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MAX_DEPTH", 3)))
    seed_fragment_pool_size = max(1, int(getattr(cfg, "LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", 256)))
    seed_fragment_pool: list[str] = []
    if use_brics_operators:
        seed_fragment_pool = build_seed_fragment_pool(
            seed_pool=seed_pool,
            rng=rng,
            sample_size=seed_fragment_pool_size,
            min_fragment_size=brics_min_fragment_size,
        )

    population = build_initial_population(seed_pool, pop_size, rng)
    init_batch = population[: min(batch_size, budget)]
    init_scores = score_task_batch(ms_task, init_batch, step=0, score_method=score_method)
    for smi, sc in zip(init_batch, init_scores.tolist()):
        archive[smi] = max(archive.get(smi, -1e18), float(sc))
        molecule_score_rows.append(
            {
                "generation": 0,
                "smiles": smi,
                "score": float(sc),
            }
        )
    population = init_batch
    pop_scores = init_scores.tolist()
    evaluated = len(init_batch)
    seen_global = set(archive.keys())
    best_so_far = float(np.max(init_scores)) if init_scores.size > 0 else float("-inf")
    stagnation_count = 0
    if init_scores.size > 0:
        init_best = float(np.max(init_scores))
        init_avg = float(np.mean(init_scores))
        best_score_rows.append(
            {
                "generation": 0,
                "evaluated": evaluated,
                "batch_size": len(init_batch),
                "best_score_generation": init_best,
                "best_score_so_far": init_best,
            }
        )
        avg_score_rows.append(
            {
                "generation": 0,
                "evaluated": evaluated,
                "batch_size": len(init_batch),
                "average_score_generation": init_avg,
                "average_score_so_far": init_avg,
            }
        )
    write_guacamol_like_reports(run_dir)
    gen = 0

    while not bool(ms_task.finished):
        if evaluated >= budget:
            break
        if max_generations > 0 and gen >= max_generations:
            break

        gen += 1
        remaining = max(0, budget - evaluated)
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        offspring: list[str] = []
        seen = set(seen_global)
        seen.update(population)
        attempts = 0
        max_attempts = this_batch * cfg.OFFSPRING_ATTEMPT_FACTOR
        parent_population = population
        parent_scores = pop_scores
        effective_tournament_k = max(2, int(tournament_k))
        mutation_steps = int(cfg.OFFSPRING_MUTATION_MAX_STEPS)
        effective_parent_pool_fraction = parent_pool_fraction
        effective_immigrant_fraction = immigrant_fraction
        effective_elite_fraction = elite_fraction
        diversity_fraction = 0.10

        if strategy == "evolution" and stagnation_patience > 0 and stagnation_count >= stagnation_patience:
            mutation_steps = min(mutation_step_cap, mutation_steps + stagnation_mutation_boost)
            effective_tournament_k = max(2, min(effective_tournament_k, 3))
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.25)
            effective_parent_pool_fraction = 1.0
            diversity_fraction = 0.20

        if (
            strategy == "evolution"
            and stagnation_patience > 0
            and stagnation_count >= (3 * stagnation_patience)
        ):
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.40)
            effective_elite_fraction = min(float(effective_elite_fraction), 0.10)
            diversity_fraction = 0.30

        if strategy == "evolution" and parent_population:
            pool_size = max(2, int(round(len(parent_population) * effective_parent_pool_fraction)))
            pool_size = min(pool_size, len(parent_population))
            parent_population = parent_population[:pool_size]
            parent_scores = parent_scores[:pool_size]
            effective_tournament_k = max(2, min(effective_tournament_k, len(parent_population)))

        immigrant_target = 0
        crossover_target = 0
        fragment_replace_target = 0
        if strategy == "evolution":
            immigrant_target = min(this_batch, int(round(this_batch * effective_immigrant_fraction)))
            non_immigrant_target = max(0, this_batch - immigrant_target)
            if use_brics_operators:
                crossover_target = min(non_immigrant_target, int(round(non_immigrant_target * crossover_fraction)))
            remaining_non_immigrant = max(0, non_immigrant_target - crossover_target)
            if use_brics_operators and seed_fragment_pool:
                fragment_replace_target = min(
                    remaining_non_immigrant,
                    int(round(non_immigrant_target * fragment_replace_fraction)),
                )

        immigrant_count = 0
        crossover_count = 0
        fragment_replace_count = 0
        mutation_count = 0
        immigrant_attempts = 0
        immigrant_max_attempts = max(1, immigrant_target * cfg.OFFSPRING_ATTEMPT_FACTOR)
        while len(offspring) < immigrant_target and immigrant_attempts < immigrant_max_attempts:
            immigrant_attempts += 1
            parent = rng.choice(seed_pool)
            child = mutate_smiles(parent, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
            candidate = child if child else parent
            if candidate and candidate not in seen:
                seen.add(candidate)
                offspring.append(candidate)
                immigrant_count += 1

        crossover_attempts = 0
        crossover_target_total = immigrant_target + crossover_target
        crossover_max_attempts = max(1, crossover_target * cfg.OFFSPRING_ATTEMPT_FACTOR)
        while (
            strategy == "evolution"
            and len(offspring) < crossover_target_total
            and crossover_attempts < crossover_max_attempts
        ):
            crossover_attempts += 1
            parent_a = tournament_select(parent_population, parent_scores, effective_tournament_k, rng)
            parent_b = tournament_select(parent_population, parent_scores, effective_tournament_k, rng)
            if parent_a == parent_b and len(parent_population) > 1:
                parent_b = rng.choice(parent_population)
            child = crossover_brics_smiles(
                parent_a,
                parent_b,
                rng,
                min_fragment_size=brics_min_fragment_size,
                max_depth=brics_max_depth,
                max_trials=3,
                extra_fragment_pool=seed_fragment_pool,
            )
            if child and child not in seen:
                seen.add(child)
                offspring.append(child)
                crossover_count += 1

        fragment_attempts = 0
        fragment_target_total = immigrant_target + crossover_target + fragment_replace_target
        fragment_max_attempts = max(1, fragment_replace_target * cfg.OFFSPRING_ATTEMPT_FACTOR)
        while (
            strategy == "evolution"
            and len(offspring) < fragment_target_total
            and fragment_attempts < fragment_max_attempts
        ):
            fragment_attempts += 1
            parent = tournament_select(parent_population, parent_scores, effective_tournament_k, rng)
            child = mutate_brics_fragment_replace(
                parent,
                rng,
                seed_fragment_pool=seed_fragment_pool,
                min_fragment_size=brics_min_fragment_size,
                max_depth=brics_max_depth,
                max_trials=3,
            )
            if child and child not in seen:
                seen.add(child)
                offspring.append(child)
                fragment_replace_count += 1

        while len(offspring) < this_batch and attempts < max_attempts:
            attempts += 1
            if strategy == "evolution":
                parent = tournament_select(parent_population, parent_scores, effective_tournament_k, rng)
            else:
                parent = rng.choice(seed_pool)
            child = mutate_smiles(parent, rng, max_steps=mutation_steps)
            if child and child not in seen:
                seen.add(child)
                offspring.append(child)
                mutation_count += 1

        while len(offspring) < this_batch and attempts < (2 * max_attempts):
            attempts += 1
            filler = rng.choice(seed_pool)
            filler_child = mutate_smiles(filler, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
            candidate = filler_child if filler_child else filler
            if candidate not in seen:
                offspring.append(candidate)
                seen.add(candidate)
                mutation_count += 1

        if len(offspring) < this_batch:
            # Last-resort sampling to keep progress moving if novelty generation stalls.
            rescue_attempts = 0
            rescue_max_attempts = this_batch * 4
            while len(offspring) < this_batch and rescue_attempts < rescue_max_attempts:
                rescue_attempts += 1
                filler = rng.choice(seed_pool)
                filler_child = mutate_smiles(filler, rng, max_steps=mutation_step_cap)
                candidate = filler_child if filler_child else filler
                if candidate not in seen:
                    offspring.append(candidate)
                    seen.add(candidate)
                    mutation_count += 1
            while len(offspring) < this_batch:
                filler = rng.choice(seed_pool)
                offspring.append(filler)
                mutation_count += 1

        off_scores = score_task_batch(ms_task, offspring, step=gen, score_method=score_method)
        evaluated += len(offspring)

        for smi, sc in zip(offspring, off_scores.tolist()):
            archive[smi] = max(archive.get(smi, -1e18), float(sc))
            seen_global.add(smi)
            molecule_score_rows.append(
                {
                    "generation": gen,
                    "smiles": smi,
                    "score": float(sc),
                }
            )

        if strategy == "evolution":
            combined: dict[str, float] = {}
            for s, sc in zip(population, pop_scores):
                combined[s] = max(combined.get(s, -1e18), float(sc))
            for s, sc in zip(offspring, off_scores.tolist()):
                combined[s] = max(combined.get(s, -1e18), float(sc))
            ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
            elite_count = max(1, min(pop_size, int(round(pop_size * effective_elite_fraction))))
            elites = ranked[:elite_count]
            tail = ranked[elite_count:]
            tail_for_band = tail
            n_remaining = max(0, pop_size - len(elites))
            diversity_count = min(n_remaining, int(round(pop_size * diversity_fraction)))
            diversity_selected: list[tuple[str, float]] = []
            if diversity_count > 0 and tail:
                if len(tail) > diversity_count:
                    diversity_selected = rng.sample(tail, diversity_count)
                    diversity_smiles = {s for s, _ in diversity_selected}
                    tail_for_band = [kv for kv in tail if kv[0] not in diversity_smiles]
                else:
                    diversity_selected = list(tail)
                    tail_for_band = []
            n_remaining = max(0, n_remaining - len(diversity_selected))
            if n_remaining > 0 and tail_for_band:
                band = tail_for_band[: max(n_remaining, n_remaining * 3)]
                if len(band) > n_remaining:
                    sampled = rng.sample(band, n_remaining)
                    sampled = sorted(sampled, key=lambda kv: kv[1], reverse=True)
                else:
                    sampled = band
            else:
                sampled = []
            selected = elites + sampled + diversity_selected
            if len(selected) < pop_size:
                selected_smiles = {s for s, _ in selected}
                for cand in ranked:
                    if cand[0] in selected_smiles:
                        continue
                    selected.append(cand)
                    selected_smiles.add(cand[0])
                    if len(selected) >= pop_size:
                        break
            population = [s for s, _ in selected[:pop_size]]
            pop_scores = [float(sc) for _, sc in selected[:pop_size]]
        else:
            population = offspring[:pop_size]
            pop_scores = off_scores.tolist()[:pop_size]

        archive_scores = list(archive.values())
        gen_best = float(np.max(off_scores)) if off_scores.size > 0 else float("nan")
        gen_avg = float(np.mean(off_scores)) if off_scores.size > 0 else float("nan")
        archive_best = max(archive_scores) if archive_scores else float("nan")
        archive_avg = float(np.mean(archive_scores)) if archive_scores else float("nan")
        if not math.isnan(archive_best):
            if archive_best > (best_so_far + 1e-9):
                best_so_far = archive_best
                stagnation_count = 0
            else:
                stagnation_count += 1
        best_score_rows.append(
            {
                "generation": gen,
                "evaluated": evaluated,
                "batch_size": len(offspring),
                "best_score_generation": gen_best,
                "best_score_so_far": archive_best,
            }
        )
        avg_score_rows.append(
            {
                "generation": gen,
                "evaluated": evaluated,
                "batch_size": len(offspring),
                "average_score_generation": gen_avg,
                "average_score_so_far": archive_avg,
            }
        )
        progress_rows.append(
            {
                "generation": gen,
                "evaluated": evaluated,
                "batch_size": len(offspring),
                "population_size": len(population),
                "best_score": archive_best,
                "average_score": archive_avg,
                "mean_top10": topk_mean(archive_scores, cfg.TOPK_MEAN_K),
                "unique_molecules": len(archive),
                "stagnation_count": stagnation_count,
                "effective_tournament_k": effective_tournament_k if strategy == "evolution" else 0,
                "effective_elite_fraction": effective_elite_fraction if strategy == "evolution" else 0.0,
                "effective_immigrant_fraction": effective_immigrant_fraction if strategy == "evolution" else 0.0,
                "diversity_fraction": diversity_fraction if strategy == "evolution" else 0.0,
                "mutation_steps": mutation_steps if strategy == "evolution" else int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
                "immigrant_offspring": immigrant_count,
                "crossover_offspring": crossover_count,
                "fragment_replace_offspring": fragment_replace_count,
                "mutation_offspring": mutation_count,
            }
        )
        every_n = max(1, int(getattr(cfg, "REPORT_WRITE_EVERY_N_GEN", 5)))
        if gen % every_n == 0:
            write_guacamol_like_reports(run_dir)

    best_score = max(archive.values()) if archive else float("nan")
    mean_top10 = topk_mean(list(archive.values()), cfg.TOPK_MEAN_K)
    elapsed = time.time() - start

    write_csv(
        run_dir / "generator_progress.csv",
        progress_rows,
        [
            "generation",
            "evaluated",
            "batch_size",
            "population_size",
            "best_score",
            "average_score",
            "mean_top10",
            "unique_molecules",
            "stagnation_count",
            "effective_tournament_k",
            "effective_elite_fraction",
            "effective_immigrant_fraction",
            "diversity_fraction",
            "mutation_steps",
            "immigrant_offspring",
            "crossover_offspring",
            "fragment_replace_offspring",
            "mutation_offspring",
        ],
    )
    top_rows = [
        {"smiles": s, "score": sc}
        for s, sc in sorted(archive.items(), key=lambda kv: kv[1], reverse=True)[: cfg.TOP_MOLECULES_TO_SAVE]
    ]
    write_csv(run_dir / "generator_top_molecules.csv", top_rows, ["smiles", "score"])
    write_tsv(
        run_dir / "molecule_scores_by_generation.tsv",
        molecule_score_rows,
        ["generation", "smiles", "score"],
    )
    write_tsv(
        run_dir / "best_score_per_generation.tsv",
        best_score_rows,
        ["generation", "evaluated", "batch_size", "best_score_generation", "best_score_so_far"],
    )
    write_tsv(
        run_dir / "average_score_per_generation.tsv",
        avg_score_rows,
        ["generation", "evaluated", "batch_size", "average_score_generation", "average_score_so_far"],
    )
    ensure_scores_csv(
        run_dir,
        strategy=strategy,
        task_name=task_name,
        molecule_score_rows=molecule_score_rows,
    )
    # Ensure final GuacaMol-style reports are in sync at task end.
    write_guacamol_like_reports(run_dir)

    main_df = getattr(ms_task, "main_df", None)
    if not isinstance(main_df, pd.DataFrame) or score_column not in main_df.columns:
        score_column = score_method

    return StrategyResult(
        strategy=strategy,
        benchmark=str(getattr(ms_task, "benchmark", "")),
        task=task_name,
        run_dir=str(run_dir),
        score_column=score_column,
        budget=budget,
        evaluated=evaluated,
        generations=gen,
        best_score=float(best_score) if not math.isnan(best_score) else float("nan"),
        mean_top10=float(mean_top10) if not math.isnan(mean_top10) else float("nan"),
        unique_molecules=len(archive),
        elapsed_seconds=float(elapsed),
    )


def ensure_molscore_import():
    try:
        from molscore import MolScoreBenchmark  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cannot import MolScore in env 'molscore'. "
            "Install missing deps first (typically: scipy, joblib, selfies). "
            "Try: conda run -n molscore conda install -y scipy joblib selfies"
        ) from exc
    return MolScoreBenchmark


def ensure_saturn_objective_import():
    try:
        from saturn_objective import SaturnObjective, SaturnObjectiveConfig  # type: ignore
    except Exception:
        try:
            from .saturn_objective import SaturnObjective, SaturnObjectiveConfig  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Cannot import SATURN objective adapter. "
                "Expected local module 'saturn_objective.py' next to this script."
            ) from exc
    return SaturnObjective, SaturnObjectiveConfig


class LocalObjectiveTask:
    """
    Adapter exposing a task-like API compatible with evolve_on_task for local objectives.
    """

    def __init__(self, objective, task_name: str, save_dir: Path, budget: int):
        self.objective = objective
        self.cfg = {"task": str(task_name), "scoring": {"method": "single"}}
        self.benchmark = "SATURN"
        self.save_dir = str(save_dir)
        self.main_df = pd.DataFrame()
        self._budget = int(budget)
        self._evaluated = 0
        self._scores_rows: list[dict] = []
        save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def finished(self) -> bool:
        return bool(self._evaluated >= self._budget)

    def __call__(
        self,
        smiles: Sequence[str],
        step: int = 0,
        canonicalize: bool = True,
        recalculate: bool = False,  # noqa: ARG002 - kept for MolScore-compatible call signatures.
    ) -> np.ndarray:
        original = [str(s) for s in smiles]
        scored_smiles: list[str] = []
        valid_mask: list[bool] = []
        for smi in original:
            canon = canonical_smiles(smi) if canonicalize else smi
            if canon:
                scored_smiles.append(canon)
                valid_mask.append(True)
            else:
                scored_smiles.append(smi)
                valid_mask.append(False)

        scores = [float(x) for x in self.objective.score_population(scored_smiles)]
        self._evaluated += len(scored_smiles)

        for idx, (smi, sc, is_valid) in enumerate(zip(scored_smiles, scores, valid_mask)):
            filt = 1.0 if is_valid else 0.0
            self._scores_rows.append(
                {
                    "step": int(step),
                    "batch_idx": int(idx),
                    "smiles": str(smi),
                    "score": float(sc),
                    "filtered_single": float(sc),
                    "valid_score": float(sc if is_valid else 0.0),
                    "valid": bool(is_valid),
                    "filter": float(filt),
                }
            )

        self.main_df = pd.DataFrame(self._scores_rows)
        self.main_df.to_csv(Path(self.save_dir) / "scores.csv", index=False)
        return np.asarray(scores, dtype=np.float64)


def _task_context(task_like):
    # MolScore task iterators can yield either task objects directly
    # or context-manager wrappers around task objects.
    if hasattr(task_like, "cfg"):
        return contextlib.nullcontext(task_like)
    if hasattr(task_like, "__enter__") and hasattr(task_like, "__exit__"):
        return task_like
    return contextlib.nullcontext(task_like)


def run_strategy(
    strategy: str,
    benchmark: str,
    custom_benchmark: str,
    include: list[str],
    exclude: list[str],
    output_root: Path,
    budget: int,
    pop_size: int,
    batch_size: int,
    max_generations: int,
    tournament_k: int,
    elite_fraction: float,
    immigrant_fraction: float,
    parent_pool_fraction: float,
    stagnation_patience: int,
    stagnation_mutation_boost: int,
    seed_pool: Sequence[str],
    seed: int,
) -> tuple[Path, pd.DataFrame]:
    MolScoreBenchmark = ensure_molscore_import()
    model_name = f"{strategy}_ga"
    benchmark_kwargs = {}
    if custom_benchmark:
        benchmark_kwargs["custom_benchmark"] = custom_benchmark
        benchmark_kwargs["benchmark"] = None
    else:
        benchmark_kwargs["benchmark"] = benchmark

    bench = MolScoreBenchmark(
        model_name=model_name,
        output_dir=str(output_root),
        budget=int(budget),
        add_benchmark_dir=True,
        include=include,
        exclude=exclude,
        model_parameters={
            "strategy": strategy,
            "population_size": pop_size,
            "batch_size": batch_size,
            "tournament_k": tournament_k,
            "elite_fraction": elite_fraction,
            "immigrant_fraction": immigrant_fraction,
            "parent_pool_fraction": parent_pool_fraction,
            "stagnation_patience": stagnation_patience,
            "stagnation_mutation_boost": stagnation_mutation_boost,
            "seed": seed,
        },
        score_invalids=False,
        **benchmark_kwargs,
    )

    out_dir = Path(bench.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[StrategyResult] = []
    task_failures: list[dict[str, object]] = []
    try:
        total_tasks = len(bench)
    except Exception:
        total_tasks = 0

    task_iter = iter(bench)
    task_idx = 0
    consecutive_setup_failures = 0
    max_consecutive_setup_failures = 5

    while True:
        try:
            task_like = next(task_iter)
        except StopIteration:
            break
        except Exception as exc:
            task_idx += 1
            consecutive_setup_failures += 1
            msg = str(exc).strip() or repr(exc)
            task_failures.append(
                {
                    "strategy": strategy,
                    "task_index": task_idx,
                    "task": f"task_{task_idx}",
                    "phase": "task_setup",
                    "error": msg,
                }
            )
            print(
                f"[{strategy}] warning task setup failed at index {task_idx}: {msg}",
                file=sys.stderr,
                flush=True,
            )
            if "server did not launch within grace period" in msg.lower():
                backoff_seconds = min(30, 5 * consecutive_setup_failures)
                print(
                    f"[{strategy}] waiting {backoff_seconds}s before continuing after server-timeout setup failure",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(backoff_seconds)
            if consecutive_setup_failures >= max_consecutive_setup_failures:
                print(
                    f"[{strategy}] aborting benchmark after {consecutive_setup_failures} consecutive task setup failures",
                    file=sys.stderr,
                    flush=True,
                )
                break
            continue

        consecutive_setup_failures = 0
        task_idx += 1
        rng = random.Random(seed + task_idx)
        task_name = f"task_{task_idx}"
        try:
            with _task_context(task_like) as ms_task:
                task_cfg = getattr(ms_task, "cfg", {}) if ms_task is not None else {}
                if isinstance(task_cfg, dict):
                    task_name = str(task_cfg.get("task", task_name))
                task_counter = f"{task_idx}/{total_tasks}" if total_tasks else str(task_idx)
                print(f"[{strategy}] task {task_counter} :: {task_name}", flush=True)
                result = evolve_on_task(
                    ms_task=ms_task,
                    strategy=strategy,
                    seed_pool=seed_pool,
                    budget=budget,
                    pop_size=pop_size,
                    batch_size=batch_size,
                    max_generations=max_generations,
                    tournament_k=tournament_k,
                    elite_fraction=elite_fraction,
                    immigrant_fraction=immigrant_fraction,
                    parent_pool_fraction=parent_pool_fraction,
                    stagnation_patience=stagnation_patience,
                    stagnation_mutation_boost=stagnation_mutation_boost,
                    rng=rng,
                )
        except Exception as exc:
            msg = str(exc).strip() or repr(exc)
            task_failures.append(
                {
                    "strategy": strategy,
                    "task_index": task_idx,
                    "task": task_name,
                    "phase": "task_run",
                    "error": msg,
                }
            )
            print(
                f"[{strategy}] warning task failed ({task_name}): {msg}",
                file=sys.stderr,
                flush=True,
            )
            if "server did not launch within grace period" in msg.lower():
                time.sleep(5)
            continue

        result.benchmark = custom_benchmark if custom_benchmark else benchmark
        rows.append(result)
        print(
            f"[{strategy}] done {result.task} :: best={result.best_score:.4f} "
            f"top10={result.mean_top10:.4f} unique={result.unique_molecules}",
            flush=True,
        )

    if task_failures:
        pd.DataFrame(task_failures).to_csv(out_dir / "strategy_task_failures.tsv", sep="\t", index=False)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(out_dir / "strategy_summary.csv", index=False)
    # Create results.csv so MolScoreBenchmark atexit hook does not force full ScoreMetrics summary.
    df.to_csv(out_dir / "results.csv", index=False)
    return out_dir, df


def run_strategy_saturn(
    strategy: str,
    output_root: Path,
    budget: int,
    pop_size: int,
    batch_size: int,
    max_generations: int,
    tournament_k: int,
    elite_fraction: float,
    immigrant_fraction: float,
    parent_pool_fraction: float,
    stagnation_patience: int,
    stagnation_mutation_boost: int,
    seed_pool: Sequence[str],
    seed: int,
    saturn_repo_root: str,
    saturn_oracle_config: str,
    saturn_oracle_config_key: str,
    saturn_apply_diversity_penalty: bool,
    saturn_diversity_bucket_size: int,
    saturn_invalid_score: float,
    saturn_allow_oracle_repeats: bool | None,
    saturn_task_name: str,
) -> tuple[Path, pd.DataFrame]:
    SaturnObjective, SaturnObjectiveConfig = ensure_saturn_objective_import()

    model_name = f"{strategy}_ga"
    out_dir = Path(output_root) / "SATURN" / model_name
    task_dir = out_dir / saturn_task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    objective = SaturnObjective(
        SaturnObjectiveConfig(
            saturn_repo_root=str(saturn_repo_root),
            oracle_config_path=str(saturn_oracle_config),
            oracle_config_key=str(saturn_oracle_config_key),
            apply_diversity_penalty=bool(saturn_apply_diversity_penalty),
            diversity_bucket_size=int(saturn_diversity_bucket_size),
            invalid_score=float(saturn_invalid_score),
            budget_override=int(budget),
            allow_oracle_repeats=saturn_allow_oracle_repeats,
        )
    )
    task = LocalObjectiveTask(
        objective=objective,
        task_name=saturn_task_name,
        save_dir=task_dir,
        budget=budget,
    )

    rng = random.Random(seed + 1)
    print(f"[{strategy}] task 1/1 :: {saturn_task_name}", flush=True)
    result = evolve_on_task(
        ms_task=task,
        strategy=strategy,
        seed_pool=seed_pool,
        budget=budget,
        pop_size=pop_size,
        batch_size=batch_size,
        max_generations=max_generations,
        tournament_k=tournament_k,
        elite_fraction=elite_fraction,
        immigrant_fraction=immigrant_fraction,
        parent_pool_fraction=parent_pool_fraction,
        stagnation_patience=stagnation_patience,
        stagnation_mutation_boost=stagnation_mutation_boost,
        rng=rng,
    )
    if hasattr(objective, "write_oracle_history"):
        try:
            oracle_history_path = objective.write_oracle_history(task_dir)
            print(f"[{strategy}] oracle history: {oracle_history_path}", flush=True)
        except Exception as exc:
            print(f"[{strategy}] warning: failed to write oracle history: {exc}", file=sys.stderr, flush=True)
    if hasattr(objective, "write_repeat_history"):
        try:
            objective.write_repeat_history(task_dir)
        except Exception as exc:
            print(f"[{strategy}] warning: failed to write repeat history: {exc}", file=sys.stderr, flush=True)
    result.benchmark = "SATURN"
    print(
        f"[{strategy}] done {result.task} :: best={result.best_score:.4f} "
        f"top10={result.mean_top10:.4f} unique={result.unique_molecules}",
        flush=True,
    )

    df = pd.DataFrame([result.__dict__])
    df.to_csv(out_dir / "strategy_summary.csv", index=False)
    df.to_csv(out_dir / "results.csv", index=False)
    return out_dir, df


def build_comparison_df(evo_df: pd.DataFrame, rnd_df: pd.DataFrame) -> pd.DataFrame:
    e = evo_df.loc[:, ["task", "best_score", "mean_top10", "unique_molecules", "evaluated"]].rename(
        columns={
            "best_score": "best_score_evolution",
            "mean_top10": "mean_top10_evolution",
            "unique_molecules": "unique_molecules_evolution",
            "evaluated": "evaluated_evolution",
        }
    )
    r = rnd_df.loc[:, ["task", "best_score", "mean_top10", "unique_molecules", "evaluated"]].rename(
        columns={
            "best_score": "best_score_random",
            "mean_top10": "mean_top10_random",
            "unique_molecules": "unique_molecules_random",
            "evaluated": "evaluated_random",
        }
    )
    c = e.merge(r, on="task", how="inner")
    c["delta_best_score"] = c["best_score_evolution"] - c["best_score_random"]
    c["delta_mean_top10"] = c["mean_top10_evolution"] - c["mean_top10_random"]
    c["delta_unique_molecules"] = c["unique_molecules_evolution"] - c["unique_molecules_random"]
    return c.sort_values("delta_best_score", ascending=False)


def main() -> int:
    args = parse_args()
    include = parse_csv_list(args.include)
    exclude = parse_csv_list(args.exclude)

    effective_seed_smiles_file, effective_seed_pool_size = resolve_initialization_inputs(
        init_population_mode=str(args.init_population_mode),
        seed_smiles_file=str(args.seed_smiles_file),
        seed_pool_size=int(args.seed_pool_size),
        graphga_zinc250k_seed_smiles_file=str(args.graphga_zinc250k_seed_smiles_file),
    )
    seed_smiles_file = cfg.resolve_from_repo(effective_seed_smiles_file)
    if str(args.init_population_mode).strip().lower() == INIT_MODE_GRAPHGA_ZINC250K and not seed_smiles_file.exists():
        raise FileNotFoundError(f"GraphGA ZINC-250k seed file not found: {seed_smiles_file}")
    saturn_repo_root = Path(args.saturn_repo_root).expanduser()
    if not saturn_repo_root.is_absolute():
        saturn_repo_root = cfg.resolve_from_repo(str(saturn_repo_root))
    else:
        saturn_repo_root = saturn_repo_root.resolve()

    saturn_oracle_config = Path(args.saturn_oracle_config).expanduser()
    if not saturn_oracle_config.is_absolute():
        saturn_oracle_config = cfg.resolve_from_repo(str(saturn_oracle_config))
    else:
        saturn_oracle_config = saturn_oracle_config.resolve()

    if args.objective_backend == "saturn":
        if not saturn_repo_root.exists():
            raise RuntimeError(f"SATURN repo root not found: {saturn_repo_root}")
        if not saturn_oracle_config.exists():
            raise RuntimeError(f"SATURN oracle config not found: {saturn_oracle_config}")

    rng = random.Random(args.seed)
    seed_pool = load_seed_pool(str(seed_smiles_file), int(effective_seed_pool_size), rng)
    if not seed_pool:
        raise RuntimeError("No valid seed molecules available.")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_base = cfg.resolve_from_repo(args.output_dir)
    case_label = args.benchmark if args.objective_backend == "molscore" else "saturn_objective"
    root = output_base / f"case_{case_label}_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "objective_backend": args.objective_backend,
        "benchmark": args.benchmark,
        "custom_benchmark": args.custom_benchmark,
        "include": include,
        "exclude": exclude,
        "budget": int(args.budget),
        "population_size": int(args.population_size),
        "batch_size": int(args.batch_size),
        "max_generations": int(args.max_generations),
        "tournament_k": int(args.tournament_k),
        "elite_fraction": float(args.elite_fraction),
        "immigrant_fraction": float(args.immigrant_fraction),
        "parent_pool_fraction": float(args.parent_pool_fraction),
        "stagnation_patience": int(args.stagnation_patience),
        "stagnation_mutation_boost": int(args.stagnation_mutation_boost),
        "seed": int(args.seed),
        "init_population_mode": str(args.init_population_mode),
        "graphga_zinc250k_seed_smiles_file": str(args.graphga_zinc250k_seed_smiles_file),
        "seed_smiles_file": str(seed_smiles_file),
        "requested_seed_pool_size": int(args.seed_pool_size),
        "effective_seed_pool_size_cap": int(effective_seed_pool_size),
        "seed_pool_size": len(seed_pool),
        "skip_random_baseline": bool(args.skip_random_baseline),
        "saturn_repo_root": str(saturn_repo_root),
        "saturn_oracle_config": str(saturn_oracle_config),
        "saturn_oracle_config_key": str(args.saturn_oracle_config_key),
        "saturn_task_name": str(args.saturn_task_name),
        "saturn_apply_diversity_penalty": bool(args.saturn_apply_diversity_penalty),
        "saturn_diversity_bucket_size": int(args.saturn_diversity_bucket_size),
        "saturn_invalid_score": float(args.saturn_invalid_score),
        "saturn_allow_oracle_repeats": args.saturn_allow_oracle_repeats,
    }
    (root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Output root: {root}", flush=True)
    if args.objective_backend == "molscore":
        evo_dir, evo_df = run_strategy(
            strategy="evolution",
            benchmark=args.benchmark,
            custom_benchmark=args.custom_benchmark,
            include=include,
            exclude=exclude,
            output_root=root / "benchmark_runs",
            budget=args.budget,
            pop_size=args.population_size,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
            tournament_k=args.tournament_k,
            elite_fraction=args.elite_fraction,
            immigrant_fraction=args.immigrant_fraction,
            parent_pool_fraction=args.parent_pool_fraction,
            stagnation_patience=args.stagnation_patience,
            stagnation_mutation_boost=args.stagnation_mutation_boost,
            seed_pool=seed_pool,
            seed=args.seed,
        )
    else:
        evo_dir, evo_df = run_strategy_saturn(
            strategy="evolution",
            output_root=root / "benchmark_runs",
            budget=args.budget,
            pop_size=args.population_size,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
            tournament_k=args.tournament_k,
            elite_fraction=args.elite_fraction,
            immigrant_fraction=args.immigrant_fraction,
            parent_pool_fraction=args.parent_pool_fraction,
            stagnation_patience=args.stagnation_patience,
            stagnation_mutation_boost=args.stagnation_mutation_boost,
            seed_pool=seed_pool,
            seed=args.seed,
            saturn_repo_root=str(saturn_repo_root),
            saturn_oracle_config=str(saturn_oracle_config),
            saturn_oracle_config_key=args.saturn_oracle_config_key,
            saturn_apply_diversity_penalty=args.saturn_apply_diversity_penalty,
            saturn_diversity_bucket_size=args.saturn_diversity_bucket_size,
            saturn_invalid_score=args.saturn_invalid_score,
            saturn_allow_oracle_repeats=args.saturn_allow_oracle_repeats,
            saturn_task_name=args.saturn_task_name,
        )

    if args.skip_random_baseline:
        print(f"Evolution summary: {evo_dir / 'strategy_summary.csv'}", flush=True)
        return 0

    if args.objective_backend == "molscore":
        rnd_dir, rnd_df = run_strategy(
            strategy="random",
            benchmark=args.benchmark,
            custom_benchmark=args.custom_benchmark,
            include=include,
            exclude=exclude,
            output_root=root / "benchmark_runs",
            budget=args.budget,
            pop_size=args.population_size,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
            tournament_k=args.tournament_k,
            elite_fraction=args.elite_fraction,
            immigrant_fraction=args.immigrant_fraction,
            parent_pool_fraction=args.parent_pool_fraction,
            stagnation_patience=args.stagnation_patience,
            stagnation_mutation_boost=args.stagnation_mutation_boost,
            seed_pool=seed_pool,
            seed=args.seed + 10000,
        )
    else:
        rnd_dir, rnd_df = run_strategy_saturn(
            strategy="random",
            output_root=root / "benchmark_runs",
            budget=args.budget,
            pop_size=args.population_size,
            batch_size=args.batch_size,
            max_generations=args.max_generations,
            tournament_k=args.tournament_k,
            elite_fraction=args.elite_fraction,
            immigrant_fraction=args.immigrant_fraction,
            parent_pool_fraction=args.parent_pool_fraction,
            stagnation_patience=args.stagnation_patience,
            stagnation_mutation_boost=args.stagnation_mutation_boost,
            seed_pool=seed_pool,
            seed=args.seed + 10000,
            saturn_repo_root=str(saturn_repo_root),
            saturn_oracle_config=str(saturn_oracle_config),
            saturn_oracle_config_key=args.saturn_oracle_config_key,
            saturn_apply_diversity_penalty=args.saturn_apply_diversity_penalty,
            saturn_diversity_bucket_size=args.saturn_diversity_bucket_size,
            saturn_invalid_score=args.saturn_invalid_score,
            saturn_allow_oracle_repeats=args.saturn_allow_oracle_repeats,
            saturn_task_name=args.saturn_task_name,
        )

    cmp_df = build_comparison_df(evo_df=evo_df, rnd_df=rnd_df)
    cmp_path = root / "comparison_evolution_vs_random.csv"
    cmp_df.to_csv(cmp_path, index=False)

    print(f"Evolution summary: {evo_dir / 'strategy_summary.csv'}", flush=True)
    print(f"Random summary:    {rnd_dir / 'strategy_summary.csv'}", flush=True)
    print(f"Comparison file:   {cmp_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
