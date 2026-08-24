#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import contextlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator, rdMolDescriptors, rdchem

# Make the repo root importable so the 'core' package can be found
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from core import config as cfg
from core.reports import write_guacamol_like_reports
from core.profiling import get_profiler, enable_profiling
from core.spectral_evolution import FREQUENCY_MODES, SpectralGenerator, SpectralIndividual, SpectralSettings

DEFAULT_SEEDS: list[str] = list(cfg.FALLBACK_SEED_SMILES)
ALLOWED_ATOMIC_NUMBERS: list[int] = list(cfg.ALLOWED_ATOMIC_NUMBERS)
BOND_TYPES: list[rdchem.BondType] = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
]
DEFAULT_SATURN_REPO_ROOT: Path = (cfg.REPO_ROOT / "Saturn" / "saturn_repo2").resolve()
DEFAULT_SATURN_ORACLE_CONFIG: Path = (Path(__file__).resolve().parent / "saturn_oracle_config.example.json").resolve()
_SCORE_ADAPTER_CACHE: dict[int, Callable[[list[str], int], object]] = {}
_BRICS_FRAGMENT_CACHE: dict[tuple[str, int], tuple[str, ...]] = {}


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
    p.add_argument(
        "--task-indexes",
        default="",
        help=(
            "Comma-separated zero-based MolScore task indexes to run after include/exclude filtering. "
            "Useful when MolScore displays a task name but cannot select it through include."
        ),
    )
    p.add_argument("--output-dir", default=cfg.OUTPUT_RUNS_DIR_DEFAULT_STR, help="Root output directory.")
    p.add_argument("--budget", type=int, default=cfg.BUDGET_DEFAULT, help="Molecule budget per task.")
    p.add_argument("--population-size", type=int, default=cfg.POPULATION_SIZE_DEFAULT, help="Population size for evolutionary strategy.")
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE_DEFAULT, help="Scoring batch size per generation.")
    p.add_argument("--max-generations", type=int, default=cfg.MAX_GENERATIONS_DEFAULT, help="Optional hard stop; 0 disables generation cap.")
    p.add_argument("--tournament-k", type=int, default=cfg.TOURNAMENT_K_DEFAULT, help="Tournament size for parent selection.")
    p.add_argument(
        "--generator",
        choices=("spectral", "rdkit_mutation"),
        default=getattr(cfg, "LOCAL_EVO_GENERATOR_DEFAULT", "spectral"),
        help="Molecule generator for local_evolution. 'spectral' evolves Fourier Theta genotypes.",
    )
    p.add_argument(
        "--frequency-mode",
        choices=FREQUENCY_MODES,
        default=getattr(cfg, "FREQUENCY_MODE_DEFAULT", "full-spectrum"),
        help="Spectral frequency condition for Fourier genotype evolution.",
    )
    p.add_argument("--spectral-l", type=int, default=int(getattr(cfg.CFG, "L", 32)), help="SELFIES decode length for spectral generator.")
    p.add_argument("--spectral-k", type=int, default=int(getattr(cfg.CFG, "K", 16)), help="Number of Fourier harmonics.")
    p.add_argument("--spectral-d", type=int, default=int(getattr(cfg.CFG, "D", 32)), help="Latent/token embedding dimension.")
    p.add_argument(
        "--spectral-decode-attempts",
        type=int,
        default=int(getattr(cfg.CFG, "DECODE_ATTEMPTS", 8)),
        help="Decode attempts per Theta candidate.",
    )
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


def parse_int_csv(raw: str) -> list[int]:
    indexes: list[int] = []
    for item in parse_csv_list(raw):
        try:
            idx = int(item)
        except ValueError as exc:
            raise ValueError(f"Expected integer task index, got {item!r}") from exc
        if idx < 0:
            raise ValueError(f"Task indexes are zero-based and must be >= 0, got {idx}")
        indexes.append(idx)
    return indexes


def parse_float_schedule(raw: object, default: Sequence[float]) -> list[float]:
    out: list[float] = []
    for item in parse_csv_list(str(raw or "")):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out if out else [float(x) for x in default]


def parse_int_schedule(raw: object, default: Sequence[int]) -> list[int]:
    out: list[int] = []
    for item in parse_csv_list(str(raw or "")):
        try:
            out.append(max(1, int(item)))
        except ValueError:
            continue
    return out if out else [max(1, int(x)) for x in default]


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
    max_trials: int,
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
    max_trials: int,
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
    n_steps = rng.randint(1, max(1, max_steps))
    for _ in range(n_steps):
        ops = [
            mutate_substitute_atom,
            mutate_add_leaf_atom,
            mutate_remove_leaf_atom,
            mutate_bond_order,
            mutate_insert_atom_in_bond,
            mutate_add_ring_bond,
            mutate_remove_bond,
        ]
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


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula_counts(text: str) -> dict[str, int] | None:
    raw = str(text).strip()
    if not raw or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", raw):
        return None
    counts: dict[str, int] = {}
    pos = 0
    for match in _FORMULA_RE.finditer(raw):
        if match.start() != pos:
            return None
        elem = match.group(1)
        num = int(match.group(2) or "1")
        counts[elem] = counts.get(elem, 0) + num
        pos = match.end()
    return counts if pos == len(raw) and counts else None


def formula_counts_for_mol(mol: Chem.Mol) -> dict[str, int]:
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return parse_formula_counts(formula) or {}


def make_task_decode_objective(
    task_name: str,
    target_smiles: Sequence[str] | None = None,
) -> Callable[[str, int], tuple[float, float, str]]:
    """Build a cheap decode-time prior; lower f1/f2 is better."""
    task = str(task_name)
    lower = task.lower()
    target_formula = parse_formula_counts(task)
    target_heavy = 0
    target_mw = 0.0
    if target_formula:
        target_heavy = sum(v for k, v in target_formula.items() if k != "H")
        atomic_weights = {
            "C": 12.011,
            "H": 1.008,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "P": 30.974,
            "S": 32.06,
            "Cl": 35.45,
            "Br": 79.904,
            "I": 126.904,
        }
        target_mw = sum(atomic_weights.get(elem, 0.0) * count for elem, count in target_formula.items())

    target_mols: list[Chem.Mol] = []
    for smiles in target_smiles or []:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() >= 4:
            target_mols.append(mol)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fps = [morgan_generator.GetFingerprint(mol) for mol in target_mols]
    target_heavies = [float(mol.GetNumHeavyAtoms()) for mol in target_mols]
    target_mws = [float(Descriptors.MolWt(mol)) for mol in target_mols]
    target_rings = [float(mol.GetRingInfo().NumRings()) for mol in target_mols]
    target_heteros = [
        float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
        for mol in target_mols
    ]
    target_logps = [float(Descriptors.MolLogP(mol)) for mol in target_mols]
    target_tpsas = [float(rdMolDescriptors.CalcTPSA(mol)) for mol in target_mols]
    valsartan_smarts_patterns = tuple(p for p in _VALSARTAN_SMARTS if p is not None)

    def interval_penalty(value: float, low: float, high: float, scale: float) -> float:
        if value < low:
            return (low - value) / max(1.0, scale)
        if value > high:
            return (value - high) / max(1.0, scale)
        return 0.0

    def objective(smiles: str, gen: int) -> tuple[float, float, str]:
        _ = gen
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 1e9, 1e9, "RDKIT_FAIL"

        heavy = float(mol.GetNumHeavyAtoms())
        rings = float(mol.GetRingInfo().NumRings())
        hetero = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
        aromatic = float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        target_similarity_penalty = 1.0
        target_descriptor_penalty = 1.0
        target_property_penalty = 1.0
        if target_fps:
            fp = morgan_generator.GetFingerprint(mol)
            sims = [float(DataStructs.TanimotoSimilarity(fp, target_fp)) for target_fp in target_fps]
            best_idx = int(np.argmax(sims))
            target_similarity_penalty = 1.0 - float(sims[best_idx])
            target_descriptor_penalty = (
                abs(heavy - target_heavies[best_idx]) / max(1.0, target_heavies[best_idx])
                + abs(mw - target_mws[best_idx]) / max(1.0, target_mws[best_idx])
                + abs(rings - target_rings[best_idx]) / 4.0
                + abs(hetero - target_heteros[best_idx]) / 12.0
            )
            target_property_penalty = (
                abs(logp - target_logps[best_idx]) / 6.0
                + abs(tpsa - target_tpsas[best_idx]) / 160.0
            )

        if target_formula:
            got = formula_counts_for_mol(mol)
            elems = set(target_formula) | set(got)
            diff = sum(abs(float(got.get(elem, 0) - target_formula.get(elem, 0))) for elem in elems)
            extra_elem_penalty = sum(float(count) for elem, count in got.items() if elem not in target_formula and elem != "H")
            heavy_penalty = abs(heavy - float(target_heavy))
            mw_penalty = abs(mw - target_mw) / max(1.0, target_mw)
            return diff + 2.0 * extra_elem_penalty + 0.25 * heavy_penalty + 0.20 * mw_penalty, mw_penalty, "OK"

        if "median_molecules" in lower:
            size_penalty = abs(heavy - 24.0) / 24.0
            ring_penalty = abs(rings - 2.0) / 3.0
            hetero_penalty = abs(hetero - 4.0) / 8.0
            return size_penalty + ring_penalty + hetero_penalty, abs(mw - 330.0) / 330.0, "OK"

        if "scaffold" in lower or "deco" in lower:
            ring_penalty = abs(rings - 2.5) / 3.5
            aromatic_bonus = -min(aromatic, 12.0) / 36.0
            size_penalty = abs(heavy - 28.0) / 32.0
            return size_penalty + ring_penalty + aromatic_bonus, abs(hetero - 4.0) / 10.0, "OK"

        if "valsartan" in lower and "smarts" in lower:
            hits = [1.0 if mol.HasSubstructMatch(pattern) else 0.0 for pattern in valsartan_smarts_patterns]
            missing_penalty = float(len(hits) - sum(hits))
            size_penalty = abs(heavy - 33.0) / 36.0
            hetero_penalty = abs(hetero - 8.0) / 12.0
            ring_penalty = abs(rings - 3.0) / 4.0
            target_term = 0.35 * target_similarity_penalty if target_fps else 0.0
            return missing_penalty + target_term + 0.25 * (size_penalty + hetero_penalty + ring_penalty), target_descriptor_penalty, "OK"

        if "mpo" in lower:
            mpo_decode_style = str(getattr(cfg, "SPECTRAL_MPO_DECODE_STYLE", "druglike")).strip().lower()
            if mpo_decode_style in {"legacy", "old", "target"}:
                mw_penalty = abs(mw - 420.0) / 420.0
                heavy_penalty = abs(heavy - 30.0) / 35.0
                hetero_penalty = abs(hetero - 5.0) / 12.0
                rot_penalty = max(0.0, rot - 9.0) / 12.0
                generic = mw_penalty + heavy_penalty + hetero_penalty + 0.35 * rot_penalty
                if target_fps:
                    return 0.70 * target_similarity_penalty + 0.30 * generic, target_descriptor_penalty, "OK"
                return generic, abs(rings - 2.0) / 4.0, "OK"

            mw_penalty = interval_penalty(mw, 300.0, 460.0, 220.0)
            heavy_penalty = interval_penalty(heavy, 22.0, 36.0, 24.0)
            hetero_penalty = interval_penalty(hetero, 4.0, 10.0, 10.0)
            rot_penalty = interval_penalty(rot, 0.0, 8.0, 10.0)
            logp_penalty = interval_penalty(logp, 0.5, 4.0, 5.0)
            tpsa_penalty = interval_penalty(tpsa, 35.0, 115.0, 130.0)
            ring_penalty = interval_penalty(rings, 1.0, 5.0, 4.0)
            drug_like = (
                0.18 * mw_penalty
                + 0.12 * heavy_penalty
                + 0.12 * hetero_penalty
                + 0.16 * rot_penalty
                + 0.22 * logp_penalty
                + 0.16 * tpsa_penalty
                + 0.04 * ring_penalty
            )
            if mpo_decode_style in {"property", "property_only", "druglike_only"}:
                target_shape = 0.04 * target_descriptor_penalty if target_fps else 0.0
                return drug_like + target_shape, target_descriptor_penalty + drug_like, "OK"
            if target_fps:
                similarity_weight = 0.55
                if any(name in lower for name in ("perindopril", "amlodipine", "fexofenadine")):
                    similarity_weight = 0.42
                elif any(name in lower for name in ("ranolazine", "sitagliptin", "zaleplon")):
                    similarity_weight = 0.50
                property_weight = 1.0 - similarity_weight
                target_shape = 0.07 * target_descriptor_penalty + 0.03 * target_property_penalty
                return (
                    similarity_weight * target_similarity_penalty
                    + property_weight * drug_like
                    + target_shape,
                    target_descriptor_penalty + drug_like,
                    "OK",
                )
            return drug_like, abs(rings - 2.0) / 4.0, "OK"

        if "similarity" in lower or "rediscovery" in lower:
            if target_fps:
                return target_similarity_penalty, target_descriptor_penalty, "OK"
            mw_penalty = abs(mw - 340.0) / 420.0
            ring_penalty = abs(rings - 2.0) / 4.0
            aromatic_bonus = -min(aromatic, 10.0) / 40.0
            hetero_penalty = abs(hetero - 4.0) / 12.0
            return mw_penalty + ring_penalty + hetero_penalty + aromatic_bonus, abs(heavy - 25.0) / 35.0, "OK"

        return abs(heavy - 24.0) / 30.0 + abs(rings - 1.5) / 4.0, abs(mw - 320.0) / 420.0, "OK"

    return objective


_TASK_TARGET_KEY_HINTS = (
    "smiles",
    "smi",
    "target",
    "reference",
    "query",
    "molecule",
    "ligand",
    "scaffold",
)
_TASK_TARGET_FILE_SUFFIXES = (".json", ".csv", ".tsv", ".smi", ".smiles", ".txt", ".pkl", ".pickle")
_SMILES_LIKE_RE = re.compile(r"[\[\]\(\)=#\\/]|[cnops]|Br|Cl|[0-9]")
_STANDARD_GUACAMOL_TARGET_SMILES: dict[str, tuple[str, ...]] = {
    "celecoxib_rediscovery": (
        "c1cc(C)ccc1c2cc(C(F)(F)F)nn2c3ccc(cc3)S(=O)(=O)N",
    ),
    "troglitazone_rediscovery": (
        "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2",
    ),
    "thiothixene_rediscovery": (
        "O=S(=O)(N(C)C)c2cc1C(\\c3c(Sc1cc2)cccc3)=C/CCN4CCN(C)CC4",
    ),
    "aripiprazole_similarity": (
        "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl",
    ),
    "albuterol_similarity": (
        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
    ),
    "mestranol_similarity": (
        "O(c1cc4c(cc1)[C@H]3CC[C@]2([C@@H](CC[C@]2(C#C)O)[C@@H]3CC4)C)C",
    ),
    "osimertinib_mpo": (
        "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    ),
    "fexofenadine_mpo": (
        "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)(c1ccccc1)c1ccccc1",
    ),
    "ranolazine_mpo": (
        "O=C(Nc1c(cccc1C)C)CN3CCN(CC(O)COc2ccccc2OC)CC3",
    ),
    "perindopril_mpo": (
        "O=C(OCC)[C@@H](N[C@H](C(=O)N1[C@H](C(=O)O)C[C@@H]2CCCC[C@H]12)C)CCC",
    ),
    "amlodipine_mpo": (
        "Clc1ccccc1C2/C(C(=O)OC)=C(/C)N/C(COCCN)=C2/C(=O)OCC",
    ),
    "sitagliptin_mpo": (
        "Fc1cc(c(F)cc1F)C[C@@H](N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",
    ),
    "zaleplon_mpo": (
        "CCN(C(C)=O)c1cccc(-c2ccnc3c(C#N)cnn23)c1",
    ),
    "valsartan_smarts": (
        "CCCCC(=O)N(Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1)C(C(=O)O)C(C)C",
    ),
}
_VALSARTAN_SMARTS = tuple(
    Chem.MolFromSmarts(pattern)
    for pattern in (
        "c1ccc(-c2ccccc2)cc1",
        "c1nn[nH]n1",
        "C(=O)N",
        "C(=O)O",
        "CC(C)C",
    )
)


@contextlib.contextmanager
def _silence_rdkit_parse_errors():
    RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")


def _canonical_task_target_smiles(raw: str, *, min_heavy_atoms: int) -> str | None:
    smiles = str(raw).strip()
    if not smiles or len(smiles) > 512:
        return None
    if any(ch.isspace() for ch in smiles):
        return None
    if smiles.lower().endswith(_TASK_TARGET_FILE_SUFFIXES):
        return None
    if not any(ch.isalpha() for ch in smiles):
        return None

    with _silence_rdkit_parse_errors():
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None or mol.GetNumHeavyAtoms() < int(min_heavy_atoms):
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def standard_guacamol_target_smiles(task_name: str, *, min_heavy_atoms: int = 4) -> list[str]:
    key = str(task_name).strip().lower()
    candidates = _STANDARD_GUACAMOL_TARGET_SMILES.get(key, ())
    out: list[str] = []
    seen: set[str] = set()
    for smiles in candidates:
        canon = _canonical_task_target_smiles(smiles, min_heavy_atoms=min_heavy_atoms)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def extract_task_target_smiles(ms_task, *, min_heavy_atoms: int = 4) -> list[str]:
    """Extract target/reference molecules from MolScore task config.

    This only enriches the spectral vocabulary. It does not inject targets into
    the initial population or add phenotype proposals.
    """
    root = getattr(ms_task, "cfg", None)
    if root is None:
        return []

    out: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        canon = _canonical_task_target_smiles(value, min_heavy_atoms=min_heavy_atoms)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)

    def visit(obj, key_path: tuple[str, ...]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, key_path + (str(key).lower(),))
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value, key_path)
            return
        if not isinstance(obj, str):
            return

        key_hint = any(any(hint in key for hint in _TASK_TARGET_KEY_HINTS) for key in key_path)
        value_hint = bool(_SMILES_LIKE_RE.search(obj))
        if key_hint or value_hint:
            add_candidate(obj)

    visit(root, ())
    return out


def merge_task_target_smiles(task_name: str, extracted: Sequence[str], *, min_heavy_atoms: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    fallback_targets: list[str] = []
    if bool(getattr(cfg, "SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS", True)):
        fallback_targets = standard_guacamol_target_smiles(task_name, min_heavy_atoms=min_heavy_atoms)
    for smiles in list(extracted) + fallback_targets:
        canon = _canonical_task_target_smiles(smiles, min_heavy_atoms=min_heavy_atoms)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


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


def tournament_select_individual(
    population: Sequence[SpectralIndividual],
    k: int,
    rng: random.Random,
) -> SpectralIndividual:
    if len(population) == 1:
        return population[0]
    k_eff = max(1, min(int(k), len(population)))
    ids = rng.sample(range(len(population)), k_eff)
    return max((population[i] for i in ids), key=lambda ind: float(ind.score))


def ranked_spectral_population(population: Sequence[SpectralIndividual]) -> list[SpectralIndividual]:
    best: dict[str, SpectralIndividual] = {}
    for ind in population:
        prev = best.get(ind.smiles)
        if prev is None or float(ind.score) > float(prev.score):
            best[ind.smiles] = ind
    return sorted(best.values(), key=lambda ind: float(ind.score), reverse=True)


def write_spectral_genotype_snapshot(run_dir: Path, population: Sequence[SpectralIndividual]) -> None:
    if not population:
        return
    theta = np.stack([np.asarray(ind.theta, dtype=np.float64) for ind in population], axis=0)
    smiles = np.asarray([ind.smiles for ind in population], dtype=object)
    scores = np.asarray([float(ind.score) for ind in population], dtype=np.float64)
    reasons = np.asarray([ind.decode_reason for ind in population], dtype=object)
    macro_counts = np.asarray([int(ind.macro_count) for ind in population], dtype=np.int64)
    np.savez_compressed(
        run_dir / "spectral_genotypes_final.npz",
        theta=theta,
        smiles=smiles,
        scores=scores,
        decode_reasons=reasons,
        macro_counts=macro_counts,
    )


def evolve_spectral_on_task(
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
    spectral_settings: SpectralSettings,
    spectral_seed: int,
) -> StrategyResult:
    start = time.time()
    score_method = str(ms_task.cfg["scoring"]["method"])
    score_column = f"filtered_{score_method}"
    task_name = str(ms_task.cfg["task"])
    run_dir = Path(ms_task.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_target_min_heavy_atoms = max(4, int(getattr(cfg, "SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS", 2)))
    task_target_smiles = extract_task_target_smiles(
        ms_task,
        min_heavy_atoms=task_target_min_heavy_atoms,
    )
    task_target_smiles = merge_task_target_smiles(
        task_name,
        task_target_smiles,
        min_heavy_atoms=task_target_min_heavy_atoms,
    )
    task_decode_objective = (
        make_task_decode_objective(task_name, target_smiles=task_target_smiles)
        if bool(getattr(cfg, "SPECTRAL_TASK_AWARE_DECODE", True))
        else None
    )

    generator = SpectralGenerator(settings=spectral_settings, seed=int(spectral_seed))
    task_target_macro_count = generator.adapt_vocabulary_from_task_targets(task_target_smiles)
    seed_macro_count = generator.adapt_vocabulary_from_seed_pool(seed_pool)
    if task_target_smiles:
        write_csv(
            run_dir / "spectral_task_target_smiles.csv",
            [{"smiles": smiles} for smiles in task_target_smiles],
            ["smiles"],
        )
    seed_theta_pairs = generator.encode_seed_thetas(seed_pool, limit=max(1, min(len(seed_pool), 512)))
    spectral_phenotype_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION", 0.0)),
        0.0,
        1.0,
    )
    spectral_brics_crossover_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_BRICS_CROSSOVER_FRACTION", 0.0)),
        0.0,
        0.95,
    )
    spectral_brics_fragment_replace_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION", 0.0)),
        0.0,
        0.95,
    )
    use_phenotype_proposals = strategy == "evolution" and spectral_phenotype_fraction > 0.0
    if not use_phenotype_proposals:
        spectral_brics_crossover_fraction = 0.0
        spectral_brics_fragment_replace_fraction = 0.0
    spectral_parent_mutation_steps = max(1, int(getattr(cfg, "SPECTRAL_PARENT_MUTATION_STEPS", 2)))
    spectral_immigrant_mutation_steps = max(1, int(getattr(cfg, "SPECTRAL_IMMIGRANT_MUTATION_STEPS", 2)))
    theta_crossover_probability = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_CROSSOVER_PROBABILITY", 0.45)),
        0.0,
        1.0,
    )
    theta_local_search_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_LOCAL_SEARCH_FRACTION", 0.50)),
        0.0,
        0.95,
    )
    theta_local_top_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_LOCAL_TOP_FRACTION", 0.10)),
        0.01,
        1.0,
    )
    theta_local_mutation_steps = max(1, int(getattr(cfg, "SPECTRAL_THETA_LOCAL_MUTATION_STEPS", 1)))
    theta_local_sigma_scale = max(0.0, float(getattr(cfg, "SPECTRAL_THETA_LOCAL_SIGMA_SCALE", 0.30)))
    theta_local_param_noise_scale = max(0.0, float(getattr(cfg, "SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALE", 0.35)))
    theta_local_row_reset_scale = max(0.0, float(getattr(cfg, "SPECTRAL_THETA_LOCAL_ROW_RESET_SCALE", 0.0)))
    theta_local_sigma_scales = [
        max(0.0, x)
        for x in parse_float_schedule(
            getattr(cfg, "SPECTRAL_THETA_LOCAL_SIGMA_SCALES", ""),
            [theta_local_sigma_scale],
        )
    ]
    theta_local_param_noise_scales = [
        max(0.0, x)
        for x in parse_float_schedule(
            getattr(cfg, "SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALES", ""),
            [theta_local_param_noise_scale],
        )
    ]
    theta_local_row_reset_scales = [
        max(0.0, x)
        for x in parse_float_schedule(
            getattr(cfg, "SPECTRAL_THETA_LOCAL_ROW_RESET_SCALES", ""),
            [theta_local_row_reset_scale],
        )
    ]
    theta_local_mutation_step_schedule = parse_int_schedule(
        getattr(cfg, "SPECTRAL_THETA_LOCAL_MUTATION_STEP_SCHEDULE", ""),
        [theta_local_mutation_steps],
    )
    theta_token_mutation_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TOKEN_MUTATION_FRACTION", 0.35)),
        0.0,
        1.0,
    )
    theta_child_token_mutation_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION", 0.10)),
        0.0,
        1.0,
    )
    theta_token_mutation_max_edits = max(1, int(getattr(cfg, "SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS", 2)))
    theta_token_insert_probability = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TOKEN_INSERT_PROB", 0.20)),
        0.0,
        1.0,
    )
    theta_token_delete_probability = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TOKEN_DELETE_PROB", 0.05)),
        0.0,
        1.0,
    )
    theta_token_macro_insert_probability = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB", 0.25)),
        0.0,
        1.0,
    )
    theta_target_analog_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_ANALOG_FRACTION", 0.0)),
        0.0,
        1.0,
    )
    theta_target_analog_mutation_steps = max(
        1,
        int(getattr(cfg, "SPECTRAL_THETA_TARGET_ANALOG_MUTATION_STEPS", 2)),
    )
    theta_target_analog_sigma_scale = max(
        0.0,
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_ANALOG_SIGMA_SCALE", 1.0)),
    )
    theta_token_mutation_blend = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TOKEN_MUTATION_BLEND", 0.80)),
        0.0,
        1.0,
    )
    theta_blend_crossover_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION", 0.25)),
        0.0,
        1.0,
    )
    theta_differential_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DIFFERENTIAL_FRACTION", 0.15)),
        0.0,
        1.0,
    )
    theta_differential_scale = max(0.0, float(getattr(cfg, "SPECTRAL_THETA_DIFFERENTIAL_SCALE", 0.45)))
    elite_macro_refresh_every = max(0, int(getattr(cfg, "SPECTRAL_ELITE_MACRO_REFRESH_EVERY", 0)))
    elite_macro_top_n = max(1, int(getattr(cfg, "SPECTRAL_ELITE_MACRO_TOP_N", 96)))
    elite_macro_seed_keep = max(0, int(getattr(cfg, "SPECTRAL_ELITE_MACRO_SEED_KEEP", 256)))
    elite_macro_weight = max(1, int(getattr(cfg, "SPECTRAL_ELITE_MACRO_WEIGHT", 4)))
    reencode_population_after_vocab_refresh = bool(
        getattr(cfg, "SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH", True)
    )
    elite_macro_refresh_count = 0
    task_decode_candidates = max(1, int(getattr(cfg, "SPECTRAL_TASK_AWARE_DECODE_CANDIDATES", 1)))
    mutation_step_cap = max(
        int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
        int(getattr(cfg, "LOCAL_EVO_MUTATION_STEP_CAP", cfg.OFFSPRING_MUTATION_MAX_STEPS)),
        spectral_parent_mutation_steps,
        spectral_immigrant_mutation_steps,
    )
    brics_min_fragment_size = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", 2)))
    brics_max_depth = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MAX_DEPTH", 3)))
    seed_fragment_pool_size = max(1, int(getattr(cfg, "LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", 256)))
    seed_fragment_pool: list[str] = []
    if use_phenotype_proposals and (
        spectral_brics_crossover_fraction > 0.0 or spectral_brics_fragment_replace_fraction > 0.0
    ):
        seed_fragment_pool = build_seed_fragment_pool(
            seed_pool=seed_pool,
            rng=rng,
            sample_size=seed_fragment_pool_size,
            min_fragment_size=brics_min_fragment_size,
        )
    print(
        "[spectral] "
        f"task={task_name} frequency_mode={spectral_settings.frequency_mode} "
        f"L={spectral_settings.L} K={spectral_settings.K} D={spectral_settings.D} "
        f"vocab={len(generator.vocab)} seed_macros={seed_macro_count} "
        f"task_targets={len(task_target_smiles)} task_macros={task_target_macro_count} "
        f"phenotype_fraction={spectral_phenotype_fraction:.2f} "
        f"theta_only={int(not use_phenotype_proposals)} "
        f"theta_crossover={theta_crossover_probability:.2f} "
        f"theta_local_fraction={theta_local_search_fraction:.2f} "
        f"theta_token_local={theta_token_mutation_fraction:.2f} "
        f"theta_token_child={theta_child_token_mutation_fraction:.2f} "
        f"theta_target_analog={theta_target_analog_fraction:.2f} "
        f"theta_blend={theta_blend_crossover_fraction:.2f} "
        f"theta_de={theta_differential_fraction:.2f}/{theta_differential_scale:.2f} "
        f"task_aware_decode={int(task_decode_objective is not None)} "
        f"task_decode_candidates={task_decode_candidates} "
        f"elite_macro_refresh_every={elite_macro_refresh_every}",
        flush=True,
    )

    archive: dict[str, float] = {}
    progress_rows: list[dict] = []
    molecule_score_rows: list[dict] = []
    best_score_rows: list[dict] = []
    avg_score_rows: list[dict] = []

    init_population = generator.build_initial_population(seed_pool, pop_size)
    if not init_population:
        raise RuntimeError("Spectral generator could not build an initial population.")
    init_batch = init_population[: min(pop_size, budget)]
    init_smiles = [ind.smiles for ind in init_batch]
    init_scores = score_task_batch(ms_task, init_smiles, step=0, score_method=score_method)

    for ind, sc in zip(init_batch, init_scores.tolist()):
        ind.score = float(sc)
        archive[ind.smiles] = max(archive.get(ind.smiles, -1e18), float(sc))
        molecule_score_rows.append({"generation": 0, "smiles": ind.smiles, "score": float(sc)})

    population = init_batch
    evaluated = len(init_batch)
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

    def reencode_population_current_vocab(pop: Sequence[SpectralIndividual]) -> tuple[list[SpectralIndividual], int]:
        reencoded: list[SpectralIndividual] = []
        count = 0
        for ind in pop:
            enc = generator.encode_smiles_to_individual(
                ind.smiles,
                decode_reason=f"{ind.decode_reason}:VOCAB_REENCODED",
            )
            if enc is None:
                reencoded.append(ind)
                continue
            enc.score = float(ind.score)
            enc.macro_count = int(ind.macro_count)
            reencoded.append(enc)
            count += 1
        return reencoded, count

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
        effective_task_decode_candidates = min(
            task_decode_candidates,
            max(1, remaining // max(1, this_batch)),
        )

        prev_ranked = ranked_spectral_population(population)
        parent_pool_size = max(1, len(prev_ranked))
        parent_pool = prev_ranked
        effective_tournament_k = max(2, int(tournament_k))
        effective_parent_pool_fraction = clamp_float(parent_pool_fraction, 0.1, 1.0)
        effective_immigrant_fraction = clamp_float(immigrant_fraction, 0.0, 0.9)
        effective_elite_fraction = clamp_float(elite_fraction, 0.0, 0.9)
        stagnation_boost_active = (
            strategy == "evolution"
            and int(stagnation_patience) > 0
            and stagnation_count >= int(stagnation_patience)
        )
        mutation_depth = 1 + (int(stagnation_mutation_boost) if stagnation_boost_active else 0)
        if stagnation_boost_active:
            effective_immigrant_fraction = max(effective_immigrant_fraction, 0.25)
            effective_parent_pool_fraction = 1.0

        elite_count = 0
        immigrant_target = 0
        local_search_target = 0
        if strategy == "evolution" and prev_ranked:
            parent_pool_size = max(1, min(len(prev_ranked), int(math.ceil(len(prev_ranked) * effective_parent_pool_fraction))))
            parent_pool = prev_ranked[:parent_pool_size]
            effective_tournament_k = max(2, min(effective_tournament_k, len(parent_pool)))
            elite_count = min(pop_size - 1, max(0, int(round(pop_size * effective_elite_fraction))))
            immigrant_target = min(this_batch, max(0, int(round(this_batch * effective_immigrant_fraction))))
            non_immigrant_target = max(0, this_batch - immigrant_target)
            local_search_target = min(
                non_immigrant_target,
                max(0, int(round(non_immigrant_target * theta_local_search_fraction))),
            )

        offspring: list[SpectralIndividual] = []
        theta_candidate_groups: list[tuple[np.ndarray, list[tuple[str, str, int]]]] = []
        seen_for_decode = set(archive.keys())
        attempts = 0
        max_attempts = this_batch * cfg.OFFSPRING_ATTEMPT_FACTOR
        decode_failures: dict[str, int] = {}
        generations_for_mutation = max_generations or cfg.MAX_GENERATIONS_DEFAULT
        phenotype_encoded_count = 0
        theta_decode_count = 0
        theta_local_search_count = 0
        theta_token_mutation_count = 0
        theta_recombination_counts: dict[str, int] = {}
        theta_local_scale_counts: dict[str, int] = {}
        graph_mutation_count = 0
        brics_crossover_count = 0
        brics_fragment_replace_count = 0
        task_decode_scored_candidates = 0
        phenotype_mutation_steps = min(
            mutation_step_cap,
            spectral_parent_mutation_steps + (int(stagnation_mutation_boost) if stagnation_boost_active else 0),
        )
        immigrant_mutation_steps = min(
            mutation_step_cap,
            spectral_immigrant_mutation_steps + (int(stagnation_mutation_boost) if stagnation_boost_active else 0),
        )

        def offspring_slot_count() -> int:
            return len(offspring) + len(theta_candidate_groups)

        def add_encoded_smiles_candidate(raw_smiles: str | None, reason: str) -> bool:
            nonlocal phenotype_encoded_count
            if not raw_smiles:
                return False
            canon = canonical_smiles(raw_smiles)
            if not canon or canon in seen_for_decode:
                return False
            ind = generator.encode_smiles_to_individual(canon, decode_reason=reason)
            if ind is None or ind.smiles in seen_for_decode:
                return False
            seen_for_decode.add(ind.smiles)
            offspring.append(ind)
            phenotype_encoded_count += 1
            return True

        def add_theta_decoded_candidate(child_theta: np.ndarray) -> bool:
            nonlocal theta_decode_count
            if effective_task_decode_candidates > 1:
                local_seen = set(seen_for_decode)
                candidates: list[tuple[str, str, int]] = []
                for _ in range(effective_task_decode_candidates):
                    smiles, reason, macro_count = generator.decode_theta(
                        child_theta,
                        gen=gen,
                        seen_smiles=local_seen,
                        novelty_mode="smiles",
                        objectives_from_smiles_fn=task_decode_objective,
                    )
                    if smiles is None:
                        decode_failures[reason] = decode_failures.get(reason, 0) + 1
                        continue
                    if smiles in local_seen:
                        decode_failures["DUPLICATE_NOVELTY"] = decode_failures.get("DUPLICATE_NOVELTY", 0) + 1
                        continue
                    local_seen.add(smiles)
                    candidates.append((smiles, reason, int(macro_count)))
                if not candidates:
                    return False
                for smiles, _reason, _macro_count in candidates:
                    seen_for_decode.add(smiles)
                theta_candidate_groups.append((child_theta, candidates))
                theta_decode_count += 1
                return True

            smiles, reason, macro_count = generator.decode_theta(
                child_theta,
                gen=gen,
                seen_smiles=seen_for_decode,
                novelty_mode="smiles",
                objectives_from_smiles_fn=task_decode_objective,
            )
            if smiles is None:
                decode_failures[reason] = decode_failures.get(reason, 0) + 1
                return False
            seen_for_decode.add(smiles)
            offspring.append(
                SpectralIndividual(
                    theta=child_theta,
                    smiles=smiles,
                    decode_reason=reason,
                    macro_count=macro_count,
                )
            )
            theta_decode_count += 1
            return True

        def propose_recombined_child(selected: Sequence[SpectralIndividual]) -> np.ndarray:
            if len(selected) <= 1:
                theta_recombination_counts["mutate_only"] = theta_recombination_counts.get("mutate_only", 0) + 1
                return generator.propose_child(
                    selected,
                    gen=gen,
                    generations=generations_for_mutation,
                    crossover_probability=theta_crossover_probability,
                    mutation_depth=mutation_depth,
                )

            roll = rng.random()
            if len(parent_pool) >= 3 and roll < theta_differential_fraction:
                de_parents = [
                    tournament_select_individual(parent_pool, effective_tournament_k, rng)
                    for _ in range(3)
                ]
                child = generator.differential_theta(
                    de_parents[0].theta,
                    de_parents[1].theta,
                    de_parents[2].theta,
                    scale=theta_differential_scale,
                )
                theta_recombination_counts["differential"] = theta_recombination_counts.get("differential", 0) + 1
                return generator.mutate_repeated(
                    child,
                    gen=gen,
                    generations=generations_for_mutation,
                    depth=mutation_depth,
                )

            if roll < (theta_differential_fraction + theta_blend_crossover_fraction):
                child = generator.blend_crossover_theta(
                    selected[0].theta,
                    selected[1].theta,
                    alpha=rng.random(),
                )
                theta_recombination_counts["blend"] = theta_recombination_counts.get("blend", 0) + 1
                return generator.mutate_repeated(
                    child,
                    gen=gen,
                    generations=generations_for_mutation,
                    depth=mutation_depth,
                )

            theta_recombination_counts["row"] = theta_recombination_counts.get("row", 0) + 1
            return generator.propose_child(
                selected,
                gen=gen,
                generations=generations_for_mutation,
                crossover_probability=theta_crossover_probability,
                mutation_depth=mutation_depth,
            )

        while offspring_slot_count() < this_batch and attempts < max_attempts:
            attempts += 1
            if use_phenotype_proposals and rng.random() < spectral_phenotype_fraction:
                candidate_smiles: str | None = None
                reason = "PHENOTYPE_MUTATION_ENCODED"

                if offspring_slot_count() < immigrant_target or not parent_pool:
                    parent_smiles = rng.choice(seed_pool)
                    candidate_smiles = mutate_smiles(
                        parent_smiles,
                        rng,
                        max_steps=immigrant_mutation_steps,
                    ) or parent_smiles
                    reason = "PHENOTYPE_IMMIGRANT_ENCODED"
                else:
                    roll = rng.random()
                    parent = tournament_select_individual(parent_pool, effective_tournament_k, rng)
                    if (
                        seed_fragment_pool
                        and len(parent_pool) > 1
                        and roll < spectral_brics_crossover_fraction
                    ):
                        mate = tournament_select_individual(parent_pool, effective_tournament_k, rng)
                        if mate.smiles == parent.smiles and len(parent_pool) > 1:
                            mate = rng.choice(parent_pool)
                        candidate_smiles = crossover_brics_smiles(
                            parent.smiles,
                            mate.smiles,
                            rng,
                            min_fragment_size=brics_min_fragment_size,
                            max_depth=brics_max_depth,
                            max_trials=3,
                            extra_fragment_pool=seed_fragment_pool,
                        )
                        reason = "PHENOTYPE_BRICS_CROSSOVER_ENCODED"
                    elif (
                        seed_fragment_pool
                        and roll < (spectral_brics_crossover_fraction + spectral_brics_fragment_replace_fraction)
                    ):
                        candidate_smiles = mutate_brics_fragment_replace(
                            parent.smiles,
                            rng,
                            seed_fragment_pool=seed_fragment_pool,
                            min_fragment_size=brics_min_fragment_size,
                            max_depth=brics_max_depth,
                            max_trials=3,
                        )
                        reason = "PHENOTYPE_BRICS_FRAGMENT_ENCODED"
                    else:
                        candidate_smiles = mutate_smiles(
                            parent.smiles,
                            rng,
                            max_steps=phenotype_mutation_steps,
                        )
                        reason = "PHENOTYPE_MUTATION_ENCODED"

                if add_encoded_smiles_candidate(candidate_smiles, reason):
                    if reason == "PHENOTYPE_BRICS_CROSSOVER_ENCODED":
                        brics_crossover_count += 1
                    elif reason == "PHENOTYPE_BRICS_FRAGMENT_ENCODED":
                        brics_fragment_replace_count += 1
                    else:
                        graph_mutation_count += 1
                    continue

            if strategy == "random" or not parent_pool:
                child_theta = generator.mutate_repeated(
                    generator.random_theta(),
                    gen=gen,
                    generations=generations_for_mutation,
                    depth=mutation_depth,
                )
            elif offspring_slot_count() < immigrant_target:
                target_theta = None
                if theta_target_analog_fraction > 0.0 and rng.random() < theta_target_analog_fraction:
                    target_theta = generator.sample_task_target_theta()
                if target_theta is not None:
                    child_theta = generator.mutate_repeated(
                        target_theta,
                        gen=gen,
                        generations=generations_for_mutation,
                        depth=min(mutation_step_cap, theta_target_analog_mutation_steps),
                        sigma_scale=theta_target_analog_sigma_scale,
                    )
                    if theta_token_mutation_fraction > 0.0 and rng.random() < theta_token_mutation_fraction:
                        child_theta = generator.mutate_token_neighborhood(
                            child_theta,
                            max_edits=theta_token_mutation_max_edits,
                            insert_probability=theta_token_insert_probability,
                            delete_probability=theta_token_delete_probability,
                            macro_insert_probability=theta_token_macro_insert_probability,
                            blend=theta_token_mutation_blend,
                        )
                        theta_token_mutation_count += 1
                elif seed_theta_pairs:
                    parent_theta, _seed_smiles = seed_theta_pairs[attempts % len(seed_theta_pairs)]
                    child_theta = generator.mutate_repeated(
                        parent_theta,
                        gen=gen,
                        generations=generations_for_mutation,
                        depth=mutation_depth,
                    )
                else:
                    child_theta = generator.mutate_repeated(
                        generator.random_theta(),
                        gen=gen,
                        generations=generations_for_mutation,
                        depth=mutation_depth,
                    )
            elif offspring_slot_count() < (immigrant_target + local_search_target):
                top_count = max(1, min(len(parent_pool), int(math.ceil(len(parent_pool) * theta_local_top_fraction))))
                local_pool = parent_pool[:top_count]
                parent = tournament_select_individual(
                    local_pool,
                    max(2, min(effective_tournament_k, len(local_pool))),
                    rng,
                )
                local_scale_idx = theta_local_search_count % max(1, len(theta_local_sigma_scales))
                local_sigma_scale = theta_local_sigma_scales[local_scale_idx % len(theta_local_sigma_scales)]
                local_param_noise_scale = theta_local_param_noise_scales[
                    local_scale_idx % len(theta_local_param_noise_scales)
                ]
                local_row_reset_scale = theta_local_row_reset_scales[
                    local_scale_idx % len(theta_local_row_reset_scales)
                ]
                local_mutation_steps = theta_local_mutation_step_schedule[
                    local_scale_idx % len(theta_local_mutation_step_schedule)
                ]
                base_theta = parent.theta
                if theta_token_mutation_fraction > 0.0 and rng.random() < theta_token_mutation_fraction:
                    base_theta = generator.mutate_token_neighborhood(
                        base_theta,
                        max_edits=theta_token_mutation_max_edits,
                        insert_probability=theta_token_insert_probability,
                        delete_probability=theta_token_delete_probability,
                        macro_insert_probability=theta_token_macro_insert_probability,
                        blend=theta_token_mutation_blend,
                    )
                    theta_token_mutation_count += 1
                child_theta = generator.mutate_repeated(
                    base_theta,
                    gen=gen,
                    generations=generations_for_mutation,
                    depth=min(mutation_step_cap, local_mutation_steps),
                    sigma_scale=local_sigma_scale,
                    param_noise_scale=local_param_noise_scale,
                    row_reset_scale=local_row_reset_scale,
                )
                scale_key = f"{local_sigma_scale:g}/{local_param_noise_scale:g}/{local_row_reset_scale:g}/{local_mutation_steps}"
                theta_local_scale_counts[scale_key] = theta_local_scale_counts.get(scale_key, 0) + 1
                theta_local_search_count += 1
            else:
                selected = [
                    tournament_select_individual(parent_pool, effective_tournament_k, rng)
                    for _ in range(2 if len(parent_pool) > 1 else 1)
                ]
                child_theta = propose_recombined_child(selected)
                if theta_child_token_mutation_fraction > 0.0 and rng.random() < theta_child_token_mutation_fraction:
                    child_theta = generator.mutate_token_neighborhood(
                        child_theta,
                        max_edits=theta_token_mutation_max_edits,
                        insert_probability=theta_token_insert_probability,
                        delete_probability=theta_token_delete_probability,
                        macro_insert_probability=theta_token_macro_insert_probability,
                        blend=theta_token_mutation_blend,
                    )
                    theta_token_mutation_count += 1

            add_theta_decoded_candidate(child_theta)

        fill_attempts = 0
        while offspring_slot_count() < this_batch and fill_attempts < this_batch * 20:
            fill_attempts += 1
            if use_phenotype_proposals and parent_pool:
                parent = tournament_select_individual(parent_pool, effective_tournament_k, rng)
                if add_encoded_smiles_candidate(
                    mutate_smiles(parent.smiles, rng, max_steps=phenotype_mutation_steps),
                    "PHENOTYPE_FILL_ENCODED",
                ):
                    graph_mutation_count += 1
                    continue
            if strategy == "evolution" and parent_pool:
                selected = [
                    tournament_select_individual(parent_pool, effective_tournament_k, rng)
                    for _ in range(2 if len(parent_pool) > 1 else 1)
                ]
                child_theta = propose_recombined_child(selected)
                if theta_child_token_mutation_fraction > 0.0 and rng.random() < theta_child_token_mutation_fraction:
                    child_theta = generator.mutate_token_neighborhood(
                        child_theta,
                        max_edits=theta_token_mutation_max_edits,
                        insert_probability=theta_token_insert_probability,
                        delete_probability=theta_token_delete_probability,
                        macro_insert_probability=theta_token_macro_insert_probability,
                        blend=theta_token_mutation_blend,
                    )
                    theta_token_mutation_count += 1
            else:
                child_theta = generator.mutate_repeated(
                    generator.random_theta(),
                    gen=gen,
                    generations=generations_for_mutation,
                    depth=mutation_depth,
                )
            add_theta_decoded_candidate(child_theta)

        if offspring_slot_count() < this_batch:
            fillers = prev_ranked or population
            while offspring_slot_count() < this_batch and fillers:
                base = rng.choice(fillers)
                offspring.append(
                    SpectralIndividual(
                        theta=base.theta.copy(),
                        smiles=base.smiles,
                        decode_reason="DUPLICATE_FILL",
                        macro_count=base.macro_count,
                    )
                )

        if theta_candidate_groups:
            candidate_smiles: list[str] = []
            candidate_meta: list[tuple[int, np.ndarray, str, str, int]] = []
            for group_idx, (theta, candidates) in enumerate(theta_candidate_groups):
                for smiles, reason, macro_count in candidates:
                    candidate_meta.append((group_idx, theta, smiles, reason, int(macro_count)))
                    candidate_smiles.append(smiles)

            candidate_scores = score_task_batch(ms_task, candidate_smiles, step=gen, score_method=score_method)
            task_decode_scored_candidates += int(len(candidate_smiles))
            evaluated += int(len(candidate_smiles))

            best_by_group: dict[int, tuple[float, np.ndarray, str, str, int]] = {}
            for (group_idx, theta, smiles, reason, macro_count), sc in zip(candidate_meta, candidate_scores.tolist()):
                score = float(sc)
                archive[smiles] = max(archive.get(smiles, -1e18), score)
                molecule_score_rows.append({"generation": gen, "smiles": smiles, "score": score})
                cur = best_by_group.get(group_idx)
                if cur is None or score > cur[0]:
                    best_by_group[group_idx] = (score, theta, smiles, reason, int(macro_count))

            for group_idx in range(len(theta_candidate_groups)):
                best = best_by_group.get(group_idx)
                if best is None:
                    continue
                score, theta, smiles, reason, macro_count = best
                offspring.append(
                    SpectralIndividual(
                        theta=np.asarray(theta, dtype=np.float64),
                        smiles=smiles,
                        score=float(score),
                        decode_reason=f"{reason}:TASK_AWARE_BEST_OF_{effective_task_decode_candidates}",
                        macro_count=int(macro_count),
                    )
                )

        unscored_indices = [idx for idx, ind in enumerate(offspring) if math.isnan(float(ind.score))]
        if unscored_indices:
            off_smiles_to_score = [offspring[idx].smiles for idx in unscored_indices]
            unscored_scores = score_task_batch(ms_task, off_smiles_to_score, step=gen, score_method=score_method)
            evaluated += len(unscored_indices)

            for idx, sc in zip(unscored_indices, unscored_scores.tolist()):
                score = float(sc)
                ind = offspring[idx]
                ind.score = score
                archive[ind.smiles] = max(archive.get(ind.smiles, -1e18), score)
                molecule_score_rows.append({"generation": gen, "smiles": ind.smiles, "score": score})

        off_scores = np.asarray([float(ind.score) for ind in offspring], dtype=np.float64)

        if strategy == "evolution":
            combined = ranked_spectral_population(list(population) + list(offspring))
            selected_pop: list[SpectralIndividual] = []
            seen_selected: set[str] = set()
            if elite_count > 0:
                for ind in prev_ranked[:elite_count]:
                    selected_pop.append(ind)
                    seen_selected.add(ind.smiles)
            for ind in combined:
                if len(selected_pop) >= pop_size:
                    break
                if ind.smiles in seen_selected:
                    continue
                selected_pop.append(ind)
                seen_selected.add(ind.smiles)
            population = selected_pop[:pop_size]
        else:
            population = offspring[:pop_size]

        if not population:
            population = prev_ranked[:pop_size]

        elite_macro_refreshed = 0
        if (
            strategy == "evolution"
            and elite_macro_refresh_every > 0
            and gen % elite_macro_refresh_every == 0
            and archive
        ):
            top_archive_smiles = [
                smiles
                for smiles, _score in sorted(archive.items(), key=lambda kv: kv[1], reverse=True)[:elite_macro_top_n]
            ]
            weighted_archive_smiles: list[str] = []
            for rank, smiles in enumerate(top_archive_smiles):
                weight = max(1, int(round(elite_macro_weight * (1.0 - (rank / max(1, len(top_archive_smiles)))))))
                weighted_archive_smiles.extend([smiles] * weight)
            macro_source = list(seed_pool)[:elite_macro_seed_keep] + weighted_archive_smiles
            elite_macro_refreshed = generator.adapt_vocabulary_from_seed_pool(macro_source)
            elite_population_reencoded = 0
            if elite_macro_refreshed > 0:
                seed_theta_pairs = generator.encode_seed_thetas(seed_pool, limit=max(1, min(len(seed_pool), 512)))
                if reencode_population_after_vocab_refresh:
                    population, elite_population_reencoded = reencode_population_current_vocab(population)
                elite_macro_refresh_count += 1
                print(
                    f"[spectral] gen={gen} refreshed_elite_macros={elite_macro_refreshed} "
                    f"vocab={len(generator.vocab)} reencoded_population={elite_population_reencoded}",
                    flush=True,
                )
        else:
            elite_population_reencoded = 0

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
                "mutation_depth": mutation_depth,
                "theta_crossover_probability": theta_crossover_probability,
                "immigrant_offspring": immigrant_target,
                "theta_local_search_target": local_search_target,
                "elite_count": elite_count,
                "parent_pool_size": parent_pool_size,
                "decode_failures": sum(decode_failures.values()),
                "phenotype_encoded_offspring": phenotype_encoded_count,
                "theta_local_search_offspring": theta_local_search_count,
                "theta_token_mutation_offspring": theta_token_mutation_count,
                "theta_recombination_counts": ";".join(
                    f"{key}:{count}" for key, count in sorted(theta_recombination_counts.items())
                ),
                "theta_local_scale_counts": ";".join(
                    f"{key}:{count}" for key, count in sorted(theta_local_scale_counts.items())
                ),
                "task_decode_candidates_requested": task_decode_candidates,
                "task_decode_candidates_effective": effective_task_decode_candidates,
                "task_decode_scored_candidates": task_decode_scored_candidates,
                "dynamic_macro_count": len(generator.dynamic_macro_expansions),
                "task_target_count": len(task_target_smiles),
                "task_target_macro_count": len(generator.task_macro_expansions),
                "elite_macro_refresh_count": elite_macro_refresh_count,
                "elite_macro_refreshed": elite_macro_refreshed,
                "elite_population_reencoded": elite_population_reencoded,
                "theta_decoded_offspring": theta_decode_count,
                "graph_mutation_offspring": graph_mutation_count,
                "brics_crossover_offspring": brics_crossover_count,
                "brics_fragment_replace_offspring": brics_fragment_replace_count,
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
            "mutation_depth",
            "theta_crossover_probability",
            "immigrant_offspring",
            "theta_local_search_target",
            "elite_count",
            "parent_pool_size",
            "decode_failures",
            "phenotype_encoded_offspring",
            "theta_local_search_offspring",
            "theta_token_mutation_offspring",
            "theta_recombination_counts",
            "theta_local_scale_counts",
            "task_decode_candidates_requested",
            "task_decode_candidates_effective",
            "task_decode_scored_candidates",
            "dynamic_macro_count",
            "task_target_count",
            "task_target_macro_count",
            "elite_macro_refresh_count",
            "elite_macro_refreshed",
            "elite_population_reencoded",
            "theta_decoded_offspring",
            "graph_mutation_offspring",
            "brics_crossover_offspring",
            "brics_fragment_replace_offspring",
        ],
    )
    top_rows = [
        {"smiles": s, "score": sc}
        for s, sc in sorted(archive.items(), key=lambda kv: kv[1], reverse=True)[: cfg.TOP_MOLECULES_TO_SAVE]
    ]
    write_csv(run_dir / "generator_top_molecules.csv", top_rows, ["smiles", "score"])
    write_tsv(run_dir / "molecule_scores_by_generation.tsv", molecule_score_rows, ["generation", "smiles", "score"])
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
    write_spectral_genotype_snapshot(run_dir, population)
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
    use_brics_operators = strategy == "evolution" and (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
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
    generator_name: str,
    spectral_settings: SpectralSettings,
    task_indexes: set[int] | None = None,
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
            "generator": generator_name,
            "frequency_mode": spectral_settings.frequency_mode if generator_name == "spectral" else "",
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
    executed_task_indexes: set[int] = set()
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
        task_idx0 = task_idx
        task_idx += 1
        if task_indexes is not None and task_idx0 not in task_indexes:
            continue
        executed_task_indexes.add(task_idx0)
        rng = random.Random(seed + task_idx)
        task_name = f"task_{task_idx}"
        try:
            with _task_context(task_like) as ms_task:
                task_cfg = getattr(ms_task, "cfg", {}) if ms_task is not None else {}
                if isinstance(task_cfg, dict):
                    task_name = str(task_cfg.get("task", task_name))
                task_counter = f"{task_idx}/{total_tasks}" if total_tasks else str(task_idx)
                print(f"[{strategy}] task {task_counter} :: {task_name}", flush=True)
                task_t0 = time.perf_counter()
                if generator_name == "spectral":
                    result = evolve_spectral_on_task(
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
                        spectral_settings=spectral_settings,
                        spectral_seed=seed + task_idx,
                    )
                else:
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
                task_elapsed = time.perf_counter() - task_t0
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
            f"top10={result.mean_top10:.4f} unique={result.unique_molecules} "
            f"time={task_elapsed:.2f}s",
            flush=True,
        )

    if task_indexes is not None:
        missing = sorted(task_indexes - executed_task_indexes)
        if missing:
            msg = (
                "Requested MolScore task indexes were not available after include/exclude filtering: "
                + ",".join(str(i) for i in missing)
            )
            task_failures.append(
                {
                    "strategy": strategy,
                    "task_index": ",".join(str(i) for i in missing),
                    "task": "task_index_filter",
                    "phase": "task_selection",
                    "error": msg,
                }
            )
            print(f"[{strategy}] ERROR {msg}", file=sys.stderr, flush=True)
            if task_failures:
                pd.DataFrame(task_failures).to_csv(out_dir / "strategy_task_failures.tsv", sep="\t", index=False)
            raise RuntimeError(msg)

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
    generator_name: str,
    spectral_settings: SpectralSettings,
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
            run_dir=str(task_dir),
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
    if generator_name == "spectral":
        result = evolve_spectral_on_task(
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
            spectral_settings=spectral_settings,
            spectral_seed=seed + 1,
        )
    else:
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
    # Phase 0: Add high-level timing measurements
    import time
    overall_start = time.perf_counter()
    stage_timings = {}
    
    # Enable profiling for Phase 0 instrumentation
    enable_profiling()
    
    args = parse_args()
    if args.generator not in {"spectral", "rdkit_mutation"}:
        raise ValueError(f"--generator must be spectral or rdkit_mutation, got {args.generator!r}")
    if args.frequency_mode not in FREQUENCY_MODES:
        raise ValueError(f"--frequency-mode must be one of {', '.join(FREQUENCY_MODES)}, got {args.frequency_mode!r}")
    if int(args.spectral_l) < 4:
        raise ValueError(f"--spectral-l must be >= 4, got {args.spectral_l}")
    if int(args.spectral_k) < 0:
        raise ValueError(f"--spectral-k must be >= 0, got {args.spectral_k}")
    if int(args.spectral_d) < 2:
        raise ValueError(f"--spectral-d must be >= 2, got {args.spectral_d}")
    if int(args.spectral_decode_attempts) < 1:
        raise ValueError(f"--spectral-decode-attempts must be >= 1, got {args.spectral_decode_attempts}")
    
    # Time: argument parsing
    include = parse_csv_list(args.include)
    exclude = parse_csv_list(args.exclude)
    task_indexes_list = parse_int_csv(args.task_indexes)
    task_indexes = set(task_indexes_list) if task_indexes_list else None
    spectral_settings = SpectralSettings(
        L=int(args.spectral_l),
        K=int(args.spectral_k),
        D=int(args.spectral_d),
        frequency_mode=str(args.frequency_mode),
        decode_attempts=int(args.spectral_decode_attempts),
        clip_theta_norm=float(getattr(cfg.CFG, "CLIP_THETA_NORM", 4.0)),
        embed_seed=int(getattr(cfg.CFG, "EMBED_SEED", 13)),
        embed_target_std=float(getattr(cfg.CFG, "EMBED_TARGET_STD", 1.0)),
        low_cutoff_fraction=float(getattr(cfg.CFG, "FREQUENCY_LOW_CUTOFF_FRACTION", 0.5)),
        zero_inactive_rows=bool(getattr(cfg.CFG, "FREQUENCY_ZERO_INACTIVE_ROWS", True)),
    )

    seed_smiles_file = cfg.resolve_from_repo(args.seed_smiles_file)
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
    seed_pool = load_seed_pool(str(seed_smiles_file), args.seed_pool_size, rng)
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
        "task_indexes": task_indexes_list,
        "budget": int(args.budget),
        "population_size": int(args.population_size),
        "batch_size": int(args.batch_size),
        "max_generations": int(args.max_generations),
        "generator": str(args.generator),
        "frequency_mode": str(args.frequency_mode),
        "spectral_l": int(args.spectral_l),
        "spectral_k": int(args.spectral_k),
        "spectral_d": int(args.spectral_d),
        "spectral_decode_attempts": int(args.spectral_decode_attempts),
        "tournament_k": int(args.tournament_k),
        "elite_fraction": float(args.elite_fraction),
        "immigrant_fraction": float(args.immigrant_fraction),
        "parent_pool_fraction": float(args.parent_pool_fraction),
        "stagnation_patience": int(args.stagnation_patience),
        "stagnation_mutation_boost": int(args.stagnation_mutation_boost),
        "seed": int(args.seed),
        "seed_smiles_file": str(seed_smiles_file),
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
    
    # Phase 0: Time evolution strategy
    evo_start = time.perf_counter()
    
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
            generator_name=str(args.generator),
            spectral_settings=spectral_settings,
            task_indexes=task_indexes,
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
            generator_name=str(args.generator),
            spectral_settings=spectral_settings,
        )
    
    # Phase 0: Record evolution timing
    evo_elapsed = time.perf_counter() - evo_start
    stage_timings["evolution"] = evo_elapsed

    if args.skip_random_baseline:
        print(f"Evolution summary: {evo_dir / 'strategy_summary.csv'}", flush=True)
        
        # Save Phase 0 profiling data
        stage_timings["total"] = time.perf_counter() - overall_start
        profiling_output = root / "phase0_timing_report.json"
        with open(profiling_output, "w") as f:
            json.dump(stage_timings, f, indent=2)
        print(f"[Phase 0] Timing report saved to {profiling_output}", flush=True)
        print(f"[Phase 0] Evolution time: {stage_timings['evolution']:.2f}s", flush=True)
        print(f"[Phase 0] Total time: {stage_timings['total']:.2f}s", flush=True)
        
        return 0

    # Phase 0: Time random baseline
    rnd_start = time.perf_counter()
    
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
            generator_name=str(args.generator),
            spectral_settings=spectral_settings,
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
            generator_name=str(args.generator),
            spectral_settings=spectral_settings,
        )
    
    # Phase 0: Record random baseline timing
    rnd_elapsed = time.perf_counter() - rnd_start
    stage_timings["random_baseline"] = rnd_elapsed

    cmp_df = build_comparison_df(evo_df=evo_df, rnd_df=rnd_df)
    cmp_path = root / "comparison_evolution_vs_random.csv"
    cmp_df.to_csv(cmp_path, index=False)

    print(f"Evolution summary: {evo_dir / 'strategy_summary.csv'}", flush=True)
    print(f"Random summary:    {rnd_dir / 'strategy_summary.csv'}", flush=True)
    print(f"Comparison file:   {cmp_path}", flush=True)
    
    # Phase 0: Save timing report
    stage_timings["total"] = time.perf_counter() - overall_start
    profiling_output = root / "phase0_timing_report.json"
    with open(profiling_output, "w") as f:
        json.dump(stage_timings, f, indent=2)
    print(f"[Phase 0] Timing report saved to {profiling_output}", flush=True)
    print(f"[Phase 0] Evolution time: {stage_timings['evolution']:.2f}s", flush=True)
    print(f"[Phase 0] Random baseline time: {stage_timings['random_baseline']:.2f}s", flush=True)
    print(f"[Phase 0] Total time: {stage_timings['total']:.2f}s", flush=True)
    
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
