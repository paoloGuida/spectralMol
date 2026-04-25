#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_ROOT = _REPO_ROOT / "core"
for _p in [str(MODULE_DIR), str(_REPO_ROOT), str(_CORE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402  (local Benchmarks/Saturn/config.py)
from evolve_vs_molscore_benchmark import (  # noqa: E402
    build_initial_population,
    build_seed_fragment_pool,
    clamp_float,
    crossover_brics_smiles,
    load_seed_pool,
    mutate_brics_fragment_replace,
    mutate_smiles,
    resolve_initialization_inputs,
    tournament_select,
)

DEFAULT_SATURN_REPO_ROOT = _CORE_ROOT.resolve()
DEFAULT_ORACLE_TEMPLATE = (MODULE_DIR / "table2_r_sa_qed_oracle_template.json").resolve()
DEFAULT_OUTPUT_ROOT = (MODULE_DIR.parent / "output" / "compare_scalar_vs_nsga2").resolve()


@dataclass(frozen=True)
class EvalRecord:
    smiles: str
    scalar: float
    rewards: np.ndarray
    raws: np.ndarray
    scaffold: str


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    budget: int
    seed: int
    evaluated: int
    generations: int
    elapsed_seconds: float
    n_unique: int
    objective_max: float
    objective_mean: float
    pct_lt_minus9: float
    pct_lt_minus10: float
    modes_lt_minus9: float
    modes_lt_minus10: float
    docking_component: str


def parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def scaffold_for_smiles(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles) or smiles
    except Exception:
        return smiles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare current scalar SATURN optimization (aggregated reward) against "
            "an NSGA-II multi-objective variant using per-component rewards."
        )
    )
    p.add_argument("--seeds", default="0", help="Comma-separated seeds.")
    p.add_argument("--budgets", default="1000", help="Comma-separated oracle budgets.")
    p.add_argument("--population-size", type=int, default=256, help="Evolution population size.")
    p.add_argument("--batch-size", type=int, default=32, help="Evaluations per generation.")
    p.add_argument("--max-generations", type=int, default=0, help="0 disables generation cap.")
    p.add_argument("--top-k", type=int, default=100, help="Top-K molecules for summary metrics.")
    p.add_argument("--seed-smiles-file", default=cfg.SEED_SMILES_FILE_DEFAULT_STR, help="Seed SMILES file.")
    p.add_argument("--seed-pool-size", type=int, default=cfg.SEED_POOL_SIZE_DEFAULT, help="Max seed-pool size.")
    p.add_argument(
        "--init-population-mode",
        choices=("current", "graphga_zinc250k"),
        default=str(getattr(cfg, "INIT_POPULATION_MODE_DEFAULT", "current")),
        help=(
            "Initialization policy. 'current' uses --seed-smiles-file; "
            "'graphga_zinc250k' uses --graphga-zinc250k-seed-smiles-file and "
            "a ZINC-250k-sized seed pool for GraphGA-style initialization."
        ),
    )
    p.add_argument(
        "--graphga-zinc250k-seed-smiles-file",
        default=str(getattr(cfg, "GRAPHGA_ZINC250K_SEED_SMILES_FILE_DEFAULT", "")),
        help="Path to ZINC-250k SMILES file used when --init-population-mode graphga_zinc250k.",
    )
    p.add_argument("--tournament-k", type=int, default=cfg.TOURNAMENT_K_DEFAULT, help="Tournament size.")
    p.add_argument(
        "--elite-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_ELITE_FRACTION", 0.15)),
        help="Elite fraction (scalar strategy).",
    )
    p.add_argument(
        "--immigrant-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_IMMIGRANT_FRACTION", 0.10)),
        help="Immigrant fraction per generation.",
    )
    p.add_argument(
        "--parent-pool-fraction",
        type=float,
        default=float(getattr(cfg, "LOCAL_EVO_PARENT_POOL_FRACTION", 0.50)),
        help="Parent pool fraction.",
    )
    p.add_argument(
        "--stagnation-patience",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_PATIENCE", 12)),
        help="Mutation boost trigger when no scalar best-score improvement.",
    )
    p.add_argument(
        "--stagnation-mutation-boost",
        type=int,
        default=int(getattr(cfg, "LOCAL_EVO_STAGNATION_MUTATION_BOOST", 2)),
        help="Extra mutation steps on stagnation.",
    )
    p.add_argument(
        "--saturn-repo-root",
        default=str(DEFAULT_SATURN_REPO_ROOT),
        help="SATURN repository root.",
    )
    p.add_argument(
        "--oracle-template",
        default=str(DEFAULT_ORACLE_TEMPLATE),
        help="Oracle template/config JSON (direct oracle payload or top-level {oracle: ...}).",
    )
    p.add_argument(
        "--oracle-config-key",
        default="oracle",
        help="Top-level key containing oracle payload if template is a full SATURN config.",
    )
    p.add_argument(
        "--quickvina-binary",
        default=str(
            (
                DEFAULT_SATURN_REPO_ROOT
                / "experimental_reproduction"
                / "synthesizability"
                / "QuickVina2-GPU-2.1"
                / "QuickVina2-GPU-2-1"
            ).resolve()
        ),
        help="QuickVina2-GPU binary path.",
    )
    p.add_argument(
        "--receptor-file",
        default=str(
            (
                DEFAULT_SATURN_REPO_ROOT
                / "experimental_reproduction"
                / "synthesizability"
                / "7uvu-2-monomers-pdbfixer.pdbqt"
            ).resolve()
        ),
        help="Docking receptor PDBQT path.",
    )
    p.add_argument(
        "--reference-ligand-file",
        default=str(
            (
                DEFAULT_SATURN_REPO_ROOT
                / "experimental_reproduction"
                / "synthesizability"
                / "7uvu-reference.pdb"
            ).resolve()
        ),
        help="Docking reference ligand PDB path.",
    )
    p.add_argument("--docking-component", default="", help="Optional docking component name override.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory.")
    p.add_argument("--run-id", default="", help="Optional run-id. Auto-generated if omitted.")
    p.add_argument("--skip-scalar", action="store_true", help="Skip scalar aggregated baseline.")
    p.add_argument("--skip-nsga2", action="store_true", help="Skip NSGA-II run.")
    return p.parse_args()


def default_run_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"compare_scalar_vs_nsga2_{stamp}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def replace_placeholders(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for token, repl in mapping.items():
            out = out.replace(token, repl)
        return out
    if isinstance(value, list):
        return [replace_placeholders(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: replace_placeholders(v, mapping) for k, v in value.items()}
    return value


def load_oracle_payload(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "components" in payload and "budget" in payload:
        return payload
    if isinstance(payload, dict) and key in payload and isinstance(payload[key], dict):
        return payload[key]
    raise ValueError(f"Cannot find oracle payload in {path} using key '{key}'.")


def resolve_oracle_config(
    *,
    template_path: Path,
    config_key: str,
    saturn_repo_root: Path,
    quickvina_binary: Path,
    receptor_file: Path,
    reference_ligand_file: Path,
    run_root: Path,
    budget: int,
) -> dict[str, Any]:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload = replace_placeholders(
        payload,
        {
            "__SATURN_REPO_ROOT__": str(saturn_repo_root),
            "__QUICKVINA_BINARY__": str(quickvina_binary),
            "__RUN_DIR__": str(run_root),
        },
    )

    oracle_cfg: dict[str, Any]
    if isinstance(payload, dict) and "components" in payload and "budget" in payload:
        oracle_cfg = payload
    elif isinstance(payload, dict) and config_key in payload and isinstance(payload[config_key], dict):
        oracle_cfg = payload[config_key]
    else:
        raise ValueError(f"Resolved template does not contain oracle payload: {template_path}")

    oracle_cfg["budget"] = int(budget)
    oracle_cfg["allow_oracle_repeats"] = False

    components = oracle_cfg.get("components", [])
    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("name", "")).lower()
            spec = comp.setdefault("specific_parameters", {})
            if not isinstance(spec, dict):
                continue
            if name == "quickvina2_gpu":
                spec["binary"] = str(quickvina_binary)
                spec["receptor"] = str(receptor_file)
                spec["reference_ligand"] = str(reference_ligand_file)
                spec["results_dir"] = str((run_root / "docking_results").resolve())

    return oracle_cfg


class SaturnComponentEvaluator:
    """
    Evaluates SATURN oracle components individually and keeps scalar aggregated reward for reporting.
    """

    def __init__(self, *, saturn_repo_root: Path, oracle_config_path: Path) -> None:
        if str(saturn_repo_root) not in sys.path:
            sys.path.insert(0, str(saturn_repo_root))
        from oracles.dataclass import OracleConfiguration  # type: ignore
        from oracles.oracle import Oracle  # type: ignore
        from oracles.reward_aggregator.reward_aggregator import RewardAggregator  # type: ignore

        oracle_payload = load_oracle_payload(oracle_config_path, "oracle")
        self._oracle = Oracle(OracleConfiguration(**oracle_payload))
        self._components = self._oracle.oracle
        self.component_names = [str(c.name) for c in self._components]
        self._weights = np.asarray([float(c.weight) for c in self._components], dtype=np.float64)
        self._aggregator_name = str(oracle_payload.get("aggregator", "product")).lower()
        self._aggregator = RewardAggregator(self._aggregator_name)
        w_sum = float(np.sum(self._weights))
        if w_sum > 0:
            self._normalized_weights = self._weights / w_sum
        else:
            self._normalized_weights = np.full_like(self._weights, 1.0 / max(1, len(self._weights)))
        self._cache: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
        self.calls = 0

    def _compute_uncached(self, canonical_smiles_list: list[str]) -> None:
        if not canonical_smiles_list:
            return
        mols = np.asarray([Chem.MolFromSmiles(s) for s in canonical_smiles_list], dtype=object)
        valid_mask = np.asarray([m is not None for m in mols], dtype=bool)
        if not np.any(valid_mask):
            for smi in canonical_smiles_list:
                self._cache[smi] = (
                    0.0,
                    np.zeros((len(self.component_names),), dtype=np.float64),
                    np.full((len(self.component_names),), np.nan, dtype=np.float64),
                )
            return

        valid_smiles = [s for s, keep in zip(canonical_smiles_list, valid_mask.tolist()) if keep]
        valid_mols = mols[valid_mask]

        reward_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        for component in self._components:
            raw_values, rewards = component.calculate_reward(valid_mols, self.calls)
            raw_rows.append(np.asarray(raw_values, dtype=np.float64))
            reward_rows.append(np.asarray(rewards, dtype=np.float64))
        reward_matrix = np.vstack(reward_rows)
        raw_matrix = np.vstack(raw_rows)
        scalar = np.asarray(self._aggregator(reward_matrix, self._weights), dtype=np.float64)

        for idx, smi in enumerate(valid_smiles):
            self._cache[smi] = (
                float(scalar[idx]),
                reward_matrix[:, idx].astype(np.float64, copy=True),
                raw_matrix[:, idx].astype(np.float64, copy=True),
            )
        self.calls += len(valid_smiles)

        invalid_smiles = [s for s, keep in zip(canonical_smiles_list, valid_mask.tolist()) if not keep]
        for smi in invalid_smiles:
            self._cache[smi] = (
                0.0,
                np.zeros((len(self.component_names),), dtype=np.float64),
                np.full((len(self.component_names),), np.nan, dtype=np.float64),
            )

    def evaluate(self, smiles_list: Sequence[str]) -> list[EvalRecord]:
        canonical_inputs: list[str | None] = [canonical_smiles(s) for s in smiles_list]
        needed: list[str] = []
        seen: set[str] = set()
        for smi in canonical_inputs:
            if smi is None or smi in self._cache or smi in seen:
                continue
            seen.add(smi)
            needed.append(smi)
        self._compute_uncached(needed)

        out: list[EvalRecord] = []
        n_obj = len(self.component_names)
        for smi in canonical_inputs:
            if smi is None:
                out.append(
                    EvalRecord(
                        smiles="",
                        scalar=0.0,
                        rewards=np.zeros((n_obj,), dtype=np.float64),
                        raws=np.full((n_obj,), np.nan, dtype=np.float64),
                        scaffold="",
                    )
                )
                continue
            scalar, rewards, raws = self._cache.get(
                smi,
                (
                    0.0,
                    np.zeros((n_obj,), dtype=np.float64),
                    np.full((n_obj,), np.nan, dtype=np.float64),
                ),
            )
            out.append(
                EvalRecord(
                    smiles=smi,
                    scalar=float(scalar),
                    rewards=np.asarray(rewards, dtype=np.float64),
                    raws=np.asarray(raws, dtype=np.float64),
                    scaffold=scaffold_for_smiles(smi),
                )
            )
        return out

    def nsga_objective_transform(self, rewards: np.ndarray) -> np.ndarray:
        """
        Apply the same weight-normalization logic used by scalar aggregation,
        but keep per-component objectives for NSGA-II.
        """
        vec = np.asarray(rewards, dtype=np.float64)
        if vec.shape[0] != self._normalized_weights.shape[0]:
            return vec
        safe = np.clip(vec, 0.0, None)
        if self._aggregator_name == "product":
            return np.power(safe, self._normalized_weights)
        if self._aggregator_name == "sum":
            return safe * self._normalized_weights
        return safe

    def nsga_objective_matrix(self, population: list[EvalRecord]) -> np.ndarray:
        if not population:
            return np.zeros((0, len(self.component_names)), dtype=np.float64)
        return np.vstack([self.nsga_objective_transform(rec.rewards) for rec in population])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    eps = 1e-12
    return bool(np.all(a >= b - eps) and np.any(a > b + eps))


def fast_non_dominated_sort(objective_matrix: np.ndarray) -> list[list[int]]:
    n = objective_matrix.shape[0]
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros((n,), dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objective_matrix[p], objective_matrix[q]):
                dominates_set[p].append(q)
            elif dominates(objective_matrix[q], objective_matrix[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def crowding_distance(objective_matrix: np.ndarray, front: list[int]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    n_obj = objective_matrix.shape[1]
    distance = {idx: 0.0 for idx in front}

    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda idx: float(objective_matrix[idx, m]))
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")

        min_v = float(objective_matrix[sorted_front[0], m])
        max_v = float(objective_matrix[sorted_front[-1], m])
        if max_v - min_v <= 1e-12:
            continue

        for i in range(1, len(sorted_front) - 1):
            prev_v = float(objective_matrix[sorted_front[i - 1], m])
            next_v = float(objective_matrix[sorted_front[i + 1], m])
            if np.isfinite(distance[sorted_front[i]]):
                distance[sorted_front[i]] += (next_v - prev_v) / (max_v - min_v)
    return distance


def nsga2_select(
    population: list[EvalRecord],
    pop_size: int,
    objective_matrix: np.ndarray | None = None,
) -> tuple[list[EvalRecord], list[int], list[float]]:
    if not population:
        return [], [], []
    if objective_matrix is None:
        objectives = np.vstack([p.rewards for p in population])
    else:
        objectives = np.asarray(objective_matrix, dtype=np.float64)
        if objectives.shape != (len(population), len(population[0].rewards)):
            raise ValueError(
                "objective_matrix must match (population_size, n_objectives). "
                f"Got {objectives.shape} for population={len(population)} and n_obj={len(population[0].rewards)}."
            )
    fronts = fast_non_dominated_sort(objectives)

    ranks = [10**9 for _ in population]
    crowd = [0.0 for _ in population]
    selected_idx: list[int] = []

    for rank, front in enumerate(fronts):
        if not front:
            continue
        dist_map = crowding_distance(objectives, front)
        for idx in front:
            ranks[idx] = rank
            crowd[idx] = float(dist_map.get(idx, 0.0))

        if len(selected_idx) + len(front) <= pop_size:
            selected_idx.extend(front)
            continue

        # Partial front fill by descending crowding distance.
        needed = pop_size - len(selected_idx)
        front_sorted = sorted(front, key=lambda idx: crowd[idx], reverse=True)
        selected_idx.extend(front_sorted[:needed])
        break

    selected = [population[i] for i in selected_idx]
    sel_rank = [ranks[i] for i in selected_idx]
    sel_crowd = [crowd[i] for i in selected_idx]
    return selected, sel_rank, sel_crowd


def unique_best_by_smiles(records: Iterable[EvalRecord]) -> dict[str, EvalRecord]:
    out: dict[str, EvalRecord] = {}
    for rec in records:
        if not rec.smiles:
            continue
        prev = out.get(rec.smiles)
        if prev is None or rec.scalar > prev.scalar:
            out[rec.smiles] = rec
    return out


def write_molecules_csv(
    path: Path,
    *,
    strategy: str,
    budget: int,
    seed: int,
    component_names: list[str],
    molecules: list[EvalRecord],
) -> None:
    rows: list[dict[str, Any]] = []
    ranked = sorted(molecules, key=lambda r: r.scalar, reverse=True)
    for rank, rec in enumerate(ranked, start=1):
        row: dict[str, Any] = {
            "strategy": strategy,
            "budget": int(budget),
            "seed": int(seed),
            "rank_by_scalar": int(rank),
            "smiles": rec.smiles,
            "scaffold": rec.scaffold,
            "scalar_objective": float(rec.scalar),
        }
        for idx, name in enumerate(component_names):
            row[f"{name}_reward"] = float(rec.rewards[idx])
            row[f"{name}_raw"] = float(rec.raws[idx]) if np.isfinite(rec.raws[idx]) else ""
        rows.append(row)

    fields = [
        "strategy",
        "budget",
        "seed",
        "rank_by_scalar",
        "smiles",
        "scaffold",
        "scalar_objective",
    ]
    for name in component_names:
        fields.extend([f"{name}_reward", f"{name}_raw"])
    write_csv(path, rows, fields)


def find_docking_index(component_names: list[str], docking_component_override: str) -> int:
    if docking_component_override:
        target = docking_component_override.strip().lower()
        for idx, name in enumerate(component_names):
            if name.lower() == target:
                return idx
        raise ValueError(f"Requested docking component '{docking_component_override}' not found in {component_names}")
    for idx, name in enumerate(component_names):
        lower = name.lower()
        if "vina" in lower or "dock" in lower:
            return idx
    return 0


def summarize_strategy(
    *,
    strategy: str,
    budget: int,
    seed: int,
    component_names: list[str],
    archive: dict[str, EvalRecord],
    evaluated: int,
    generations: int,
    elapsed_seconds: float,
    top_k: int,
    docking_component_override: str,
) -> StrategyResult:
    molecules = sorted(archive.values(), key=lambda r: r.scalar, reverse=True)
    top = molecules[: max(1, int(top_k))]
    dock_idx = find_docking_index(component_names, docking_component_override)
    docking_name = component_names[dock_idx]

    docking_vals = np.asarray([rec.raws[dock_idx] for rec in top], dtype=np.float64)
    valid = np.isfinite(docking_vals)
    denom = int(valid.sum())
    if denom > 0:
        lt9_mask = (docking_vals < -9.0) & valid
        lt10_mask = (docking_vals < -10.0) & valid
        pct_lt_minus9 = float(100.0 * np.sum(lt9_mask) / denom)
        pct_lt_minus10 = float(100.0 * np.sum(lt10_mask) / denom)
        modes_lt_minus9 = float(
            len({rec.scaffold for rec, keep in zip(top, lt9_mask.tolist()) if keep and rec.scaffold})
        )
        modes_lt_minus10 = float(
            len({rec.scaffold for rec, keep in zip(top, lt10_mask.tolist()) if keep and rec.scaffold})
        )
    else:
        pct_lt_minus9 = float("nan")
        pct_lt_minus10 = float("nan")
        modes_lt_minus9 = float("nan")
        modes_lt_minus10 = float("nan")

    scalar_vals = np.asarray([rec.scalar for rec in top], dtype=np.float64)
    return StrategyResult(
        strategy=strategy,
        budget=int(budget),
        seed=int(seed),
        evaluated=int(evaluated),
        generations=int(generations),
        elapsed_seconds=float(elapsed_seconds),
        n_unique=int(len(archive)),
        objective_max=float(np.max(scalar_vals)) if scalar_vals.size else float("nan"),
        objective_mean=float(np.mean(scalar_vals)) if scalar_vals.size else float("nan"),
        pct_lt_minus9=pct_lt_minus9,
        pct_lt_minus10=pct_lt_minus10,
        modes_lt_minus9=modes_lt_minus9,
        modes_lt_minus10=modes_lt_minus10,
        docking_component=docking_name,
    )


def generate_offspring_scalar(
    *,
    rng: random.Random,
    population: list[EvalRecord],
    seed_pool: list[str],
    archive_smiles: set[str],
    this_batch: int,
    tournament_k: int,
    parent_pool_fraction: float,
    immigrant_fraction: float,
    mutation_steps: int,
    mutation_step_cap: int,
    crossover_fraction: float,
    fragment_replace_fraction: float,
    seed_fragment_pool: Sequence[str],
    brics_min_fragment_size: int,
    brics_max_depth: int,
) -> list[str]:
    pop_sorted = sorted(population, key=lambda r: r.scalar, reverse=True)
    pool_size = max(2, int(round(len(pop_sorted) * parent_pool_fraction))) if pop_sorted else 0
    pool_size = min(pool_size, len(pop_sorted))
    if pool_size <= 0:
        parent_pop = pop_sorted
        parent_scores: list[float] = []
    else:
        parent_pop = pop_sorted[:pool_size]
        parent_scores = [rec.scalar for rec in parent_pop]

    offspring: list[str] = []
    seen = set(archive_smiles)
    seen.update(rec.smiles for rec in population if rec.smiles)

    max_attempts = max(1, this_batch * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    immigrant_target = min(this_batch, int(round(this_batch * immigrant_fraction)))
    non_immigrant_target = max(0, this_batch - immigrant_target)
    crossover_target = 0
    fragment_replace_target = 0
    use_brics_operators = (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
    if non_immigrant_target > 0:
        if use_brics_operators:
            crossover_target = min(non_immigrant_target, int(round(non_immigrant_target * crossover_fraction)))
        remaining_non_immigrant = max(0, non_immigrant_target - crossover_target)
        if use_brics_operators and seed_fragment_pool:
            fragment_replace_target = min(
                remaining_non_immigrant,
                int(round(non_immigrant_target * fragment_replace_fraction)),
            )
    mutation_target = max(0, this_batch - immigrant_target - crossover_target - fragment_replace_target)

    def choose_parent() -> str:
        if parent_pop:
            return tournament_select([r.smiles for r in parent_pop], parent_scores, tournament_k, rng)
        return rng.choice(seed_pool)

    immigrant_attempts = 0
    immigrant_max = max(1, immigrant_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while len(offspring) < immigrant_target and immigrant_attempts < immigrant_max:
        immigrant_attempts += 1
        base = rng.choice(seed_pool)
        child = mutate_smiles(base, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
        candidate = child if child else base
        can = canonical_smiles(candidate)
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    crossover_attempts = 0
    crossover_target_total = immigrant_target + crossover_target
    crossover_max = max(1, crossover_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while (
        crossover_target > 0
        and len(offspring) < crossover_target_total
        and crossover_attempts < crossover_max
    ):
        crossover_attempts += 1
        parent_a = choose_parent()
        parent_b = choose_parent()
        if parent_a == parent_b:
            continue
        child = crossover_brics_smiles(
            parent_a,
            parent_b,
            rng=rng,
            min_fragment_size=brics_min_fragment_size,
            max_depth=brics_max_depth,
            max_trials=3,
            extra_fragment_pool=seed_fragment_pool,
        )
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    fragment_attempts = 0
    fragment_target_total = immigrant_target + crossover_target + fragment_replace_target
    fragment_max = max(1, fragment_replace_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while (
        fragment_replace_target > 0
        and len(offspring) < fragment_target_total
        and fragment_attempts < fragment_max
    ):
        fragment_attempts += 1
        parent = choose_parent()
        child = mutate_brics_fragment_replace(
            parent,
            rng=rng,
            seed_fragment_pool=seed_fragment_pool,
            min_fragment_size=brics_min_fragment_size,
            max_depth=brics_max_depth,
            max_trials=3,
        )
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    attempts = 0
    mutation_target_total = immigrant_target + crossover_target + fragment_replace_target + mutation_target
    while len(offspring) < mutation_target_total and attempts < max_attempts:
        attempts += 1
        parent = choose_parent()
        child = mutate_smiles(parent, rng, max_steps=mutation_steps)
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    while len(offspring) < this_batch and attempts < 2 * max_attempts:
        attempts += 1
        filler = rng.choice(seed_pool)
        filler_child = mutate_smiles(filler, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
        candidate = filler_child if filler_child else filler
        can = canonical_smiles(candidate)
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    if len(offspring) < this_batch:
        rescue_attempts = 0
        rescue_max_attempts = this_batch * 4
        while len(offspring) < this_batch and rescue_attempts < rescue_max_attempts:
            rescue_attempts += 1
            filler = rng.choice(seed_pool)
            filler_child = mutate_smiles(filler, rng, max_steps=mutation_step_cap)
            candidate = filler_child if filler_child else filler
            can = canonical_smiles(candidate)
            if can and can not in seen:
                seen.add(can)
                offspring.append(can)
    while len(offspring) < this_batch:
        filler = canonical_smiles(rng.choice(seed_pool))
        if filler:
            offspring.append(filler)
    return offspring


def generate_offspring_nsga2(
    *,
    rng: random.Random,
    population: list[EvalRecord],
    ranks: list[int],
    crowd: list[float],
    seed_pool: list[str],
    archive_smiles: set[str],
    this_batch: int,
    tournament_k: int,
    parent_pool_fraction: float,
    immigrant_fraction: float,
    mutation_steps: int,
    mutation_step_cap: int,
    crossover_fraction: float,
    fragment_replace_fraction: float,
    seed_fragment_pool: Sequence[str],
    brics_min_fragment_size: int,
    brics_max_depth: int,
) -> list[str]:
    if not population:
        return generate_offspring_scalar(
            rng=rng,
            population=[],
            seed_pool=seed_pool,
            archive_smiles=archive_smiles,
            this_batch=this_batch,
            tournament_k=tournament_k,
            parent_pool_fraction=parent_pool_fraction,
            immigrant_fraction=immigrant_fraction,
            mutation_steps=mutation_steps,
            mutation_step_cap=mutation_step_cap,
            crossover_fraction=crossover_fraction,
            fragment_replace_fraction=fragment_replace_fraction,
            seed_fragment_pool=seed_fragment_pool,
            brics_min_fragment_size=brics_min_fragment_size,
            brics_max_depth=brics_max_depth,
        )

    order = sorted(range(len(population)), key=lambda i: (ranks[i], -crowd[i], -population[i].scalar))
    pool_size = max(2, int(round(len(order) * parent_pool_fraction)))
    pool_size = min(pool_size, len(order))
    parent_ids = order[:pool_size]

    def choose_parent() -> str:
        k_eff = max(1, min(tournament_k, len(parent_ids)))
        sampled = rng.sample(parent_ids, k_eff)
        best = sampled[0]
        for idx in sampled[1:]:
            if ranks[idx] < ranks[best]:
                best = idx
                continue
            if ranks[idx] == ranks[best] and crowd[idx] > crowd[best]:
                best = idx
        return population[best].smiles

    offspring: list[str] = []
    seen = set(archive_smiles)
    seen.update(rec.smiles for rec in population if rec.smiles)
    max_attempts = max(1, this_batch * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    immigrant_target = min(this_batch, int(round(this_batch * immigrant_fraction)))
    non_immigrant_target = max(0, this_batch - immigrant_target)
    crossover_target = 0
    fragment_replace_target = 0
    use_brics_operators = (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
    if non_immigrant_target > 0:
        if use_brics_operators:
            crossover_target = min(non_immigrant_target, int(round(non_immigrant_target * crossover_fraction)))
        remaining_non_immigrant = max(0, non_immigrant_target - crossover_target)
        if use_brics_operators and seed_fragment_pool:
            fragment_replace_target = min(
                remaining_non_immigrant,
                int(round(non_immigrant_target * fragment_replace_fraction)),
            )
    mutation_target = max(0, this_batch - immigrant_target - crossover_target - fragment_replace_target)

    immigrant_attempts = 0
    immigrant_max = max(1, immigrant_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while len(offspring) < immigrant_target and immigrant_attempts < immigrant_max:
        immigrant_attempts += 1
        base = rng.choice(seed_pool)
        child = mutate_smiles(base, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
        candidate = child if child else base
        can = canonical_smiles(candidate)
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    crossover_attempts = 0
    crossover_target_total = immigrant_target + crossover_target
    crossover_max = max(1, crossover_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while (
        crossover_target > 0
        and len(offspring) < crossover_target_total
        and crossover_attempts < crossover_max
    ):
        crossover_attempts += 1
        parent_a = choose_parent()
        parent_b = choose_parent()
        if parent_a == parent_b:
            continue
        child = crossover_brics_smiles(
            parent_a,
            parent_b,
            rng=rng,
            min_fragment_size=brics_min_fragment_size,
            max_depth=brics_max_depth,
            max_trials=3,
            extra_fragment_pool=seed_fragment_pool,
        )
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    fragment_attempts = 0
    fragment_target_total = immigrant_target + crossover_target + fragment_replace_target
    fragment_max = max(1, fragment_replace_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    while (
        fragment_replace_target > 0
        and len(offspring) < fragment_target_total
        and fragment_attempts < fragment_max
    ):
        fragment_attempts += 1
        parent = choose_parent()
        child = mutate_brics_fragment_replace(
            parent,
            rng=rng,
            seed_fragment_pool=seed_fragment_pool,
            min_fragment_size=brics_min_fragment_size,
            max_depth=brics_max_depth,
            max_trials=3,
        )
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    attempts = 0
    mutation_target_total = immigrant_target + crossover_target + fragment_replace_target + mutation_target
    while len(offspring) < mutation_target_total and attempts < max_attempts:
        attempts += 1
        parent = choose_parent()
        child = mutate_smiles(parent, rng, max_steps=mutation_steps)
        can = canonical_smiles(child) if child else None
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    while len(offspring) < this_batch and attempts < 2 * max_attempts:
        attempts += 1
        filler = rng.choice(seed_pool)
        filler_child = mutate_smiles(filler, rng, max_steps=min(mutation_step_cap, mutation_steps + 1))
        candidate = filler_child if filler_child else filler
        can = canonical_smiles(candidate)
        if can and can not in seen:
            seen.add(can)
            offspring.append(can)

    if len(offspring) < this_batch:
        rescue_attempts = 0
        rescue_max_attempts = this_batch * 4
        while len(offspring) < this_batch and rescue_attempts < rescue_max_attempts:
            rescue_attempts += 1
            filler = rng.choice(seed_pool)
            filler_child = mutate_smiles(filler, rng, max_steps=mutation_step_cap)
            candidate = filler_child if filler_child else filler
            can = canonical_smiles(candidate)
            if can and can not in seen:
                seen.add(can)
                offspring.append(can)

    while len(offspring) < this_batch:
        filler = canonical_smiles(rng.choice(seed_pool))
        if filler:
            offspring.append(filler)
    return offspring


def run_scalar_strategy(
    *,
    evaluator: SaturnComponentEvaluator,
    seed_pool: list[str],
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
) -> tuple[list[EvalRecord], dict[str, EvalRecord], int, int, float]:
    t0 = time.time()
    mutation_step_cap = max(
        int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
        int(getattr(cfg, "LOCAL_EVO_MUTATION_STEP_CAP", cfg.OFFSPRING_MUTATION_MAX_STEPS)),
    )
    crossover_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_CROSSOVER_FRACTION", 0.0)), 0.0, 0.95)
    fragment_replace_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_FRAGMENT_REPLACE_FRACTION", 0.0)), 0.0, 0.95)
    brics_disabled = bool(getattr(cfg, "LOCAL_EVO_DISABLE_BRICS", False))
    if brics_disabled:
        crossover_fraction = 0.0
        fragment_replace_fraction = 0.0
    use_brics_operators = (not brics_disabled) and (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
    brics_min_fragment_size = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", 2)))
    brics_max_depth = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MAX_DEPTH", 3)))
    seed_fragment_pool_size = max(1, int(getattr(cfg, "LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", 256)))
    elite_fraction = clamp_float(elite_fraction, 0.0, 0.9)
    immigrant_fraction = clamp_float(immigrant_fraction, 0.0, 0.9)
    parent_pool_fraction = clamp_float(parent_pool_fraction, 0.1, 1.0)
    seed_fragment_pool: list[str] = []
    if use_brics_operators:
        seed_fragment_pool = build_seed_fragment_pool(
            seed_pool=seed_pool,
            rng=rng,
            sample_size=seed_fragment_pool_size,
            min_fragment_size=brics_min_fragment_size,
        )

    init_pool = build_initial_population(seed_pool, pop_size, rng)
    init_batch = init_pool[: min(batch_size, budget)]
    population = evaluator.evaluate(init_batch)
    archive = unique_best_by_smiles(population)
    evaluated = len(init_batch)
    generations = 0

    best_so_far = max((r.scalar for r in population), default=float("-inf"))
    stagnation_count = 0

    while evaluated < budget:
        if max_generations > 0 and generations >= max_generations:
            break
        generations += 1
        remaining = budget - evaluated
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        mutation_steps = int(cfg.OFFSPRING_MUTATION_MAX_STEPS)
        eff_tournament_k = max(2, int(tournament_k))
        effective_parent_pool_fraction = parent_pool_fraction
        effective_immigrant_fraction = immigrant_fraction
        effective_elite_fraction = elite_fraction
        diversity_fraction = 0.10
        if stagnation_patience > 0 and stagnation_count >= stagnation_patience:
            mutation_steps = min(mutation_step_cap, mutation_steps + max(0, stagnation_mutation_boost))
            eff_tournament_k = max(2, min(eff_tournament_k, 3))
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.25)
            effective_parent_pool_fraction = 1.0
            diversity_fraction = 0.20
        if (
            stagnation_patience > 0
            and stagnation_count >= (3 * stagnation_patience)
        ):
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.40)
            effective_elite_fraction = min(float(effective_elite_fraction), 0.10)
            diversity_fraction = 0.30

        offspring_smiles = generate_offspring_scalar(
            rng=rng,
            population=population,
            seed_pool=seed_pool,
            archive_smiles=set(archive.keys()),
            this_batch=this_batch,
            tournament_k=eff_tournament_k,
            parent_pool_fraction=effective_parent_pool_fraction,
            immigrant_fraction=effective_immigrant_fraction,
            mutation_steps=mutation_steps,
            mutation_step_cap=mutation_step_cap,
            crossover_fraction=crossover_fraction,
            fragment_replace_fraction=fragment_replace_fraction,
            seed_fragment_pool=seed_fragment_pool,
            brics_min_fragment_size=brics_min_fragment_size,
            brics_max_depth=brics_max_depth,
        )
        offspring = evaluator.evaluate(offspring_smiles)
        evaluated += len(offspring_smiles)

        combined = unique_best_by_smiles(population + offspring)
        ranked = sorted(combined.values(), key=lambda r: r.scalar, reverse=True)

        elite_count = max(1, min(pop_size, int(round(pop_size * effective_elite_fraction))))
        elites = ranked[:elite_count]
        tail = ranked[elite_count:]
        tail_for_band = tail
        n_remaining = max(0, pop_size - len(elites))
        diversity_count = min(n_remaining, int(round(pop_size * diversity_fraction)))
        diversity_selected: list[EvalRecord] = []
        if diversity_count > 0 and tail:
            if len(tail) > diversity_count:
                diversity_selected = rng.sample(tail, diversity_count)
                diversity_smiles = {rec.smiles for rec in diversity_selected}
                tail_for_band = [rec for rec in tail if rec.smiles not in diversity_smiles]
            else:
                diversity_selected = list(tail)
                tail_for_band = []
        n_remaining = max(0, n_remaining - len(diversity_selected))
        sampled: list[EvalRecord]
        if n_remaining > 0 and tail_for_band:
            band = tail_for_band[: max(n_remaining, n_remaining * 3)]
            if len(band) > n_remaining:
                sampled = rng.sample(band, n_remaining)
                sampled = sorted(sampled, key=lambda r: r.scalar, reverse=True)
            else:
                sampled = band
        else:
            sampled = []
        selected = elites + sampled + diversity_selected
        if len(selected) < pop_size:
            selected_smiles = {rec.smiles for rec in selected}
            for cand in ranked:
                if cand.smiles in selected_smiles:
                    continue
                selected.append(cand)
                selected_smiles.add(cand.smiles)
                if len(selected) >= pop_size:
                    break
        population = selected[:pop_size]

        for rec in offspring:
            prev = archive.get(rec.smiles)
            if prev is None or rec.scalar > prev.scalar:
                archive[rec.smiles] = rec

        current_best = max((r.scalar for r in archive.values()), default=float("-inf"))
        if current_best > best_so_far + 1e-9:
            best_so_far = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1

    elapsed = time.time() - t0
    all_molecules = sorted(archive.values(), key=lambda r: r.scalar, reverse=True)
    return all_molecules, archive, evaluated, generations, elapsed


def run_nsga2_strategy(
    *,
    evaluator: SaturnComponentEvaluator,
    seed_pool: list[str],
    budget: int,
    pop_size: int,
    batch_size: int,
    max_generations: int,
    tournament_k: int,
    immigrant_fraction: float,
    parent_pool_fraction: float,
    stagnation_patience: int,
    stagnation_mutation_boost: int,
    rng: random.Random,
) -> tuple[list[EvalRecord], dict[str, EvalRecord], int, int, float]:
    t0 = time.time()
    mutation_step_cap = max(
        int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
        int(getattr(cfg, "LOCAL_EVO_MUTATION_STEP_CAP", cfg.OFFSPRING_MUTATION_MAX_STEPS)),
    )
    crossover_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_CROSSOVER_FRACTION", 0.0)), 0.0, 0.95)
    fragment_replace_fraction = clamp_float(float(getattr(cfg, "LOCAL_EVO_FRAGMENT_REPLACE_FRACTION", 0.0)), 0.0, 0.95)
    brics_disabled = bool(getattr(cfg, "LOCAL_EVO_DISABLE_BRICS", False))
    if brics_disabled:
        crossover_fraction = 0.0
        fragment_replace_fraction = 0.0
    use_brics_operators = (not brics_disabled) and (crossover_fraction > 0.0 or fragment_replace_fraction > 0.0)
    brics_min_fragment_size = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MIN_FRAGMENT_SIZE", 2)))
    brics_max_depth = max(1, int(getattr(cfg, "LOCAL_EVO_BRICS_MAX_DEPTH", 3)))
    seed_fragment_pool_size = max(1, int(getattr(cfg, "LOCAL_EVO_SEED_FRAGMENT_POOL_SIZE", 256)))
    immigrant_fraction = clamp_float(immigrant_fraction, 0.0, 0.9)
    parent_pool_fraction = clamp_float(parent_pool_fraction, 0.1, 1.0)
    seed_fragment_pool: list[str] = []
    if use_brics_operators:
        seed_fragment_pool = build_seed_fragment_pool(
            seed_pool=seed_pool,
            rng=rng,
            sample_size=seed_fragment_pool_size,
            min_fragment_size=brics_min_fragment_size,
        )

    init_pool = build_initial_population(seed_pool, pop_size, rng)
    init_batch = init_pool[: min(batch_size, budget)]
    population = evaluator.evaluate(init_batch)
    archive = unique_best_by_smiles(population)
    evaluated = len(init_batch)
    generations = 0

    best_so_far = max((r.scalar for r in population), default=float("-inf"))
    stagnation_count = 0

    while evaluated < budget:
        if max_generations > 0 and generations >= max_generations:
            break
        generations += 1
        remaining = budget - evaluated
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        # Parent metadata from current NSGA-II population.
        population_objectives = evaluator.nsga_objective_matrix(population)
        population, ranks, crowd = nsga2_select(
            population,
            len(population),
            objective_matrix=population_objectives,
        )

        mutation_steps = int(cfg.OFFSPRING_MUTATION_MAX_STEPS)
        eff_tournament_k = max(2, int(tournament_k))
        effective_parent_pool_fraction = parent_pool_fraction
        effective_immigrant_fraction = immigrant_fraction
        if stagnation_patience > 0 and stagnation_count >= stagnation_patience:
            mutation_steps = min(mutation_step_cap, mutation_steps + max(0, stagnation_mutation_boost))
            eff_tournament_k = max(2, min(eff_tournament_k, 3))
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.25)
            effective_parent_pool_fraction = 1.0
        if (
            stagnation_patience > 0
            and stagnation_count >= (3 * stagnation_patience)
        ):
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.40)

        offspring_smiles = generate_offspring_nsga2(
            rng=rng,
            population=population,
            ranks=ranks,
            crowd=crowd,
            seed_pool=seed_pool,
            archive_smiles=set(archive.keys()),
            this_batch=this_batch,
            tournament_k=eff_tournament_k,
            parent_pool_fraction=effective_parent_pool_fraction,
            immigrant_fraction=effective_immigrant_fraction,
            mutation_steps=mutation_steps,
            mutation_step_cap=mutation_step_cap,
            crossover_fraction=crossover_fraction,
            fragment_replace_fraction=fragment_replace_fraction,
            seed_fragment_pool=seed_fragment_pool,
            brics_min_fragment_size=brics_min_fragment_size,
            brics_max_depth=brics_max_depth,
        )
        offspring = evaluator.evaluate(offspring_smiles)
        evaluated += len(offspring_smiles)

        combined = unique_best_by_smiles(population + offspring)
        combined_records = list(combined.values())
        combined_objectives = evaluator.nsga_objective_matrix(combined_records)
        population, _, _ = nsga2_select(
            combined_records,
            pop_size,
            objective_matrix=combined_objectives,
        )

        for rec in offspring:
            prev = archive.get(rec.smiles)
            if prev is None or rec.scalar > prev.scalar:
                archive[rec.smiles] = rec

        current_best = max((r.scalar for r in archive.values()), default=float("-inf"))
        if current_best > best_so_far + 1e-9:
            best_so_far = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1

    elapsed = time.time() - t0
    all_molecules = sorted(archive.values(), key=lambda r: r.scalar, reverse=True)
    return all_molecules, archive, evaluated, generations, elapsed


def result_to_row(res: StrategyResult) -> dict[str, Any]:
    return {
        "strategy": res.strategy,
        "budget": res.budget,
        "seed": res.seed,
        "evaluated": res.evaluated,
        "generations": res.generations,
        "elapsed_seconds": res.elapsed_seconds,
        "n_unique": res.n_unique,
        "objective_max": res.objective_max,
        "objective_mean": res.objective_mean,
        "pct_lt_minus9": res.pct_lt_minus9,
        "pct_lt_minus10": res.pct_lt_minus10,
        "modes_lt_minus9": res.modes_lt_minus9,
        "modes_lt_minus10": res.modes_lt_minus10,
        "docking_component": res.docking_component,
    }


def main() -> int:
    args = parse_args()

    seeds = parse_int_csv(args.seeds)
    budgets = parse_int_csv(args.budgets)

    effective_seed_smiles_file, effective_seed_pool_size = resolve_initialization_inputs(
        init_population_mode=str(args.init_population_mode),
        seed_smiles_file=str(args.seed_smiles_file),
        seed_pool_size=int(args.seed_pool_size),
        graphga_zinc250k_seed_smiles_file=str(args.graphga_zinc250k_seed_smiles_file),
    )
    seed_smiles_file = Path(effective_seed_smiles_file).expanduser()
    if not seed_smiles_file.is_absolute():
        seed_smiles_file = cfg.resolve_from_repo(str(seed_smiles_file))
    else:
        seed_smiles_file = seed_smiles_file.resolve()
    if str(args.init_population_mode).strip().lower() == "graphga_zinc250k" and not seed_smiles_file.exists():
        raise FileNotFoundError(f"GraphGA ZINC-250k seed file not found: {seed_smiles_file}")

    saturn_repo_root = Path(args.saturn_repo_root).expanduser().resolve()
    template_path = Path(args.oracle_template).expanduser().resolve()
    quickvina_binary = Path(args.quickvina_binary).expanduser().resolve()
    receptor_file = Path(args.receptor_file).expanduser().resolve()
    reference_ligand_file = Path(args.reference_ligand_file).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    run_id = args.run_id.strip() if args.run_id.strip() else default_run_id()
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if not saturn_repo_root.exists():
        raise FileNotFoundError(f"SATURN repo root not found: {saturn_repo_root}")
    if not template_path.exists():
        raise FileNotFoundError(f"Oracle template/config not found: {template_path}")

    # Only validate docking files when the oracle template actually uses a docking component.
    _DOCKING_NAMES = frozenset({"quickvina2_gpu", "quickvina2", "dockstream", "geam_oracle"})
    _tpl_raw = json.loads(template_path.read_text(encoding="utf-8"))
    _tpl_oracle = _tpl_raw.get(args.oracle_config_key, _tpl_raw)
    _needs_docking = any(
        str(c.get("name", "")).lower() in _DOCKING_NAMES
        for c in _tpl_oracle.get("components", [])
        if isinstance(c, dict)
    )
    if _needs_docking:
        if not quickvina_binary.exists():
            raise FileNotFoundError(f"QuickVina binary not found: {quickvina_binary}")
        if not receptor_file.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_file}")
        if not reference_ligand_file.exists():
            raise FileNotFoundError(f"Reference ligand file not found: {reference_ligand_file}")

    run_params = vars(args).copy()
    run_params["effective_seed_pool_size_cap"] = int(effective_seed_pool_size)
    run_params["resolved_seed_smiles_file"] = str(seed_smiles_file)
    run_params["resolved_saturn_repo_root"] = str(saturn_repo_root)
    run_params["resolved_template_path"] = str(template_path)
    run_params["resolved_quickvina_binary"] = str(quickvina_binary)
    run_params["resolved_receptor_file"] = str(receptor_file)
    run_params["resolved_reference_ligand_file"] = str(reference_ligand_file)
    run_params["run_id"] = run_id
    (run_root / "run_parameters.json").write_text(json.dumps(run_params, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []

    for budget in budgets:
        for seed in seeds:
            print(f"[compare] budget={budget} seed={seed} starting", flush=True)
            case_root = run_root / "runs" / f"budget_{int(budget)}" / f"seed_{int(seed)}"
            case_root.mkdir(parents=True, exist_ok=True)

            rng_scalar = random.Random(seed)
            rng_nsga = random.Random(seed)
            seed_pool = load_seed_pool(str(seed_smiles_file), int(effective_seed_pool_size), rng_scalar)

            # Scalar baseline (current behavior: optimize aggregated score).
            if not args.skip_scalar:
                scalar_root = case_root / "scalar_aggregated"
                scalar_root.mkdir(parents=True, exist_ok=True)
                scalar_cfg = resolve_oracle_config(
                    template_path=template_path,
                    config_key=args.oracle_config_key,
                    saturn_repo_root=saturn_repo_root,
                    quickvina_binary=quickvina_binary,
                    receptor_file=receptor_file,
                    reference_ligand_file=reference_ligand_file,
                    run_root=scalar_root,
                    budget=int(budget),
                )
                scalar_cfg_path = scalar_root / "oracle_config.resolved.json"
                scalar_cfg_path.write_text(json.dumps({"oracle": scalar_cfg}, indent=2), encoding="utf-8")

                scalar_eval = SaturnComponentEvaluator(
                    saturn_repo_root=saturn_repo_root,
                    oracle_config_path=scalar_cfg_path,
                )
                scalar_mols, scalar_archive, eval_calls, gens, elapsed = run_scalar_strategy(
                    evaluator=scalar_eval,
                    seed_pool=seed_pool,
                    budget=int(budget),
                    pop_size=int(args.population_size),
                    batch_size=int(args.batch_size),
                    max_generations=int(args.max_generations),
                    tournament_k=int(args.tournament_k),
                    elite_fraction=float(args.elite_fraction),
                    immigrant_fraction=float(args.immigrant_fraction),
                    parent_pool_fraction=float(args.parent_pool_fraction),
                    stagnation_patience=int(args.stagnation_patience),
                    stagnation_mutation_boost=int(args.stagnation_mutation_boost),
                    rng=rng_scalar,
                )
                write_molecules_csv(
                    scalar_root / "molecules_all.csv",
                    strategy="scalar_aggregated",
                    budget=int(budget),
                    seed=int(seed),
                    component_names=scalar_eval.component_names,
                    molecules=scalar_mols,
                )
                scalar_summary = summarize_strategy(
                    strategy="scalar_aggregated",
                    budget=int(budget),
                    seed=int(seed),
                    component_names=scalar_eval.component_names,
                    archive=scalar_archive,
                    evaluated=eval_calls,
                    generations=gens,
                    elapsed_seconds=elapsed,
                    top_k=int(args.top_k),
                    docking_component_override=str(args.docking_component),
                )
                summary_rows.append(result_to_row(scalar_summary))
                print(
                    "[compare] scalar done "
                    f"budget={budget} seed={seed} pct<-9={scalar_summary.pct_lt_minus9:.2f} "
                    f"pct<-10={scalar_summary.pct_lt_minus10:.2f} time={elapsed/60.0:.1f}m",
                    flush=True,
                )

            # NSGA-II multi-objective (optimize per-component rewards directly).
            if not args.skip_nsga2:
                nsga_root = case_root / "nsga2_multiobjective"
                nsga_root.mkdir(parents=True, exist_ok=True)
                nsga_cfg = resolve_oracle_config(
                    template_path=template_path,
                    config_key=args.oracle_config_key,
                    saturn_repo_root=saturn_repo_root,
                    quickvina_binary=quickvina_binary,
                    receptor_file=receptor_file,
                    reference_ligand_file=reference_ligand_file,
                    run_root=nsga_root,
                    budget=int(budget),
                )
                nsga_cfg_path = nsga_root / "oracle_config.resolved.json"
                nsga_cfg_path.write_text(json.dumps({"oracle": nsga_cfg}, indent=2), encoding="utf-8")

                nsga_eval = SaturnComponentEvaluator(
                    saturn_repo_root=saturn_repo_root,
                    oracle_config_path=nsga_cfg_path,
                )
                nsga_mols, nsga_archive, eval_calls, gens, elapsed = run_nsga2_strategy(
                    evaluator=nsga_eval,
                    seed_pool=seed_pool,
                    budget=int(budget),
                    pop_size=int(args.population_size),
                    batch_size=int(args.batch_size),
                    max_generations=int(args.max_generations),
                    tournament_k=int(args.tournament_k),
                    immigrant_fraction=float(args.immigrant_fraction),
                    parent_pool_fraction=float(args.parent_pool_fraction),
                    stagnation_patience=int(args.stagnation_patience),
                    stagnation_mutation_boost=int(args.stagnation_mutation_boost),
                    rng=rng_nsga,
                )
                write_molecules_csv(
                    nsga_root / "molecules_all.csv",
                    strategy="nsga2_multiobjective",
                    budget=int(budget),
                    seed=int(seed),
                    component_names=nsga_eval.component_names,
                    molecules=nsga_mols,
                )
                nsga_summary = summarize_strategy(
                    strategy="nsga2_multiobjective",
                    budget=int(budget),
                    seed=int(seed),
                    component_names=nsga_eval.component_names,
                    archive=nsga_archive,
                    evaluated=eval_calls,
                    generations=gens,
                    elapsed_seconds=elapsed,
                    top_k=int(args.top_k),
                    docking_component_override=str(args.docking_component),
                )
                summary_rows.append(result_to_row(nsga_summary))
                print(
                    "[compare] nsga2 done "
                    f"budget={budget} seed={seed} pct<-9={nsga_summary.pct_lt_minus9:.2f} "
                    f"pct<-10={nsga_summary.pct_lt_minus10:.2f} time={elapsed/60.0:.1f}m",
                    flush=True,
                )

    if not summary_rows:
        raise RuntimeError("No strategies ran. Remove --skip-scalar/--skip-nsga2 or adjust flags.")

    summary_path = run_root / "scalar_vs_nsga2_summary.csv"
    fields = [
        "strategy",
        "budget",
        "seed",
        "evaluated",
        "generations",
        "elapsed_seconds",
        "n_unique",
        "objective_max",
        "objective_mean",
        "pct_lt_minus9",
        "pct_lt_minus10",
        "modes_lt_minus9",
        "modes_lt_minus10",
        "docking_component",
    ]
    write_csv(summary_path, summary_rows, fields)

    print()
    print(f"Run ID: {run_id}")
    print(f"Output root: {run_root}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
