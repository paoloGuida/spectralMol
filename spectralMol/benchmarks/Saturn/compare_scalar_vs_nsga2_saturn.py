#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_ROOT = _REPO_ROOT / "core"
for _p in [str(_CORE_ROOT), str(_REPO_ROOT), str(MODULE_DIR)]:
    if _p in sys.path:
        sys.path.remove(_p)
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
from core.spectral_evolution import (  # noqa: E402
    FREQUENCY_MODES,
    SpectralGenerator,
    SpectralIndividual,
    SpectralSettings,
)
try:  # noqa: E402
    from oracles.synthesizability.sascorer import calculateScore as _calculate_sa_score
except Exception:  # pragma: no cover - optional local cheap prefilter
    _calculate_sa_score = None

DEFAULT_SATURN_REPO_ROOT = _CORE_ROOT.resolve()
DEFAULT_ORACLE_TEMPLATE = (MODULE_DIR / "table2_r_sa_qed_oracle_template.json").resolve()
DEFAULT_OUTPUT_ROOT = (MODULE_DIR.parent / "output" / "compare_scalar_vs_nsga2").resolve()


def _cfg_float(attr: str, default: float, env_names: Sequence[str] = ()) -> float:
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    try:
        return float(getattr(cfg, attr, default))
    except Exception:
        return float(default)


def _cfg_int(attr: str, default: int, env_names: Sequence[str] = ()) -> int:
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    try:
        return int(getattr(cfg, attr, default))
    except Exception:
        return int(default)


def _cfg_bool(attr: str, default: bool, env_names: Sequence[str] = ()) -> bool:
    for name in env_names:
        raw = os.environ.get(name, "").strip().lower()
        if not raw:
            continue
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(getattr(cfg, attr, default))
    except Exception:
        return bool(default)


def _cfg_token_list(attr: str, default: str, env_names: Sequence[str] = ()) -> list[str]:
    raw = ""
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if raw:
            break
    if not raw:
        raw = str(getattr(cfg, attr, default))
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        tok = part.strip()
        if not tok:
            continue
        if not (tok.startswith("[") and tok.endswith("]")):
            tok = f"[{tok}]"
        out.append(tok)
    return out


@dataclass(frozen=True)
class EvalRecord:
    smiles: str
    scalar: float
    rewards: np.ndarray
    raws: np.ndarray
    scaffold: str
    theta: np.ndarray | None = None
    decode_reason: str = ""
    macro_count: int = 0


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
    p.add_argument(
        "--frequency-mode",
        choices=FREQUENCY_MODES,
        default="full-spectrum",
        help="SpectralMol Fourier frequency mode used by theta NSGA-II.",
    )
    p.add_argument("--spectral-l", type=int, default=32, help="SpectralMol decoded sequence length.")
    p.add_argument("--spectral-k", type=int, default=16, help="SpectralMol Fourier mode truncation K.")
    p.add_argument("--spectral-d", type=int, default=32, help="SpectralMol latent dimension.")
    p.add_argument("--spectral-decode-attempts", type=int, default=8, help="Best-of-k theta decode attempts.")
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
    p.add_argument(
        "--per-seed-seed-smiles-dir",
        default="",
        help=(
            "Optional directory containing old-benchmark seed-specific initialization "
            "files named 'seed_<seed>.smi'. When present, each seed uses its "
            "corresponding 256-molecule population."
        ),
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
    p.add_argument(
        "--nsga2-genotype",
        choices=("theta", "smiles"),
        default="theta",
        help=(
            "Representation evolved by the NSGA-II strategy. 'theta' is SpectralMol's "
            "Fourier genotype; 'smiles' keeps the legacy string/BRICS NSGA-II path."
        ),
    )
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
        saturn_path = str(saturn_repo_root)
        core_path = str(_CORE_ROOT.resolve())
        cleaned_path: list[str] = []
        for p in sys.path:
            try:
                if str(Path(p).resolve()) == core_path:
                    continue
            except Exception:
                pass
            if p != saturn_path:
                cleaned_path.append(p)
        sys.path = cleaned_path
        sys.path.insert(0, saturn_path)
        for module_name in list(sys.modules):
            if module_name == "oracles" or module_name.startswith("oracles."):
                del sys.modules[module_name]
        importlib.invalidate_caches()
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

    @staticmethod
    def _is_docking_component(component: Any) -> bool:
        name = str(getattr(component, "name", "")).lower()
        return "vina" in name or "dock" in name

    @staticmethod
    def _looks_like_full_docking_failure(raw_values: np.ndarray) -> bool:
        raw = np.asarray(raw_values, dtype=np.float64)
        if raw.size == 0:
            return False
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return True
        return bool(np.all(finite >= 50.0))

    def _calculate_component_reward(
        self,
        component: Any,
        mols: np.ndarray,
        oracle_calls: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_docking_component(component):
            raw, rewards = component.calculate_reward(mols, oracle_calls)
            return np.asarray(raw, dtype=np.float64), np.asarray(rewards, dtype=np.float64)

        n = int(len(mols))
        chunk_size = max(1, int(getattr(cfg, "SATURN_DOCKING_CHUNK_SIZE", 32)))
        retry_chunk_size = max(1, int(getattr(cfg, "SATURN_DOCKING_RETRY_CHUNK_SIZE", 8)))
        raw_all = np.full((n,), 99.9, dtype=np.float64)
        reward_all = np.zeros((n,), dtype=np.float64)

        def run_chunk(start: int, end: int, call_offset: int) -> tuple[np.ndarray, np.ndarray]:
            sub_mols = np.asarray(mols[start:end], dtype=object)
            raw, rewards = component.calculate_reward(sub_mols, oracle_calls + call_offset)
            return np.asarray(raw, dtype=np.float64), np.asarray(rewards, dtype=np.float64)

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            raw, rewards = run_chunk(start, end, start)
            if (
                self._looks_like_full_docking_failure(raw)
                and retry_chunk_size < (end - start)
            ):
                for sub_start in range(start, end, retry_chunk_size):
                    sub_end = min(end, sub_start + retry_chunk_size)
                    sub_raw, sub_rewards = run_chunk(sub_start, sub_end, 1_000_000 + sub_start)
                    raw_all[sub_start:sub_end] = sub_raw[: sub_end - sub_start]
                    reward_all[sub_start:sub_end] = sub_rewards[: sub_end - sub_start]
            else:
                raw_all[start:end] = raw[: end - start]
                reward_all[start:end] = rewards[: end - start]
        return raw_all, reward_all

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
            raw_values, rewards = self._calculate_component_reward(component, valid_mols, self.calls)
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

    @staticmethod
    def _raw_objective_value(name: str, raw: float, reward: float) -> float:
        lower = name.strip().lower()
        raw_ok = bool(np.isfinite(raw))
        reward_ok = bool(np.isfinite(reward))

        if ("vina" in lower or "dock" in lower) and raw_ok:
            # Docking is reported as lower-is-better binding energy. SATURN uses
            # 99.9-style sentinels for failed docking, so push those out of the
            # Pareto front instead of treating them as finite weak binders.
            if raw >= 50.0:
                return -1.0e6
            value = -float(raw)
            cap = float(getattr(cfg, "SPECTRAL_THETA_NSGA_DOCKING_OBJECTIVE_CAP", 0.0))
            if np.isfinite(cap) and cap > 0.0:
                value = min(value, cap)
            return value
        if "qed" in lower and raw_ok:
            return float(raw)
        if (lower in {"sa", "sa_score"} or lower.startswith("sa_") or "synthetic" in lower) and raw_ok:
            return -float(raw)
        if reward_ok:
            return float(reward)
        return -1.0e6

    def nsga_objective_record(self, rec: EvalRecord) -> np.ndarray:
        if not bool(getattr(cfg, "SPECTRAL_THETA_USE_RAW_NSGA_OBJECTIVES", True)):
            return self.nsga_objective_transform(rec.rewards)

        values: list[float] = []
        for idx, name in enumerate(self.component_names):
            raw = float(rec.raws[idx]) if idx < len(rec.raws) else float("nan")
            reward = float(rec.rewards[idx]) if idx < len(rec.rewards) else float("nan")
            values.append(self._raw_objective_value(name, raw, reward))
        return np.asarray(values, dtype=np.float64)

    def nsga_objective_matrix(self, population: list[EvalRecord]) -> np.ndarray:
        if not population:
            return np.zeros((0, len(self.component_names)), dtype=np.float64)
        return np.vstack([self.nsga_objective_record(rec) for rec in population])


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
            "decode_reason": rec.decode_reason,
            "macro_count": int(rec.macro_count),
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
        "decode_reason",
        "macro_count",
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


def find_sa_index(component_names: Sequence[str]) -> int | None:
    for idx, name in enumerate(component_names):
        lower = str(name).strip().lower()
        if lower in {"sa", "sa_score"} or lower.startswith("sa_") or "synthetic" in lower:
            return idx
    return None


def find_qed_index(component_names: Sequence[str]) -> int | None:
    for idx, name in enumerate(component_names):
        lower = str(name).strip().lower()
        if lower == "qed" or lower.startswith("qed_"):
            return idx
    return None


def raw_docking_value(rec: EvalRecord, docking_idx: int | None) -> float:
    if docking_idx is None or docking_idx < 0 or docking_idx >= len(rec.raws):
        return float("inf")
    value = float(rec.raws[docking_idx])
    if not np.isfinite(value) or value >= 50.0:
        return float("inf")
    return value


def qed_value_from_smiles(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    try:
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def decode_reason_prefix(rec: EvalRecord) -> str:
    return str(rec.decode_reason or "").split(":", 1)[0].strip()


def is_seed_encoded_record(rec: EvalRecord) -> bool:
    return decode_reason_prefix(rec) in {"SEED_ENCODED", "SEED_FILL_ENCODED"}


def is_generated_theta_record(rec: EvalRecord) -> bool:
    return bool(rec.smiles and rec.theta is not None and not is_seed_encoded_record(rec))


def write_theta_progress_tsv(
    path,
    *,
    seed: int,
    generation: int,
    evaluated: int,
    archive: dict[str, EvalRecord],
    population: Sequence[EvalRecord],
    docking_idx: int | None,
    elapsed_seconds: float,
    event: str,
) -> None:
    if path is None:
        return
    records = list(archive.values())

    def count_hits(threshold: float, *, generated: bool | None = None) -> int:
        total = 0
        for rec in records:
            if generated is True and not is_generated_theta_record(rec):
                continue
            if generated is False and not is_seed_encoded_record(rec):
                continue
            if raw_docking_value(rec, docking_idx) < threshold:
                total += 1
        return total

    def best_docking(*, generated: bool | None = None) -> float:
        values: list[float] = []
        for rec in records:
            if generated is True and not is_generated_theta_record(rec):
                continue
            if generated is False and not is_seed_encoded_record(rec):
                continue
            value = raw_docking_value(rec, docking_idx)
            if value < float("inf"):
                values.append(value)
        return min(values) if values else float("nan")

    row = {
        "event": str(event),
        "seed": int(seed),
        "generation": int(generation),
        "evaluated": int(evaluated),
        "archive_size": int(len(records)),
        "population_size": int(len(population)),
        "all_lt_minus9": count_hits(-9.0),
        "generated_lt_minus9": count_hits(-9.0, generated=True),
        "seed_lt_minus9": count_hits(-9.0, generated=False),
        "all_lt_minus10": count_hits(-10.0),
        "generated_lt_minus10": count_hits(-10.0, generated=True),
        "seed_lt_minus10": count_hits(-10.0, generated=False),
        "best_docking": best_docking(),
        "best_generated_docking": best_docking(generated=True),
        "elapsed_seconds": float(elapsed_seconds),
    }
    fieldnames = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(
        "[theta-progress] "
        f"event={row['event']} gen={row['generation']} evaluated={row['evaluated']} "
        f"archive={row['archive_size']} all<-9={row['all_lt_minus9']} "
        f"gen<-9={row['generated_lt_minus9']} all<-10={row['all_lt_minus10']} "
        f"best={row['best_docking']}",
        file=sys.stderr,
        flush=True,
    )


def best_docking_target_smiles(
    population: Sequence[EvalRecord],
    *,
    docking_idx: int | None,
    sa_idx: int | None = None,
    max_sa: float | None = None,
    min_qed: float | None = None,
    limit: int,
) -> list[str]:
    ranked: list[EvalRecord] = []
    for rec in population:
        if not rec.smiles or raw_docking_value(rec, docking_idx) >= float("inf"):
            continue
        if max_sa is not None and sa_idx is not None and 0 <= sa_idx < len(rec.raws):
            sa_value = float(rec.raws[sa_idx])
            if np.isfinite(sa_value) and sa_value > float(max_sa):
                continue
        if min_qed is not None:
            qed_value = qed_value_from_smiles(rec.smiles)
            if (not np.isfinite(qed_value)) or qed_value < float(min_qed):
                continue
        ranked.append(rec)
    ranked.sort(key=lambda rec: (raw_docking_value(rec, docking_idx), -rec.scalar, rec.smiles))
    out: list[str] = []
    seen: set[str] = set()
    for rec in ranked:
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(rec.smiles)
        if len(out) >= max(0, int(limit)):
            break
    return out


def best_docking_theta_anchors(
    population: Sequence[EvalRecord],
    *,
    docking_idx: int | None,
    sa_idx: int | None = None,
    max_sa: float | None = None,
    min_qed: float | None = None,
    limit: int,
    scaffold_diverse: bool = False,
    max_per_scaffold: int = 1,
    scaffold_diverse_top_keep: int = 0,
) -> list[np.ndarray]:
    ranked: list[EvalRecord] = []
    for rec in population:
        if rec.theta is None or not rec.smiles or raw_docking_value(rec, docking_idx) >= float("inf"):
            continue
        if max_sa is not None and sa_idx is not None and 0 <= sa_idx < len(rec.raws):
            sa_value = float(rec.raws[sa_idx])
            if np.isfinite(sa_value) and sa_value > float(max_sa):
                continue
        if min_qed is not None:
            qed_value = qed_value_from_smiles(rec.smiles)
            if (not np.isfinite(qed_value)) or qed_value < float(min_qed):
                continue
        ranked.append(rec)
    ranked.sort(key=lambda rec: (raw_docking_value(rec, docking_idx), -rec.scalar, rec.smiles))

    out: list[np.ndarray] = []
    seen: set[str] = set()
    scaffold_counts: dict[str, int] = {}

    if scaffold_diverse:
        top_keep = min(max(0, int(scaffold_diverse_top_keep)), max(0, int(limit)))
        for rec in ranked:
            if len(out) >= top_keep:
                break
            if rec.smiles in seen:
                continue
            seen.add(rec.smiles)
            scaffold = str(rec.scaffold or rec.smiles)
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            out.append(np.asarray(rec.theta, dtype=np.float64).copy())
        if len(out) >= max(0, int(limit)):
            return out

        max_per = max(1, int(max_per_scaffold))
        deferred: list[EvalRecord] = []
        for rec in ranked:
            if rec.smiles in seen:
                continue
            scaffold = str(rec.scaffold or rec.smiles)
            if scaffold_counts.get(scaffold, 0) >= max_per:
                deferred.append(rec)
                continue
            seen.add(rec.smiles)
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            out.append(np.asarray(rec.theta, dtype=np.float64).copy())
            if len(out) >= max(0, int(limit)):
                return out
        ranked = deferred

    for rec in ranked:
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(np.asarray(rec.theta, dtype=np.float64).copy())
        if len(out) >= max(0, int(limit)):
            break
    return out


def best_generated_docking_theta_anchors(
    population: Sequence[EvalRecord],
    *,
    docking_idx: int | None,
    sa_idx: int | None = None,
    max_sa: float | None = None,
    min_qed: float | None = None,
    max_docking: float | None = None,
    limit: int,
    scaffold_diverse: bool = False,
    max_per_scaffold: int = 1,
    scaffold_diverse_top_keep: int = 0,
) -> list[np.ndarray]:
    ranked: list[EvalRecord] = []
    for rec in population:
        docking_value = raw_docking_value(rec, docking_idx)
        if not is_generated_theta_record(rec) or docking_value >= float("inf"):
            continue
        if max_docking is not None and np.isfinite(max_docking) and docking_value > float(max_docking):
            continue
        if max_sa is not None and sa_idx is not None and 0 <= sa_idx < len(rec.raws):
            sa_value = float(rec.raws[sa_idx])
            if np.isfinite(sa_value) and sa_value > float(max_sa):
                continue
        if min_qed is not None:
            qed_value = qed_value_from_smiles(rec.smiles)
            if (not np.isfinite(qed_value)) or qed_value < float(min_qed):
                continue
        ranked.append(rec)
    ranked.sort(key=lambda rec: (raw_docking_value(rec, docking_idx), -rec.scalar, rec.smiles))

    out: list[np.ndarray] = []
    seen: set[str] = set()
    scaffold_counts: dict[str, int] = {}

    if scaffold_diverse:
        top_keep = min(max(0, int(scaffold_diverse_top_keep)), max(0, int(limit)))
        for rec in ranked:
            if len(out) >= top_keep:
                break
            if rec.smiles in seen:
                continue
            seen.add(rec.smiles)
            scaffold = str(rec.scaffold or rec.smiles)
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            out.append(np.asarray(rec.theta, dtype=np.float64).copy())
        if len(out) >= max(0, int(limit)):
            return out

        max_per = max(1, int(max_per_scaffold))
        deferred: list[EvalRecord] = []
        for rec in ranked:
            if rec.smiles in seen:
                continue
            scaffold = str(rec.scaffold or rec.smiles)
            if scaffold_counts.get(scaffold, 0) >= max_per:
                deferred.append(rec)
                continue
            seen.add(rec.smiles)
            scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            out.append(np.asarray(rec.theta, dtype=np.float64).copy())
            if len(out) >= max(0, int(limit)):
                return out
        ranked = deferred

    for rec in ranked:
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(np.asarray(rec.theta, dtype=np.float64).copy())
        if len(out) >= max(0, int(limit)):
            break
    return out


def reencode_theta_records(generator: SpectralGenerator, records: Sequence[EvalRecord]) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    for rec in records:
        if not rec.smiles:
            out.append(rec)
            continue
        ind = generator.encode_smiles_to_individual(rec.smiles, decode_reason=rec.decode_reason)
        if ind is None:
            out.append(rec)
            continue
        out.append(replace(rec, theta=np.asarray(ind.theta, dtype=np.float64).copy()))
    return out


def inject_docking_elites(
    population: Sequence[EvalRecord],
    archive: dict[str, EvalRecord],
    *,
    docking_idx: int | None,
    pop_size: int,
    elite_fraction: float,
    generated_elite_fraction: float = 0.0,
    generated_max_docking: float | None = None,
) -> list[EvalRecord]:
    elite_count = min(
        max(0, int(pop_size)),
        max(0, int(round(max(0.0, float(elite_fraction)) * max(0, int(pop_size))))),
    )
    generated_elite_count = min(
        max(0, int(pop_size)),
        max(0, int(round(max(0.0, float(generated_elite_fraction)) * max(0, int(pop_size))))),
    )
    if elite_count <= 0 and generated_elite_count <= 0:
        return list(population)[: max(0, int(pop_size))]

    elite_candidates = [
        rec
        for rec in archive.values()
        if rec.smiles and rec.theta is not None and raw_docking_value(rec, docking_idx) < float("inf")
    ]
    elite_candidates.sort(key=lambda rec: (raw_docking_value(rec, docking_idx), -rec.scalar, rec.smiles))
    generated_candidates = [
        rec
        for rec in elite_candidates
        if is_generated_theta_record(rec)
        and (
            generated_max_docking is None
            or not np.isfinite(generated_max_docking)
            or raw_docking_value(rec, docking_idx) <= float(generated_max_docking)
        )
    ]

    out: list[EvalRecord] = []
    seen: set[str] = set()
    for rec in generated_candidates[:generated_elite_count]:
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(rec)
    target_elite_count = min(max(0, int(pop_size)), elite_count + generated_elite_count)
    for rec in elite_candidates:
        if len(out) >= target_elite_count:
            break
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(rec)
    for rec in population:
        if len(out) >= int(pop_size):
            break
        if not rec.smiles or rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(rec)
    for rec in elite_candidates[elite_count:]:
        if len(out) >= int(pop_size):
            break
        if rec.smiles in seen:
            continue
        seen.add(rec.smiles)
        out.append(rec)
    return out[: max(0, int(pop_size))]


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


def attach_theta_records(
    evaluator: SaturnComponentEvaluator,
    individuals: Sequence[SpectralIndividual],
) -> list[EvalRecord]:
    records = evaluator.evaluate([ind.smiles for ind in individuals])
    out: list[EvalRecord] = []
    for rec, ind in zip(records, individuals):
        if not rec.smiles:
            continue
        out.append(
            EvalRecord(
                smiles=rec.smiles,
                scalar=rec.scalar,
                rewards=rec.rewards,
                raws=rec.raws,
                scaffold=rec.scaffold,
                theta=np.asarray(ind.theta, dtype=np.float64).copy(),
                decode_reason=str(ind.decode_reason),
                macro_count=int(ind.macro_count),
            )
        )
    return out


def build_spectral_settings_from_args(args: argparse.Namespace) -> SpectralSettings:
    return SpectralSettings(
        L=int(args.spectral_l),
        K=int(args.spectral_k),
        D=int(args.spectral_d),
        frequency_mode=str(args.frequency_mode),
        decode_attempts=int(args.spectral_decode_attempts),
    )


def theta_record_to_individual(rec: EvalRecord) -> SpectralIndividual | None:
    if rec.theta is None or not rec.smiles:
        return None
    return SpectralIndividual(
        theta=np.asarray(rec.theta, dtype=np.float64).copy(),
        smiles=rec.smiles,
        score=float(rec.scalar),
        decode_reason=str(rec.decode_reason or "EVALUATED_THETA"),
        macro_count=int(rec.macro_count),
    )


def generate_offspring_theta_nsga2(
    *,
    generator: SpectralGenerator,
    rng: random.Random,
    population: list[EvalRecord],
    ranks: list[int],
    crowd: list[float],
    seed_theta_pairs: Sequence[tuple[np.ndarray, str]],
    hit_theta_anchors: Sequence[np.ndarray],
    generated_hit_theta_anchors: Sequence[np.ndarray],
    archive_smiles: set[str],
    this_batch: int,
    gen: int,
    max_generations: int,
    tournament_k: int,
    parent_pool_fraction: float,
    immigrant_fraction: float,
    mutation_steps: int,
    mutation_step_cap: int,
    crossover_probability: float,
    docking_idx: int | None = None,
    evaluated: int = 0,
    budget: int = 0,
) -> list[SpectralIndividual]:
    theta_ids = [i for i, rec in enumerate(population) if rec.theta is not None and rec.smiles]
    order = sorted(theta_ids, key=lambda i: (ranks[i], -crowd[i], -population[i].scalar))
    pool_size = max(2, int(round(len(order) * parent_pool_fraction))) if order else 0
    parent_ids = order[: min(pool_size, len(order))]

    docking_parent_ids: list[int] = []
    if docking_idx is not None:
        finite_docking_ids = [i for i in theta_ids if raw_docking_value(population[i], docking_idx) < float("inf")]
        finite_docking_ids.sort(
            key=lambda i: (
                raw_docking_value(population[i], docking_idx),
                ranks[i],
                -crowd[i],
                -population[i].scalar,
            )
        )
        if finite_docking_ids:
            top_fraction = clamp_float(
                float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_TOP_FRACTION", 0.25)),
                0.01,
                1.0,
            )
            top_n = max(1, int(round(len(finite_docking_ids) * top_fraction)))
            docking_parent_ids = finite_docking_ids[:top_n]

    def choose_parent() -> EvalRecord | None:
        if not parent_ids:
            return None
        k_eff = max(1, min(int(tournament_k), len(parent_ids)))
        sampled = rng.sample(parent_ids, k_eff)
        best = sampled[0]
        for idx in sampled[1:]:
            if ranks[idx] < ranks[best]:
                best = idx
                continue
            if ranks[idx] == ranks[best] and crowd[idx] > crowd[best]:
                best = idx
        return population[best]

    def choose_docking_parent() -> EvalRecord | None:
        if not docking_parent_ids:
            return choose_parent()
        k_eff = max(1, min(int(tournament_k), len(docking_parent_ids)))
        sampled = rng.sample(docking_parent_ids, k_eff)
        best = sampled[0]
        for idx in sampled[1:]:
            lhs = (raw_docking_value(population[idx], docking_idx), ranks[idx], -crowd[idx], -population[idx].scalar)
            rhs = (raw_docking_value(population[best], docking_idx), ranks[best], -crowd[best], -population[best].scalar)
            if lhs < rhs:
                best = idx
        return population[best]

    candidate_multiplier = max(1.0, float(getattr(cfg, "SPECTRAL_THETA_CANDIDATE_POOL_MULTIPLIER", 1.0)))
    preselect_by_proxy = bool(getattr(cfg, "SPECTRAL_THETA_PRESELECT_BY_CHEAP_PROXY", False))
    preselect_scaffold_diverse = _cfg_bool(
        "SPECTRAL_THETA_PRESELECT_SCAFFOLD_DIVERSE",
        False,
        ("SATURN_THETA_PRESELECT_SCAFFOLD_DIVERSE", "MOLSCORE_SATURN_THETA_PRESELECT_SCAFFOLD_DIVERSE"),
    )
    preselect_max_per_scaffold = max(
        1,
        _cfg_int(
            "SPECTRAL_THETA_PRESELECT_MAX_PER_SCAFFOLD",
            4,
            ("SATURN_THETA_PRESELECT_MAX_PER_SCAFFOLD", "MOLSCORE_SATURN_THETA_PRESELECT_MAX_PER_SCAFFOLD"),
        ),
    )
    preselect_scaffold_top_keep = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_PRESELECT_SCAFFOLD_TOP_KEEP",
            0,
            ("SATURN_THETA_PRESELECT_SCAFFOLD_TOP_KEEP", "MOLSCORE_SATURN_THETA_PRESELECT_SCAFFOLD_TOP_KEEP"),
        ),
    )
    candidate_batch = int(this_batch)
    if preselect_by_proxy:
        candidate_batch = max(int(this_batch), int(round(int(this_batch) * candidate_multiplier)))

    offspring: list[SpectralIndividual] = []
    seen = set(archive_smiles)
    seen.update(rec.smiles for rec in population if rec.smiles)
    max_attempts = max(1, candidate_batch * int(cfg.OFFSPRING_ATTEMPT_FACTOR))
    immigrant_target = min(candidate_batch, int(round(candidate_batch * immigrant_fraction)))
    child_token_mutation_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION", 0.10)),
        0.0,
        1.0,
    )
    token_mutation_max_edits = max(1, int(getattr(cfg, "SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS", 2)))
    token_insert_prob = clamp_float(float(getattr(cfg, "SPECTRAL_THETA_TOKEN_INSERT_PROB", 0.20)), 0.0, 1.0)
    token_delete_prob = clamp_float(float(getattr(cfg, "SPECTRAL_THETA_TOKEN_DELETE_PROB", 0.05)), 0.0, 1.0)
    token_macro_prob = clamp_float(float(getattr(cfg, "SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB", 0.25)), 0.0, 1.0)
    token_blend = clamp_float(float(getattr(cfg, "SPECTRAL_THETA_TOKEN_MUTATION_BLEND", 0.80)), 0.0, 1.0)
    docking_focus_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_FRACTION", 0.50)),
        0.0,
        0.95,
    )
    docking_focus_steps = max(1, int(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_MUTATION_STEPS", 2)))
    docking_focus_target_sample_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_TARGET_SAMPLE_FRACTION", 0.35)),
        0.0,
        1.0,
    )
    docking_focus_sigma_scale = max(0.0, float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_SIGMA_SCALE", 0.08)))
    docking_focus_param_noise_scale = max(
        0.0,
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_PARAM_NOISE_SCALE", 0.12)),
    )
    docking_focus_row_reset_scale = max(
        0.0,
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_ROW_RESET_SCALE", 0.0)),
    )
    docking_focus_token_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_TOKEN_MUTATION_FRACTION", 0.15)),
        0.0,
        1.0,
    )
    docking_focus_token_blend = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_TOKEN_BLEND", 0.35)),
        0.0,
        1.0,
    )
    docking_focus_token_macro_prob = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_FOCUS_TOKEN_MACRO_INSERT_PROB", 0.20)),
        0.0,
        1.0,
    )
    target_token_analog_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_FRACTION", 0.0)),
        0.0,
        1.0,
    )
    target_token_analog_max_edits = max(
        1,
        int(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_MAX_EDITS", 1)),
    )
    target_token_analog_insert_prob = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_INSERT_PROB", 0.20)),
        0.0,
        1.0,
    )
    target_token_analog_delete_prob = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_DELETE_PROB", 0.0)),
        0.0,
        1.0,
    )
    target_token_analog_macro_prob = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_MACRO_INSERT_PROB", 0.10)),
        0.0,
        1.0,
    )
    target_token_analog_blend = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_BLEND", 1.0)),
        0.0,
        1.0,
    )
    target_token_analog_expand_macros = bool(
        getattr(cfg, "SPECTRAL_THETA_TARGET_TOKEN_ANALOG_EXPAND_MACROS", True)
    )
    hit_neighborhood_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_FRACTION",
            0.0,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION"),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_min_count = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_MIN_COUNT",
            0,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT"),
        ),
    )
    hit_neighborhood_max_edits = max(
        1,
        _cfg_int(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_MAX_EDITS",
            1,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_MAX_EDITS", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_MAX_EDITS"),
        ),
    )
    hit_neighborhood_insert_prob = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_INSERT_PROB",
            0.05,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_PROB", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_PROB"),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_delete_prob = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_DELETE_PROB",
            0.0,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_DELETE_PROB", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_DELETE_PROB"),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_macro_prob = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB",
            0.0,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB"),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_blend = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_BLEND",
            1.0,
            ("SATURN_THETA_HIT_NEIGHBORHOOD_BLEND", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_BLEND"),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_expand_macros = _cfg_bool(
        "SPECTRAL_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS",
        True,
        ("SATURN_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS", "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS"),
    )
    hit_neighborhood_target_sample_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION",
            1.0,
            (
                "SATURN_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION",
                "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_anchor_bias = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS",
            8.0,
            (
                "SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS",
                "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS",
            ),
        ),
    )
    hit_neighborhood_edge_position_prob = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB",
            0.65,
            (
                "SATURN_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB",
                "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB",
            ),
        ),
        0.0,
        1.0,
    )
    hit_neighborhood_insert_tokens = _cfg_token_list(
        "SPECTRAL_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS",
        "[F],[Cl],[C],[O],[N]",
        (
            "SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS",
            "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS",
        ),
    )
    include_task_target_anchors = _cfg_bool(
        "SPECTRAL_THETA_INCLUDE_TASK_TARGET_ANCHORS",
        False,
        ("SATURN_THETA_INCLUDE_TASK_TARGET_ANCHORS", "MOLSCORE_SATURN_THETA_INCLUDE_TASK_TARGET_ANCHORS"),
    )
    task_target_anchor_limit = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_TASK_TARGET_ANCHOR_LIMIT",
            0,
            ("SATURN_THETA_TASK_TARGET_ANCHOR_LIMIT", "MOLSCORE_SATURN_THETA_TASK_TARGET_ANCHOR_LIMIT"),
        ),
    )
    hit_site_scan_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_SITE_SCAN_FRACTION",
            0.0,
            (
                "SATURN_THETA_HIT_SITE_SCAN_FRACTION",
                "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    hit_site_scan_max_edits = max(
        1,
        _cfg_int(
            "SPECTRAL_THETA_HIT_SITE_SCAN_MAX_EDITS",
            2,
            (
                "SATURN_THETA_HIT_SITE_SCAN_MAX_EDITS",
                "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_MAX_EDITS",
            ),
        ),
    )
    hit_site_scan_blend = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_SITE_SCAN_BLEND",
            1.0,
            (
                "SATURN_THETA_HIT_SITE_SCAN_BLEND",
                "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_BLEND",
            ),
        ),
        0.0,
        1.0,
    )
    hit_site_scan_motif_choice_probability = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
            0.0,
            (
                "SATURN_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
                "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
            ),
        ),
        0.0,
        1.0,
    )
    hit_site_scan_tokens = _cfg_token_list(
        "SPECTRAL_THETA_HIT_SITE_SCAN_TOKENS",
        "[F],[Cl],[Br],[C],[=C],[N],[O],[S],[AMIDE],[BENZAMIDE],[PHENETHYL],[PHENETHYL_AMIDE],[PHENOXY_ETHYL]",
        (
            "SATURN_THETA_HIT_SITE_SCAN_TOKENS",
            "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_TOKENS",
        ),
    )
    hit_microjitter_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_HIT_MICROJITTER_FRACTION",
            1.0,
            (
                "SATURN_THETA_HIT_MICROJITTER_FRACTION",
                "MOLSCORE_SATURN_THETA_HIT_MICROJITTER_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    hit_microjitter_sigma_min = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_HIT_MICROJITTER_SIGMA_MIN",
            0.002,
            (
                "SATURN_THETA_HIT_MICROJITTER_SIGMA_MIN",
                "MOLSCORE_SATURN_THETA_HIT_MICROJITTER_SIGMA_MIN",
            ),
        ),
    )
    hit_microjitter_sigma_max = max(
        hit_microjitter_sigma_min,
        _cfg_float(
            "SPECTRAL_THETA_HIT_MICROJITTER_SIGMA_MAX",
            0.030,
            (
                "SATURN_THETA_HIT_MICROJITTER_SIGMA_MAX",
                "MOLSCORE_SATURN_THETA_HIT_MICROJITTER_SIGMA_MAX",
            ),
        ),
    )
    generated_hit_anchor_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_GENERATED_HIT_ANCHOR_FRACTION",
            0.85,
            (
                "SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION",
                "MOLSCORE_SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    late_generated_hit_anchor_max_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING",
        ),
    )
    late_generated_hit_anchor_max = (
        late_generated_hit_anchor_max_raw
        if np.isfinite(late_generated_hit_anchor_max_raw)
        else None
    )
    late_generated_hit_anchor_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA",
        ),
    )
    late_generated_hit_anchor_max_sa = (
        late_generated_hit_anchor_max_sa_raw
        if np.isfinite(late_generated_hit_anchor_max_sa_raw) and late_generated_hit_anchor_max_sa_raw < 99.0
        else None
    )
    late_generated_hit_anchor_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED",
        ),
    )
    late_generated_hit_anchor_min_qed = (
        late_generated_hit_anchor_min_qed_raw
        if np.isfinite(late_generated_hit_anchor_min_qed_raw) and late_generated_hit_anchor_min_qed_raw > 0.0
        else None
    )
    generated_hit_jitter_sigma_scale = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE",
            0.55,
            (
                "SATURN_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE",
                "MOLSCORE_SATURN_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE",
            ),
        ),
    )
    late_spike_start_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_LATE_DEEP_SPIKE_START_FRACTION",
            0.0,
            (
                "SATURN_THETA_LATE_DEEP_SPIKE_START_FRACTION",
                "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_START_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    late_spike_active = False
    if late_spike_start_fraction > 0.0 and budget > 0:
        progress = float(evaluated) / max(1.0, float(budget))
        if progress >= late_spike_start_fraction:
            late_spike_active = True
            hit_neighborhood_insert_tokens = _cfg_token_list(
                "SPECTRAL_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS",
                ",".join(hit_neighborhood_insert_tokens),
                (
                    "SATURN_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS",
                    "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS",
                ),
            )
            hit_site_scan_tokens = _cfg_token_list(
                "SPECTRAL_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS",
                ",".join(hit_site_scan_tokens),
                (
                    "SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS",
                    "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS",
                ),
            )
            hit_site_scan_fraction = clamp_float(
                _cfg_float(
                    "SPECTRAL_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION",
                    hit_site_scan_fraction,
                    (
                        "SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION",
                        "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION",
                    ),
                ),
                0.0,
                1.0,
            )
            hit_site_scan_motif_choice_probability = clamp_float(
                _cfg_float(
                    "SPECTRAL_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
                    hit_site_scan_motif_choice_probability,
                    (
                        "SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
                        "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY",
                    ),
                ),
                0.0,
                1.0,
            )
            hit_microjitter_fraction = clamp_float(
                _cfg_float(
                    "SPECTRAL_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION",
                    hit_microjitter_fraction,
                    (
                        "SATURN_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION",
                        "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION",
                    ),
                ),
                0.0,
                1.0,
            )
            generated_hit_anchor_fraction = clamp_float(
                _cfg_float(
                    "SPECTRAL_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION",
                    generated_hit_anchor_fraction,
                    (
                        "SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION",
                        "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION",
                    ),
                ),
                0.0,
                1.0,
            )
            if late_generated_hit_anchor_max is not None:
                generated_hit_anchor_max = late_generated_hit_anchor_max
            if late_generated_hit_anchor_max_sa is not None:
                generated_hit_anchor_max_sa = late_generated_hit_anchor_max_sa
            if late_generated_hit_anchor_min_qed is not None:
                generated_hit_anchor_min_qed = late_generated_hit_anchor_min_qed
    hit_neighborhood_bypass_preselect = _cfg_bool(
        "SPECTRAL_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT",
        True,
        (
            "SATURN_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT",
            "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT",
        ),
    )
    priority_preselect_by_proxy = _cfg_bool(
        "SPECTRAL_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY",
        False,
        (
            "SATURN_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY",
            "MOLSCORE_SATURN_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY",
        ),
    )
    priority_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_PRIORITY_MIN_QED",
        0.0,
        ("SATURN_THETA_PRIORITY_MIN_QED", "MOLSCORE_SATURN_THETA_PRIORITY_MIN_QED"),
    )
    priority_min_qed = (
        priority_min_qed_raw if np.isfinite(priority_min_qed_raw) and priority_min_qed_raw > 0.0 else None
    )
    priority_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_PRIORITY_MAX_SA",
        99.0,
        ("SATURN_THETA_PRIORITY_MAX_SA", "MOLSCORE_SATURN_THETA_PRIORITY_MAX_SA"),
    )
    priority_max_sa = (
        priority_max_sa_raw if np.isfinite(priority_max_sa_raw) and priority_max_sa_raw < 99.0 else None
    )
    priority_min_mw_raw = _cfg_float(
        "SPECTRAL_THETA_PRIORITY_MIN_MW",
        0.0,
        ("SATURN_THETA_PRIORITY_MIN_MW", "MOLSCORE_SATURN_THETA_PRIORITY_MIN_MW"),
    )
    priority_min_mw = (
        priority_min_mw_raw if np.isfinite(priority_min_mw_raw) and priority_min_mw_raw > 0.0 else None
    )
    priority_max_mw_raw = _cfg_float(
        "SPECTRAL_THETA_PRIORITY_MAX_MW",
        9999.0,
        ("SATURN_THETA_PRIORITY_MAX_MW", "MOLSCORE_SATURN_THETA_PRIORITY_MAX_MW"),
    )
    priority_max_mw = (
        priority_max_mw_raw if np.isfinite(priority_max_mw_raw) and priority_max_mw_raw < 9999.0 else None
    )
    late_priority_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED",
        ),
    )
    late_priority_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA",
        ),
    )
    late_priority_min_mw_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW",
        ),
    )
    late_priority_max_mw_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW",
        float("nan"),
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW",
        ),
    )
    if late_spike_active:
        if np.isfinite(late_priority_min_qed_raw) and late_priority_min_qed_raw > 0.0:
            priority_min_qed = late_priority_min_qed_raw
        if np.isfinite(late_priority_max_sa_raw) and late_priority_max_sa_raw < 99.0:
            priority_max_sa = late_priority_max_sa_raw
        if np.isfinite(late_priority_min_mw_raw) and late_priority_min_mw_raw > 0.0:
            priority_min_mw = late_priority_min_mw_raw
        if np.isfinite(late_priority_max_mw_raw) and late_priority_max_mw_raw < 9999.0:
            priority_max_mw = late_priority_max_mw_raw
    generated_site_scan_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_SITE_SCAN_MIN_QED",
        0.0,
        (
            "SATURN_THETA_GENERATED_SITE_SCAN_MIN_QED",
            "MOLSCORE_SATURN_THETA_GENERATED_SITE_SCAN_MIN_QED",
        ),
    )
    generated_site_scan_min_qed = (
        generated_site_scan_min_qed_raw
        if np.isfinite(generated_site_scan_min_qed_raw) and generated_site_scan_min_qed_raw > 0.0
        else None
    )
    generated_site_scan_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_SITE_SCAN_MAX_SA",
        99.0,
        (
            "SATURN_THETA_GENERATED_SITE_SCAN_MAX_SA",
            "MOLSCORE_SATURN_THETA_GENERATED_SITE_SCAN_MAX_SA",
        ),
    )
    generated_site_scan_max_sa = (
        generated_site_scan_max_sa_raw
        if np.isfinite(generated_site_scan_max_sa_raw) and generated_site_scan_max_sa_raw < 99.0
        else None
    )
    hit_site_scan_max_attempts_cap = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS",
            0,
            (
                "SATURN_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS",
                "MOLSCORE_SATURN_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS",
            ),
        ),
    )
    hit_microjitter_max_attempts_cap = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_HIT_MICROJITTER_MAX_ATTEMPTS",
            0,
            (
                "SATURN_THETA_HIT_MICROJITTER_MAX_ATTEMPTS",
                "MOLSCORE_SATURN_THETA_HIT_MICROJITTER_MAX_ATTEMPTS",
            ),
        ),
    )
    hit_neighborhood_max_attempts_cap = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS",
            0,
            (
                "SATURN_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS",
                "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS",
            ),
        ),
    )
    hit_pool_use_candidate_batch = _cfg_bool(
        "SPECTRAL_THETA_HIT_POOL_USE_CANDIDATE_BATCH",
        False,
        (
            "SATURN_THETA_HIT_POOL_USE_CANDIDATE_BATCH",
            "MOLSCORE_SATURN_THETA_HIT_POOL_USE_CANDIDATE_BATCH",
        ),
    )
    preselect_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_MIN_QED",
        0.0,
        ("SATURN_THETA_PRESELECT_MIN_QED", "MOLSCORE_SATURN_THETA_PRESELECT_MIN_QED"),
    )
    preselect_min_qed = (
        preselect_min_qed_raw if np.isfinite(preselect_min_qed_raw) and preselect_min_qed_raw > 0.0 else None
    )
    preselect_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_MAX_SA",
        99.0,
        ("SATURN_THETA_PRESELECT_MAX_SA", "MOLSCORE_SATURN_THETA_PRESELECT_MAX_SA"),
    )
    preselect_max_sa = (
        preselect_max_sa_raw if np.isfinite(preselect_max_sa_raw) and preselect_max_sa_raw < 99.0 else None
    )
    preselect_min_mw_raw = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_MIN_MW",
        0.0,
        ("SATURN_THETA_PRESELECT_MIN_MW", "MOLSCORE_SATURN_THETA_PRESELECT_MIN_MW"),
    )
    preselect_min_mw = (
        preselect_min_mw_raw if np.isfinite(preselect_min_mw_raw) and preselect_min_mw_raw > 0.0 else None
    )
    preselect_max_mw_raw = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_MAX_MW",
        9999.0,
        ("SATURN_THETA_PRESELECT_MAX_MW", "MOLSCORE_SATURN_THETA_PRESELECT_MAX_MW"),
    )
    preselect_max_mw = (
        preselect_max_mw_raw if np.isfinite(preselect_max_mw_raw) and preselect_max_mw_raw < 9999.0 else None
    )
    preselect_docking_motif_weight = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT",
            0.0,
            (
                "SATURN_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT",
                "MOLSCORE_SATURN_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT",
            ),
        ),
    )
    preselect_qed_floor = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_QED_FLOOR",
        0.0,
        ("SATURN_THETA_PRESELECT_QED_FLOOR", "MOLSCORE_SATURN_THETA_PRESELECT_QED_FLOOR"),
    )
    preselect_qed_floor_weight = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_PRESELECT_QED_FLOOR_WEIGHT",
            0.0,
            (
                "SATURN_THETA_PRESELECT_QED_FLOOR_WEIGHT",
                "MOLSCORE_SATURN_THETA_PRESELECT_QED_FLOOR_WEIGHT",
            ),
        ),
    )
    preselect_sa_ceiling = _cfg_float(
        "SPECTRAL_THETA_PRESELECT_SA_CEILING",
        99.0,
        ("SATURN_THETA_PRESELECT_SA_CEILING", "MOLSCORE_SATURN_THETA_PRESELECT_SA_CEILING"),
    )
    preselect_sa_ceiling_weight = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_PRESELECT_SA_CEILING_WEIGHT",
            0.0,
            (
                "SATURN_THETA_PRESELECT_SA_CEILING_WEIGHT",
                "MOLSCORE_SATURN_THETA_PRESELECT_SA_CEILING_WEIGHT",
            ),
        ),
    )
    preselect_br_penalty_weight = max(
        0.0,
        _cfg_float(
            "SPECTRAL_THETA_PRESELECT_BR_PENALTY_WEIGHT",
            0.0,
            (
                "SATURN_THETA_PRESELECT_BR_PENALTY_WEIGHT",
                "MOLSCORE_SATURN_THETA_PRESELECT_BR_PENALTY_WEIGHT",
            ),
        ),
    )
    br_pattern = Chem.MolFromSmarts("[Br]") if preselect_br_penalty_weight > 0.0 else None
    docking_motif_patterns: list[tuple[Chem.Mol, float]] = []
    if preselect_docking_motif_weight > 0.0:
        for smarts, weight in (
            ("[CX4](F)(F)F", 0.30),
            ("[c][O][c,C]", 0.22),
            ("[#6]=[#6]=[#7]", 0.22),
            ("[#6]=[#6]=[#8]", 0.16),
            ("[c][Cl,Br,F]", 0.12),
            ("[CX3](=O)[NX3,NX2]", 0.18),
            ("[c][CX4][c]", 0.12),
        ):
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                docking_motif_patterns.append((pattern, float(weight)))

    target_fps = []
    if preselect_by_proxy:
        for target_smiles in list(getattr(generator, "task_target_smiles", []) or []):
            mol = Chem.MolFromSmiles(target_smiles)
            if mol is None:
                continue
            try:
                target_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            except Exception:
                continue

    def cheap_property_values(smiles: str) -> tuple[float, float, float, float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0, 0.0, 9.0, 0.0
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            max_sim = max((float(DataStructs.TanimotoSimilarity(fp, tfp)) for tfp in target_fps), default=0.0)
        except Exception:
            max_sim = 0.0
        try:
            qed = float(QED.qed(mol))
        except Exception:
            qed = 0.0
        if _calculate_sa_score is not None:
            try:
                sa = float(_calculate_sa_score(mol))
            except Exception:
                sa = 9.0
        else:
            sa = 9.0
        try:
            mw = float(Descriptors.MolWt(mol))
        except Exception:
            mw = 0.0
        return max_sim, qed, sa, mw

    def docking_motif_score(smiles: str) -> float:
        if preselect_docking_motif_weight <= 0.0 or not docking_motif_patterns:
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        score = 0.0
        for pattern, weight in docking_motif_patterns:
            try:
                if mol.HasSubstructMatch(pattern):
                    score += weight
            except Exception:
                continue
        return max(0.0, min(1.0, score))

    def cheap_proxy_key(ind: SpectralIndividual) -> tuple[float, float, float, float, float, float, str]:
        mol = Chem.MolFromSmiles(ind.smiles)
        if mol is None:
            return (-1.0e9, 0.0, 0.0, 0.0, -99.0, -1.0e9, ind.smiles)
        max_sim, qed, sa, mw = cheap_property_values(ind.smiles)

        sa_score = max(0.0, min(1.0, (10.0 - sa) / 9.0))
        target_mw = float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_TARGET_MW", 340.0))
        mw_scale = max(1.0, float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_MW_SCALE", 140.0)))
        mw_score = math.exp(-abs(mw - target_mw) / mw_scale)
        motif_score = docking_motif_score(ind.smiles)
        score = (
            float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_SIM_WEIGHT", 3.0)) * max_sim
            + float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_QED_WEIGHT", 0.5)) * qed
            + float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_SA_WEIGHT", 0.8)) * sa_score
            + float(getattr(cfg, "SPECTRAL_THETA_PRESELECT_MW_WEIGHT", 0.2)) * mw_score
            + preselect_docking_motif_weight * motif_score
        )
        if preselect_qed_floor_weight > 0.0 and np.isfinite(preselect_qed_floor) and preselect_qed_floor > 0.0:
            score -= preselect_qed_floor_weight * max(0.0, preselect_qed_floor - qed)
        if preselect_sa_ceiling_weight > 0.0 and np.isfinite(preselect_sa_ceiling) and preselect_sa_ceiling < 99.0:
            score -= preselect_sa_ceiling_weight * max(0.0, sa - preselect_sa_ceiling)
        if br_pattern is not None:
            try:
                if mol.HasSubstructMatch(br_pattern):
                    score -= preselect_br_penalty_weight
            except Exception:
                pass
        return (score, motif_score, max_sim, qed, -sa, -abs(mw - target_mw), ind.smiles)

    def priority_child_passes_filter(ind: SpectralIndividual) -> bool:
        reason = str(ind.decode_reason)
        generated_site_scan_filter = (
            "GENERATED_HIT_SITE_SCAN" in reason
            and (generated_site_scan_min_qed is not None or generated_site_scan_max_sa is not None)
        )
        if (
            priority_min_qed is None
            and priority_max_sa is None
            and priority_min_mw is None
            and priority_max_mw is None
            and not generated_site_scan_filter
        ):
            return True
        _max_sim, qed, sa, mw = cheap_property_values(ind.smiles)
        if generated_site_scan_min_qed is not None and ((not np.isfinite(qed)) or qed < generated_site_scan_min_qed):
            return False
        if generated_site_scan_max_sa is not None and ((not np.isfinite(sa)) or sa > generated_site_scan_max_sa):
            return False
        if priority_min_qed is not None and ((not np.isfinite(qed)) or qed < priority_min_qed):
            return False
        if priority_max_sa is not None and ((not np.isfinite(sa)) or sa > priority_max_sa):
            return False
        if priority_min_mw is not None and ((not np.isfinite(mw)) or mw < priority_min_mw):
            return False
        if priority_max_mw is not None and ((not np.isfinite(mw)) or mw > priority_max_mw):
            return False
        return True

    def candidate_passes_preselect_filter(ind: SpectralIndividual) -> bool:
        if (
            preselect_min_qed is None
            and preselect_max_sa is None
            and preselect_min_mw is None
            and preselect_max_mw is None
        ):
            return True
        _max_sim, qed, sa, mw = cheap_property_values(ind.smiles)
        if preselect_min_qed is not None and ((not np.isfinite(qed)) or qed < preselect_min_qed):
            return False
        if preselect_max_sa is not None and ((not np.isfinite(sa)) or sa > preselect_max_sa):
            return False
        if preselect_min_mw is not None and ((not np.isfinite(mw)) or mw < preselect_min_mw):
            return False
        if preselect_max_mw is not None and ((not np.isfinite(mw)) or mw > preselect_max_mw):
            return False
        return True

    def decode_child(theta: np.ndarray, reason_prefix: str) -> bool:
        if len(offspring) >= candidate_batch:
            return False
        smiles, reason, macro_count = generator.decode_theta(
            theta,
            gen=int(gen),
            seen_smiles=seen,
            novelty_mode="smiles",
        )
        if not smiles or smiles in seen:
            return False
        seen.add(smiles)
        offspring.append(
            SpectralIndividual(
                theta=generator.apply_frequency_mode(theta),
                smiles=smiles,
                decode_reason=f"{reason_prefix}:{reason}",
                macro_count=int(macro_count),
            )
        )
        return True

    priority_smiles: set[str] = set()
    priority_property_rejections = 0

    def decode_priority_child(theta: np.ndarray, reason_prefix: str) -> bool:
        nonlocal priority_property_rejections
        before = len(offspring)
        if late_spike_active and reason_prefix.startswith("THETA_"):
            reason_prefix = "THETA_LATE_" + reason_prefix[len("THETA_") :]
        ok = decode_child(theta, reason_prefix)
        if ok and len(offspring) > before:
            child = offspring[-1]
            if not priority_child_passes_filter(child):
                seen.discard(child.smiles)
                offspring.pop()
                priority_property_rejections += 1
                return False
            priority_smiles.add(child.smiles)
        return ok

    hit_neighborhood_target = 0
    if hit_neighborhood_fraction > 0.0 or hit_neighborhood_min_count > 0:
        hit_target_base = candidate_batch if (preselect_by_proxy and hit_pool_use_candidate_batch) else int(this_batch)
        hit_neighborhood_target = max(
            hit_neighborhood_min_count,
            int(round(int(hit_target_base) * hit_neighborhood_fraction)),
        )
        hit_neighborhood_target = min(candidate_batch, hit_neighborhood_target)

    task_target_anchor_pool: list[np.ndarray] = []
    if include_task_target_anchors:
        task_target_pairs = list(getattr(generator, "task_target_theta_pairs", []) or [])
        if task_target_anchor_limit > 0:
            task_target_pairs = task_target_pairs[:task_target_anchor_limit]
        for theta, _smiles in task_target_pairs:
            task_target_anchor_pool.append(np.asarray(theta, dtype=np.float64).copy())

    anchor_pool = task_target_anchor_pool + list(hit_theta_anchors)
    generated_anchor_pool = list(generated_hit_theta_anchors)

    def sample_anchor(pool: Sequence[np.ndarray]) -> np.ndarray | None:
        if not pool:
            return None
        if hit_neighborhood_anchor_bias > 0.0 and len(pool) > 1:
            denom = max(1, len(pool) - 1)
            weights = [math.exp(-hit_neighborhood_anchor_bias * (idx / denom)) for idx in range(len(pool))]
            return np.asarray(rng.choices(list(pool), weights=weights, k=1)[0], dtype=np.float64).copy()
        return np.asarray(rng.choice(list(pool)), dtype=np.float64).copy()

    def choose_hit_anchor_theta() -> tuple[np.ndarray | None, bool]:
        if generated_anchor_pool and rng.random() < generated_hit_anchor_fraction:
            theta = sample_anchor(generated_anchor_pool)
            if theta is not None:
                return theta, True
        theta = sample_anchor(anchor_pool)
        if theta is not None:
            return theta, False
        theta = sample_anchor(generated_anchor_pool)
        if theta is not None:
            return theta, True
        return None, False

    def jitter_hit_anchor_theta(source_theta: np.ndarray, *, generated_anchor: bool = False) -> np.ndarray:
        if hit_microjitter_sigma_max <= 0.0:
            return generator.apply_frequency_mode(source_theta)
        if hit_microjitter_sigma_min > 0.0 and hit_microjitter_sigma_max > hit_microjitter_sigma_min:
            sigma = math.exp(
                rng.uniform(
                    math.log(hit_microjitter_sigma_min),
                    math.log(hit_microjitter_sigma_max),
                )
            )
        else:
            sigma = hit_microjitter_sigma_max
        if generated_anchor:
            sigma *= generated_hit_jitter_sigma_scale
        noise = generator.rng.normal(0.0, float(sigma), size=np.asarray(source_theta).shape)
        return generator.apply_frequency_mode(np.asarray(source_theta, dtype=np.float64) + noise)

    hit_site_target = min(
        hit_neighborhood_target,
        int(round(hit_neighborhood_target * hit_site_scan_fraction)),
    )
    hit_site_attempts = 0
    hit_site_success = 0
    hit_site_max_attempts = max(1, hit_site_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR) * 8)
    if hit_site_scan_max_attempts_cap > 0:
        hit_site_max_attempts = min(hit_site_max_attempts, hit_site_scan_max_attempts_cap)
    while (
        hit_site_target > 0
        and hit_site_success < hit_site_target
        and len(priority_smiles) < hit_neighborhood_target
        and len(offspring) < candidate_batch
        and hit_site_attempts < hit_site_max_attempts
    ):
        hit_site_attempts += 1
        source_theta, generated_anchor = choose_hit_anchor_theta()
        if source_theta is None:
            break
        theta = generator.mutate_token_site_scan(
            np.asarray(source_theta, dtype=np.float64).copy(),
            max_edits=hit_site_scan_max_edits,
            blend=hit_site_scan_blend,
            expand_macros_before_edit=True,
            edge_position_probability=hit_neighborhood_edge_position_prob,
            motif_choice_probability=hit_site_scan_motif_choice_probability,
            scan_tokens=hit_site_scan_tokens,
        )
        before = len(priority_smiles)
        reason_prefix = "THETA_GENERATED_HIT_SITE_SCAN" if generated_anchor else "THETA_HIT_SITE_SCAN"
        decode_priority_child(theta, reason_prefix)
        if len(priority_smiles) > before:
            hit_site_success += 1

    hit_attempts = 0
    remaining_hit_target = max(0, hit_neighborhood_target - len(priority_smiles))
    hit_micro_target = min(
        remaining_hit_target,
        int(round(hit_neighborhood_target * hit_microjitter_fraction)),
    )
    hit_micro_attempts = 0
    hit_micro_success = 0
    hit_micro_max_attempts = max(1, hit_micro_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR) * 8)
    if hit_microjitter_max_attempts_cap > 0:
        hit_micro_max_attempts = min(hit_micro_max_attempts, hit_microjitter_max_attempts_cap)
    while (
        hit_micro_target > 0
        and hit_micro_success < hit_micro_target
        and len(offspring) < candidate_batch
        and hit_micro_attempts < hit_micro_max_attempts
    ):
        hit_micro_attempts += 1
        source_theta, generated_anchor = choose_hit_anchor_theta()
        if source_theta is None:
            break
        before = len(priority_smiles)
        reason_prefix = "THETA_GENERATED_HIT_MICROJITTER" if generated_anchor else "THETA_HIT_MICROJITTER"
        decode_priority_child(
            jitter_hit_anchor_theta(source_theta, generated_anchor=generated_anchor),
            reason_prefix,
        )
        if len(priority_smiles) > before:
            hit_micro_success += 1

    hit_max_attempts = max(1, hit_neighborhood_target * int(cfg.OFFSPRING_ATTEMPT_FACTOR) * 4)
    if hit_neighborhood_max_attempts_cap > 0:
        hit_max_attempts = min(hit_max_attempts, hit_neighborhood_max_attempts_cap)
    while (
        hit_neighborhood_target > 0
        and len(priority_smiles) < hit_neighborhood_target
        and len(offspring) < candidate_batch
        and hit_attempts < hit_max_attempts
    ):
        hit_attempts += 1
        source_theta = None
        generated_anchor = False
        if (hit_theta_anchors or generated_hit_theta_anchors) and rng.random() < hit_neighborhood_target_sample_fraction:
            source_theta, generated_anchor = choose_hit_anchor_theta()
        if source_theta is None and hit_neighborhood_target_sample_fraction > 0.0:
            source_theta = generator.sample_task_target_theta()
        if source_theta is None:
            parent = choose_docking_parent()
            if parent is not None and parent.theta is not None:
                source_theta = np.asarray(parent.theta, dtype=np.float64).copy()
        if source_theta is None:
            break
        theta = generator.mutate_token_neighborhood(
            np.asarray(source_theta, dtype=np.float64).copy(),
            max_edits=hit_neighborhood_max_edits,
            insert_probability=hit_neighborhood_insert_prob,
            delete_probability=hit_neighborhood_delete_prob,
            macro_insert_probability=hit_neighborhood_macro_prob,
            blend=hit_neighborhood_blend,
            expand_macros_before_edit=hit_neighborhood_expand_macros,
            edge_position_probability=hit_neighborhood_edge_position_prob,
            insert_tokens=hit_neighborhood_insert_tokens,
        )
        reason_prefix = "THETA_GENERATED_HIT_NEIGHBORHOOD" if generated_anchor else "THETA_HIT_NEIGHBORHOOD"
        decode_priority_child(theta, reason_prefix)

    if hit_neighborhood_target > 0 and (len(priority_smiles) == 0 or int(gen) <= 3 or int(gen) % 10 == 0):
        print(
            "[theta-hit-neighborhood] "
            f"gen={int(gen)} target={hit_neighborhood_target} attempts={hit_attempts} "
            f"success={len(priority_smiles)} micro_success={hit_micro_success} "
            f"micro_attempts={hit_micro_attempts} site_success={hit_site_success} "
            f"site_attempts={hit_site_attempts} anchors={len(hit_theta_anchors)} "
            f"site_max_attempts={hit_site_max_attempts} "
            f"micro_max_attempts={hit_micro_max_attempts} "
            f"hit_max_attempts={hit_max_attempts} "
            f"task_target_anchors={len(task_target_anchor_pool)} "
            f"generated_anchors={len(generated_hit_theta_anchors)} "
            f"priority_property_rejections={priority_property_rejections} "
            f"generated_anchor_fraction={generated_hit_anchor_fraction:.3g} "
            f"hit_pool_candidate_batch={int(hit_pool_use_candidate_batch)} "
            f"anchor_bias={hit_neighborhood_anchor_bias:.3g} "
            f"task_target_thetas={len(getattr(generator, 'task_target_theta_pairs', []) or [])}",
            file=sys.stderr,
            flush=True,
        )

    attempts = 0
    while len(offspring) < immigrant_target and attempts < max_attempts:
        attempts += 1
        if seed_theta_pairs and rng.random() < 0.85:
            base_theta, _seed_smiles = rng.choice(list(seed_theta_pairs))
            theta = np.asarray(base_theta, dtype=np.float64).copy()
        else:
            theta = generator.random_theta()
        theta = generator.mutate_repeated(
            theta,
            gen=int(gen),
            generations=max(1, int(max_generations) if int(max_generations) > 0 else int(gen) + 2),
            depth=min(mutation_step_cap, mutation_steps + 1),
        )
        if rng.random() < child_token_mutation_fraction:
            theta = generator.mutate_token_neighborhood(
                theta,
                max_edits=token_mutation_max_edits,
                insert_probability=token_insert_prob,
                delete_probability=token_delete_prob,
                macro_insert_probability=token_macro_prob,
                blend=token_blend,
            )
        decode_child(theta, "THETA_IMMIGRANT")

    focus_target = int(round(candidate_batch * docking_focus_fraction))
    focus_limit = min(candidate_batch, len(offspring) + max(0, focus_target))
    focus_attempts = 0
    while docking_parent_ids and len(offspring) < focus_limit and focus_attempts < max_attempts:
        focus_attempts += 1
        parent = choose_docking_parent()
        target_theta = None
        using_target_theta = False
        if docking_focus_target_sample_fraction > 0.0 and rng.random() < docking_focus_target_sample_fraction:
            target_theta = generator.sample_task_target_theta()
        if target_theta is not None:
            theta = np.asarray(target_theta, dtype=np.float64).copy()
            using_target_theta = True
        elif parent is not None and parent.theta is not None:
            theta = np.asarray(parent.theta, dtype=np.float64).copy()
        else:
            break
        if using_target_theta and rng.random() < target_token_analog_fraction:
            theta = generator.mutate_token_neighborhood(
                theta,
                max_edits=target_token_analog_max_edits,
                insert_probability=target_token_analog_insert_prob,
                delete_probability=target_token_analog_delete_prob,
                macro_insert_probability=target_token_analog_macro_prob,
                blend=target_token_analog_blend,
                expand_macros_before_edit=target_token_analog_expand_macros,
            )
            decode_child(theta, "THETA_TARGET_TOKEN_ANALOG")
            continue
        theta = generator.mutate_repeated(
            theta,
            gen=int(gen),
            generations=max(1, int(max_generations) if int(max_generations) > 0 else int(gen) + 2),
            depth=min(mutation_step_cap, docking_focus_steps),
            sigma_scale=docking_focus_sigma_scale,
            param_noise_scale=docking_focus_param_noise_scale,
            row_reset_scale=docking_focus_row_reset_scale,
        )
        if rng.random() < docking_focus_token_fraction:
            theta = generator.mutate_token_neighborhood(
                theta,
                max_edits=1,
                insert_probability=0.10,
                delete_probability=0.0,
                macro_insert_probability=docking_focus_token_macro_prob,
                blend=docking_focus_token_blend,
            )
        decode_child(theta, "THETA_DOCKING_MICRO")

    attempts = 0
    while len(offspring) < candidate_batch and attempts < max_attempts:
        attempts += 1
        left = choose_parent()
        if left is None or left.theta is None:
            theta = generator.random_theta()
        else:
            theta = np.asarray(left.theta, dtype=np.float64).copy()
            right = choose_parent()
            if (
                right is not None
                and right.theta is not None
                and right.smiles != left.smiles
                and rng.random() < crossover_probability
            ):
                if rng.random() < float(getattr(cfg, "SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION", 0.25)):
                    theta = generator.blend_crossover_theta(left.theta, right.theta)
                else:
                    theta = generator.crossover_theta(left.theta, right.theta)
        theta = generator.mutate_repeated(
            theta,
            gen=int(gen),
            generations=max(1, int(max_generations) if int(max_generations) > 0 else int(gen) + 2),
            depth=mutation_steps,
        )
        if rng.random() < child_token_mutation_fraction:
            theta = generator.mutate_token_neighborhood(
                theta,
                max_edits=token_mutation_max_edits,
                insert_probability=token_insert_prob,
                delete_probability=token_delete_prob,
                macro_insert_probability=token_macro_prob,
                blend=token_blend,
            )
        decode_child(theta, "THETA_CHILD")

    rescue_attempts = 0
    while len(offspring) < candidate_batch and rescue_attempts < max(8, candidate_batch * 8):
        rescue_attempts += 1
        theta = generator.random_theta()
        decode_child(theta, "THETA_RANDOM_RESCUE")

    if preselect_by_proxy and len(offspring) > int(this_batch):
        selected: list[SpectralIndividual] = []
        selected_smiles: set[str] = set()
        selected_scaffolds: dict[str, int] = {}

        def add_preselected(ind: SpectralIndividual) -> None:
            selected_smiles.add(ind.smiles)
            if preselect_scaffold_diverse:
                scaffold = scaffold_for_smiles(ind.smiles) or ind.smiles
                selected_scaffolds[scaffold] = selected_scaffolds.get(scaffold, 0) + 1
            selected.append(ind)

        def choose_scaffold_diverse(candidates: Sequence[SpectralIndividual], slots: int) -> list[SpectralIndividual]:
            if slots <= 0:
                return []
            chosen: list[SpectralIndividual] = []
            deferred: list[SpectralIndividual] = []
            scaffold_counts = dict(selected_scaffolds)
            top_keep = min(max(0, preselect_scaffold_top_keep), int(this_batch))
            for ind in candidates:
                if len(chosen) >= slots:
                    break
                if ind.smiles in selected_smiles:
                    continue
                if preselect_scaffold_diverse and len(selected) + len(chosen) >= top_keep:
                    scaffold = scaffold_for_smiles(ind.smiles) or ind.smiles
                    if scaffold_counts.get(scaffold, 0) >= preselect_max_per_scaffold:
                        deferred.append(ind)
                        continue
                chosen.append(ind)
                if preselect_scaffold_diverse:
                    scaffold = scaffold_for_smiles(ind.smiles) or ind.smiles
                    scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            if len(chosen) < slots:
                for ind in deferred:
                    if len(chosen) >= slots:
                        break
                    if ind.smiles in selected_smiles or any(prev.smiles == ind.smiles for prev in chosen):
                        continue
                    chosen.append(ind)
                    if preselect_scaffold_diverse:
                        scaffold = scaffold_for_smiles(ind.smiles) or ind.smiles
                        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
            return chosen

        if hit_neighborhood_bypass_preselect and priority_smiles:
            priority = [ind for ind in offspring if ind.smiles in priority_smiles]
            if priority_preselect_by_proxy:
                priority = sorted(priority, key=cheap_proxy_key, reverse=True)
            for ind in choose_scaffold_diverse(priority, int(this_batch)):
                if ind.smiles in selected_smiles:
                    continue
                add_preselected(ind)

        remaining_slots = max(0, int(this_batch) - len(selected))
        ranked_all = sorted(
            (ind for ind in offspring if ind.smiles not in selected_smiles),
            key=cheap_proxy_key,
            reverse=True,
        )
        ranked = [ind for ind in ranked_all if candidate_passes_preselect_filter(ind)]
        if len(ranked) < remaining_slots:
            already = {ind.smiles for ind in ranked}
            ranked.extend(ind for ind in ranked_all if ind.smiles not in already)
        for ind in choose_scaffold_diverse(ranked, remaining_slots):
            if ind.smiles in selected_smiles:
                continue
            add_preselected(ind)
        for ind in selected:
            if ":CHEAP_PRESELECT" not in ind.decode_reason:
                ind.decode_reason = f"{ind.decode_reason}:CHEAP_PRESELECT"
        return selected

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
    init_batch = init_pool[: min(pop_size, budget)]
    population = evaluator.evaluate(init_batch)
    archive = unique_best_by_smiles(population)
    evaluated = len(init_batch)
    generations = 0
    progress_every = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_PROGRESS_EVERY",
            0,
            ("SATURN_THETA_PROGRESS_EVERY", "MOLSCORE_SATURN_THETA_PROGRESS_EVERY"),
        ),
    )
    progress_path = snapshot_dir / "theta_progress.tsv" if snapshot_dir is not None and progress_every > 0 else None
    write_theta_progress_tsv(
        progress_path,
        seed=seed,
        generation=generations,
        evaluated=evaluated,
        archive=archive,
        population=population,
        docking_idx=docking_idx,
        elapsed_seconds=time.time() - t0,
        event="initial",
    )

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

        if progress_path is not None and (generations % progress_every == 0 or evaluated >= budget):
            write_theta_progress_tsv(
                progress_path,
                seed=seed,
                generation=generations,
                evaluated=evaluated,
                archive=archive,
                population=population,
                docking_idx=docking_idx,
                elapsed_seconds=time.time() - t0,
                event="loop",
            )

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
    init_batch = init_pool[: min(pop_size, budget)]
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


def run_theta_nsga2_strategy(
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
    spectral_seed: int,
    spectral_settings: SpectralSettings,
    snapshot_dir: Path | None = None,
    strategy_name: str = "theta_nsga2_multiobjective",
    seed: int = 0,
) -> tuple[list[EvalRecord], dict[str, EvalRecord], int, int, float]:
    t0 = time.time()
    generator = SpectralGenerator(spectral_settings, seed=int(spectral_seed))
    generator.adapt_vocabulary_from_seed_pool(seed_pool)
    docking_idx = find_docking_index(evaluator.component_names, "")
    sa_idx = find_sa_index(evaluator.component_names)
    qed_idx = find_qed_index(evaluator.component_names)

    mutation_step_cap = max(
        int(cfg.OFFSPRING_MUTATION_MAX_STEPS),
        int(getattr(cfg, "LOCAL_EVO_MUTATION_STEP_CAP", cfg.OFFSPRING_MUTATION_MAX_STEPS)),
    )
    immigrant_fraction = clamp_float(immigrant_fraction, 0.0, 0.9)
    parent_pool_fraction = clamp_float(parent_pool_fraction, 0.1, 1.0)
    crossover_probability = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_CROSSOVER_PROBABILITY", 0.45)),
        0.0,
        1.0,
    )
    docking_elite_fraction = clamp_float(
        float(getattr(cfg, "SPECTRAL_THETA_DOCKING_ELITE_FRACTION", 0.15)),
        0.0,
        0.50,
    )
    generated_docking_elite_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_GENERATED_DOCKING_ELITE_FRACTION",
            0.25,
            (
                "SATURN_THETA_GENERATED_DOCKING_ELITE_FRACTION",
                "MOLSCORE_SATURN_THETA_GENERATED_DOCKING_ELITE_FRACTION",
            ),
        ),
        0.0,
        0.50,
    )
    generated_docking_elite_max_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING",
        -6.5,
        (
            "SATURN_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING",
            "MOLSCORE_SATURN_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING",
        ),
    )
    generated_docking_elite_max = (
        generated_docking_elite_max_raw if np.isfinite(generated_docking_elite_max_raw) else None
    )
    generated_hit_anchor_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_GENERATED_HIT_ANCHOR_FRACTION",
            0.85,
            (
                "SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION",
                "MOLSCORE_SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    generated_hit_anchor_max_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING",
        -8.0,
        (
            "SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING",
            "MOLSCORE_SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING",
        ),
    )
    generated_hit_anchor_max = generated_hit_anchor_max_raw if np.isfinite(generated_hit_anchor_max_raw) else None
    generated_hit_anchor_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_HIT_ANCHOR_MAX_SA",
        99.0,
        (
            "SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_SA",
            "MOLSCORE_SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_SA",
        ),
    )
    generated_hit_anchor_max_sa = (
        generated_hit_anchor_max_sa_raw
        if np.isfinite(generated_hit_anchor_max_sa_raw) and generated_hit_anchor_max_sa_raw < 99.0
        else None
    )
    generated_hit_anchor_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_GENERATED_HIT_ANCHOR_MIN_QED",
        0.0,
        (
            "SATURN_THETA_GENERATED_HIT_ANCHOR_MIN_QED",
            "MOLSCORE_SATURN_THETA_GENERATED_HIT_ANCHOR_MIN_QED",
        ),
    )
    generated_hit_anchor_min_qed = (
        generated_hit_anchor_min_qed_raw
        if np.isfinite(generated_hit_anchor_min_qed_raw) and generated_hit_anchor_min_qed_raw > 0.0
        else None
    )
    scaffold_diverse_anchors = _cfg_bool(
        "SPECTRAL_THETA_SCAFFOLD_DIVERSE_ANCHORS",
        False,
        ("SATURN_THETA_SCAFFOLD_DIVERSE_ANCHORS", "MOLSCORE_SATURN_THETA_SCAFFOLD_DIVERSE_ANCHORS"),
    )
    anchor_max_per_scaffold = max(
        1,
        _cfg_int(
            "SPECTRAL_THETA_ANCHOR_MAX_PER_SCAFFOLD",
            1,
            ("SATURN_THETA_ANCHOR_MAX_PER_SCAFFOLD", "MOLSCORE_SATURN_THETA_ANCHOR_MAX_PER_SCAFFOLD"),
        ),
    )
    scaffold_diverse_top_keep_requested = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_SCAFFOLD_DIVERSE_TOP_KEEP",
            0,
            (
                "SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP",
                "MOLSCORE_SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP",
            ),
        ),
    )
    scaffold_diverse_top_keep_fraction = clamp_float(
        _cfg_float(
            "SPECTRAL_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION",
            0.0,
            (
                "SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION",
                "MOLSCORE_SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION",
            ),
        ),
        0.0,
        1.0,
    )
    preselect_scaffold_diverse_cfg = _cfg_bool(
        "SPECTRAL_THETA_PRESELECT_SCAFFOLD_DIVERSE",
        False,
        ("SATURN_THETA_PRESELECT_SCAFFOLD_DIVERSE", "MOLSCORE_SATURN_THETA_PRESELECT_SCAFFOLD_DIVERSE"),
    )
    preselect_max_per_scaffold_cfg = max(
        1,
        _cfg_int(
            "SPECTRAL_THETA_PRESELECT_MAX_PER_SCAFFOLD",
            4,
            ("SATURN_THETA_PRESELECT_MAX_PER_SCAFFOLD", "MOLSCORE_SATURN_THETA_PRESELECT_MAX_PER_SCAFFOLD"),
        ),
    )
    preselect_scaffold_top_keep_cfg = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_PRESELECT_SCAFFOLD_TOP_KEEP",
            0,
            ("SATURN_THETA_PRESELECT_SCAFFOLD_TOP_KEEP", "MOLSCORE_SATURN_THETA_PRESELECT_SCAFFOLD_TOP_KEEP"),
        ),
    )
    hit_target_limit = max(0, int(getattr(cfg, "SPECTRAL_THETA_HIT_TARGET_COUNT", 48)))
    hit_target_max_sa_raw = float(getattr(cfg, "SPECTRAL_THETA_HIT_TARGET_MAX_SA", 99.0))
    hit_target_max_sa = hit_target_max_sa_raw if np.isfinite(hit_target_max_sa_raw) and hit_target_max_sa_raw < 99.0 else None
    hit_target_min_qed_raw = float(getattr(cfg, "SPECTRAL_THETA_HIT_TARGET_MIN_QED", 0.0))
    hit_target_min_qed = (
        hit_target_min_qed_raw
        if np.isfinite(hit_target_min_qed_raw) and hit_target_min_qed_raw > 0.0
        else None
    )
    refresh_hit_targets_every = max(0, int(getattr(cfg, "SPECTRAL_THETA_REFRESH_HIT_TARGETS_EVERY", 0)))
    late_archive_max_docking_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING",
        0.0,
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING",
        ),
    )
    late_archive_max_docking = (
        late_archive_max_docking_raw
        if np.isfinite(late_archive_max_docking_raw) and late_archive_max_docking_raw < 0.0
        else None
    )
    late_archive_min_qed_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED",
        0.0,
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED",
        ),
    )
    late_archive_min_qed = (
        late_archive_min_qed_raw
        if np.isfinite(late_archive_min_qed_raw) and late_archive_min_qed_raw > 0.0
        else None
    )
    late_archive_max_sa_raw = _cfg_float(
        "SPECTRAL_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA",
        99.0,
        (
            "SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA",
            "MOLSCORE_SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA",
        ),
    )
    late_archive_max_sa = (
        late_archive_max_sa_raw
        if np.isfinite(late_archive_max_sa_raw) and late_archive_max_sa_raw < 99.0
        else None
    )
    print(
        "[theta-config] "
        f"cfg_file={getattr(cfg, '__file__', '<unknown>')} "
        f"hit_neighborhood_fraction={_cfg_float('SPECTRAL_THETA_HIT_NEIGHBORHOOD_FRACTION', 0.0, ('SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION', 'MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION'))} "
        f"hit_neighborhood_min_count={_cfg_int('SPECTRAL_THETA_HIT_NEIGHBORHOOD_MIN_COUNT', 0, ('SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT', 'MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT'))} "
        f"hit_target_count={hit_target_limit} "
        f"generated_docking_elite_fraction={generated_docking_elite_fraction} "
        f"generated_hit_anchor_fraction={generated_hit_anchor_fraction} "
        f"generated_hit_anchor_max={generated_hit_anchor_max} "
        f"generated_hit_anchor_max_sa={generated_hit_anchor_max_sa} "
        f"generated_hit_anchor_min_qed={generated_hit_anchor_min_qed} "
        f"scaffold_diverse_anchors={int(scaffold_diverse_anchors)} "
        f"anchor_max_per_scaffold={anchor_max_per_scaffold} "
        f"scaffold_diverse_top_keep={scaffold_diverse_top_keep_requested} "
        f"scaffold_diverse_top_keep_fraction={scaffold_diverse_top_keep_fraction} "
        f"preselect_scaffold_diverse={int(preselect_scaffold_diverse_cfg)} "
        f"preselect_max_per_scaffold={preselect_max_per_scaffold_cfg} "
        f"preselect_scaffold_top_keep={preselect_scaffold_top_keep_cfg} "
        f"hit_target_max_sa={hit_target_max_sa} "
        f"hit_target_min_qed={hit_target_min_qed} "
        f"late_archive_max_docking={late_archive_max_docking} "
        f"late_archive_min_qed={late_archive_min_qed} "
        f"late_archive_max_sa={late_archive_max_sa}",
        file=sys.stderr,
        flush=True,
    )

    def keep_late_spike_archive_record(rec: EvalRecord) -> bool:
        if not str(rec.decode_reason).startswith("THETA_LATE_"):
            return True
        if late_archive_max_docking is not None:
            if raw_docking_value(rec, docking_idx) >= late_archive_max_docking:
                return False
        if late_archive_min_qed is not None and qed_idx is not None and 0 <= qed_idx < len(rec.raws):
            qed_value = float(rec.raws[qed_idx])
            if not np.isfinite(qed_value) or qed_value < late_archive_min_qed:
                return False
        if late_archive_max_sa is not None and sa_idx is not None and 0 <= sa_idx < len(rec.raws):
            sa_value = float(rec.raws[sa_idx])
            if not np.isfinite(sa_value) or sa_value > late_archive_max_sa:
                return False
        return True

    seed_theta_pairs = generator.encode_seed_thetas(seed_pool, limit=max(len(seed_pool), pop_size))
    init_individuals = generator.build_initial_population(seed_pool, pop_size)
    init_batch = init_individuals[: min(pop_size, budget)]
    population = attach_theta_records(evaluator, init_batch)
    last_hit_targets: tuple[str, ...] = ()
    if bool(getattr(cfg, "SPECTRAL_THETA_ADAPT_HIT_TARGET_MACROS", True)):
        hit_targets = best_docking_target_smiles(
            population,
            docking_idx=docking_idx,
            sa_idx=sa_idx,
            max_sa=hit_target_max_sa,
            min_qed=hit_target_min_qed,
            limit=hit_target_limit,
        )
        if not hit_targets and hit_target_max_sa is not None:
            hit_targets = best_docking_target_smiles(
                population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                min_qed=hit_target_min_qed,
                limit=hit_target_limit,
            )
        if not hit_targets and hit_target_min_qed is not None:
            hit_targets = best_docking_target_smiles(
                population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                min_qed=None,
                limit=hit_target_limit,
            )
        if hit_targets:
            last_hit_targets = tuple(hit_targets)
            generator.adapt_vocabulary_from_task_targets(hit_targets)
            population = reencode_theta_records(generator, population)
            seed_theta_pairs = generator.encode_seed_thetas(seed_pool, limit=max(len(seed_pool), pop_size))
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_molecules_csv(
            snapshot_dir / "initial_population.csv",
            strategy=f"{strategy_name}_initial_population",
            budget=int(budget),
            seed=int(seed),
            component_names=evaluator.component_names,
            molecules=population,
        )
    archive = unique_best_by_smiles(population)
    evaluated = len(init_batch)
    generations = 0
    progress_every = max(
        0,
        _cfg_int(
            "SPECTRAL_THETA_PROGRESS_EVERY",
            0,
            ("SATURN_THETA_PROGRESS_EVERY", "MOLSCORE_SATURN_THETA_PROGRESS_EVERY"),
        ),
    )
    progress_path = snapshot_dir / "theta_progress.tsv" if snapshot_dir is not None and progress_every > 0 else None
    write_theta_progress_tsv(
        progress_path,
        seed=seed,
        generation=generations,
        evaluated=evaluated,
        archive=archive,
        population=population,
        docking_idx=docking_idx,
        elapsed_seconds=time.time() - t0,
        event="initial",
    )

    best_so_far = max((r.scalar for r in population), default=float("-inf"))
    stagnation_count = 0
    late_archive_filtered = 0

    while evaluated < budget:
        if max_generations > 0 and generations >= max_generations:
            break
        generations += 1
        remaining = budget - evaluated
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        population_objectives = evaluator.nsga_objective_matrix(population)
        population, ranks, crowd = nsga2_select(
            population,
            len(population),
            objective_matrix=population_objectives,
        )
        population = inject_docking_elites(
            population,
            archive,
            docking_idx=docking_idx,
            pop_size=pop_size,
            elite_fraction=docking_elite_fraction,
            generated_elite_fraction=generated_docking_elite_fraction,
            generated_max_docking=generated_docking_elite_max,
        )
        population_objectives = evaluator.nsga_objective_matrix(population)
        population, ranks, crowd = nsga2_select(
            population,
            len(population),
            objective_matrix=population_objectives,
        )
        hit_anchor_limit = max(
            1,
            _cfg_int(
                "SPECTRAL_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT",
                hit_target_limit if hit_target_limit > 0 else 48,
                (
                    "SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT",
                    "MOLSCORE_SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT",
                ),
            ),
        )
        scaffold_diverse_top_keep = scaffold_diverse_top_keep_requested
        if scaffold_diverse_top_keep <= 0 and scaffold_diverse_top_keep_fraction > 0.0:
            scaffold_diverse_top_keep = int(round(hit_anchor_limit * scaffold_diverse_top_keep_fraction))
        scaffold_diverse_top_keep = min(max(0, int(scaffold_diverse_top_keep)), hit_anchor_limit)
        hit_theta_anchors = best_docking_theta_anchors(
            list(archive.values()) + population,
            docking_idx=docking_idx,
            sa_idx=sa_idx,
            max_sa=hit_target_max_sa,
            min_qed=hit_target_min_qed,
            limit=hit_anchor_limit,
            scaffold_diverse=scaffold_diverse_anchors,
            max_per_scaffold=anchor_max_per_scaffold,
            scaffold_diverse_top_keep=scaffold_diverse_top_keep,
        )
        if not hit_theta_anchors and hit_target_max_sa is not None:
            hit_theta_anchors = best_docking_theta_anchors(
                list(archive.values()) + population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                min_qed=hit_target_min_qed,
                limit=hit_anchor_limit,
                scaffold_diverse=scaffold_diverse_anchors,
                max_per_scaffold=anchor_max_per_scaffold,
                scaffold_diverse_top_keep=scaffold_diverse_top_keep,
            )
        if not hit_theta_anchors and hit_target_min_qed is not None:
            hit_theta_anchors = best_docking_theta_anchors(
                list(archive.values()) + population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                min_qed=None,
                limit=hit_anchor_limit,
                scaffold_diverse=scaffold_diverse_anchors,
                max_per_scaffold=anchor_max_per_scaffold,
                scaffold_diverse_top_keep=scaffold_diverse_top_keep,
            )
        generated_hit_theta_anchors = best_generated_docking_theta_anchors(
            list(archive.values()) + population,
            docking_idx=docking_idx,
            sa_idx=sa_idx,
            max_sa=generated_hit_anchor_max_sa,
            min_qed=generated_hit_anchor_min_qed,
            max_docking=generated_hit_anchor_max,
            limit=hit_anchor_limit,
            scaffold_diverse=scaffold_diverse_anchors,
            max_per_scaffold=anchor_max_per_scaffold,
            scaffold_diverse_top_keep=scaffold_diverse_top_keep,
        )
        if not generated_hit_theta_anchors and generated_hit_anchor_max is not None:
            generated_hit_theta_anchors = best_generated_docking_theta_anchors(
                list(archive.values()) + population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=generated_hit_anchor_max_sa,
                min_qed=generated_hit_anchor_min_qed,
                max_docking=None,
                limit=hit_anchor_limit,
                scaffold_diverse=scaffold_diverse_anchors,
                max_per_scaffold=anchor_max_per_scaffold,
                scaffold_diverse_top_keep=scaffold_diverse_top_keep,
            )
        if not generated_hit_theta_anchors and generated_hit_anchor_max_sa is not None:
            generated_hit_theta_anchors = best_generated_docking_theta_anchors(
                list(archive.values()) + population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                min_qed=generated_hit_anchor_min_qed,
                max_docking=generated_hit_anchor_max,
                limit=hit_anchor_limit,
                scaffold_diverse=scaffold_diverse_anchors,
                max_per_scaffold=anchor_max_per_scaffold,
                scaffold_diverse_top_keep=scaffold_diverse_top_keep,
            )
        if (
            not generated_hit_theta_anchors
            and generated_hit_anchor_max_sa is not None
            and generated_hit_anchor_max is not None
        ):
            generated_hit_theta_anchors = best_generated_docking_theta_anchors(
                list(archive.values()) + population,
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=None,
                max_docking=None,
                limit=hit_anchor_limit,
                scaffold_diverse=scaffold_diverse_anchors,
                max_per_scaffold=anchor_max_per_scaffold,
                scaffold_diverse_top_keep=scaffold_diverse_top_keep,
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
        if stagnation_patience > 0 and stagnation_count >= (3 * stagnation_patience):
            effective_immigrant_fraction = max(float(effective_immigrant_fraction), 0.40)

        offspring_individuals = generate_offspring_theta_nsga2(
            generator=generator,
            rng=rng,
            population=population,
            ranks=ranks,
            crowd=crowd,
            seed_theta_pairs=seed_theta_pairs,
            hit_theta_anchors=hit_theta_anchors,
            generated_hit_theta_anchors=generated_hit_theta_anchors,
            archive_smiles=set(archive.keys()),
            this_batch=this_batch,
            gen=generations,
            max_generations=max_generations,
            tournament_k=eff_tournament_k,
            parent_pool_fraction=effective_parent_pool_fraction,
            immigrant_fraction=effective_immigrant_fraction,
            mutation_steps=mutation_steps,
            mutation_step_cap=mutation_step_cap,
            crossover_probability=crossover_probability,
            docking_idx=docking_idx,
            evaluated=evaluated,
            budget=budget,
        )
        offspring = attach_theta_records(evaluator, offspring_individuals)
        evaluated += len(offspring_individuals)

        for rec in offspring:
            if not keep_late_spike_archive_record(rec):
                late_archive_filtered += 1
                continue
            prev = archive.get(rec.smiles)
            if prev is None or rec.scalar > prev.scalar:
                archive[rec.smiles] = rec

        combined = unique_best_by_smiles(population + offspring)
        combined_records = list(combined.values())
        combined_objectives = evaluator.nsga_objective_matrix(combined_records)
        population, _, _ = nsga2_select(
            combined_records,
            pop_size,
            objective_matrix=combined_objectives,
        )
        population = inject_docking_elites(
            population,
            archive,
            docking_idx=docking_idx,
            pop_size=pop_size,
            elite_fraction=docking_elite_fraction,
            generated_elite_fraction=generated_docking_elite_fraction,
            generated_max_docking=generated_docking_elite_max,
        )

        if (
            refresh_hit_targets_every > 0
            and generations % refresh_hit_targets_every == 0
            and bool(getattr(cfg, "SPECTRAL_THETA_ADAPT_HIT_TARGET_MACROS", True))
        ):
            hit_targets = best_docking_target_smiles(
                list(archive.values()),
                docking_idx=docking_idx,
                sa_idx=sa_idx,
                max_sa=hit_target_max_sa,
                min_qed=hit_target_min_qed,
                limit=hit_target_limit,
            )
            if not hit_targets and hit_target_max_sa is not None:
                hit_targets = best_docking_target_smiles(
                    list(archive.values()),
                    docking_idx=docking_idx,
                    sa_idx=sa_idx,
                    max_sa=None,
                    min_qed=hit_target_min_qed,
                    limit=hit_target_limit,
                )
            if not hit_targets and hit_target_min_qed is not None:
                hit_targets = best_docking_target_smiles(
                    list(archive.values()),
                    docking_idx=docking_idx,
                    sa_idx=sa_idx,
                    max_sa=None,
                    min_qed=None,
                    limit=hit_target_limit,
                )
            hit_target_key = tuple(hit_targets)
            if hit_targets and hit_target_key != last_hit_targets:
                last_hit_targets = hit_target_key
                generator.adapt_vocabulary_from_task_targets(hit_targets)
                population = reencode_theta_records(generator, population)
                archive = unique_best_by_smiles(reencode_theta_records(generator, list(archive.values())))
                seed_theta_pairs = generator.encode_seed_thetas(seed_pool, limit=max(len(seed_pool), pop_size))

        current_best = max((r.scalar for r in archive.values()), default=float("-inf"))
        if current_best > best_so_far + 1e-9:
            best_so_far = current_best
            stagnation_count = 0
        else:
            stagnation_count += 1

        if progress_path is not None and (generations % progress_every == 0 or evaluated >= budget):
            write_theta_progress_tsv(
                progress_path,
                seed=seed,
                generation=generations,
                evaluated=evaluated,
                archive=archive,
                population=population,
                docking_idx=docking_idx,
                elapsed_seconds=time.time() - t0,
                event="loop",
            )

    elapsed = time.time() - t0
    if late_archive_filtered:
        print(
            "[theta-archive-filter] "
            f"filtered_late_spike={late_archive_filtered} "
            f"max_docking={late_archive_max_docking} "
            f"min_qed={late_archive_min_qed} "
            f"max_sa={late_archive_max_sa}",
            file=sys.stderr,
            flush=True,
        )
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        write_molecules_csv(
            snapshot_dir / "final_population.csv",
            strategy=f"{strategy_name}_final_population",
            budget=int(budget),
            seed=int(seed),
            component_names=evaluator.component_names,
            molecules=population,
        )
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
    per_seed_seed_smiles_dir = (
        Path(args.per_seed_seed_smiles_dir).expanduser()
        if str(args.per_seed_seed_smiles_dir).strip()
        else None
    )
    if per_seed_seed_smiles_dir is not None and not per_seed_seed_smiles_dir.is_absolute():
        per_seed_seed_smiles_dir = cfg.resolve_from_repo(str(per_seed_seed_smiles_dir))
    if per_seed_seed_smiles_dir is not None:
        per_seed_seed_smiles_dir = per_seed_seed_smiles_dir.resolve()
        if not per_seed_seed_smiles_dir.exists():
            raise FileNotFoundError(f"Per-seed seed directory not found: {per_seed_seed_smiles_dir}")

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
    run_params["resolved_per_seed_seed_smiles_dir"] = (
        str(per_seed_seed_smiles_dir) if per_seed_seed_smiles_dir is not None else ""
    )
    run_params["resolved_saturn_repo_root"] = str(saturn_repo_root)
    run_params["resolved_template_path"] = str(template_path)
    run_params["resolved_quickvina_binary"] = str(quickvina_binary)
    run_params["resolved_receptor_file"] = str(receptor_file)
    run_params["resolved_reference_ligand_file"] = str(reference_ligand_file)
    run_params["run_id"] = run_id
    (run_root / "run_parameters.json").write_text(json.dumps(run_params, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    spectral_settings = build_spectral_settings_from_args(args)

    for budget in budgets:
        for seed in seeds:
            print(f"[compare] budget={budget} seed={seed} starting", flush=True)
            case_root = run_root / "runs" / f"budget_{int(budget)}" / f"seed_{int(seed)}"
            case_root.mkdir(parents=True, exist_ok=True)

            rng_scalar = random.Random(seed)
            rng_nsga = random.Random(seed)
            seed_smiles_file_for_seed = seed_smiles_file
            if per_seed_seed_smiles_dir is not None:
                candidate = per_seed_seed_smiles_dir / f"seed_{int(seed)}.smi"
                if not candidate.exists():
                    raise FileNotFoundError(
                        f"Expected old-benchmark per-seed seed file missing: {candidate} "
                        f"(required for seed={int(seed)})"
                    )
                seed_smiles_file_for_seed = candidate
            print(f"[compare] seed_pool_file={seed_smiles_file_for_seed}", flush=True)
            seed_pool = load_seed_pool(str(seed_smiles_file_for_seed), int(effective_seed_pool_size), rng_scalar)

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
            # By default this is the manuscript SpectralMol path: survival and parent
            # selection are Pareto-based, while offspring are generated only by
            # crossover/mutation of the Fourier genotype theta.
            if not args.skip_nsga2:
                nsga_strategy = (
                    "theta_nsga2_multiobjective"
                    if str(args.nsga2_genotype).strip().lower() == "theta"
                    else "smiles_nsga2_multiobjective"
                )
                nsga_root = case_root / nsga_strategy
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
                if str(args.nsga2_genotype).strip().lower() == "theta":
                    nsga_mols, nsga_archive, eval_calls, gens, elapsed = run_theta_nsga2_strategy(
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
                        spectral_seed=int(seed) + 1,
                        spectral_settings=spectral_settings,
                        snapshot_dir=nsga_root / "snapshots",
                        strategy_name=nsga_strategy,
                        seed=int(seed),
                    )
                else:
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
                    strategy=nsga_strategy,
                    budget=int(budget),
                    seed=int(seed),
                    component_names=nsga_eval.component_names,
                    molecules=nsga_mols,
                )
                nsga_summary = summarize_strategy(
                    strategy=nsga_strategy,
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
                    f"strategy={nsga_strategy} budget={budget} seed={seed} pct<-9={nsga_summary.pct_lt_minus9:.2f} "
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
