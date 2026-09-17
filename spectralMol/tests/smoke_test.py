#!/usr/bin/env python3
"""
Smoke test for molevoDrugDiscovery.

Checks:
  1. Core package imports (config, vocab, fourier_theta, embedding, smiles_theta_encoder, reports)
  2. Oracle infrastructure imports (Oracle, QED, SAScore, reward aggregator)
  3. GuacaMol benchmark: 1 task, 50-molecule budget, 2 generations
  4. Saturn benchmark: scalar strategy only, QED+SA oracle (no docking), 50-molecule budget

Usage:
    conda run -p /path/to/molevoDrugDiscovery python tests/smoke_test.py

All output is written under tests/smoke_output/ so it does not pollute the repo.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TESTS_DIR / "smoke_output"
PYTHON = sys.executable

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _subprocess_env() -> dict[str, str]:
    """Return an environment dict that ensures subprocesses can import `core`."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(REPO_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _ok(name: str, detail: str = "") -> None:
    _results.append((name, True, detail))
    print(f"  [ PASS ] {name}" + (f"  — {detail}" if detail else ""))


def _fail(name: str, detail: str = "") -> None:
    _results.append((name, False, detail))
    print(f"  [ FAIL ] {name}" + (f"  — {detail}" if detail else ""))


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Core import checks
# ──────────────────────────────────────────────────────────────────────────────

def check_core_imports() -> None:
    _section("1. Core package imports")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    modules = [
        ("core.config",               "cfg"),
        ("core.vocab",                "vocab"),
        ("core.fourier_theta",        "ft"),
        ("core.embedding",            "emb"),
        ("core.smiles_theta_encoder", "enc"),
        ("core.reports",              "rep"),
    ]
    for module_path, alias in modules:
        try:
            import importlib
            importlib.import_module(module_path)
            _ok(f"import {module_path}")
        except Exception as exc:
            _fail(f"import {module_path}", str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Oracle imports
# ──────────────────────────────────────────────────────────────────────────────

def check_oracle_imports() -> None:
    _section("2. Oracle infrastructure imports")
    core_root = REPO_ROOT / "core"
    if str(core_root) not in sys.path:
        sys.path.insert(0, str(core_root))

    checks = [
        ("oracles.oracle",                          "Oracle"),
        ("oracles.dataclass",                       "OracleConfiguration"),
        ("oracles.physchem.qed",                    "QED"),
        ("oracles.synthesizability.sa_score",       "SAScore"),
        ("oracles.reward_aggregator.reward_aggregator", "RewardAggregator"),
        ("diversity_filter.diversity_filter",       "DiversityFilter"),
        ("utils.chemistry_utils",                   "canonicalize_smiles"),
    ]
    for module_path, symbol in checks:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, symbol)
            _ok(f"import {module_path}.{symbol}")
        except Exception as exc:
            _fail(f"import {module_path}.{symbol}", str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Oracle functional test (QED + SA on a handful of SMILES)
# ──────────────────────────────────────────────────────────────────────────────

def check_oracle_functional() -> None:
    _section("3. Oracle functional test (QED + SA, no docking)")
    core_root = REPO_ROOT / "core"
    if str(core_root) not in sys.path:
        sys.path.insert(0, str(core_root))

    try:
        import json
        import numpy as np
        from rdkit import Chem
        from oracles.oracle import Oracle
        from oracles.dataclass import OracleConfiguration

        config_path = TESTS_DIR / "test_oracle_qed_sa.json"
        raw = json.loads(config_path.read_text())["oracle"]

        # Oracle.construct_oracle expects components as raw dicts (unpacked with **),
        # so pass raw["components"] directly rather than pre-building dataclasses.
        oracle_config = OracleConfiguration(
            components=raw["components"],
            budget=raw["budget"],
            allow_oracle_repeats=raw["allow_oracle_repeats"],
            aggregator=raw["aggregator"],
        )
        oracle = Oracle(oracle_config)

        test_smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "invalid!!smiles"]
        mols = [Chem.MolFromSmiles(s) for s in test_smiles]
        valid_mols = np.array([m for m in mols if m is not None])
        # RewardAggregator expects shape (n_components, n_smiles); stack per-component arrays.
        component_rewards = np.stack([comp(valid_mols) for comp in oracle.oracle], axis=0)
        scores = oracle.aggregator(component_rewards, oracle.oracle_weights)
        assert len(scores) == len(valid_mols), "Score count mismatch"
        assert all(0.0 <= float(s) <= 1.0 for s in scores), f"Scores out of range: {scores}"
        _ok("Oracle QED+SA scores", f"scores={[f'{float(s):.3f}' for s in scores]}")
    except Exception as exc:
        _fail("Oracle QED+SA scores", traceback.format_exc(limit=3))


# ──────────────────────────────────────────────────────────────────────────────
# 4. GuacaMol benchmark smoke run
# ──────────────────────────────────────────────────────────────────────────────

def check_guacamol_benchmark() -> None:
    _section("4. GuacaMol benchmark (1 task, budget=50, 2 generations)")
    out_dir = OUTPUT_DIR / "guacamol"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "Benchmarks" / "Guacamol" / "evolve_vs_molscore_benchmark.py"
    cmd = [
        PYTHON, str(script),
        "--benchmark", "GuacaMol",
        "--include", "Albuterol_similarity",
        "--budget", "50",
        "--population-size", "8",
        "--batch-size", "8",
        "--max-generations", "2",
        "--tournament-k", "4",
        "--seed", "42",
        "--seed-pool-size", "50",
        "--skip-random-baseline",
        "--output-dir", str(out_dir),
    ]

    env = _subprocess_env()
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT), env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            _ok("GuacaMol evolve_vs_molscore_benchmark", f"elapsed={elapsed:.1f}s, output={out_dir}")
        else:
            # Print last 30 lines of stderr for diagnosis
            stderr_tail = "\n".join(result.stderr.strip().splitlines()[-30:])
            _fail("GuacaMol evolve_vs_molscore_benchmark",
                  f"exit={result.returncode}\n{textwrap.indent(stderr_tail, '      ')}")
    except subprocess.TimeoutExpired:
        _fail("GuacaMol evolve_vs_molscore_benchmark", "Timed out after 300 s")
    except Exception as exc:
        _fail("GuacaMol evolve_vs_molscore_benchmark", str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Saturn benchmark smoke run (scalar only, QED+SA, no docking)
# ──────────────────────────────────────────────────────────────────────────────

def check_saturn_benchmark() -> None:
    _section("5. Saturn benchmark (scalar only, QED+SA oracle, budget=50)")
    out_dir = OUTPUT_DIR / "saturn"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "Benchmarks" / "Saturn" / "compare_scalar_vs_nsga2_saturn.py"
    oracle_cfg = TESTS_DIR / "test_oracle_qed_sa.json"
    cmd = [
        PYTHON, str(script),
        "--seeds", "0",
        "--budgets", "50",
        "--population-size", "8",
        "--batch-size", "8",
        "--max-generations", "2",
        "--top-k", "10",
        "--tournament-k", "4",
        "--oracle-template", str(oracle_cfg),
        "--saturn-repo-root", str(REPO_ROOT / "core"),
        "--skip-nsga2",
        "--output-dir", str(out_dir),
    ]

    env = _subprocess_env()
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT / "Benchmarks" / "Saturn"), env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            _ok("Saturn compare_scalar_vs_nsga2", f"elapsed={elapsed:.1f}s, output={out_dir}")
        else:
            stderr_tail = "\n".join(result.stderr.strip().splitlines()[-30:])
            _fail("Saturn compare_scalar_vs_nsga2",
                  f"exit={result.returncode}\n{textwrap.indent(stderr_tail, '      ')}")
    except subprocess.TimeoutExpired:
        _fail("Saturn compare_scalar_vs_nsga2", "Timed out after 300 s")
    except Exception as exc:
        _fail("Saturn compare_scalar_vs_nsga2", str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary() -> int:
    _section("Summary")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"  {passed} passed  /  {failed} failed  /  {passed + failed} total\n")
    if failed:
        print("  Failed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    • {name}")
                if detail:
                    for line in detail.splitlines():
                        print(f"        {line}")
    return 1 if failed else 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("molevoDrugDiscovery smoke test")
    print(f"  REPO_ROOT : {REPO_ROOT}")
    print(f"  PYTHON    : {PYTHON}")
    print(f"  OUTPUT    : {OUTPUT_DIR}")

    check_core_imports()
    check_oracle_imports()
    check_oracle_functional()
    check_guacamol_benchmark()
    check_saturn_benchmark()

    sys.exit(print_summary())
