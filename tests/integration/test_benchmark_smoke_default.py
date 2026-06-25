from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.integration
@pytest.mark.smoke
@pytest.mark.requires_molscore
def test_guacamol_smoke_run(repo_root: Path, subprocess_env: dict[str, str], tmp_path: Path) -> None:
    if not _has_module("molscore"):
        pytest.skip("molscore is required for benchmark smoke tests")

    script = repo_root / "benchmarks" / "Guacamol" / "evolve_vs_molscore_benchmark.py"
    out_dir = tmp_path / "guacamol_smoke"

    cmd = [
        sys.executable,
        str(script),
        "--benchmark",
        "GuacaMol",
        "--include",
        "Albuterol_similarity",
        "--budget",
        "40",
        "--population-size",
        "8",
        "--batch-size",
        "8",
        "--max-generations",
        "2",
        "--tournament-k",
        "4",
        "--seed",
        "7",
        "--seed-pool-size",
        "40",
        "--skip-random-baseline",
        "--output-dir",
        str(out_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.integration
@pytest.mark.smoke
@pytest.mark.requires_molscore
def test_saturn_smoke_run_scalar_only(repo_root: Path, subprocess_env: dict[str, str], tmp_path: Path) -> None:
    if not _has_module("molscore"):
        pytest.skip("molscore is required for benchmark smoke tests")

    script = repo_root / "benchmarks" / "Saturn" / "compare_scalar_vs_nsga2_saturn.py"
    out_dir = tmp_path / "saturn_smoke"
    oracle_cfg = repo_root / "tests" / "test_oracle_qed_sa.json"

    cmd = [
        sys.executable,
        str(script),
        "--seeds",
        "0",
        "--budgets",
        "40",
        "--population-size",
        "8",
        "--batch-size",
        "8",
        "--max-generations",
        "2",
        "--top-k",
        "10",
        "--tournament-k",
        "4",
        "--oracle-template",
        str(oracle_cfg),
        "--saturn-repo-root",
        str(repo_root / "core"),
        "--skip-nsga2",
        "--output-dir",
        str(out_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(repo_root / "benchmarks" / "Saturn"),
        env=subprocess_env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert result.returncode == 0, result.stderr[-4000:]
