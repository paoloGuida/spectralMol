"""Lightweight checks for the public command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in completed.stdout
    assert "--dry-run" in completed.stdout


def test_public_profiles_cover_all_guacamol_tasks() -> None:
    profiles = json.loads((ROOT / "configs" / "guacamol_task_profiles.json").read_text())
    assert sorted(map(int, profiles)) == list(range(20))


def test_saturn_profile_keeps_theta_as_the_only_genotype() -> None:
    environment = json.loads((ROOT / "configs" / "saturn_table8_v119_environment.json").read_text())
    assert environment["MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION"] == "0"
    assert environment["MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION"] == "0"
    assert environment["MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION"] == "0"
