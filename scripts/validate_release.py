#!/usr/bin/env python3
"""Validate the portable release package without running expensive benchmarks."""

from __future__ import annotations

import gzip
import hashlib
import json
import py_compile
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "run.py",
    "pyproject.toml",
    "LICENSE",
    "CITATION.cff",
    "configs/test.toml",
    "configs/guacamol_manuscript.toml",
    "configs/saturn_table8.toml",
    "configs/guacamol_task_profiles.json",
    "configs/saturn_table8_v119_environment.json",
    "spectralMol/benchmarks/Guacamol/model_specs.json",
    "spectralMol/benchmarks/Saturn/table2_r_sa_qed_oracle_template.json",
    "reproducibility/manuscript_2026/inputs/guacamol/shared_initial_population_seed_7.smi",
    "reproducibility/manuscript_2026/inputs/guacamol/chembl.filtered.smi.gz",
    "reproducibility/manuscript_2026/inputs/saturn/seed_sets/seed_0.smi",
    "reproducibility/manuscript_2026/inputs/saturn/seed_sets/seed_9.smi",
    "reproducibility/manuscript_2026/inputs/saturn/docking/7uvu-reference.pdb",
    "reproducibility/manuscript_2026/inputs/saturn/docking/7uvu-2-monomers-pdbfixer.pdbqt",
    "reproducibility/manuscript_2026/results/guacamol/guacamol_per_seed_aggregate.tsv",
    "reproducibility/manuscript_2026/results/saturn/table8_spectralmol.tsv",
)
TEXT_SUFFIXES = {
    ".cff", ".cfg", ".csv", ".env", ".ini", ".json", ".md", ".py", ".smi",
    ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
FORBIDDEN = (
    "i" + "bex",
    "KA" + "UST",
    "/i" + "bex/",
    "#SB" + "ATCH",
    "module" + " load",
    "/ho" + "me/",
    "/Us" + "ers/",
    "col" + "leoe",
)


def _tree_files() -> list[Path]:
    ignored = {".git", ".venv", "results", "__pycache__", ".pytest_cache"}
    return [path for path in ROOT.rglob("*") if path.is_file() and not any(part in ignored for part in path.parts)]


def _check_required() -> None:
    missing = [value for value in REQUIRED if not (ROOT / value).is_file() or (ROOT / value).stat().st_size == 0]
    if missing:
        raise RuntimeError("Missing required release files:\n  " + "\n  ".join(missing))


def _check_python() -> None:
    cache = Path("/tmp") / "spectralmol-release-pycache"
    for path in _tree_files():
        if path.suffix == ".py":
            target = cache / path.relative_to(ROOT).with_suffix(".pyc")
            target.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(path), cfile=str(target), doraise=True)


def _check_config_snapshots() -> None:
    profiles = json.loads((ROOT / "configs/guacamol_task_profiles.json").read_text(encoding="utf-8"))
    saturn = json.loads((ROOT / "configs/saturn_table8_v119_environment.json").read_text(encoding="utf-8"))
    if sorted(map(int, profiles)) != list(range(20)):
        raise RuntimeError("GuacaMol task-profile snapshot must cover task indexes 0 through 19")
    if any(profile["environment"].get("SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION") != "0" for profile in profiles.values()):
        raise RuntimeError("GuacaMol task profiles are not theta-only")
    theta_only = (
        "MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION",
        "MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION",
        "MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION",
    )
    if any(saturn.get(key) != "0" for key in theta_only):
        raise RuntimeError("SATURN profile is not theta-only")


def _check_archives() -> None:
    for path in _tree_files():
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass


def _check_portability() -> None:
    findings: list[str] = []
    for path in _tree_files():
        relative = path.relative_to(ROOT)
        lower_name = str(relative).lower()
        for marker in FORBIDDEN:
            if marker.lower() in lower_name:
                findings.append(f"filename:{relative}:{marker}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in FORBIDDEN:
            if marker.lower() in text.lower():
                findings.append(f"text:{relative}:{marker}")
        if re.search(r"(?:password|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][^'\"]+", text, re.IGNORECASE):
            findings.append(f"possible-secret:{relative}")
    findings.extend(f"shell-script:{path.relative_to(ROOT)}" for path in _tree_files() if path.suffix == ".sh")
    if findings:
        raise RuntimeError("Release portability audit failed:\n  " + "\n  ".join(findings))


def _check_manifest() -> None:
    manifest = ROOT / "reproducibility/manuscript_2026/provenance/SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, value = line.split(maxsplit=1)
        path = ROOT / value.lstrip("* ")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {path.relative_to(ROOT)}")


def main() -> int:
    _check_required()
    _check_python()
    _check_config_snapshots()
    _check_archives()
    _check_portability()
    _check_manifest()
    print("SpectralMol public-release validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
