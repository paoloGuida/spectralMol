from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest


SAFE_CORE_IMPORTS = [
    "core.config",
    "core.vocab",
    "core.fourier_theta",
    "core.embedding",
    "core.reports",
    "core.smiles_theta_encoder",
    "core.oracles.oracle",
    "core.oracles.dataclass",
    "core.oracles.reward_aggregator.reward_aggregator",
    "core.diversity_filter.diversity_filter",
    "core.utils.chemistry_utils",
]


BENCHMARK_IMPORTS = [
    "benchmarks.Guacamol.evolve_vs_molscore_benchmark",
    "benchmarks.Guacamol.benchmark_compare_models",
    "benchmarks.Guacamol.compare_local_graphga_runs",
    "benchmarks.Guacamol.run_graphga_example_wrapper",
    "benchmarks.Guacamol.run_smiles_rnn_example",
    "benchmarks.Guacamol.run_crem_example",
    "benchmarks.Saturn.compare_scalar_vs_nsga2_saturn",
    "benchmarks.Saturn.evolve_vs_molscore_benchmark",
    "benchmarks.Saturn.guacamol_reports",
]


@pytest.mark.unit
@pytest.mark.parametrize("module_name", SAFE_CORE_IMPORTS)
def test_core_package_imports(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.unit
@pytest.mark.parametrize("module_name", BENCHMARK_IMPORTS)
def test_benchmark_package_imports(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.unit
def test_all_core_modules_importable_except_external(repo_root: Path) -> None:
    """
    Broad import sweep over core modules, skipping known external/binary-heavy paths.
    """
    core_pkg = importlib.import_module("core")
    core_path = Path(core_pkg.__file__).resolve().parent

    skip_fragments = (
        "core.oracles.xtb",
        "core.oracles.docking",
    )
    skip_modules = set(KNOWN_IMPORT_ISSUES.keys())

    failures: list[tuple[str, str]] = []
    for module_info in pkgutil.walk_packages([str(core_path)], prefix="core."):
        module_name = module_info.name
        if any(fragment in module_name for fragment in skip_fragments):
            continue
        if module_name in skip_modules:
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, str(exc)))

    assert not failures, f"Import failures: {failures}"
