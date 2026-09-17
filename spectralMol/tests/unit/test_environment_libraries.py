from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "pandas",
        "scipy",
        "joblib",
        "tqdm",
        "rdkit",
        "selfies",
        "guacamol",
        "molscore",
        "openbabel",
        "dask",
        "distributed",
    ],
)
def test_required_environment_libraries_import(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.parametrize(
    "module_name",
    [
        "cudf",
        "dask_cuda",
        "rmm",
    ],
)
def test_optional_gpu_libraries_import_if_available(module_name: str) -> None:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"Optional GPU library unavailable: {module_name} ({exc})")
    assert mod is not None
