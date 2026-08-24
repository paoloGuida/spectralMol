from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"


def _ensure_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_path(REPO_ROOT)
_ensure_path(CORE_ROOT)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    py_paths: list[str] = [str(REPO_ROOT), str(CORE_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    return env


@pytest.fixture(scope="session")
def xtb_available() -> bool:
    return shutil.which("xtb") is not None


@pytest.fixture(scope="session")
def quickvina_available() -> bool:
    return (
        shutil.which("QuickVina2-GPU-2-1") is not None
        or shutil.which("qvina2") is not None
        or shutil.which("qvina02") is not None
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "requires_xtb" in item.keywords and shutil.which("xtb") is None:
        pytest.skip("xtb binary is not available in PATH")

    if "requires_quickvina2" in item.keywords:
        has_quickvina = (
            shutil.which("QuickVina2-GPU-2-1") is not None
            or shutil.which("qvina2") is not None
            or shutil.which("qvina02") is not None
        )
        if not has_quickvina:
            pytest.skip("QuickVina2 binary is not available in PATH")
