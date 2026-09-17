#!/usr/bin/env python3
"""Repository entry point for the portable SpectralMol CLI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "spectralMol"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spectralmol_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
