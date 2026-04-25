#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from molscore.manager import MolScore


def patch_molscore_compat() -> None:
    """
    Provide a compatibility layer for newer pandas/MolScore behavior without
    changing MolScore_examples/GraphGA sources.
    """

    original_run_diversity_filter = MolScore.run_diversity_filter

    def _run_diversity_filter_safe(self, df):
        if df is None:
            return df
        if "occurrences" not in df.columns:
            df = df.copy()
            df["occurrences"] = 0

        method = str(self.cfg.get("scoring", {}).get("method", "")).strip()
        if method and method not in df.columns:
            filtered_method = f"filtered_{method}"
            if filtered_method in df.columns:
                df = df.copy()
                df[method] = df[filtered_method]
        return original_run_diversity_filter(self, df)

    def _score_only_compat(self, smiles: list, step: int = None, flt: bool = False):
        if step is not None:
            self.step = step
        run_smiles = list(dict.fromkeys(smiles))
        file_names = [f"{self.step}_{i}" for i, _ in enumerate(run_smiles)]
        self.run_scoring_functions(smiles=run_smiles, file_names=file_names)
        self.update_maxmin(self.results_df)
        self.results_df = self.compute_score(self.results_df)
        if self.results_df is None:
            self.results_df = pd.DataFrame({"smiles": run_smiles})
        if "occurrences" not in self.results_df.columns:
            self.results_df = self.results_df.copy()
            self.results_df["occurrences"] = 0

        if self.diversity_filter is not None:
            self.results_df = self.run_diversity_filter(self.results_df)
            score_col = f"filtered_{self.cfg['scoring']['method']}"
        else:
            score_col = self.cfg["scoring"]["method"]

        scores = []
        for smi in smiles:
            sel = self.results_df.loc[self.results_df.smiles == smi, score_col]
            scores.append(float(sel.iloc[0]) if len(sel) else 0.0)
        if not flt:
            scores = np.array(scores, dtype=np.float32)

        self.batch_df = None
        self.exists_df = pd.DataFrame()
        self.results_df = None
        return scores

    MolScore.run_diversity_filter = _run_diversity_filter_safe
    MolScore.score_only = _score_only_compat


def load_graphga_module(graphga_script: Path):
    if not graphga_script.exists():
        raise FileNotFoundError(f"GraphGA script not found: {graphga_script}")

    module_dir = str(graphga_script.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("graphga_example_module", str(graphga_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load GraphGA module from: {graphga_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Wrapper around MolScore_examples GraphGA runner with compatibility patches."
    )
    parser.add_argument(
        "--graphga-script",
        required=True,
        help="Path to MolScore_examples/GraphGA/molscore_GB_GA.py",
    )
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def main(argv: list[str] | None = None) -> int:
    known, remaining = parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    graphga_script = Path(known.graphga_script).resolve()
    module = load_graphga_module(graphga_script)

    # Apply the final patch after module import to override GraphGA's internal patch.
    patch_molscore_compat()

    # Delegate argument parsing/execution to GraphGA script.
    sys.argv = [str(graphga_script), *remaining]
    args = module.get_args()
    module.main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
