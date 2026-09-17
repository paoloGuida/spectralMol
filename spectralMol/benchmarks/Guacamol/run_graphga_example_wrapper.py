#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
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
    parser.add_argument(
        "--task-indexes",
        default="",
        help="Comma-separated zero-based MolScore task indexes to run after include/exclude filtering.",
    )
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def _parse_task_indexes(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _graphga_starting_population(module, generator, args):
    if not getattr(args, "starting_population_file", ""):
        return None
    start_pool = generator.load_smiles_from_file(args.starting_population_file)
    if len(start_pool) < args.population_size:
        raise ValueError(
            f"starting_population_file has {len(start_pool)} molecules, "
            f"but population_size={args.population_size}"
        )
    return start_pool[: args.population_size]


def run_graphga_task_indexes(module, args, task_indexes: list[int]) -> None:
    generator = module.GB_GA(
        smi_file=args.smiles_file,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        n_jobs=args.n_jobs,
        random_start=args.random_start,
        patience=args.patience,
    )
    starting_population = _graphga_starting_population(module, generator, args)
    include = _parse_csv(getattr(args, "include", ""))
    exclude = _parse_csv(getattr(args, "exclude", ""))
    selected = set(int(i) for i in task_indexes)

    if args.molscore in module.MolScoreBenchmark.presets:
        scoring_function = module.MolScoreBenchmark(
            model_name=args.model_name,
            output_dir=args.output_dir,
            budget=args.budget,
            benchmark=args.molscore,
            include=include,
            exclude=exclude,
        )
    elif os.path.isdir(args.molscore):
        scoring_function = module.MolScoreBenchmark(
            model_name=args.model_name,
            output_dir=args.output_dir,
            budget=args.budget,
            custom_benchmark=args.molscore,
            include=include,
            exclude=exclude,
        )
    else:
        raise ValueError("--task-indexes requires MolScoreBenchmark preset or custom benchmark directory.")

    matched = 0
    for idx, task in enumerate(scoring_function):
        if idx not in selected:
            continue
        matched += 1
        task_name = getattr(task, "cfg", {}).get("task", f"task_{idx}")
        print(f"[graphga-wrapper] running task_index={idx} task={task_name}", flush=True)
        final_population_smiles = generator.generate_optimized_molecules(
            scoring_function=task,
            starting_population=starting_population,
        )
        with open(os.path.join(task.save_dir, "final_population.smi"), "w") as f:
            for smi in final_population_smiles:
                f.write(smi + "\n")

    if matched != len(selected):
        raise ValueError(
            f"Requested task indexes {sorted(selected)} but matched {matched} tasks after include/exclude filtering."
        )


def main(argv: list[str] | None = None) -> int:
    known, remaining = parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    graphga_script = Path(known.graphga_script).resolve()
    module = load_graphga_module(graphga_script)

    # Apply the final patch after module import to override GraphGA's internal patch.
    patch_molscore_compat()

    # Delegate argument parsing/execution to GraphGA script.
    sys.argv = [str(graphga_script), *remaining]
    args = module.get_args()
    task_indexes = _parse_task_indexes(known.task_indexes)
    if task_indexes:
        run_graphga_task_indexes(module, args, task_indexes)
    else:
        module.main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
