#!/usr/bin/env python3
"""Summarize ablation comparison outputs into consolidated tables.

This script is intended for runs produced by:
- slurm_scripts/compare_ablation_dask_rapids_container_v100.sbatch

It reads one compare_* directory and emits:
- ablation_overall_summary_<timestamp>.tsv
- ablation_task_delta_summary_<timestamp>.tsv
- ablation_summary_<timestamp>.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CONFIG_DIR = "molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container"


@dataclass
class OverallRow:
    model: str
    condition: str
    rank: float
    sum_task_best_score: float
    mean_task_best_score: float
    sum_task_avg_score: float
    mean_task_avg_score: float
    mean_task_std_best: float
    n_tasks: int
    total_runs: int
    mean_elapsed_seconds: float
    min_elapsed_seconds: float
    max_elapsed_seconds: float
    n_failed: int
    budget_effective: int
    evals_per_second_mean: float
    speedup_vs_full: float
    delta_best_vs_full: float


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _pick_latest_run(config_dir: Path, run_glob: str) -> Path:
    runs = sorted([p for p in config_dir.glob(run_glob) if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories matching {run_glob!r} under {config_dir}")
    return runs[-1]


def _condition_from_model(model: str, model_to_mode: dict[str, str]) -> str:
    if model in model_to_mode and model_to_mode[model]:
        return str(model_to_mode[model])

    lowered = model.lower()
    if lowered == "local_evolution":
        return "full-spectrum"
    if lowered.endswith("_low_only") or lowered.endswith("-low-only"):
        return "low-only"
    if lowered.endswith("_high_only") or lowered.endswith("-high-only"):
        return "high-only"
    if lowered.endswith("_random_matrix") or lowered.endswith("-random-matrix"):
        return "random-matrix"
    if lowered.endswith("_full_spectrum") or lowered.endswith("-full-spectrum"):
        return "full-spectrum"
    return "unknown"


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _write_overall_tsv(out_path: Path, rows: list[OverallRow]) -> None:
    fieldnames = [
        "model",
        "condition",
        "rank",
        "sum_task_best_score",
        "mean_task_best_score",
        "sum_task_avg_score",
        "mean_task_avg_score",
        "mean_task_std_best",
        "n_tasks",
        "total_runs",
        "mean_elapsed_seconds",
        "min_elapsed_seconds",
        "max_elapsed_seconds",
        "n_failed",
        "budget_effective",
        "evals_per_second_mean",
        "speedup_vs_full",
        "delta_best_vs_full",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "model": r.model,
                    "condition": r.condition,
                    "rank": _fmt(r.rank),
                    "sum_task_best_score": _fmt(r.sum_task_best_score),
                    "mean_task_best_score": _fmt(r.mean_task_best_score),
                    "sum_task_avg_score": _fmt(r.sum_task_avg_score),
                    "mean_task_avg_score": _fmt(r.mean_task_avg_score),
                    "mean_task_std_best": _fmt(r.mean_task_std_best),
                    "n_tasks": r.n_tasks,
                    "total_runs": r.total_runs,
                    "mean_elapsed_seconds": _fmt(r.mean_elapsed_seconds),
                    "min_elapsed_seconds": _fmt(r.min_elapsed_seconds),
                    "max_elapsed_seconds": _fmt(r.max_elapsed_seconds),
                    "n_failed": r.n_failed,
                    "budget_effective": r.budget_effective,
                    "evals_per_second_mean": _fmt(r.evals_per_second_mean),
                    "speedup_vs_full": _fmt(r.speedup_vs_full),
                    "delta_best_vs_full": _fmt(r.delta_best_vs_full),
                }
            )


def _write_task_delta_tsv(out_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    out_path: Path,
    *,
    config_dir: Path,
    run_dir: Path,
    rows: list[OverallRow],
    task_deltas_path: Path,
    overall_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Ablation Summary")
    lines.append("")
    lines.append(f"Config directory: {config_dir}")
    lines.append(f"Run directory: {run_dir}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Condition | Model | Mean Best Score | Delta vs Full | Mean Elapsed (s) | Speedup vs Full | Evals/s | n_failed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {cond} | {model} | {best} | {delta} | {elapsed} | {speedup}x | {eps} | {failed} |".format(
                cond=r.condition,
                model=r.model,
                best=_fmt(r.mean_task_best_score),
                delta=_fmt(r.delta_best_vs_full),
                elapsed=_fmt(r.mean_elapsed_seconds),
                speedup=_fmt(r.speedup_vs_full),
                eps=_fmt(r.evals_per_second_mean),
                failed=r.n_failed,
            )
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Overall TSV: {overall_path}")
    lines.append(f"- Task delta TSV: {task_deltas_path}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Full-spectrum baseline is selected by condition label full-spectrum.")
    lines.append("- If full-spectrum is missing, deltas and speedup are computed against the first sorted row.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ablation run outputs.")
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        help="Directory that contains compare_* run folders.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit run directory name under config-dir.",
    )
    parser.add_argument(
        "--run-glob",
        default="compare_*",
        help="Glob used to detect latest run when --run-dir is omitted.",
    )
    parser.add_argument(
        "--output-prefix",
        default="ablation_summary",
        help="Output file prefix written under config-dir.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir).resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if args.run_dir.strip():
        run_dir = (config_dir / args.run_dir.strip()).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        run_dir = _pick_latest_run(config_dir, args.run_glob).resolve()

    comparison_overall_path = run_dir / "comparison_overall.tsv"
    comparison_task_path = run_dir / "comparison_task_summary.tsv"
    runtime_summary_path = run_dir / "model_runtime_summary.tsv"
    run_meta_path = run_dir / "run_meta.json"

    comparison_overall = _read_tsv(comparison_overall_path)
    comparison_task = _read_tsv(comparison_task_path)
    runtime_summary = _read_tsv(runtime_summary_path)
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    runtime_by_model = {row["model"]: row for row in runtime_summary}
    model_to_mode = dict(run_meta.get("fourier_mode_by_model") or {})
    budget_effective = int(run_meta.get("budget_effective", 0))

    rows: list[OverallRow] = []
    for row in comparison_overall:
        model = row["model"]
        rt = runtime_by_model.get(model)
        if rt is None:
            continue

        n_runs = int(rt.get("n_runs", "0") or 0)
        mean_elapsed = float(rt.get("mean_elapsed_seconds", "nan"))
        eps_mean = 0.0
        if mean_elapsed > 0 and n_runs > 0 and budget_effective > 0:
            # Equivalent to budget_effective / mean_elapsed_seconds.
            eps_mean = float((budget_effective * n_runs) / (mean_elapsed * n_runs))

        rows.append(
            OverallRow(
                model=model,
                condition=_condition_from_model(model, model_to_mode),
                rank=float(row.get("rank", "nan")),
                sum_task_best_score=float(row["sum_task_best_score"]),
                mean_task_best_score=float(row["mean_task_best_score"]),
                sum_task_avg_score=float(row["sum_task_avg_score"]),
                mean_task_avg_score=float(row["mean_task_avg_score"]),
                mean_task_std_best=float(row["mean_task_std_best"]),
                n_tasks=int(float(row["n_tasks"])),
                total_runs=int(float(row["total_runs"])),
                mean_elapsed_seconds=mean_elapsed,
                min_elapsed_seconds=float(rt.get("min_elapsed_seconds", "nan")),
                max_elapsed_seconds=float(rt.get("max_elapsed_seconds", "nan")),
                n_failed=int(rt.get("n_failed", "0") or 0),
                budget_effective=budget_effective,
                evals_per_second_mean=eps_mean,
                speedup_vs_full=0.0,
                delta_best_vs_full=0.0,
            )
        )

    if not rows:
        raise RuntimeError(f"No model rows found in {comparison_overall_path}")

    rows = sorted(rows, key=lambda x: x.mean_task_best_score, reverse=True)
    baseline = next((r for r in rows if r.condition == "full-spectrum"), rows[0])

    for r in rows:
        if r.mean_elapsed_seconds > 0 and baseline.mean_elapsed_seconds > 0:
            r.speedup_vs_full = baseline.mean_elapsed_seconds / r.mean_elapsed_seconds
        else:
            r.speedup_vs_full = 0.0
        r.delta_best_vs_full = r.mean_task_best_score - baseline.mean_task_best_score

    baseline_task = {
        row["task"]: float(row["mean_best_score"])
        for row in comparison_task
        if row.get("model") == baseline.model
    }

    task_delta_rows: list[dict[str, str]] = []
    for row in comparison_task:
        model = row["model"]
        task = row["task"]
        model_best = float(row["mean_best_score"])
        base_best = baseline_task.get(task, float("nan"))
        delta = model_best - base_best if base_best == base_best else float("nan")
        task_delta_rows.append(
            {
                "model": model,
                "condition": _condition_from_model(model, model_to_mode),
                "task": task,
                "mean_best_score": _fmt(model_best),
                "baseline_full_mean_best_score": _fmt(base_best),
                "delta_best_vs_full": _fmt(delta),
                "mean_avg_score": _fmt(float(row.get("mean_avg_score", "nan"))),
                "std_best_score": _fmt(float(row.get("std_best_score", "nan"))),
                "n_runs": row.get("n_runs", ""),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    overall_out = config_dir / f"{args.output_prefix}_overall_{stamp}.tsv"
    task_delta_out = config_dir / f"{args.output_prefix}_task_delta_{stamp}.tsv"
    md_out = config_dir / f"{args.output_prefix}_{stamp}.md"

    _write_overall_tsv(overall_out, rows)
    _write_task_delta_tsv(task_delta_out, task_delta_rows)
    _write_markdown(
        md_out,
        config_dir=config_dir,
        run_dir=run_dir,
        rows=rows,
        task_deltas_path=task_delta_out,
        overall_path=overall_out,
    )

    print(f"Wrote: {overall_out}")
    print(f"Wrote: {task_delta_out}")
    print(f"Wrote: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
