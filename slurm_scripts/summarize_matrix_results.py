#!/usr/bin/env python3
"""Summarize 4-way matrix outputs into consolidated wall-clock and throughput tables.

Default target configs are the containerized matrix runs:
- baseline_cpu_container
- dask_cpu_container
- rapids_v100_container
- dask_rapids_v100_container

For each config, this script picks the latest compare_* run directory unless --run-dir
is provided for a config (config=run_dir_name form).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIGS = [
    "baseline_cpu_container",
    "dask_cpu_container",
    "rapids_v100_container",
    "dask_rapids_v100_container",
]


@dataclass
class ConfigSummary:
    config: str
    run_dir: str
    mean_elapsed_seconds: float
    campaign_wallclock_seconds: float
    n_runs: int
    n_failed: int
    budget_effective: int
    total_evaluations: int
    evals_per_second_mean: float
    evals_per_second_campaign: float
    speedup_vs_baseline_mean: float = 0.0
    speedup_vs_baseline_campaign: float = 0.0
    time_reduction_pct_mean: float = 0.0
    time_reduction_pct_campaign: float = 0.0


def _parse_iso_utc(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _pick_latest_run(config_dir: Path, pattern: str) -> Path:
    runs = sorted([p for p in config_dir.glob(pattern) if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories matching {pattern!r} under {config_dir}")
    return runs[-1]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _weighted_mean_elapsed(runtime_summary_rows: list[dict[str, str]]) -> tuple[float, int, int]:
    weighted_sum = 0.0
    total_runs = 0
    total_failed = 0
    for row in runtime_summary_rows:
        n_runs = int(row["n_runs"])
        mean_elapsed = float(row["mean_elapsed_seconds"])
        n_failed = int(row.get("n_failed", "0"))
        weighted_sum += mean_elapsed * n_runs
        total_runs += n_runs
        total_failed += n_failed
    if total_runs <= 0:
        raise ValueError("No runs found in model_runtime_summary.tsv")
    return (weighted_sum / total_runs, total_runs, total_failed)


def _campaign_wallclock_seconds(runtime_rows: list[dict[str, str]]) -> float:
    started = []
    finished = []
    for row in runtime_rows:
        if row.get("started_at_utc") and row.get("finished_at_utc"):
            started.append(_parse_iso_utc(row["started_at_utc"]))
            finished.append(_parse_iso_utc(row["finished_at_utc"]))
    if not started or not finished:
        # Fallback: sum elapsed when timestamps are missing.
        return float(sum(float(r["elapsed_seconds"]) for r in runtime_rows if r.get("elapsed_seconds")))
    return float((max(finished) - min(started)).total_seconds())


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _write_tsv(out_path: Path, rows: Iterable[ConfigSummary]) -> None:
    fieldnames = [
        "config",
        "run_dir",
        "mean_elapsed_seconds",
        "campaign_wallclock_seconds",
        "n_runs",
        "n_failed",
        "budget_effective",
        "total_evaluations",
        "evals_per_second_mean",
        "evals_per_second_campaign",
        "speedup_vs_baseline_mean",
        "speedup_vs_baseline_campaign",
        "time_reduction_pct_mean",
        "time_reduction_pct_campaign",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "config": r.config,
                    "run_dir": r.run_dir,
                    "mean_elapsed_seconds": _fmt(r.mean_elapsed_seconds),
                    "campaign_wallclock_seconds": _fmt(r.campaign_wallclock_seconds),
                    "n_runs": r.n_runs,
                    "n_failed": r.n_failed,
                    "budget_effective": r.budget_effective,
                    "total_evaluations": r.total_evaluations,
                    "evals_per_second_mean": _fmt(r.evals_per_second_mean),
                    "evals_per_second_campaign": _fmt(r.evals_per_second_campaign),
                    "speedup_vs_baseline_mean": _fmt(r.speedup_vs_baseline_mean),
                    "speedup_vs_baseline_campaign": _fmt(r.speedup_vs_baseline_campaign),
                    "time_reduction_pct_mean": _fmt(r.time_reduction_pct_mean),
                    "time_reduction_pct_campaign": _fmt(r.time_reduction_pct_campaign),
                }
            )


def _write_markdown(out_path: Path, rows: list[ConfigSummary], base_dir: Path) -> None:
    lines = []
    lines.append("# Matrix Wall-Clock and Throughput Summary")
    lines.append("")
    lines.append(f"Base directory: {base_dir}")
    lines.append("")
    lines.append("| Config | Run Dir | Mean elapsed (s) | Campaign wall-clock (s) | Evals/s (mean) | Evals/s (campaign) | Speedup mean | Speedup campaign | n_failed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {config} | {run_dir} | {mean_elapsed} | {campaign_wallclock} | {eps_mean} | {eps_campaign} | {sp_mean}x | {sp_campaign}x | {n_failed} |".format(
                config=r.config,
                run_dir=r.run_dir,
                mean_elapsed=_fmt(r.mean_elapsed_seconds),
                campaign_wallclock=_fmt(r.campaign_wallclock_seconds),
                eps_mean=_fmt(r.evals_per_second_mean),
                eps_campaign=_fmt(r.evals_per_second_campaign),
                sp_mean=_fmt(r.speedup_vs_baseline_mean),
                sp_campaign=_fmt(r.speedup_vs_baseline_campaign),
                n_failed=r.n_failed,
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- Evals/s (mean) = budget_effective / mean_elapsed_seconds")
    lines.append("- Evals/s (campaign) = total_evaluations / campaign_wallclock_seconds")
    lines.append("- total_evaluations = budget_effective * n_runs")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize comparison_matrix runtime + throughput.")
    parser.add_argument(
        "--base-dir",
        default="molscore/outputs/comparison_matrix",
        help="Base comparison_matrix directory.",
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config directories to summarize.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Optional explicit run dir mapping in form config=run_dir_name. Can be repeated.",
    )
    parser.add_argument(
        "--run-glob",
        default="compare_*",
        help="Glob used to detect latest run dir when --run-dir is not provided.",
    )
    parser.add_argument(
        "--output-prefix",
        default="matrix_summary",
        help="Output filename prefix under base-dir.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    explicit_run_dirs: dict[str, str] = {}
    for item in args.run_dir:
        if "=" not in item:
            raise ValueError(f"Invalid --run-dir {item!r}; expected config=run_dir_name")
        cfg, run_name = item.split("=", 1)
        explicit_run_dirs[cfg.strip()] = run_name.strip()

    summaries: list[ConfigSummary] = []
    for cfg in configs:
        cfg_dir = base_dir / cfg
        if not cfg_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {cfg_dir}")

        if cfg in explicit_run_dirs:
            run_dir = cfg_dir / explicit_run_dirs[cfg]
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found for {cfg}: {run_dir}")
        else:
            run_dir = _pick_latest_run(cfg_dir, args.run_glob)

        runtime_summary_path = run_dir / "model_runtime_summary.tsv"
        runtime_path = run_dir / "model_runtime.tsv"
        run_meta_path = run_dir / "run_meta.json"

        runtime_summary_rows = _read_tsv(runtime_summary_path)
        runtime_rows = _read_tsv(runtime_path)
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

        mean_elapsed, n_runs, n_failed = _weighted_mean_elapsed(runtime_summary_rows)
        campaign_wallclock = _campaign_wallclock_seconds(runtime_rows)

        budget_effective = int(run_meta["budget_effective"])
        total_evaluations = int(budget_effective * n_runs)
        evals_per_second_mean = float(total_evaluations / (mean_elapsed * n_runs))
        evals_per_second_campaign = float(total_evaluations / campaign_wallclock) if campaign_wallclock > 0 else 0.0

        summaries.append(
            ConfigSummary(
                config=cfg,
                run_dir=run_dir.name,
                mean_elapsed_seconds=mean_elapsed,
                campaign_wallclock_seconds=campaign_wallclock,
                n_runs=n_runs,
                n_failed=n_failed,
                budget_effective=budget_effective,
                total_evaluations=total_evaluations,
                evals_per_second_mean=evals_per_second_mean,
                evals_per_second_campaign=evals_per_second_campaign,
            )
        )

    baseline = summaries[0]
    if baseline.mean_elapsed_seconds <= 0 or baseline.campaign_wallclock_seconds <= 0:
        raise ValueError("Baseline elapsed values must be > 0")

    for s in summaries:
        s.speedup_vs_baseline_mean = baseline.mean_elapsed_seconds / s.mean_elapsed_seconds
        s.speedup_vs_baseline_campaign = baseline.campaign_wallclock_seconds / s.campaign_wallclock_seconds
        s.time_reduction_pct_mean = (1.0 - (s.mean_elapsed_seconds / baseline.mean_elapsed_seconds)) * 100.0
        s.time_reduction_pct_campaign = (1.0 - (s.campaign_wallclock_seconds / baseline.campaign_wallclock_seconds)) * 100.0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_tsv = base_dir / f"{args.output_prefix}_{stamp}.tsv"
    out_md = base_dir / f"{args.output_prefix}_{stamp}.md"

    _write_tsv(out_tsv, summaries)
    _write_markdown(out_md, summaries, base_dir)

    print(f"Wrote: {out_tsv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
