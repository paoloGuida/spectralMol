#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONDITIONS = ("full-spectrum", "high-only", "low-only", "random-matrix")

MANUSCRIPT_TARGETS: dict[str, dict[str, float]] = {
    "high-only": {
        "mean_best_score": 0.7542,
        "delta_vs_full": 0.0050,
        "mean_elapsed_seconds": 1331.3,
        "speedup_vs_full": 1.156,
        "evals_per_second": 15.023,
    },
    "full-spectrum": {
        "mean_best_score": 0.7492,
        "delta_vs_full": 0.0,
        "mean_elapsed_seconds": 1538.6,
        "speedup_vs_full": 1.0,
        "evals_per_second": 12.999,
    },
    "low-only": {
        "mean_best_score": 0.7367,
        "delta_vs_full": -0.0125,
        "mean_elapsed_seconds": 1548.1,
        "speedup_vs_full": 0.994,
        "evals_per_second": 12.919,
    },
    "random-matrix": {
        "mean_best_score": 0.6469,
        "delta_vs_full": -0.1023,
        "mean_elapsed_seconds": 1256.1,
        "speedup_vs_full": 1.225,
        "evals_per_second": 15.923,
    },
}


def mean(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_seed(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def parse_task_index(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"task_(\d+)_.*", part)
        if match:
            return int(match.group(1))
    return None


def read_dicts(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    except Exception:
        return []


def collect_condition(condition: str, root: Path, model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for compare_root in sorted({p.parent for p in root.rglob("comparison_task_best_seed_summary.tsv")}):
        seed = parse_seed(compare_root)
        task_index = parse_task_index(compare_root)

        best_path = compare_root / "comparison_task_best_seed_summary.tsv"
        best_records = [r for r in read_dicts(best_path) if r.get("model") == model]
        if not best_records:
            continue
        best = best_records[0]

        raw_path = compare_root / "comparison_raw.tsv"
        raw_records = [r for r in read_dicts(raw_path) if r.get("model") == model]
        raw = raw_records[0] if raw_records else {}

        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "task_index": task_index,
                "task": best.get("task", raw.get("task", "")),
                "best_score": fnum(best.get("best_seed_score", raw.get("best_score"))),
                "avg_score": fnum(best.get("best_seed_avg_score", raw.get("avg_score"))),
                "evaluated": fnum(best.get("best_seed_n_scored", raw.get("n_scored"))),
                "scores_csv": best.get("scores_csv", raw.get("scores_csv", "")),
                "compare_root": str(compare_root),
            }
        )

        runtime_path = compare_root / "model_runtime.tsv"
        for record in read_dicts(runtime_path):
            if record.get("model") != model:
                continue
            runtime_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "task_index": task_index,
                    "task": best.get("task", raw.get("task", "")),
                    "status": record.get("status", ""),
                    "return_code": record.get("return_code", ""),
                    "elapsed_seconds": fnum(record.get("elapsed_seconds")),
                    "source": str(runtime_path),
                }
            )
    return rows, runtime_rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    for row in runtime_rows:
        runtime_by_condition[str(row["condition"])].append(row)

    summary: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        cond_rows = by_condition.get(condition, [])
        cond_runtime = runtime_by_condition.get(condition, [])
        best_scores = [fnum(r["best_score"]) for r in cond_rows]
        evaluated = [fnum(r["evaluated"]) for r in cond_rows]
        elapsed = [fnum(r["elapsed_seconds"]) for r in cond_runtime]
        total_evaluated = sum(x for x in evaluated if math.isfinite(x))
        total_elapsed = sum(x for x in elapsed if math.isfinite(x))
        failed = sum(1 for r in cond_runtime if str(r.get("status", "")).lower() not in {"ok", ""})
        summary.append(
            {
                "condition": condition,
                "mean_best_score": mean(best_scores),
                "std_best_score": std(best_scores),
                "mean_elapsed_seconds": mean(elapsed),
                "std_elapsed_seconds": std(elapsed),
                "total_evaluated": total_evaluated,
                "total_elapsed_seconds": total_elapsed,
                "evals_per_second": total_evaluated / total_elapsed if total_elapsed > 0 else float("nan"),
                "budget_over_mean_elapsed": float(budget) / mean(elapsed) if mean(elapsed) > 0 else float("nan"),
                "n_rows": len(cond_rows),
                "n_tasks": len({r["task_index"] for r in cond_rows if r["task_index"] is not None}),
                "n_seeds": len({r["seed"] for r in cond_rows if r["seed"] is not None}),
                "n_failed": failed,
            }
        )

    full = next((r for r in summary if r["condition"] == "full-spectrum"), None)
    full_best = fnum(full["mean_best_score"]) if full else float("nan")
    full_elapsed = fnum(full["mean_elapsed_seconds"]) if full else float("nan")
    for row in summary:
        row["delta_vs_full"] = fnum(row["mean_best_score"]) - full_best
        elapsed = fnum(row["mean_elapsed_seconds"])
        row["speedup_vs_full"] = full_elapsed / elapsed if elapsed > 0 else float("nan")
    return summary


def task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, str], list[float]] = defaultdict(list)
    task_names: dict[tuple[str, int | None, str], str] = {}
    for row in rows:
        key = (str(row["condition"]), row["task_index"], str(row["task"]))
        grouped[key].append(fnum(row["best_score"]))
        task_names[key] = str(row["task"])
    out = []
    for (condition, task_index, task), vals in sorted(grouped.items(), key=lambda kv: (kv[0][1] if kv[0][1] is not None else 999, kv[0][0])):
        out.append(
            {
                "condition": condition,
                "task_index": task_index,
                "task": task_names[(condition, task_index, task)],
                "mean_best_score": mean(vals),
                "std_best_score": std(vals),
                "n_seeds": len(vals),
            }
        )
    return out


def build_checks(summary_rows: list[dict[str, Any]], expected_tasks: int, expected_seeds: int) -> list[dict[str, Any]]:
    rows_by_condition = {str(r["condition"]): r for r in summary_rows}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({"check": name, "passed": int(bool(passed)), "observed": observed, "expected": expected})

    present_conditions = [
        condition
        for condition, row in rows_by_condition.items()
        if int(row.get("n_rows", 0)) > 0
    ]
    add(
        "all_four_conditions_present",
        all(condition in present_conditions for condition in CONDITIONS),
        ",".join(present_conditions),
        ",".join(CONDITIONS),
    )
    for condition in CONDITIONS:
        row = rows_by_condition.get(condition)
        if not row:
            continue
        add(f"{condition}: expected task count", int(row["n_tasks"]) == expected_tasks, row["n_tasks"], expected_tasks)
        add(f"{condition}: expected seed count", int(row["n_seeds"]) == expected_seeds, row["n_seeds"], expected_seeds)
        add(f"{condition}: no failed model runs", int(row["n_failed"]) == 0, row["n_failed"], 0)

    full = rows_by_condition.get("full-spectrum")
    high = rows_by_condition.get("high-only")
    low = rows_by_condition.get("low-only")
    random_matrix = rows_by_condition.get("random-matrix")
    if full and high:
        add("claim: high-only mean best score exceeds full-spectrum", fnum(high["delta_vs_full"]) > 0, high["delta_vs_full"], "> 0")
        add("claim: high-only is faster than full-spectrum", fnum(high["speedup_vs_full"]) > 1, high["speedup_vs_full"], "> 1")
    if full and low:
        add("claim: low-only underperforms full-spectrum", fnum(low["delta_vs_full"]) < 0, low["delta_vs_full"], "< 0")
    if random_matrix:
        sorted_by_delta = sorted(summary_rows, key=lambda r: fnum(r["delta_vs_full"]))
        sorted_by_speed = sorted(summary_rows, key=lambda r: fnum(r["speedup_vs_full"]), reverse=True)
        add("claim: random-matrix has largest quality drop", sorted_by_delta[0]["condition"] == "random-matrix", sorted_by_delta[0]["condition"], "random-matrix")
        add("claim: random-matrix is fastest", sorted_by_speed[0]["condition"] == "random-matrix", sorted_by_speed[0]["condition"], "random-matrix")

    for condition, targets in MANUSCRIPT_TARGETS.items():
        row = rows_by_condition.get(condition)
        if not row:
            continue
        for metric, target in targets.items():
            observed = fnum(row.get(metric))
            add(
                f"manuscript target: {condition} {metric}",
                math.isfinite(observed),
                observed,
                target,
            )
    return checks


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                vals.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GuacaMol frequency-mode ablation outputs.")
    parser.add_argument("--input-root", required=True, help="Root with condition/seed/task comparison outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for ablation summary tables.")
    parser.add_argument("--model", default="spectralmol")
    parser.add_argument("--budget", type=int, default=20000)
    parser.add_argument("--expected-seeds", type=int, default=6)
    parser.add_argument("--expected-tasks", type=int, default=20)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    all_runtime: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_root = input_root / condition
        rows, runtime = collect_condition(condition, condition_root, args.model)
        all_rows.extend(rows)
        all_runtime.extend(runtime)

    if not all_rows:
        raise SystemExit(f"No comparison_task_best_seed_summary.tsv files found under {input_root}")

    summary_rows = summarize(all_rows, all_runtime, args.budget)
    task_rows = task_summary(all_rows)
    check_rows = build_checks(summary_rows, args.expected_tasks, args.expected_seeds)

    raw_fields = ["condition", "seed", "task_index", "task", "best_score", "avg_score", "evaluated", "scores_csv", "compare_root"]
    runtime_fields = ["condition", "seed", "task_index", "task", "status", "return_code", "elapsed_seconds", "source"]
    summary_fields = [
        "condition",
        "mean_best_score",
        "std_best_score",
        "delta_vs_full",
        "mean_elapsed_seconds",
        "std_elapsed_seconds",
        "speedup_vs_full",
        "evals_per_second",
        "budget_over_mean_elapsed",
        "n_rows",
        "n_tasks",
        "n_seeds",
        "n_failed",
    ]
    task_fields = ["condition", "task_index", "task", "mean_best_score", "std_best_score", "n_seeds"]
    check_fields = ["check", "passed", "observed", "expected"]

    write_tsv(output_dir / "ablation_raw_rows.tsv", all_rows, raw_fields)
    write_tsv(output_dir / "ablation_runtime_rows.tsv", all_runtime, runtime_fields)
    write_tsv(output_dir / "ablation_condition_summary.tsv", summary_rows, summary_fields)
    write_tsv(output_dir / "ablation_task_summary.tsv", task_rows, task_fields)
    write_tsv(output_dir / "ablation_claim_check.tsv", check_rows, check_fields)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "model": args.model,
        "budget": args.budget,
        "expected_seeds": args.expected_seeds,
        "expected_tasks": args.expected_tasks,
        "conditions": list(CONDITIONS),
        "manuscript_targets": MANUSCRIPT_TARGETS,
    }
    (output_dir / "ablation_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        "# GuacaMol Frequency-Mode Ablation",
        "",
        f"Generated: {manifest['created_at_utc']}",
        f"Input root: `{input_root}`",
        "",
        "## Condition Summary",
        "",
        markdown_table(summary_rows, summary_fields),
        "",
        "## Claim Checks",
        "",
        markdown_table(check_rows, check_fields),
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote ablation summary to: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']:14s} mean_best={fnum(row['mean_best_score']):.6f} "
            f"delta_vs_full={fnum(row['delta_vs_full']):+.6f} "
            f"elapsed={fnum(row['mean_elapsed_seconds']):.2f}s "
            f"speedup={fnum(row['speedup_vs_full']):.3f} "
            f"n={row['n_rows']} tasks={row['n_tasks']} seeds={row['n_seeds']} failed={row['n_failed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
