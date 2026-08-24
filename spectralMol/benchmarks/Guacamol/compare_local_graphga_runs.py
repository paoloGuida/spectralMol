#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LOCAL_MODEL_KEY = "local_evolution"
GRAPH_MODEL_KEY = "graphga_example"

META_NUMERIC_BLACKLIST = {
    "smiles",
    "batch_idx",
    "absolute_time",
    "total_time",
    "batch_time",
    "score_time",
    "score_time_mean",
    "score_time_sum",
    "n_jobs",
    "step",
    "iteration",
    "iter",
    "rank",
    "index",
    "idx",
    "job_id",
    "task_id",
    "generation",
    "gen",
    "is_valid",
    "valid",
    "validity",
    "passes_filters",
    "filter",
    "valid_score",
    "raw_score",
    "score_component",
    "elapsed_time",
    "elapsed_seconds",
    "wall_time",
}

PREFERRED_SCORE_COLUMNS = (
    "filtered_score",
    "score",
    "Score",
    "total_score",
    "filtered_single",
    "single",
    "filtered_gmean",
    "gmean",
    "filtered_amean",
    "amean",
    "filtered_wmean",
    "wmean",
    "filtered_wsum",
    "wsum",
    "filtered_sum",
    "sum",
)


@dataclass
class TaskMetric:
    task: str
    best_score: float | None
    runtime_seconds: float | None
    score_column: str | None
    score_source: str


@dataclass
class BenchmarkCompareResult:
    benchmark: str
    local_compare_dir: Path
    graph_compare_dir: Path
    seed: int
    rows: list[dict[str, object]]
    total_local_score: float
    total_graph_score: float
    total_local_runtime: float | None
    total_graph_runtime: float | None
    local_model_runtime_seconds: float | None
    graph_model_runtime_seconds: float | None
    local_model_runtime_status: str | None
    graph_model_runtime_status: str | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare one local_evolution run and one graphga_example run "
            "from output_single_model_all_benchmarks."
        )
    )
    parser.add_argument(
        "--root",
        default=str(repo_root / "output_single_model_all_benchmarks"),
        help="Path to output_single_model_all_benchmarks directory.",
    )
    parser.add_argument("--local-run", default="", help="Local-evolution run dir name (e.g. run_local_evolution_20260312_143403).")
    parser.add_argument("--graph-run", default="", help="GraphGA run dir name (e.g. run_graphga_example_20260309_114955).")
    parser.add_argument(
        "--benchmark",
        default="GuacaMol",
        help="Benchmark name, comma list, or 'all' for all common benchmarks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to compare (default: auto-select common seed).",
    )
    parser.add_argument(
        "--exact-scores",
        action="store_true",
        help="Read per-task scores.csv for best scores (more accurate, slower).",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available local/graph runs and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs" / "run_pair_comparisons"),
        help="Directory where comparison reports will be written.",
    )
    return parser.parse_args()


def parse_seed_from_dirname(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    raw = name.split("_", 1)[1]
    try:
        return int(raw)
    except ValueError:
        return None


def parse_float(raw: object) -> float | None:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def sec_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem = seconds - (hours * 3600 + minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{rem:05.2f}"


def list_model_runs(root: Path, model_key: str) -> list[Path]:
    model_dir = root / model_key
    if not model_dir.exists():
        return []
    runs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def list_benchmarks(run_dir: Path) -> list[str]:
    benchmarks: list[str] = []
    for candidate in sorted(run_dir.iterdir()):
        if not candidate.is_dir():
            continue
        has_compare = any(x.is_dir() and x.name.startswith("compare_") for x in candidate.iterdir())
        if has_compare:
            benchmarks.append(candidate.name)
    return benchmarks


def choose_latest_compare_dir(bench_dir: Path) -> Path:
    compare_dirs = [p for p in bench_dir.iterdir() if p.is_dir() and p.name.startswith("compare_")]
    if not compare_dirs:
        raise FileNotFoundError(f"No compare_* directory found under {bench_dir}")
    compare_dirs.sort(key=lambda p: p.name)
    return compare_dirs[-1]


def choose_run_interactive(label: str, runs: list[Path]) -> str:
    if not runs:
        raise RuntimeError(f"No {label} runs found.")
    print(f"\nAvailable {label} runs:")
    for idx, run in enumerate(runs, start=1):
        print(f"  {idx:2d}. {run.name}")
    prompt = f"Select {label} run [1]: "
    choice = input(prompt).strip()
    if not choice:
        return runs[0].name
    pick = int(choice)
    if pick < 1 or pick > len(runs):
        raise ValueError(f"Invalid selection for {label}: {choice}")
    return runs[pick - 1].name


def pick_run_name(
    provided: str,
    label: str,
    runs: list[Path],
) -> str:
    if provided:
        names = {p.name for p in runs}
        if provided not in names:
            raise FileNotFoundError(f"{label} run not found: {provided}")
        return provided
    if sys.stdin.isatty():
        return choose_run_interactive(label, runs)
    if not runs:
        raise RuntimeError(f"No {label} runs available for auto-select.")
    selected = runs[0].name
    print(f"[info] stdin is not interactive; auto-selected latest {label} run: {selected}")
    return selected


def detect_task_column(fieldnames: Iterable[str]) -> str:
    fields = list(fieldnames)
    lower_to_raw = {f.lower(): f for f in fields}
    for key in ("task", "benchmark_task", "target"):
        if key in lower_to_raw:
            return lower_to_raw[key]
    raise ValueError(f"Could not detect task column in fields: {fields}")


def detect_results_best_score_column(fieldnames: Iterable[str]) -> str | None:
    fields = list(fieldnames)
    lower_to_raw = {f.lower(): f for f in fields}

    for key in ("best_score", "score", "totalscore", "total_score", "guacamol_score"):
        if key in lower_to_raw:
            return lower_to_raw[key]

    score_suffix = [f for f in fields if f.lower().endswith("_score")]
    if score_suffix:
        score_suffix.sort(key=lambda x: (x.lower().startswith("guacamol"), x.lower()), reverse=True)
        return score_suffix[0]

    for key in PREFERRED_SCORE_COLUMNS:
        if key.lower() in lower_to_raw:
            return lower_to_raw[key.lower()]
    return None


def extract_task_scoring_method(task_dir: Path) -> str | None:
    config_files = sorted(task_dir.glob("*_config.json"))
    for cfg in config_files:
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        scoring = data.get("scoring")
        if isinstance(scoring, dict):
            method = scoring.get("method")
            if isinstance(method, str) and method.strip():
                return method.strip()
    return None


def detect_scores_csv_column(header: list[str], preferred_method: str | None = None) -> tuple[int, str] | None:
    lower_to_idx = {col.lower(): i for i, col in enumerate(header)}

    preferred: list[str] = []
    if preferred_method:
        preferred.append(f"filtered_{preferred_method}")
        preferred.append(preferred_method)
    preferred.extend(PREFERRED_SCORE_COLUMNS)

    for key in preferred:
        idx = lower_to_idx.get(key.lower())
        if idx is not None:
            return idx, header[idx]

    candidates: list[tuple[int, str]] = []
    for i, col in enumerate(header):
        cl = col.lower()
        if cl in META_NUMERIC_BLACKLIST:
            continue
        if cl.startswith("filter_") and cl not in {
            "filtered_score",
            "filtered_single",
            "filtered_gmean",
            "filtered_amean",
            "filtered_wsum",
            "filtered_wmean",
            "filtered_sum",
        }:
            continue
        if cl.endswith("_time") or cl.endswith("_sec") or cl.endswith("_seconds"):
            continue
        candidates.append((i, col))

    for i, col in candidates:
        cl = col.lower()
        if "score" in cl or "mean" in cl or "objective" in cl or "fitness" in cl:
            return i, col

    if candidates:
        return candidates[0]
    return None


def compute_exact_best_score(task_dir: Path) -> tuple[float | None, str | None]:
    scores_csv = task_dir / "scores.csv"
    if not scores_csv.exists():
        return None, None

    preferred_method = extract_task_scoring_method(task_dir)
    with scores_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return None, None

        selected = detect_scores_csv_column(header, preferred_method=preferred_method)
        if selected is None:
            return None, None
        idx, score_col = selected

        best: float | None = None
        for row in reader:
            if idx >= len(row):
                continue
            value = parse_float(row[idx])
            if value is None:
                continue
            if best is None or value > best:
                best = value
    return best, score_col


def find_seed_dir(runs_model_dir: Path, requested_seed: int | None = None) -> tuple[int, Path]:
    seed_dirs: list[tuple[int, Path]] = []
    if not runs_model_dir.exists():
        raise FileNotFoundError(f"Missing runs model dir: {runs_model_dir}")
    for candidate in runs_model_dir.iterdir():
        if not candidate.is_dir():
            continue
        seed = parse_seed_from_dirname(candidate.name)
        if seed is None:
            continue
        seed_dirs.append((seed, candidate))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs found under: {runs_model_dir}")
    seed_dirs.sort(key=lambda x: x[0])
    if requested_seed is not None:
        for seed, path in seed_dirs:
            if seed == requested_seed:
                return seed, path
        raise FileNotFoundError(f"Requested seed {requested_seed} not found under {runs_model_dir}")
    return seed_dirs[0]


def find_results_csv(seed_dir: Path) -> Path:
    direct = [p for p in seed_dir.glob("*/results.csv") if p.is_file() and p.name == "results.csv"]
    if direct:
        direct.sort(key=lambda p: p.stat().st_mtime)
        return direct[-1]
    recursive = [p for p in seed_dir.rglob("results.csv") if p.is_file() and p.name == "results.csv"]
    if not recursive:
        raise FileNotFoundError(f"No results.csv found under {seed_dir}")
    recursive.sort(key=lambda p: p.stat().st_mtime)
    return recursive[-1]


def resolve_task_dir(
    benchmark_run_dir: Path,
    task_name: str,
    run_dir_hint: str | None = None,
) -> Path | None:
    if run_dir_hint:
        hinted = Path(run_dir_hint)
        if hinted.exists():
            return hinted
        local_name = hinted.name
        local_candidate = benchmark_run_dir / local_name
        if local_candidate.exists():
            return local_candidate

    candidates = [p for p in benchmark_run_dir.glob(f"*_{task_name}") if p.is_dir()]
    if candidates:
        candidates.sort(key=lambda p: p.name)
        return candidates[-1]

    # Fallback for uncommon naming.
    deep = [p for p in benchmark_run_dir.rglob("*") if p.is_dir() and p.name.endswith(f"_{task_name}")]
    if deep:
        deep.sort(key=lambda p: p.name)
        return deep[-1]
    return None


def infer_graphga_runtime_from_log(
    compare_dir: Path,
    model_key: str,
    seed: int,
    task_order: list[str],
) -> dict[str, float]:
    log_file = compare_dir / "logs" / f"{model_key}_seed_{seed}.log"
    if not log_file.exists():
        return {}

    pattern = re.compile(r"^(\d+)\s+\|\s+max:.*\|\s+([0-9.]+)\s+sec/gen\s+\|")
    segments: list[list[float]] = []
    current: list[float] = []
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        gen = int(match.group(1))
        sec = float(match.group(2))
        if gen == 0 and current:
            segments.append(current)
            current = []
        current.append(sec)
    if current:
        segments.append(current)

    if len(segments) != len(task_order):
        return {}

    runtimes: dict[str, float] = {}
    for task, segment in zip(task_order, segments):
        runtimes[task] = float(sum(segment))
    return runtimes


def read_model_runtime_for_seed(compare_dir: Path, seed: int) -> tuple[float | None, str | None]:
    model_runtime_tsv = compare_dir / "model_runtime.tsv"
    if not model_runtime_tsv.exists():
        return None, None
    with model_runtime_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fallback: tuple[float | None, str | None] = (None, None)
        for row in reader:
            elapsed = parse_float(row.get("elapsed_seconds"))
            status = row.get("status")
            row_seed = parse_float(row.get("seed"))
            if fallback[0] is None:
                fallback = (elapsed, status)
            if row_seed is not None and int(row_seed) == seed:
                return elapsed, status
    return fallback


def extract_task_metrics(
    compare_dir: Path,
    model_key: str,
    seed: int,
    exact_scores: bool,
) -> tuple[dict[str, TaskMetric], list[str], Path]:
    model_seed_root = compare_dir / "runs" / model_key / f"seed_{seed}"
    if not model_seed_root.exists():
        raise FileNotFoundError(f"Missing seed dir: {model_seed_root}")
    results_csv = find_results_csv(model_seed_root)
    benchmark_run_dir = results_csv.parent

    with results_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        task_col = detect_task_column(fieldnames)
        score_col = detect_results_best_score_column(fieldnames)
        if score_col is None:
            raise ValueError(f"No score column detected in {results_csv}")

        task_order: list[str] = []
        metrics: dict[str, TaskMetric] = {}
        run_dir_col = next((c for c in fieldnames if c.lower() == "run_dir"), None)
        elapsed_col = next((c for c in fieldnames if c.lower() == "elapsed_seconds"), None)

        for row in reader:
            task = str(row.get(task_col, "")).strip()
            if not task:
                continue
            task_order.append(task)
            best_score = parse_float(row.get(score_col))
            runtime = parse_float(row.get(elapsed_col)) if elapsed_col else None
            score_source = "results.csv"
            selected_score_col = score_col

            if exact_scores:
                run_dir_hint = str(row.get(run_dir_col, "")).strip() if run_dir_col else None
                task_dir = resolve_task_dir(benchmark_run_dir, task, run_dir_hint=run_dir_hint)
                if task_dir is not None:
                    exact_best, exact_col = compute_exact_best_score(task_dir)
                    if exact_best is not None:
                        best_score = exact_best
                        selected_score_col = exact_col
                        score_source = "scores.csv"
                    elif exact_col:
                        selected_score_col = exact_col

            metrics[task] = TaskMetric(
                task=task,
                best_score=best_score,
                runtime_seconds=runtime,
                score_column=selected_score_col,
                score_source=score_source,
            )

    if model_key == GRAPH_MODEL_KEY and any(m.runtime_seconds is None for m in metrics.values()):
        inferred = infer_graphga_runtime_from_log(compare_dir, model_key=model_key, seed=seed, task_order=task_order)
        for task, seconds in inferred.items():
            metric = metrics.get(task)
            if metric is not None and metric.runtime_seconds is None:
                metric.runtime_seconds = seconds
    return metrics, task_order, benchmark_run_dir


def select_benchmarks(local_run_dir: Path, graph_run_dir: Path, benchmark_arg: str) -> list[str]:
    local_bench = set(list_benchmarks(local_run_dir))
    graph_bench = set(list_benchmarks(graph_run_dir))
    common = sorted(local_bench & graph_bench)
    if not common:
        raise RuntimeError("No common benchmarks found between selected runs.")

    raw = benchmark_arg.strip()
    if raw.lower() == "all":
        return common

    requested = [x.strip() for x in raw.split(",") if x.strip()]
    missing = [b for b in requested if b not in common]
    if missing:
        raise ValueError(f"Requested benchmark(s) not common to both runs: {missing}. Common: {common}")
    return requested


def choose_common_seed(local_compare_dir: Path, graph_compare_dir: Path, requested_seed: int | None) -> int:
    local_seeds = {
        seed
        for seed, _p in sorted(
            (
                (parse_seed_from_dirname(p.name), p)
                for p in (local_compare_dir / "runs" / LOCAL_MODEL_KEY).iterdir()
                if p.is_dir()
            ),
            key=lambda x: (x[0] is None, x[0]),
        )
        if seed is not None
    }
    graph_seeds = {
        seed
        for seed, _p in sorted(
            (
                (parse_seed_from_dirname(p.name), p)
                for p in (graph_compare_dir / "runs" / GRAPH_MODEL_KEY).iterdir()
                if p.is_dir()
            ),
            key=lambda x: (x[0] is None, x[0]),
        )
        if seed is not None
    }
    common = sorted(local_seeds & graph_seeds)
    if not common:
        raise RuntimeError(
            f"No common seeds between {local_compare_dir} and {graph_compare_dir}. "
            f"local seeds={sorted(local_seeds)}, graph seeds={sorted(graph_seeds)}"
        )
    if requested_seed is not None:
        if requested_seed not in common:
            raise RuntimeError(
                f"Requested seed {requested_seed} not available in both runs. "
                f"Common seeds: {common}"
            )
        return requested_seed
    return common[0]


def compare_one_benchmark(
    benchmark: str,
    local_compare_dir: Path,
    graph_compare_dir: Path,
    seed: int,
    exact_scores: bool,
) -> BenchmarkCompareResult:
    local_metrics, _local_order, _local_bench_dir = extract_task_metrics(
        local_compare_dir,
        model_key=LOCAL_MODEL_KEY,
        seed=seed,
        exact_scores=exact_scores,
    )
    graph_metrics, _graph_order, _graph_bench_dir = extract_task_metrics(
        graph_compare_dir,
        model_key=GRAPH_MODEL_KEY,
        seed=seed,
        exact_scores=exact_scores,
    )

    tasks = sorted(set(local_metrics) | set(graph_metrics))
    rows: list[dict[str, object]] = []

    total_local_score = 0.0
    total_graph_score = 0.0
    local_runtime_acc = 0.0
    graph_runtime_acc = 0.0
    have_local_runtime = False
    have_graph_runtime = False

    for task in tasks:
        local = local_metrics.get(task)
        graph = graph_metrics.get(task)
        local_score = local.best_score if local else None
        graph_score = graph.best_score if graph else None
        score_delta = None
        if local_score is not None and graph_score is not None:
            score_delta = local_score - graph_score
        if local_score is not None:
            total_local_score += local_score
        if graph_score is not None:
            total_graph_score += graph_score

        local_runtime = local.runtime_seconds if local else None
        graph_runtime = graph.runtime_seconds if graph else None
        runtime_delta = None
        if local_runtime is not None and graph_runtime is not None:
            runtime_delta = local_runtime - graph_runtime
        if local_runtime is not None:
            local_runtime_acc += local_runtime
            have_local_runtime = True
        if graph_runtime is not None:
            graph_runtime_acc += graph_runtime
            have_graph_runtime = True

        rows.append(
            {
                "task": task,
                "local_best_score": local_score,
                "graphga_best_score": graph_score,
                "best_score_delta_local_minus_graphga": score_delta,
                "local_runtime_seconds": local_runtime,
                "graphga_runtime_seconds": graph_runtime,
                "runtime_delta_seconds_local_minus_graphga": runtime_delta,
                "local_runtime_hms": sec_to_hms(local_runtime),
                "graphga_runtime_hms": sec_to_hms(graph_runtime),
                "local_score_column": local.score_column if local else None,
                "graphga_score_column": graph.score_column if graph else None,
                "local_score_source": local.score_source if local else None,
                "graphga_score_source": graph.score_source if graph else None,
            }
        )

    total_local_runtime = local_runtime_acc if have_local_runtime else None
    total_graph_runtime = graph_runtime_acc if have_graph_runtime else None

    local_model_runtime, local_model_status = read_model_runtime_for_seed(local_compare_dir, seed)
    graph_model_runtime, graph_model_status = read_model_runtime_for_seed(graph_compare_dir, seed)

    rows.append(
        {
            "task": "__TOTAL__",
            "local_best_score": total_local_score,
            "graphga_best_score": total_graph_score,
            "best_score_delta_local_minus_graphga": total_local_score - total_graph_score,
            "local_runtime_seconds": total_local_runtime,
            "graphga_runtime_seconds": total_graph_runtime,
            "runtime_delta_seconds_local_minus_graphga": (
                (total_local_runtime - total_graph_runtime)
                if (total_local_runtime is not None and total_graph_runtime is not None)
                else None
            ),
            "local_runtime_hms": sec_to_hms(total_local_runtime),
            "graphga_runtime_hms": sec_to_hms(total_graph_runtime),
            "local_score_column": None,
            "graphga_score_column": None,
            "local_score_source": None,
            "graphga_score_source": None,
        }
    )

    return BenchmarkCompareResult(
        benchmark=benchmark,
        local_compare_dir=local_compare_dir,
        graph_compare_dir=graph_compare_dir,
        seed=seed,
        rows=rows,
        total_local_score=total_local_score,
        total_graph_score=total_graph_score,
        total_local_runtime=total_local_runtime,
        total_graph_runtime=total_graph_runtime,
        local_model_runtime_seconds=local_model_runtime,
        graph_model_runtime_seconds=graph_model_runtime,
        local_model_runtime_status=local_model_status,
        graph_model_runtime_status=graph_model_status,
    )


def write_benchmark_report(result: BenchmarkCompareResult, out_dir: Path, exact_scores: bool) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_tsv = out_dir / f"{result.benchmark}_task_comparison.tsv"
    summary_txt = out_dir / f"{result.benchmark}_summary.txt"

    fields = [
        "task",
        "local_best_score",
        "graphga_best_score",
        "best_score_delta_local_minus_graphga",
        "local_runtime_seconds",
        "graphga_runtime_seconds",
        "runtime_delta_seconds_local_minus_graphga",
        "local_runtime_hms",
        "graphga_runtime_hms",
        "local_score_column",
        "graphga_score_column",
        "local_score_source",
        "graphga_score_source",
    ]
    with report_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(result.rows)

    with summary_txt.open("w", encoding="utf-8") as fh:
        fh.write(f"benchmark\t{result.benchmark}\n")
        fh.write(f"seed\t{result.seed}\n")
        fh.write(f"score_mode\t{'exact' if exact_scores else 'results'}\n")
        fh.write(f"local_compare_dir\t{result.local_compare_dir}\n")
        fh.write(f"graph_compare_dir\t{result.graph_compare_dir}\n")
        fh.write(f"local_total_best_score\t{result.total_local_score:.12f}\n")
        fh.write(f"graphga_total_best_score\t{result.total_graph_score:.12f}\n")
        fh.write(f"best_score_delta_local_minus_graphga\t{(result.total_local_score - result.total_graph_score):.12f}\n")
        fh.write(f"local_total_runtime_seconds\t{'' if result.total_local_runtime is None else f'{result.total_local_runtime:.6f}'}\n")
        fh.write(f"graphga_total_runtime_seconds\t{'' if result.total_graph_runtime is None else f'{result.total_graph_runtime:.6f}'}\n")
        fh.write(
            "runtime_delta_seconds_local_minus_graphga\t"
            + (
                ""
                if (result.total_local_runtime is None or result.total_graph_runtime is None)
                else f"{(result.total_local_runtime - result.total_graph_runtime):.6f}"
            )
            + "\n"
        )
        fh.write(f"local_total_runtime_hms\t{sec_to_hms(result.total_local_runtime)}\n")
        fh.write(f"graphga_total_runtime_hms\t{sec_to_hms(result.total_graph_runtime)}\n")
        fh.write(
            f"local_model_runtime_seconds(model_runtime.tsv)\t"
            f"{'' if result.local_model_runtime_seconds is None else f'{result.local_model_runtime_seconds:.6f}'}\n"
        )
        fh.write(
            f"graphga_model_runtime_seconds(model_runtime.tsv)\t"
            f"{'' if result.graph_model_runtime_seconds is None else f'{result.graph_model_runtime_seconds:.6f}'}\n"
        )
        fh.write(f"local_model_status\t{result.local_model_runtime_status or ''}\n")
        fh.write(f"graph_model_status\t{result.graph_model_runtime_status or ''}\n")
    return report_tsv, summary_txt


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] root directory not found: {root}", file=sys.stderr)
        return 1

    local_runs = list_model_runs(root, LOCAL_MODEL_KEY)
    graph_runs = list_model_runs(root, GRAPH_MODEL_KEY)

    if args.list_runs:
        print(f"Root: {root}")
        print(f"\n{LOCAL_MODEL_KEY} runs ({len(local_runs)}):")
        for run in local_runs:
            print(f"  - {run.name}")
        print(f"\n{GRAPH_MODEL_KEY} runs ({len(graph_runs)}):")
        for run in graph_runs:
            print(f"  - {run.name}")
        return 0

    try:
        local_run_name = pick_run_name(args.local_run.strip(), LOCAL_MODEL_KEY, local_runs)
        graph_run_name = pick_run_name(args.graph_run.strip(), GRAPH_MODEL_KEY, graph_runs)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    local_run_dir = root / LOCAL_MODEL_KEY / local_run_name
    graph_run_dir = root / GRAPH_MODEL_KEY / graph_run_name

    try:
        benchmarks = select_benchmarks(local_run_dir, graph_run_dir, args.benchmark)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_root = Path(args.output_dir).resolve() / f"{local_run_name}__vs__{graph_run_name}__{timestamp}"
    report_root.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    print(f"[info] local run : {local_run_name}")
    print(f"[info] graph run : {graph_run_name}")
    print(f"[info] benchmarks: {', '.join(benchmarks)}")
    print(f"[info] output    : {report_root}")
    print(f"[info] score mode: {'exact' if args.exact_scores else 'results'}")

    for benchmark in benchmarks:
        local_compare = choose_latest_compare_dir(local_run_dir / benchmark)
        graph_compare = choose_latest_compare_dir(graph_run_dir / benchmark)

        try:
            seed = choose_common_seed(local_compare, graph_compare, args.seed)
            result = compare_one_benchmark(
                benchmark=benchmark,
                local_compare_dir=local_compare,
                graph_compare_dir=graph_compare,
                seed=seed,
                exact_scores=args.exact_scores,
            )
            bench_dir = report_root / benchmark
            report_tsv, summary_txt = write_benchmark_report(result, bench_dir, exact_scores=args.exact_scores)
            index_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "local_total_best_score": result.total_local_score,
                    "graphga_total_best_score": result.total_graph_score,
                    "best_score_delta_local_minus_graphga": result.total_local_score - result.total_graph_score,
                    "local_total_runtime_seconds": result.total_local_runtime,
                    "graphga_total_runtime_seconds": result.total_graph_runtime,
                    "runtime_delta_seconds_local_minus_graphga": (
                        None
                        if (result.total_local_runtime is None or result.total_graph_runtime is None)
                        else (result.total_local_runtime - result.total_graph_runtime)
                    ),
                    "report_tsv": str(report_tsv),
                    "summary_txt": str(summary_txt),
                }
            )
            print(f"[ok] {benchmark}: {report_tsv}")
        except Exception as exc:
            print(f"[error] benchmark {benchmark} failed: {exc}", file=sys.stderr)
            index_rows.append(
                {
                    "benchmark": benchmark,
                    "seed": "",
                    "local_total_best_score": "",
                    "graphga_total_best_score": "",
                    "best_score_delta_local_minus_graphga": "",
                    "local_total_runtime_seconds": "",
                    "graphga_total_runtime_seconds": "",
                    "runtime_delta_seconds_local_minus_graphga": "",
                    "report_tsv": "",
                    "summary_txt": "",
                    "error": str(exc),
                }
            )

    index_tsv = report_root / "comparison_index.tsv"
    index_fields = [
        "benchmark",
        "seed",
        "local_total_best_score",
        "graphga_total_best_score",
        "best_score_delta_local_minus_graphga",
        "local_total_runtime_seconds",
        "graphga_total_runtime_seconds",
        "runtime_delta_seconds_local_minus_graphga",
        "report_tsv",
        "summary_txt",
        "error",
    ]
    with index_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=index_fields, delimiter="\t")
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    print(f"[done] index: {index_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
