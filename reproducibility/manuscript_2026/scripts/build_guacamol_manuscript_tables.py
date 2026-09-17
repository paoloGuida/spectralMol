#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


TASK_FAMILIES = {
    0: "rediscovery",
    1: "rediscovery",
    2: "rediscovery",
    3: "similarity",
    4: "similarity",
    5: "similarity",
    6: "isomer",
    7: "isomer",
    8: "median",
    9: "median",
    10: "mpo",
    11: "mpo",
    12: "mpo",
    13: "mpo",
    14: "mpo",
    15: "mpo",
    16: "mpo",
    17: "smarts",
    18: "hopping",
    19: "hopping",
}


SUMMARY_NAMES = [
    "comparison_task_best_seed_summary.tsv",
    "comparison_generation_summary.tsv",
    "comparison_generation_raw.tsv",
    "comparison_generation_model_summary.tsv",
    "model_runtime_summary.tsv",
    "model_runtime.tsv",
    "comparison_raw.tsv",
]


def _float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else float("nan")


def _path_meta(path: Path) -> tuple[str, int]:
    text = str(path)
    seed_match = re.search(r"/seed_([^/]+)/", text)
    task_match = re.search(r"/task_(\d+)_", text)
    seed = seed_match.group(1) if seed_match else ""
    task_idx = int(task_match.group(1)) if task_match else -1
    return seed, task_idx


def _open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "wt") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def concat_tsv_files(paths: list[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    fieldnames: list[str] | None = None
    with _open_text(out_path, "wt") as out_f:
        writer: csv.DictWriter | None = None
        for path in paths:
            seed, task_idx = _path_meta(path)
            with path.open() as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames is None:
                    continue
                extra = ["seed", "task_index", "source_file"]
                if fieldnames is None:
                    fieldnames = extra + reader.fieldnames
                    writer = csv.DictWriter(out_f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                for row in reader:
                    row = dict(row)
                    row.update({"seed": seed, "task_index": task_idx, "source_file": str(path)})
                    assert writer is not None
                    writer.writerow(row)
                    written += 1
    return written


def load_best_summary(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_root.glob("seed_*/task_*/compare_GuacaMol_*/comparison_task_best_seed_summary.tsv")):
        seed, task_idx = _path_meta(path)
        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                score = _float(row.get("best_seed_score") or row.get("best_score"))
                rows.append(
                    {
                        "seed": seed,
                        "task_index": task_idx,
                        "task_family": TASK_FAMILIES.get(task_idx, "unknown"),
                        "model": row.get("model", ""),
                        "task": row.get("task", f"task_{task_idx}"),
                        "best_seed": row.get("best_seed", seed),
                        "best_seed_score": _fmt(score),
                        "best_seed_avg_score": row.get("best_seed_avg_score", ""),
                        "best_seed_n_scored": row.get("best_seed_n_scored", ""),
                        "n_runs": row.get("n_runs", ""),
                        "score_column": row.get("score_column", ""),
                        "scores_csv": row.get("scores_csv", ""),
                        "source_file": str(path),
                    }
                )
    return rows


def aggregate_scores(best_rows: list[dict], out_dir: Path) -> None:
    score_by_model_seed_task: dict[str, dict[str, dict[int, tuple[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in best_rows:
        model = row["model"]
        seed = row["seed"]
        task_idx = int(row["task_index"])
        score_by_model_seed_task[model][seed][task_idx] = (row["task"], _float(row["best_seed_score"]))

    seeds = sorted({row["seed"] for row in best_rows}, key=lambda s: int(s) if s.isdigit() else s)
    models = sorted({row["model"] for row in best_rows})

    per_seed_rows: list[dict] = []
    for seed in seeds:
        row = {"seed": seed}
        complete = True
        for model in models:
            task_scores = score_by_model_seed_task[model].get(seed, {})
            row[f"{model}_n_tasks"] = len(task_scores)
            if len(task_scores) == 20:
                total = sum(v for _, v in task_scores.values())
                row[f"{model}_aggregate"] = _fmt(total)
            else:
                row[f"{model}_aggregate"] = ""
                complete = False
        if "spectralmol" in models and "graphga" in models:
            s = _float(row.get("spectralmol_aggregate"))
            g = _float(row.get("graphga_aggregate"))
            row["delta_spectralmol_minus_graphga"] = _fmt(s - g) if not math.isnan(s) and not math.isnan(g) else ""
            row["spectralmol_win"] = int(s > g) if not math.isnan(s) and not math.isnan(g) else ""
        row["complete"] = int(complete)
        per_seed_rows.append(row)

    per_seed_fields = [
        "seed",
        "spectralmol_n_tasks",
        "spectralmol_aggregate",
        "graphga_n_tasks",
        "graphga_aggregate",
        "delta_spectralmol_minus_graphga",
        "spectralmol_win",
        "complete",
    ]
    write_tsv(out_dir / "guacamol_per_seed_aggregate.tsv", per_seed_rows, per_seed_fields)

    aggregate_rows: list[dict] = []
    for model in models:
        vals = [_float(r.get(f"{model}_aggregate")) for r in per_seed_rows if r.get("complete") == 1]
        vals = [v for v in vals if not math.isnan(v)]
        aggregate_rows.append(
            {
                "metric": f"{model}_aggregate",
                "n": len(vals),
                "mean": _fmt(mean(vals)) if vals else "",
                "std": _fmt(_std(vals)) if vals else "",
                "min": _fmt(min(vals)) if vals else "",
                "max": _fmt(max(vals)) if vals else "",
            }
        )
    deltas = [_float(r.get("delta_spectralmol_minus_graphga")) for r in per_seed_rows if r.get("complete") == 1]
    deltas = [v for v in deltas if not math.isnan(v)]
    aggregate_rows.append(
        {
            "metric": "delta_spectralmol_minus_graphga",
            "n": len(deltas),
            "mean": _fmt(mean(deltas)) if deltas else "",
            "std": _fmt(_std(deltas)) if deltas else "",
            "min": _fmt(min(deltas)) if deltas else "",
            "max": _fmt(max(deltas)) if deltas else "",
        }
    )
    aggregate_rows.append(
        {
            "metric": "spectralmol_seed_wins",
            "n": len(per_seed_rows),
            "mean": str(sum(int(r.get("spectralmol_win") or 0) for r in per_seed_rows)),
            "std": "",
            "min": "",
            "max": "",
        }
    )
    write_tsv(out_dir / "guacamol_table_update_summary.tsv", aggregate_rows, ["metric", "n", "mean", "std", "min", "max"])

    task_rows: list[dict] = []
    for task_idx in range(20):
        task_name = ""
        row = {"task_index": task_idx, "task_family": TASK_FAMILIES.get(task_idx, "unknown")}
        means: dict[str, float] = {}
        for model in models:
            vals: list[float] = []
            for seed in seeds:
                if task_idx in score_by_model_seed_task[model].get(seed, {}):
                    task, score = score_by_model_seed_task[model][seed][task_idx]
                    task_name = task_name or task
                    vals.append(score)
            means[model] = mean(vals) if vals else float("nan")
            row[f"{model}_mean"] = _fmt(means[model])
            row[f"{model}_std"] = _fmt(_std(vals))
            row[f"{model}_n"] = len(vals)
        row["task"] = task_name or f"task_{task_idx}"
        row["delta_spectralmol_minus_graphga_mean"] = _fmt(means.get("spectralmol", float("nan")) - means.get("graphga", float("nan")))
        task_rows.append(row)
    task_fields = [
        "task_index",
        "task",
        "task_family",
        "spectralmol_mean",
        "spectralmol_std",
        "spectralmol_n",
        "graphga_mean",
        "graphga_std",
        "graphga_n",
        "delta_spectralmol_minus_graphga_mean",
    ]
    write_tsv(out_dir / "guacamol_per_task_mean_scores.tsv", task_rows, task_fields)
    write_tsv(out_dir / "guacamol_figure3_task_comparison.tsv", task_rows, task_fields)


def top_n_scores(scores_csv: Path, score_column: str, n: int) -> list[dict]:
    heap: list[tuple[float, int, dict]] = []
    counter = 0
    with scores_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if score_column not in reader.fieldnames:
            # Fall back to common final-score columns used by MolScore.
            for candidate in ("filtered_single", "single", "Score", "score", "total_score", "gmean"):
                if candidate in reader.fieldnames:
                    score_column = candidate
                    break
        for row in reader:
            score = _float(row.get(score_column))
            if math.isnan(score):
                continue
            item = {
                "score": _fmt(score),
                "smiles": row.get("smiles", ""),
                "step": row.get("step", row.get("generation", "")),
                "batch_idx": row.get("batch_idx", ""),
                "valid": row.get("valid", ""),
                "unique": row.get("unique", ""),
                "qed": row.get("desc_QED", ""),
                "sa": row.get("desc_SAscore", ""),
                "molwt": row.get("desc_MolWt", ""),
            }
            packed = (score, counter, item)
            if len(heap) < n:
                heapq.heappush(heap, packed)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, packed)
            counter += 1
    return [item for _, _, item in sorted(heap, key=lambda x: x[0], reverse=True)]


def build_top200(best_rows: list[dict], out_path: Path, n: int) -> int:
    fields = [
        "model",
        "seed",
        "task_index",
        "task",
        "task_family",
        "rank",
        "score",
        "smiles",
        "step",
        "batch_idx",
        "valid",
        "unique",
        "qed",
        "sa",
        "molwt",
        "score_column",
        "scores_csv",
    ]
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(out_path, "wt") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in best_rows:
            scores_csv = Path(row["scores_csv"])
            if not scores_csv.exists():
                continue
            top_rows = top_n_scores(scores_csv, row.get("score_column", ""), n)
            for rank, top in enumerate(top_rows, 1):
                out = {
                    **top,
                    "model": row["model"],
                    "seed": row["seed"],
                    "task_index": row["task_index"],
                    "task": row["task"],
                    "task_family": row["task_family"],
                    "rank": rank,
                    "score_column": row.get("score_column", ""),
                    "scores_csv": row.get("scores_csv", ""),
                }
                writer.writerow(out)
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--extract-top200", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_rows = load_best_summary(input_root)
    best_fields = [
        "seed",
        "task_index",
        "task",
        "task_family",
        "model",
        "best_seed",
        "best_seed_score",
        "best_seed_avg_score",
        "best_seed_n_scored",
        "n_runs",
        "score_column",
        "scores_csv",
        "source_file",
    ]
    write_tsv(out_dir / "guacamol_all_task_best_seed_summary.tsv", best_rows, best_fields)
    aggregate_scores(best_rows, out_dir)

    counts = {"best_summary_rows": len(best_rows)}
    for name in SUMMARY_NAMES[1:]:
        paths = sorted(input_root.glob(f"seed_*/task_*/compare_GuacaMol_*/{name}"))
        suffix = ".tsv.gz" if name in {"comparison_generation_raw.tsv", "comparison_generation_summary.tsv"} else ".tsv"
        out_name = "guacamol_all_" + name.replace(".tsv", suffix)
        counts[out_name] = concat_tsv_files(paths, out_dir / out_name)

    score_index_fields = [
        "seed",
        "task_index",
        "task",
        "task_family",
        "model",
        "score_column",
        "scores_csv",
        "scores_csv_exists",
    ]
    score_index_rows = [
        {
            "seed": row["seed"],
            "task_index": row["task_index"],
            "task": row["task"],
            "task_family": row["task_family"],
            "model": row["model"],
            "score_column": row["score_column"],
            "scores_csv": row["scores_csv"],
            "scores_csv_exists": int(Path(row["scores_csv"]).exists()),
        }
        for row in best_rows
    ]
    write_tsv(out_dir / "guacamol_scores_csv_index.tsv", score_index_rows, score_index_fields)

    if args.extract_top200:
        counts["guacamol_top200_scores_by_seed_task_model.tsv.gz"] = build_top200(
            best_rows,
            out_dir / "guacamol_top200_scores_by_seed_task_model.tsv.gz",
            args.top_n,
        )

    manifest = {
        "input_root": str(input_root),
        "output_dir": str(out_dir),
        "summary_file_count": len(list(input_root.glob("seed_*/task_*/compare_GuacaMol_*/comparison_task_best_seed_summary.tsv"))),
        "counts": counts,
    }
    (out_dir / "guacamol_table_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
