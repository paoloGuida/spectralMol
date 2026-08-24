#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


SATURN_REFERENCE = {
    -9.0: {
        "method": "Saturn*",
        "successful_replicates": 10,
        "modes_mean": 38.0,
        "modes_std": 13.0,
        "yield_mean": 83.0,
        "yield_std": 33.0,
        "qed_mean": 0.80,
        "qed_std": 0.05,
        "sa_mean": 2.10,
        "sa_std": 0.10,
        "molwt_mean": 343.3,
        "molwt_std": 6.2,
        "source": "Guo et al. Saturn Experiment 2, as reported in manuscript Table 8",
    },
    -10.0: {
        "method": "Saturn*",
        "successful_replicates": 9,
        "modes_mean": 3.0,
        "modes_std": 1.0,
        "yield_mean": 4.0,
        "yield_std": 2.0,
        "qed_mean": 0.72,
        "qed_std": 0.16,
        "sa_mean": 2.26,
        "sa_std": 0.16,
        "molwt_mean": 379.9,
        "molwt_std": 18.8,
        "source": "Guo et al. Saturn Experiment 2, as reported in manuscript Table 8",
    },
}


@dataclass(frozen=True)
class MoleculeRow:
    seed: int
    budget: int
    strategy: str
    source_file: str
    population_stage: str
    rank_by_scalar: int
    smiles: str
    scaffold: str
    scalar_objective: float
    docking_raw: float
    qed: float
    sa: float
    molwt: float
    decode_reason: str
    macro_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate SpectralMol theta NSGA-II SATURN outputs into manuscript Table 8 "
            "replacement data and figure-ready TSV files."
        )
    )
    p.add_argument(
        "--input-root",
        action="append",
        required=True,
        help=(
            "Root containing compare_scalar_vs_nsga2_saturn.py outputs. Can be repeated. "
            "The script searches recursively for theta_nsga2_multiobjective CSV files."
        ),
    )
    p.add_argument("--output-dir", required=True, help="Directory where reproducibility data will be written.")
    p.add_argument("--strategy", default="theta_nsga2_multiobjective", help="Strategy directory/name to aggregate.")
    p.add_argument(
        "--expected-seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated expected seeds. Defaults to old Saturn benchmark seed_0.smi ... seed_9.smi.",
    )
    p.add_argument("--budget", type=int, default=1000, help="Expected oracle-call budget.")
    p.add_argument("--thresholds", default="-9,-10", help="Comma-separated docking thresholds in kcal/mol.")
    return p.parse_args()


def parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_csv(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def detect_column(fieldnames: Iterable[str], *, contains: list[str], suffix: str | None = None) -> str | None:
    fields = list(fieldnames)
    lowered = [(f, f.lower()) for f in fields]
    for f, low in lowered:
        if suffix and not low.endswith(suffix.lower()):
            continue
        if all(token.lower() in low for token in contains):
            return f
    return None


def molwt_from_smiles(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return float(Descriptors.MolWt(mol))


def infer_seed_budget(path: Path, row: dict[str, str]) -> tuple[int, int]:
    seed = safe_int(row.get("seed"), -1)
    budget = safe_int(row.get("budget"), -1)
    parts = path.parts
    for part in parts:
        if seed < 0 and part.startswith("seed_"):
            seed = safe_int(part.replace("seed_", ""), seed)
        if budget < 0 and part.startswith("budget_"):
            budget = safe_int(part.replace("budget_", ""), budget)
    return seed, budget


def load_molecule_file(path: Path, *, stage: str, strategy: str) -> list[MoleculeRow]:
    rows = read_csv(path)
    if not rows:
        return []
    fields = rows[0].keys()
    docking_col = (
        detect_column(fields, contains=["vina"], suffix="_raw")
        or detect_column(fields, contains=["dock"], suffix="_raw")
        or detect_column(fields, contains=["quickvina"], suffix="_raw")
    )
    qed_col = detect_column(fields, contains=["qed"], suffix="_raw") or detect_column(fields, contains=["qed"], suffix="_reward")
    sa_col = (
        detect_column(fields, contains=["sa_score"], suffix="_raw")
        or detect_column(fields, contains=["sa"], suffix="_raw")
        or detect_column(fields, contains=["synth"], suffix="_raw")
    )
    missing = [name for name, col in (("docking", docking_col), ("qed", qed_col), ("sa", sa_col)) if col is None]
    if missing:
        raise ValueError(f"{path} is missing required component columns: {', '.join(missing)}")

    out: list[MoleculeRow] = []
    for row in rows:
        smiles = str(row.get("smiles", "")).strip()
        if not smiles:
            continue
        seed, budget = infer_seed_budget(path, row)
        out.append(
            MoleculeRow(
                seed=seed,
                budget=budget,
                strategy=str(row.get("strategy", strategy) or strategy),
                source_file=str(path),
                population_stage=stage,
                rank_by_scalar=safe_int(row.get("rank_by_scalar"), 0),
                smiles=smiles,
                scaffold=str(row.get("scaffold", "") or smiles),
                scalar_objective=safe_float(row.get("scalar_objective")),
                docking_raw=safe_float(row.get(docking_col)),
                qed=safe_float(row.get(qed_col)),
                sa=safe_float(row.get(sa_col)),
                molwt=molwt_from_smiles(smiles),
                decode_reason=str(row.get("decode_reason", "")),
                macro_count=safe_int(row.get("macro_count"), 0),
            )
        )
    return out


def find_input_files(input_roots: list[Path], strategy: str) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    molecules: list[Path] = []
    initial: list[Path] = []
    final: list[Path] = []
    summaries: list[Path] = []
    for root in input_roots:
        molecules.extend(root.glob(f"**/{strategy}/molecules_all.csv"))
        initial.extend(root.glob(f"**/{strategy}/snapshots/initial_population.csv"))
        final.extend(root.glob(f"**/{strategy}/snapshots/final_population.csv"))
        summaries.extend(root.glob("**/scalar_vs_nsga2_summary.csv"))
    return sorted(set(molecules)), sorted(set(initial)), sorted(set(final)), sorted(set(summaries))


def row_to_dict(row: MoleculeRow) -> dict[str, Any]:
    return {
        "seed": row.seed,
        "budget": row.budget,
        "strategy": row.strategy,
        "population_stage": row.population_stage,
        "rank_by_scalar": row.rank_by_scalar,
        "smiles": row.smiles,
        "scaffold": row.scaffold,
        "scalar_objective": row.scalar_objective,
        "docking_raw": row.docking_raw,
        "qed": row.qed,
        "sa": row.sa,
        "molwt": row.molwt,
        "decode_reason": row.decode_reason,
        "macro_count": row.macro_count,
        "source_file": row.source_file,
    }


def table8_rows(molecules: list[MoleculeRow], expected_seeds: list[int], thresholds: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    by_seed = {seed: [m for m in molecules if m.seed == seed] for seed in expected_seeds}

    for threshold in thresholds:
        seed_modes: list[float] = []
        seed_yields: list[float] = []
        seed_qed_means: list[float] = []
        seed_sa_means: list[float] = []
        seed_molwt_means: list[float] = []
        successful = 0
        pooled_hits: list[MoleculeRow] = []
        for seed in expected_seeds:
            hits = [m for m in by_seed.get(seed, []) if math.isfinite(m.docking_raw) and m.docking_raw < threshold]
            pooled_hits.extend(hits)
            modes = len({m.scaffold for m in hits if m.scaffold})
            yld = len(hits)
            seed_modes.append(float(modes))
            seed_yields.append(float(yld))
            if yld > 0:
                successful += 1
                q_mean, _ = mean_std(m.qed for m in hits)
                sa_mean, _ = mean_std(m.sa for m in hits)
                mw_mean, _ = mean_std(m.molwt for m in hits)
                seed_qed_means.append(q_mean)
                seed_sa_means.append(sa_mean)
                seed_molwt_means.append(mw_mean)
            per_seed_rows.append(
                {
                    "threshold": threshold,
                    "seed": seed,
                    "hit_yield": yld,
                    "modes": modes,
                    "qed_mean_hits": seed_qed_means[-1] if yld > 0 else "",
                    "sa_mean_hits": seed_sa_means[-1] if yld > 0 else "",
                    "molwt_mean_hits": seed_molwt_means[-1] if yld > 0 else "",
                }
            )

        modes_mean, modes_std = mean_std(seed_modes)
        yield_mean, yield_std = mean_std(seed_yields)
        qed_mean, qed_std = mean_std(seed_qed_means)
        sa_mean, sa_std = mean_std(seed_sa_means)
        mw_mean, mw_std = mean_std(seed_molwt_means)
        pooled_qed_mean, pooled_qed_std = mean_std(m.qed for m in pooled_hits)
        pooled_sa_mean, pooled_sa_std = mean_std(m.sa for m in pooled_hits)
        pooled_mw_mean, pooled_mw_std = mean_std(m.molwt for m in pooled_hits)
        table_rows.append(
            {
                "threshold": threshold,
                "method": "SpectralMol",
                "successful_replicates": successful,
                "n_expected_seeds": len(expected_seeds),
                "modes_mean": modes_mean,
                "modes_std": modes_std,
                "yield_mean": yield_mean,
                "yield_std": yield_std,
                "qed_mean": qed_mean,
                "qed_std": qed_std,
                "sa_mean": sa_mean,
                "sa_std": sa_std,
                "molwt_mean": mw_mean,
                "molwt_std": mw_std,
                "pooled_qed_mean": pooled_qed_mean,
                "pooled_qed_std": pooled_qed_std,
                "pooled_sa_mean": pooled_sa_mean,
                "pooled_sa_std": pooled_sa_std,
                "pooled_molwt_mean": pooled_mw_mean,
                "pooled_molwt_std": pooled_mw_std,
            }
        )
    return per_seed_rows, table_rows


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    eps = 1e-12
    return bool(np.all(a >= b - eps) and np.any(a > b + eps))


def f1_indices(objectives: np.ndarray) -> list[int]:
    n = objectives.shape[0]
    out: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[j], objectives[i]):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return out


def crowding_distance(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}
    distance = {idx: 0.0 for idx in front}
    for m in range(objectives.shape[1]):
        sorted_front = sorted(front, key=lambda idx: float(objectives[idx, m]))
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        min_v = float(objectives[sorted_front[0], m])
        max_v = float(objectives[sorted_front[-1], m])
        if max_v - min_v <= 1e-12:
            continue
        for pos in range(1, len(sorted_front) - 1):
            idx = sorted_front[pos]
            if math.isfinite(distance[idx]):
                prev_v = float(objectives[sorted_front[pos - 1], m])
                next_v = float(objectives[sorted_front[pos + 1], m])
                distance[idx] += (next_v - prev_v) / (max_v - min_v)
    return distance


def pareto_rows(population: list[MoleculeRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [
        m for m in population
        if math.isfinite(m.qed) and math.isfinite(m.sa) and math.isfinite(m.docking_raw)
    ]
    if not valid:
        return [], []
    objectives = np.asarray([[m.qed, -m.sa, -m.docking_raw] for m in valid], dtype=np.float64)
    front = f1_indices(objectives)
    crowd = crowding_distance(objectives, front)
    front_set = set(front)
    pop_rows: list[dict[str, Any]] = []
    f1_rows_out: list[dict[str, Any]] = []
    for idx, mol in enumerate(valid):
        row = row_to_dict(mol)
        row["is_f1"] = int(idx in front_set)
        row["crowding_distance"] = crowd.get(idx, "")
        pop_rows.append(row)
        if idx in front_set:
            f1_rows_out.append(row)
    return pop_rows, f1_rows_out


def fmt(mean: float, std: float, digits: int = 1) -> str:
    if not math.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def main() -> int:
    args = parse_args()
    input_roots = [Path(p).expanduser().resolve() for p in args.input_root]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = parse_int_csv(args.expected_seeds)
    thresholds = parse_float_csv(args.thresholds)

    molecule_files, initial_files, final_files, summary_files = find_input_files(input_roots, args.strategy)
    if not molecule_files:
        raise SystemExit(f"No {args.strategy}/molecules_all.csv files found under {input_roots}")

    archive_rows: list[MoleculeRow] = []
    for path in molecule_files:
        archive_rows.extend(load_molecule_file(path, stage="archive", strategy=args.strategy))
    initial_rows: list[MoleculeRow] = []
    for path in initial_files:
        initial_rows.extend(load_molecule_file(path, stage="initial", strategy=args.strategy))
    final_rows: list[MoleculeRow] = []
    for path in final_files:
        final_rows.extend(load_molecule_file(path, stage="final", strategy=args.strategy))

    table8_per_seed, spectral_table = table8_rows(archive_rows, expected_seeds, thresholds)
    manuscript_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        ref = dict(SATURN_REFERENCE.get(float(threshold), {}))
        if ref:
            ref["threshold"] = threshold
            ref["n_expected_seeds"] = 10
            manuscript_rows.append(ref)
        manuscript_rows.extend(row for row in spectral_table if float(row["threshold"]) == float(threshold))

    molecule_fields = [
        "seed", "budget", "strategy", "population_stage", "rank_by_scalar", "smiles", "scaffold",
        "scalar_objective", "docking_raw", "qed", "sa", "molwt", "decode_reason", "macro_count", "source_file",
    ]
    write_tsv(output_dir / "all_molecules_standardized.tsv", [row_to_dict(r) for r in archive_rows], molecule_fields)
    write_tsv(
        output_dir / "table8_per_seed.tsv",
        table8_per_seed,
        ["threshold", "seed", "hit_yield", "modes", "qed_mean_hits", "sa_mean_hits", "molwt_mean_hits"],
    )
    table_fields = [
        "threshold", "method", "successful_replicates", "n_expected_seeds",
        "modes_mean", "modes_std", "yield_mean", "yield_std",
        "qed_mean", "qed_std", "sa_mean", "sa_std", "molwt_mean", "molwt_std",
        "pooled_qed_mean", "pooled_qed_std", "pooled_sa_mean", "pooled_sa_std",
        "pooled_molwt_mean", "pooled_molwt_std", "source",
    ]
    write_tsv(output_dir / "table8_spectralmol.tsv", spectral_table, table_fields)
    write_tsv(output_dir / "table8_manuscript_replacement.tsv", manuscript_rows, table_fields)

    figure6_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for mol in archive_rows:
            if math.isfinite(mol.docking_raw) and mol.docking_raw < threshold:
                row = row_to_dict(mol)
                row["threshold"] = threshold
                figure6_rows.append(row)
    write_tsv(output_dir / "figure6_qed_sa_docking_hits.tsv", figure6_rows, ["threshold"] + molecule_fields)

    if initial_rows or final_rows:
        write_tsv(
            output_dir / "figure7_initial_final_population.tsv",
            [row_to_dict(r) for r in initial_rows + final_rows],
            molecule_fields,
        )

    pareto_population_source = final_rows if final_rows else archive_rows
    pareto_pop, pareto_f1 = pareto_rows(pareto_population_source)
    pareto_fields = molecule_fields + ["is_f1", "crowding_distance"]
    write_tsv(output_dir / "figure8_population_with_f1_flag.tsv", pareto_pop, pareto_fields)
    write_tsv(output_dir / "figure8_f1_pareto_front.tsv", pareto_f1, pareto_fields)
    write_tsv(output_dir / "figure9_f1_crowding_distances.tsv", pareto_f1, pareto_fields)

    copied_summary_rows: list[dict[str, Any]] = []
    for path in summary_files:
        for row in read_csv(path):
            row = dict(row)
            row["source_file"] = str(path)
            copied_summary_rows.append(row)
    if copied_summary_rows:
        fields = sorted({key for row in copied_summary_rows for key in row})
        write_tsv(output_dir / "strategy_summaries.tsv", copied_summary_rows, fields)

    manifest = {
        "input_roots": [str(p) for p in input_roots],
        "strategy": args.strategy,
        "budget": args.budget,
        "expected_seeds": expected_seeds,
        "thresholds": thresholds,
        "molecules_all_files": [str(p) for p in molecule_files],
        "initial_population_files": [str(p) for p in initial_files],
        "final_population_files": [str(p) for p in final_files],
        "summary_files": [str(p) for p in summary_files],
        "notes": [
            "Table 8 Modes and Yield are per-seed values summarized across all expected seeds; missing/no-hit seeds contribute zero.",
            "Table 8 QED, SA, and MolWt columns are means of per-seed hit means over successful seeds.",
            "Pooled property statistics are also saved for auditing but are not the primary manuscript replacement columns.",
            "Figure 8/9 use final_population snapshots when available; otherwise all_molecules archive is used as fallback.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_lines = [
        "# SATURN Theta NSGA-II Analysis",
        "",
        f"Input roots: {', '.join(str(p) for p in input_roots)}",
        f"Strategy: `{args.strategy}`",
        f"Expected seeds: {', '.join(map(str, expected_seeds))}",
        "",
        "## Table 8 Replacement Rows",
        "",
        "| Threshold | Method | Replicates | Modes | Yield | QED | SA | MolWt |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manuscript_rows:
        method = row.get("method", "")
        md_lines.append(
            "| "
            f"{float(row['threshold']):.0f} | {method} | {row.get('successful_replicates', '')} | "
            f"{fmt(safe_float(row.get('modes_mean')), safe_float(row.get('modes_std')))} | "
            f"{fmt(safe_float(row.get('yield_mean')), safe_float(row.get('yield_std')))} | "
            f"{fmt(safe_float(row.get('qed_mean')), safe_float(row.get('qed_std')), 2)} | "
            f"{fmt(safe_float(row.get('sa_mean')), safe_float(row.get('sa_std')), 2)} | "
            f"{fmt(safe_float(row.get('molwt_mean')), safe_float(row.get('molwt_std')), 1)} |"
        )
    md_lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `table8_manuscript_replacement.tsv`: Saturn reference rows plus updated SpectralMol rows.",
            "- `table8_spectralmol.tsv`: SpectralMol-only Table 8 statistics.",
            "- `table8_per_seed.tsv`: per-seed hit counts and scaffold modes.",
            "- `all_molecules_standardized.tsv`: normalized molecule-level archive.",
            "- `figure6_qed_sa_docking_hits.tsv`: QED-SA scatter inputs for docking-hit thresholds.",
            "- `figure7_initial_final_population.tsv`: initial/final population projections when snapshots are available.",
            "- `figure8_population_with_f1_flag.tsv`: final population with F1 membership flag.",
            "- `figure8_f1_pareto_front.tsv`: first non-dominated front.",
            "- `figure9_f1_crowding_distances.tsv`: crowding-distance data for F1.",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote analysis to: {output_dir}")
    print(f"Archive molecules: {len(archive_rows)}")
    print(f"Initial snapshot molecules: {len(initial_rows)}")
    print(f"Final snapshot molecules: {len(final_rows)}")
    print(f"F1 molecules: {len(pareto_f1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
