#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import DataStructs

from analyze_frequency_feature_changes import (
    CONDITIONS,
    FRAGMENT_NAMES,
    MolFeatures,
    canonicalize,
    fnum,
    format_float,
    frac,
    mean,
    mol_features,
    parse_context,
    std,
    write_tsv,
)


def best_parent_match(child: MolFeatures, parents: list[tuple[int, MolFeatures]]) -> tuple[int, MolFeatures, float]:
    similarities = DataStructs.BulkTanimotoSimilarity(child.fp, [parent.fp for _rank, parent in parents])
    best_idx = max(range(len(similarities)), key=similarities.__getitem__)
    rank, parent = parents[best_idx]
    return int(rank), parent, float(similarities[best_idx])


def feature_delta_row(
    context: dict[str, Any],
    child_record: dict[str, Any],
    child_features: MolFeatures,
    parent_rank: int,
    parent_features: MolFeatures,
    parent_similarity: float,
) -> dict[str, Any]:
    same_scaffold = bool(child_features.scaffold and child_features.scaffold == parent_features.scaffold)
    scaffold_changed = not same_scaffold
    fg_delta_l1 = sum(abs(a - b) for a, b in zip(child_features.fragments, parent_features.fragments))
    fg_changed = fg_delta_l1 > 0

    return {
        "condition": context.get("condition", ""),
        "seed": context.get("seed"),
        "task_index": context.get("task_index"),
        "task": context.get("task", ""),
        "generation": child_record.get("generation"),
        "uid": child_record.get("uid"),
        "score": child_record.get("score"),
        "smiles": child_features.smiles,
        "decode_reason": child_record.get("decode_reason", ""),
        "lineage_operator": child_record.get("lineage_operator", ""),
        "best_parent_rank": parent_rank,
        "best_parent_uid": child_record.get(f"parent{'' if parent_rank == 1 else parent_rank}_uid", ""),
        "best_parent_smiles": parent_features.smiles,
        "best_parent_tanimoto": parent_similarity,
        "generated_scaffold": child_features.scaffold,
        "best_parent_scaffold": parent_features.scaffold,
        "same_scaffold": int(same_scaffold),
        "scaffold_changed": int(scaffold_changed),
        "functional_group_changed": int(fg_changed),
        "same_scaffold_functional_group_changed": int(same_scaffold and fg_changed),
        "functional_group_delta_l1": fg_delta_l1,
        "heavy_atom_delta": child_features.heavy_atoms - parent_features.heavy_atoms,
        "abs_heavy_atom_delta": abs(child_features.heavy_atoms - parent_features.heavy_atoms),
        "ring_delta": child_features.rings - parent_features.rings,
        "abs_ring_delta": abs(child_features.rings - parent_features.rings),
        "aromatic_ring_delta": child_features.aromatic_rings - parent_features.aromatic_rings,
        "abs_aromatic_ring_delta": abs(child_features.aromatic_rings - parent_features.aromatic_rings),
        "hetero_atom_delta": child_features.hetero_atoms - parent_features.hetero_atoms,
        "abs_hetero_atom_delta": abs(child_features.hetero_atoms - parent_features.hetero_atoms),
        "rotatable_bond_delta": child_features.rotatable_bonds - parent_features.rotatable_bonds,
        "abs_rotatable_bond_delta": abs(child_features.rotatable_bonds - parent_features.rotatable_bonds),
        "hba_delta": child_features.hba - parent_features.hba,
        "hbd_delta": child_features.hbd - parent_features.hbd,
        "formal_charge_delta": child_features.formal_charge - parent_features.formal_charge,
        "tpsa_delta": child_features.tpsa - parent_features.tpsa,
        "abs_tpsa_delta": abs(child_features.tpsa - parent_features.tpsa),
        "logp_delta": child_features.logp - parent_features.logp,
        "abs_logp_delta": abs(child_features.logp - parent_features.logp),
    }


def lineage_paths(input_root: Path, source: str) -> list[Path]:
    if source == "top-lineage":
        paths = sorted(input_root.rglob("generator_top_molecules_lineage.csv"))
        if paths:
            return paths
    return sorted(input_root.rglob("molecule_lineage_by_generation.tsv"))


def read_lineage_records(path: Path, source: str, min_generation: int) -> tuple[list[dict[str, Any]], int]:
    delimiter = "," if path.name.endswith(".csv") else "\t"
    records_by_smiles: dict[str, dict[str, Any]] = {}
    missing_parent_rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            child = canonicalize(row.get("smiles", ""))
            if not child:
                continue
            generation = int(fnum(row.get("birth_generation") or row.get("generation") or 0))
            if generation < min_generation:
                continue
            parents: list[tuple[int, str]] = []
            for rank in (1, 2, 3):
                key = "parent_smiles" if rank == 1 else f"parent{rank}_smiles"
                parent = canonicalize(row.get(key, ""))
                if parent:
                    parents.append((rank, parent))
            if not parents:
                missing_parent_rows += 1
                continue
            row = dict(row)
            row["smiles"] = child
            row["generation"] = generation
            row["_parents"] = parents
            score = fnum(row.get("score"))
            previous = records_by_smiles.get(child)
            if previous is None or score > fnum(previous.get("score")):
                records_by_smiles[child] = row
    records = list(records_by_smiles.values())
    records.sort(key=lambda r: fnum(r.get("score")), reverse=True)
    return records, missing_parent_rows


def collect_rows(
    input_root: Path,
    top_n_per_run: int,
    min_generation: int,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    paths = lineage_paths(input_root, source)

    for export_path in paths:
        context = parse_context(export_path)
        if context.get("condition") not in CONDITIONS:
            continue
        records, missing_parent_rows = read_lineage_records(export_path, source, min_generation)
        selected = records[:top_n_per_run]
        valid_selected = 0
        for record in selected:
            child_features = mol_features(str(record.get("smiles", "")))
            parents: list[tuple[int, MolFeatures]] = []
            for rank, parent_smiles in record.get("_parents", []):
                parent_features = mol_features(parent_smiles)
                if parent_features is not None:
                    parents.append((int(rank), parent_features))
            if child_features is None or not parents:
                continue
            parent_rank, parent_features, similarity = best_parent_match(child_features, parents)
            rows.append(feature_delta_row(context, record, child_features, parent_rank, parent_features, similarity))
            valid_selected += 1

        run_rows.append(
            {
                "condition": context.get("condition", ""),
                "seed": context.get("seed"),
                "task_index": context.get("task_index"),
                "task": context.get("task", ""),
                "export_path": str(export_path),
                "source": source,
                "generated_with_parent": len(records),
                "missing_parent_rows": missing_parent_rows,
                "selected_top_n": len(selected),
                "valid_selected": valid_selected,
            }
        )
    return rows, run_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    summary: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        cond_rows = by_condition.get(condition, [])
        summary.append(
            {
                "condition": condition,
                "n_molecules": len(cond_rows),
                "n_tasks": len({r["task_index"] for r in cond_rows if r.get("task_index") is not None}),
                "n_seeds": len({r["seed"] for r in cond_rows if r.get("seed") is not None}),
                "mean_score": mean([fnum(r["score"]) for r in cond_rows]),
                "std_score": std([fnum(r["score"]) for r in cond_rows]),
                "mean_best_parent_tanimoto": mean([fnum(r["best_parent_tanimoto"]) for r in cond_rows]),
                "std_best_parent_tanimoto": std([fnum(r["best_parent_tanimoto"]) for r in cond_rows]),
                "scaffold_changed_fraction": frac([bool(int(r["scaffold_changed"])) for r in cond_rows]),
                "same_scaffold_fraction": frac([bool(int(r["same_scaffold"])) for r in cond_rows]),
                "functional_group_changed_fraction": frac([bool(int(r["functional_group_changed"])) for r in cond_rows]),
                "same_scaffold_functional_group_changed_fraction": frac(
                    [bool(int(r["same_scaffold_functional_group_changed"])) for r in cond_rows]
                ),
                "mean_functional_group_delta_l1": mean([fnum(r["functional_group_delta_l1"]) for r in cond_rows]),
                "mean_abs_heavy_atom_delta": mean([fnum(r["abs_heavy_atom_delta"]) for r in cond_rows]),
                "mean_abs_ring_delta": mean([fnum(r["abs_ring_delta"]) for r in cond_rows]),
                "mean_abs_aromatic_ring_delta": mean([fnum(r["abs_aromatic_ring_delta"]) for r in cond_rows]),
                "mean_abs_hetero_atom_delta": mean([fnum(r["abs_hetero_atom_delta"]) for r in cond_rows]),
                "mean_abs_rotatable_bond_delta": mean([fnum(r["abs_rotatable_bond_delta"]) for r in cond_rows]),
                "mean_abs_tpsa_delta": mean([fnum(r["abs_tpsa_delta"]) for r in cond_rows]),
                "mean_abs_logp_delta": mean([fnum(r["abs_logp_delta"]) for r in cond_rows]),
            }
        )
    return summary


def summarize_by_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), row.get("task_index"), str(row.get("task", "")))].append(row)

    out: list[dict[str, Any]] = []
    for (condition, task_index, task), group_rows in sorted(
        grouped.items(),
        key=lambda kv: (kv[0][1] if kv[0][1] is not None else 999, kv[0][0]),
    ):
        out.append(
            {
                "condition": condition,
                "task_index": task_index,
                "task": task,
                "n_molecules": len(group_rows),
                "mean_best_parent_tanimoto": mean([fnum(r["best_parent_tanimoto"]) for r in group_rows]),
                "scaffold_changed_fraction": frac([bool(int(r["scaffold_changed"])) for r in group_rows]),
                "same_scaffold_functional_group_changed_fraction": frac(
                    [bool(int(r["same_scaffold_functional_group_changed"])) for r in group_rows]
                ),
                "mean_abs_heavy_atom_delta": mean([fnum(r["abs_heavy_atom_delta"]) for r in group_rows]),
                "mean_abs_ring_delta": mean([fnum(r["abs_ring_delta"]) for r in group_rows]),
            }
        )
    return out


def interpretation_checks(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition = {str(r["condition"]): r for r in summary_rows}
    high = by_condition.get("high-only")
    low = by_condition.get("low-only")
    full = by_condition.get("full-spectrum")
    random_matrix = by_condition.get("random-matrix")
    checks: list[dict[str, Any]] = []

    def add(check: str, passed: bool, observed: Any, expected: str) -> None:
        checks.append({"check": check, "passed": int(bool(passed)), "observed": observed, "expected": expected})

    if high and low:
        add(
            "high-only children are closer to their recorded parents than low-only",
            fnum(high["mean_best_parent_tanimoto"]) > fnum(low["mean_best_parent_tanimoto"]),
            f"{high['mean_best_parent_tanimoto']} vs {low['mean_best_parent_tanimoto']}",
            "high > low",
        )
        add(
            "high-only preserves parent scaffold more often than low-only",
            fnum(high["same_scaffold_fraction"]) > fnum(low["same_scaffold_fraction"]),
            f"{high['same_scaffold_fraction']} vs {low['same_scaffold_fraction']}",
            "high > low",
        )
        add(
            "low-only changes parent scaffold more often than high-only",
            fnum(low["scaffold_changed_fraction"]) > fnum(high["scaffold_changed_fraction"]),
            f"{low['scaffold_changed_fraction']} vs {high['scaffold_changed_fraction']}",
            "low > high",
        )
        add(
            "low-only has larger parent-child heavy-atom shifts than high-only",
            fnum(low["mean_abs_heavy_atom_delta"]) > fnum(high["mean_abs_heavy_atom_delta"]),
            f"{low['mean_abs_heavy_atom_delta']} vs {high['mean_abs_heavy_atom_delta']}",
            "low > high",
        )
        add(
            "high-only has more same-scaffold functional-group changes than low-only",
            fnum(high["same_scaffold_functional_group_changed_fraction"])
            > fnum(low["same_scaffold_functional_group_changed_fraction"]),
            (
                f"{high['same_scaffold_functional_group_changed_fraction']} vs "
                f"{low['same_scaffold_functional_group_changed_fraction']}"
            ),
            "high > low",
        )
    if full and random_matrix:
        add(
            "random-matrix differs from full-spectrum in explicit parent-child structural profile",
            any(
                abs(fnum(random_matrix[k]) - fnum(full[k])) > 1e-9
                for k in (
                    "mean_best_parent_tanimoto",
                    "scaffold_changed_fraction",
                    "mean_abs_heavy_atom_delta",
                )
            ),
            "profile differs",
            "non-identical",
        )
    return checks


def write_markdown(
    output_dir: Path,
    input_root: Path,
    summary_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    top_n_per_run: int,
    source: str,
) -> None:
    lines: list[str] = []
    lines.append("# GuacaMol Frequency Parent-Lineage Feature-Change Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Input root: `{input_root}`")
    lines.append(f"Source: `{source}`")
    lines.append(f"Molecule set: top {top_n_per_run} unique generated molecules per task/seed run with recorded parent smiles.")
    lines.append("")
    lines.append(
        "Each child is compared with the most similar recorded parent among parent, parent2, and parent3. "
        "This avoids the nearest-seed proxy used by the previous feature-change analysis."
    )
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    headers = [
        "condition",
        "n_molecules",
        "mean_best_parent_tanimoto",
        "scaffold_changed_fraction",
        "same_scaffold_fraction",
        "same_scaffold_functional_group_changed_fraction",
        "mean_functional_group_delta_l1",
        "mean_abs_heavy_atom_delta",
        "mean_abs_ring_delta",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary_rows:
        vals = []
        for key in headers:
            value = row.get(key, "")
            vals.append(str(value) if key in {"condition", "n_molecules"} else format_float(value))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Interpretation Checks")
    lines.append("")
    lines.append("| check | passed | observed | expected |")
    lines.append("| --- | --- | --- | --- |")
    for row in checks:
        lines.append(f"| {row['check']} | {row['passed']} | {row['observed']} | {row['expected']} |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `frequency_lineage_feature_rows.tsv`: molecule-level parent-child structural-change rows.")
    lines.append("- `frequency_lineage_feature_summary.tsv`: condition-level summary.")
    lines.append("- `frequency_lineage_feature_task_summary.tsv`: task-level summary.")
    lines.append("- `frequency_lineage_feature_interpretation_checks.tsv`: checks for the low/global vs high/local interpretation.")
    lines.append("- `frequency_lineage_feature_run_coverage.tsv`: per task/seed lineage export coverage.")
    lines.append("- `frequency_lineage_feature_manifest.json`: run metadata.")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n-per-run", type=int, default=100)
    parser.add_argument("--min-generation", type=int, default=1)
    parser.add_argument(
        "--source",
        choices=("top-lineage", "lineage"),
        default="top-lineage",
        help="Use generator_top_molecules_lineage.csv when present, or full molecule_lineage_by_generation.tsv exports.",
    )
    args = parser.parse_args(argv)

    rows, run_rows = collect_rows(
        args.input_root,
        top_n_per_run=args.top_n_per_run,
        min_generation=args.min_generation,
        source=args.source,
    )
    if not rows:
        raise SystemExit(f"No generated molecule rows with explicit parent lineage found under {args.input_root}")

    summary_rows = summarize(rows)
    task_rows = summarize_by_task(rows)
    checks = interpretation_checks(summary_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    row_fields = [
        "condition",
        "seed",
        "task_index",
        "task",
        "generation",
        "uid",
        "score",
        "smiles",
        "decode_reason",
        "lineage_operator",
        "best_parent_rank",
        "best_parent_uid",
        "best_parent_smiles",
        "best_parent_tanimoto",
        "generated_scaffold",
        "best_parent_scaffold",
        "same_scaffold",
        "scaffold_changed",
        "functional_group_changed",
        "same_scaffold_functional_group_changed",
        "functional_group_delta_l1",
        "heavy_atom_delta",
        "abs_heavy_atom_delta",
        "ring_delta",
        "abs_ring_delta",
        "aromatic_ring_delta",
        "abs_aromatic_ring_delta",
        "hetero_atom_delta",
        "abs_hetero_atom_delta",
        "rotatable_bond_delta",
        "abs_rotatable_bond_delta",
        "hba_delta",
        "hbd_delta",
        "formal_charge_delta",
        "tpsa_delta",
        "abs_tpsa_delta",
        "logp_delta",
        "abs_logp_delta",
    ]
    summary_fields = list(summary_rows[0])
    task_fields = list(task_rows[0]) if task_rows else []
    run_fields = list(run_rows[0]) if run_rows else []

    write_tsv(output_dir / "frequency_lineage_feature_rows.tsv", rows, row_fields)
    write_tsv(output_dir / "frequency_lineage_feature_summary.tsv", summary_rows, summary_fields)
    write_tsv(output_dir / "frequency_lineage_feature_task_summary.tsv", task_rows, task_fields)
    write_tsv(
        output_dir / "frequency_lineage_feature_interpretation_checks.tsv",
        checks,
        ["check", "passed", "observed", "expected"],
    )
    write_tsv(output_dir / "frequency_lineage_feature_run_coverage.tsv", run_rows, run_fields)
    (output_dir / "frequency_lineage_feature_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input_root": str(args.input_root),
                "top_n_per_run": args.top_n_per_run,
                "min_generation": args.min_generation,
                "source": args.source,
                "conditions": list(CONDITIONS),
                "fragment_names": list(FRAGMENT_NAMES),
                "n_rows": len(rows),
                "n_run_exports": len(run_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_markdown(output_dir, args.input_root, summary_rows, checks, args.top_n_per_run, args.source)

    print(f"Wrote parent-lineage feature-change analysis to: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']:14s} n={row['n_molecules']:5d} "
            f"parent_sim={format_float(row['mean_best_parent_tanimoto'])} "
            f"scaffold_changed={format_float(row['scaffold_changed_fraction'])} "
            f"same_scaf_fg={format_float(row['same_scaffold_functional_group_changed_fraction'])} "
            f"heavy_delta={format_float(row['mean_abs_heavy_atom_delta'])} "
            f"ring_delta={format_float(row['mean_abs_ring_delta'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
