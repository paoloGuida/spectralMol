#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from rdkit.Chem import rdFingerprintGenerator

    _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception:  # pragma: no cover - old RDKit fallback
    _MORGAN_GENERATOR = None


CONDITIONS = ("full-spectrum", "high-only", "low-only", "random-matrix")
FRAGMENT_NAMES = tuple(name for name in dir(Fragments) if name.startswith("fr_"))
rdBase.DisableLog("rdApp.*")


@dataclass(frozen=True)
class MolFeatures:
    smiles: str
    scaffold: str
    heavy_atoms: int
    rings: int
    aromatic_rings: int
    hetero_atoms: int
    rotatable_bonds: int
    hba: int
    hbd: int
    formal_charge: int
    tpsa: float
    logp: float
    fp: Any
    fragments: tuple[int, ...]


_FEATURE_CACHE: dict[str, MolFeatures | None] = {}


def mean(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def frac(values: list[bool]) -> float:
    return sum(1 for x in values if x) / len(values) if values else float("nan")


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def canonicalize(smiles: str) -> str | None:
    if not smiles or smiles.lower() == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fingerprint(mol: Chem.Mol) -> Any:
    if _MORGAN_GENERATOR is not None:
        return _MORGAN_GENERATOR.GetFingerprint(mol)
    from rdkit.Chem import AllChem

    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""
    return canonicalize(scaffold) or ""


def mol_features(smiles: str) -> MolFeatures | None:
    canon = canonicalize(smiles)
    if canon is None:
        return None
    cached = _FEATURE_CACHE.get(canon)
    if cached is not None or canon in _FEATURE_CACHE:
        return cached

    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        _FEATURE_CACHE[canon] = None
        return None

    fragment_values: list[int] = []
    for name in FRAGMENT_NAMES:
        fn = getattr(Fragments, name)
        try:
            fragment_values.append(int(fn(mol)))
        except Exception:
            fragment_values.append(0)

    features = MolFeatures(
        smiles=canon,
        scaffold=scaffold_smiles(mol),
        heavy_atoms=int(mol.GetNumHeavyAtoms()),
        rings=int(rdMolDescriptors.CalcNumRings(mol)),
        aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        hetero_atoms=int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        formal_charge=int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        logp=float(Crippen.MolLogP(mol)),
        fp=fingerprint(mol),
        fragments=tuple(fragment_values),
    )
    _FEATURE_CACHE[canon] = features
    return features


def parse_context(path: Path) -> dict[str, Any]:
    parts = path.parts
    context: dict[str, Any] = {"condition": "", "seed": None, "task_index": None, "task": ""}

    for condition in CONDITIONS:
        if condition in parts:
            context["condition"] = condition
            break

    for part in parts:
        seed_match = re.fullmatch(r"seed_(\d+)", part)
        if seed_match and context["seed"] is None:
            context["seed"] = int(seed_match.group(1))
        task_match = re.fullmatch(r"task_(\d+)_(.+)", part)
        if task_match:
            context["task_index"] = int(task_match.group(1))
            context["task"] = task_match.group(2)

    if not context["task"]:
        try:
            export_idx = parts.index("exports")
            context["task"] = parts[export_idx + 3]
        except Exception:
            pass
    return context


def read_molecule_scores(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    initial: dict[str, dict[str, Any]] = {}
    generated: dict[str, dict[str, Any]] = {}
    duplicate_seed_rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            canon = canonicalize(row.get("smiles", ""))
            if not canon:
                continue
            gen = int(fnum(row.get("generation")))
            score = fnum(row.get("score"))
            record = {
                "smiles": canon,
                "score": score,
                "generation": gen,
            }
            if gen <= 0:
                if canon not in initial or score > fnum(initial[canon].get("score")):
                    initial[canon] = record
            else:
                if canon in initial:
                    duplicate_seed_rows += 1
                    continue
                previous = generated.get(canon)
                if previous is None or score > fnum(previous.get("score")):
                    generated[canon] = record
    return initial, generated, duplicate_seed_rows


def find_compare_root(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith("compare_GuacaMol_"):
            return parent
    return None


def read_seed_population(compare_root: Path, seed: int | None) -> dict[str, dict[str, Any]]:
    if seed is None:
        candidates = sorted((compare_root / "shared_initial_population").glob("seed_*.smi"))
    else:
        candidates = [compare_root / "shared_initial_population" / f"seed_{seed}.smi"]
    initial: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                raw = line.strip().split()[0] if line.strip() else ""
                canon = canonicalize(raw)
                if canon and canon not in initial:
                    initial[canon] = {"smiles": canon, "score": float("nan"), "generation": 0}
    return initial


def read_generator_top(path: Path, initial: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    generated: dict[str, dict[str, Any]] = {}
    duplicate_seed_rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canon = canonicalize(row.get("smiles", ""))
            if not canon:
                continue
            if canon in initial:
                duplicate_seed_rows += 1
                continue
            score = fnum(row.get("score"))
            previous = generated.get(canon)
            if previous is None or score > fnum(previous.get("score")):
                generated[canon] = {"smiles": canon, "score": score, "generation": ""}
    return generated, duplicate_seed_rows


def nearest_seed(generated: MolFeatures, seed_features: list[MolFeatures]) -> tuple[MolFeatures, float]:
    similarities = DataStructs.BulkTanimotoSimilarity(generated.fp, [seed.fp for seed in seed_features])
    best_idx = max(range(len(similarities)), key=similarities.__getitem__)
    return seed_features[best_idx], float(similarities[best_idx])


def feature_delta_row(
    context: dict[str, Any],
    generated_record: dict[str, Any],
    generated_features: MolFeatures,
    seed_features: MolFeatures,
    nearest_similarity: float,
) -> dict[str, Any]:
    same_scaffold = bool(generated_features.scaffold and generated_features.scaffold == seed_features.scaffold)
    scaffold_changed = not same_scaffold
    fg_delta_l1 = sum(abs(a - b) for a, b in zip(generated_features.fragments, seed_features.fragments))
    fg_changed = fg_delta_l1 > 0

    return {
        "condition": context.get("condition", ""),
        "seed": context.get("seed"),
        "task_index": context.get("task_index"),
        "task": context.get("task", ""),
        "generation": generated_record.get("generation"),
        "score": generated_record.get("score"),
        "smiles": generated_features.smiles,
        "nearest_seed_smiles": seed_features.smiles,
        "nearest_seed_tanimoto": nearest_similarity,
        "generated_scaffold": generated_features.scaffold,
        "nearest_seed_scaffold": seed_features.scaffold,
        "same_scaffold": int(same_scaffold),
        "scaffold_changed": int(scaffold_changed),
        "functional_group_changed": int(fg_changed),
        "same_scaffold_functional_group_changed": int(same_scaffold and fg_changed),
        "functional_group_delta_l1": fg_delta_l1,
        "heavy_atom_delta": generated_features.heavy_atoms - seed_features.heavy_atoms,
        "abs_heavy_atom_delta": abs(generated_features.heavy_atoms - seed_features.heavy_atoms),
        "ring_delta": generated_features.rings - seed_features.rings,
        "abs_ring_delta": abs(generated_features.rings - seed_features.rings),
        "aromatic_ring_delta": generated_features.aromatic_rings - seed_features.aromatic_rings,
        "abs_aromatic_ring_delta": abs(generated_features.aromatic_rings - seed_features.aromatic_rings),
        "hetero_atom_delta": generated_features.hetero_atoms - seed_features.hetero_atoms,
        "abs_hetero_atom_delta": abs(generated_features.hetero_atoms - seed_features.hetero_atoms),
        "rotatable_bond_delta": generated_features.rotatable_bonds - seed_features.rotatable_bonds,
        "abs_rotatable_bond_delta": abs(generated_features.rotatable_bonds - seed_features.rotatable_bonds),
        "hba_delta": generated_features.hba - seed_features.hba,
        "hbd_delta": generated_features.hbd - seed_features.hbd,
        "formal_charge_delta": generated_features.formal_charge - seed_features.formal_charge,
        "tpsa_delta": generated_features.tpsa - seed_features.tpsa,
        "abs_tpsa_delta": abs(generated_features.tpsa - seed_features.tpsa),
        "logp_delta": generated_features.logp - seed_features.logp,
        "abs_logp_delta": abs(generated_features.logp - seed_features.logp),
    }


def collect_rows(
    input_root: Path,
    top_n_per_run: int,
    min_generation: int,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    if source == "generator-top":
        export_paths = sorted(input_root.rglob("generator_top_molecules.csv"))
    else:
        export_paths = sorted(input_root.rglob("molecules_scores_by_generation.tsv"))

    for export_path in export_paths:
        context = parse_context(export_path)
        if context.get("condition") not in CONDITIONS:
            continue
        if source == "generator-top":
            compare_root = find_compare_root(export_path)
            initial = read_seed_population(compare_root, context.get("seed")) if compare_root else {}
            generated, seed_duplicate_rows = read_generator_top(export_path, initial)
        else:
            initial, generated, seed_duplicate_rows = read_molecule_scores(export_path)
        seed_features = [f for f in (mol_features(smi) for smi in initial) if f is not None]
        generated_records = [
            record
            for record in generated.values()
            if source == "generator-top" or int(record.get("generation", 0)) >= min_generation
        ]
        generated_records.sort(key=lambda r: fnum(r.get("score")), reverse=True)
        selected = generated_records[:top_n_per_run]

        valid_selected = 0
        for record in selected:
            gen_features = mol_features(str(record["smiles"]))
            if gen_features is None or not seed_features:
                continue
            seed_match, similarity = nearest_seed(gen_features, seed_features)
            rows.append(feature_delta_row(context, record, gen_features, seed_match, similarity))
            valid_selected += 1

        run_rows.append(
            {
                "condition": context.get("condition", ""),
                "seed": context.get("seed"),
                "task_index": context.get("task_index"),
                "task": context.get("task", ""),
                "export_path": str(export_path),
                "source": source,
                "initial_unique": len(initial),
                "generated_unique_novel": len(generated_records),
                "seed_duplicate_rows": seed_duplicate_rows,
                "selected_top_n": len(selected),
                "valid_selected": valid_selected,
            }
        )
    return rows, run_rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    summary: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        cond_rows = by_condition.get(condition, [])
        scores = [fnum(r["score"]) for r in cond_rows]
        similarities = [fnum(r["nearest_seed_tanimoto"]) for r in cond_rows]
        scaffold_changed = [bool(int(r["scaffold_changed"])) for r in cond_rows]
        same_scaffold = [bool(int(r["same_scaffold"])) for r in cond_rows]
        fg_changed = [bool(int(r["functional_group_changed"])) for r in cond_rows]
        same_scaf_fg_changed = [bool(int(r["same_scaffold_functional_group_changed"])) for r in cond_rows]

        summary.append(
            {
                "condition": condition,
                "n_molecules": len(cond_rows),
                "n_tasks": len({r["task_index"] for r in cond_rows if r.get("task_index") is not None}),
                "n_seeds": len({r["seed"] for r in cond_rows if r.get("seed") is not None}),
                "mean_score": mean(scores),
                "std_score": std(scores),
                "mean_nearest_seed_tanimoto": mean(similarities),
                "std_nearest_seed_tanimoto": std(similarities),
                "scaffold_changed_fraction": frac(scaffold_changed),
                "same_scaffold_fraction": frac(same_scaffold),
                "functional_group_changed_fraction": frac(fg_changed),
                "same_scaffold_functional_group_changed_fraction": frac(same_scaf_fg_changed),
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
    for (condition, task_index, task), group_rows in sorted(grouped.items(), key=lambda kv: (kv[0][1] if kv[0][1] is not None else 999, kv[0][0])):
        out.append(
            {
                "condition": condition,
                "task_index": task_index,
                "task": task,
                "n_molecules": len(group_rows),
                "mean_nearest_seed_tanimoto": mean([fnum(r["nearest_seed_tanimoto"]) for r in group_rows]),
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
            "high-only keeps molecules closer to initial population than low-only",
            fnum(high["mean_nearest_seed_tanimoto"]) > fnum(low["mean_nearest_seed_tanimoto"]),
            f"{high['mean_nearest_seed_tanimoto']} vs {low['mean_nearest_seed_tanimoto']}",
            "high > low",
        )
        add(
            "high-only preserves scaffold more often than low-only",
            fnum(high["same_scaffold_fraction"]) > fnum(low["same_scaffold_fraction"]),
            f"{high['same_scaffold_fraction']} vs {low['same_scaffold_fraction']}",
            "high > low",
        )
        add(
            "low-only changes scaffold more often than high-only",
            fnum(low["scaffold_changed_fraction"]) > fnum(high["scaffold_changed_fraction"]),
            f"{low['scaffold_changed_fraction']} vs {high['scaffold_changed_fraction']}",
            "low > high",
        )
        add(
            "low-only has larger heavy-atom shifts than high-only",
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
            "random-matrix is not identical to full-spectrum in structural-change profile",
            any(
                abs(fnum(random_matrix[k]) - fnum(full[k])) > 1e-9
                for k in (
                    "mean_nearest_seed_tanimoto",
                    "scaffold_changed_fraction",
                    "mean_abs_heavy_atom_delta",
                )
            ),
            "profile differs",
            "non-identical",
        )
    return checks


def format_float(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_markdown(
    output_dir: Path,
    input_root: Path,
    summary_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    top_n_per_run: int,
    source: str,
) -> None:
    lines: list[str] = []
    lines.append("# GuacaMol Frequency Feature-Change Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Input root: `{input_root}`")
    lines.append(f"Source: `{source}`")
    lines.append(f"Molecule set: top {top_n_per_run} unique, non-initial generated molecules per task/seed run.")
    lines.append("")
    lines.append(
        "Nearest initial-population molecule is used as a parent proxy because explicit parent lineage is not stored in the GuacaMol exports."
    )
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    headers = [
        "condition",
        "n_molecules",
        "mean_nearest_seed_tanimoto",
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
    lines.append("- `frequency_feature_rows.tsv`: molecule-level structural-change rows.")
    lines.append("- `frequency_feature_summary.tsv`: condition-level summary.")
    lines.append("- `frequency_feature_task_summary.tsv`: task-level summary.")
    lines.append("- `frequency_feature_interpretation_checks.tsv`: checks for the low/global vs high/local interpretation.")
    lines.append("- `frequency_feature_run_coverage.tsv`: per task/seed export coverage.")
    lines.append("- `frequency_feature_manifest.json`: run metadata.")
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n-per-run", type=int, default=100)
    parser.add_argument("--min-generation", type=int, default=1)
    parser.add_argument(
        "--source",
        choices=("generator-top", "exports"),
        default="generator-top",
        help="Use final generator_top_molecules.csv files or full molecules_scores_by_generation.tsv exports.",
    )
    args = parser.parse_args(argv)

    rows, run_rows = collect_rows(
        args.input_root,
        top_n_per_run=args.top_n_per_run,
        min_generation=args.min_generation,
        source=args.source,
    )
    if not rows:
        raise SystemExit(f"No generated molecule rows found under {args.input_root}")

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
        "score",
        "smiles",
        "nearest_seed_smiles",
        "nearest_seed_tanimoto",
        "generated_scaffold",
        "nearest_seed_scaffold",
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

    write_tsv(output_dir / "frequency_feature_rows.tsv", rows, row_fields)
    write_tsv(output_dir / "frequency_feature_summary.tsv", summary_rows, summary_fields)
    write_tsv(output_dir / "frequency_feature_task_summary.tsv", task_rows, task_fields)
    write_tsv(output_dir / "frequency_feature_interpretation_checks.tsv", checks, ["check", "passed", "observed", "expected"])
    write_tsv(output_dir / "frequency_feature_run_coverage.tsv", run_rows, run_fields)
    (output_dir / "frequency_feature_manifest.json").write_text(
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

    print(f"Wrote feature-change analysis to: {output_dir}")
    for row in summary_rows:
        print(
            f"{row['condition']:14s} n={row['n_molecules']:5d} "
            f"sim={format_float(row['mean_nearest_seed_tanimoto'])} "
            f"scaffold_changed={format_float(row['scaffold_changed_fraction'])} "
            f"same_scaf_fg={format_float(row['same_scaffold_functional_group_changed_fraction'])} "
            f"heavy_delta={format_float(row['mean_abs_heavy_atom_delta'])} "
            f"ring_delta={format_float(row['mean_abs_ring_delta'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
