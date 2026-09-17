#!/usr/bin/env python3
"""
Regenerate manuscript figures from the packaged fresh GuacaMol and SATURN data.

Default usage from this package:

    python scripts/update_manuscript_figures.py

Useful options:

    python scripts/update_manuscript_figures.py --formats png,pdf,svg
    python scripts/update_manuscript_figures.py --input-dir /path/to/manuscript_update_20260824
    python scripts/update_manuscript_figures.py --output-dir /path/to/figures_updated

The script intentionally reads only compact manuscript-update files, not the
full 52 GB GuacaMol scratch root.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from pathlib import Path
from typing import Callable


SATURN_REFERENCE = {
    -9.0: {"qed": 0.80, "sa": 2.10},
    -10.0: {"qed": 0.72, "sa": 2.26},
}

MODEL_LABEL = {
    "spectralmol": "SpectralMol",
    "graphga": "GraphGA",
    "graphGA": "GraphGA",
    "evolution_ga": "SpectralMol",
}

MODEL_COLOR = {
    "spectralmol": "#1f77b4",
    "graphga": "#ff7f0e",
    "graphGA": "#ff7f0e",
    "evolution_ga": "#1f77b4",
}


def require_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "Could not import plotting dependencies. Install/use an environment with "
            "pandas, numpy, and matplotlib, then rerun this script.\n"
            f"Original error: {exc}"
        ) from exc

    try:
        import seaborn as sns
    except Exception:
        sns = None

    return plt, np, pd, sns


def read_table(path: Path, pd):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip")
    return pd.read_csv(path, sep="\t")


def as_num(series, pd):
    return pd.to_numeric(series, errors="coerce")


def ensure_columns(df, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def save_figure(fig, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        out = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        paths.append(out)
    return paths


def clean_task_label(name: str) -> str:
    return str(name).replace("_", " ")


def figure3_guacamol_task_comparison(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "guacamol" / "guacamol_figure3_task_comparison.tsv"
    df = read_table(path, pd)
    ensure_columns(
        df,
        ["task_index", "task", "task_family", "spectralmol_mean", "graphga_mean", "delta_spectralmol_minus_graphga_mean"],
        path,
    )
    df = df.sort_values(["task_family", "task_index"]).reset_index(drop=True)
    y = np.arange(len(df))
    spec = as_num(df["spectralmol_mean"], pd).to_numpy()
    graph = as_num(df["graphga_mean"], pd).to_numpy()
    delta = as_num(df["delta_spectralmol_minus_graphga_mean"], pd).to_numpy()
    abs_delta = np.abs(delta)

    fig_h = max(8.0, 0.38 * len(df) + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    norm = plt.Normalize(vmin=0, vmax=max(0.5, float(np.nanmax(abs_delta))))
    cmap = plt.cm.viridis

    for i, (s, g, d) in enumerate(zip(spec, graph, abs_delta)):
        ax.plot([g, s], [i, i], color=cmap(norm(d)), linewidth=2.4, alpha=0.9, solid_capstyle="round")
    ax.scatter(graph, y, s=46, color=MODEL_COLOR["graphga"], edgecolor="white", linewidth=0.7, label="GraphGA", zorder=3)
    ax.scatter(spec, y, s=54, color=MODEL_COLOR["spectralmol"], edgecolor="white", linewidth=0.7, label="SpectralMol", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([clean_task_label(t) for t in df["task"]], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("Normalized GuacaMol score")
    ax.set_title("Figure 3 update: GuacaMol task-by-task comparison")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", frameon=False)

    # Add subtle family separators.
    last_family = None
    for i, fam in enumerate(df["task_family"]):
        if last_family is not None and fam != last_family:
            ax.axhline(i - 0.5, color="#9a9a9a", linewidth=0.7, alpha=0.55)
        last_family = fam

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("|SpectralMol - GraphGA|")
    fig.tight_layout()
    return save_figure(fig, out_dir, "figure3_guacamol_task_comparison", formats, dpi)


def figure4_guacamol_trajectories(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "guacamol" / "guacamol_all_comparison_generation_summary.tsv.gz"
    df = read_table(path, pd)
    ensure_columns(df, ["task_index", "task", "model", "generation", "mean_best_score_so_far"], path)
    df["generation"] = as_num(df["generation"], pd)
    df["mean_best_score_so_far"] = as_num(df["mean_best_score_so_far"], pd)
    df = df.dropna(subset=["generation", "mean_best_score_so_far"])
    grouped = (
        df.groupby(["task_index", "task", "model", "generation"], as_index=False)["mean_best_score_so_far"]
        .mean()
        .sort_values(["task_index", "model", "generation"])
    )

    tasks = grouped[["task_index", "task"]].drop_duplicates().sort_values("task_index")
    fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (_, task_row) in zip(axes, tasks.iterrows()):
        task_idx = task_row["task_index"]
        task_df = grouped[grouped["task_index"] == task_idx]
        for model, model_df in task_df.groupby("model"):
            key = "spectralmol" if str(model).lower().startswith("spectral") else "graphga"
            ax.plot(
                model_df["generation"],
                model_df["mean_best_score_so_far"],
                label=MODEL_LABEL.get(key, str(model)),
                color=MODEL_COLOR.get(key, None),
                linewidth=1.7,
            )
        ax.set_title(clean_task_label(task_row["task"]), fontsize=8)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
    for ax in axes[len(tasks) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("Generation")
    fig.supylabel("Mean running best score")
    fig.suptitle("Figure 4 update: GuacaMol best-score trajectories", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.965])
    return save_figure(fig, out_dir, "figure4_guacamol_best_score_trajectories", formats, dpi)


def figure5_guacamol_top200_distributions(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "guacamol" / "guacamol_top200_scores_by_seed_task_model.tsv.gz"
    df = read_table(path, pd)
    ensure_columns(df, ["task_index", "task", "model", "score"], path)
    df["score"] = as_num(df["score"], pd)
    df = df.dropna(subset=["score"])
    tasks = df[["task_index", "task"]].drop_duplicates().sort_values("task_index")

    fig, axes = plt.subplots(5, 4, figsize=(16, 14), sharey=True)
    axes = axes.ravel()
    for ax, (_, task_row) in zip(axes, tasks.iterrows()):
        task_df = df[df["task_index"] == task_row["task_index"]]
        spec = task_df[task_df["model"].str.lower().str.contains("spectral")]["score"].to_numpy()
        graph = task_df[task_df["model"].str.lower().str.contains("graph")]["score"].to_numpy()
        data = [spec, graph]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.75, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], [MODEL_COLOR["spectralmol"], MODEL_COLOR["graphga"]]):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.55)
        if "cmeans" in parts:
            parts["cmeans"].set_color("#222222")
            parts["cmeans"].set_linewidth(1.0)
        ax.boxplot(
            data,
            positions=[0, 1],
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "alpha": 0.8, "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.0},
        )
        ax.set_title(clean_task_label(task_row["task"]), fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SpectralMol", "GraphGA"], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
    for ax in axes[len(tasks) :]:
        ax.axis("off")
    fig.supylabel("Top-200 molecule score")
    fig.suptitle("Figure 5 update: GuacaMol top-200 score distributions", y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.965])
    return save_figure(fig, out_dir, "figure5_guacamol_top200_score_distributions", formats, dpi)


def figure6_saturn_qed_sa(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "saturn" / "figure6_qed_sa_docking_hits.tsv"
    df = read_table(path, pd)
    ensure_columns(df, ["threshold", "qed", "sa", "docking_raw"], path)
    for col in ["threshold", "qed", "sa", "docking_raw"]:
        df[col] = as_num(df[col], pd)

    thresholds = [-9.0, -10.0]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    for ax, threshold in zip(axes, thresholds):
        sub = df[df["threshold"] == threshold].copy()
        sc = ax.scatter(
            sub["qed"],
            sub["sa"],
            c=sub["docking_raw"],
            cmap="magma_r",
            s=18,
            alpha=0.72,
            linewidths=0,
        )
        ref = SATURN_REFERENCE[threshold]
        ax.axvline(ref["qed"], color="#2b6cb0", linestyle="--", linewidth=1.1, label="Saturn mean QED")
        ax.axhline(ref["sa"], color="#c53030", linestyle="--", linewidth=1.1, label="Saturn mean SA")
        ax.set_title(f"Docking score < {threshold:.0f} kcal/mol (n={len(sub)})")
        ax.set_xlabel("QED")
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("SA score")
    cbar = fig.colorbar(sc, ax=axes, pad=0.02)
    cbar.set_label("Docking score (kcal/mol)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Figure 6 update: SATURN QED-SA distributions for docking hits", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save_figure(fig, out_dir, "figure6_saturn_qed_sa_docking_hits", formats, dpi)


def figure7_saturn_initial_final(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "saturn" / "figure7_initial_final_population.tsv"
    df = read_table(path, pd)
    ensure_columns(df, ["population_stage", "docking_raw", "qed", "sa"], path)
    for col in ["docking_raw", "qed", "sa"]:
        df[col] = as_num(df[col], pd)
    stages = ["initial", "final"]
    colors = {"initial": "#7f7f7f", "final": MODEL_COLOR["spectralmol"]}
    metrics = [
        ("docking_raw", "Docking score (kcal/mol)", (-13, -3)),
        ("qed", "QED", (0, 1)),
        ("sa", "SA score", (1, 6)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (col, label, xlim) in zip(axes, metrics):
        for stage in stages:
            vals = df.loc[df["population_stage"] == stage, col].dropna()
            ax.hist(vals, bins=42, density=True, alpha=0.45, color=colors[stage], label=stage.capitalize())
        if col == "docking_raw":
            ax.axvline(-9, color="#444444", linestyle="--", linewidth=1.0)
            ax.axvline(-10, color="#444444", linestyle=":", linewidth=1.0)
        ax.set_xlabel(label)
        ax.set_xlim(*xlim)
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 7 update: SATURN initial vs final population distributions", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save_figure(fig, out_dir, "figure7_saturn_initial_final_distributions", formats, dpi)


def figure8_saturn_pareto(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    pop_path = input_dir / "saturn" / "figure8_population_with_f1_flag.tsv"
    f1_path = input_dir / "saturn" / "figure8_f1_pareto_front.tsv"
    pop = read_table(pop_path, pd)
    f1 = read_table(f1_path, pd)
    ensure_columns(pop, ["docking_raw", "qed", "sa", "is_f1"], pop_path)
    ensure_columns(f1, ["docking_raw", "qed", "sa"], f1_path)
    for frame in [pop, f1]:
        for col in ["docking_raw", "qed", "sa"]:
            frame[col] = as_num(frame[col], pd)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    axes[0].scatter(pop["docking_raw"], pop["qed"], s=12, color="#bdbdbd", alpha=0.35, label="Final population")
    axes[0].scatter(f1["docking_raw"], f1["qed"], s=28, color=MODEL_COLOR["spectralmol"], alpha=0.85, label="F1 Pareto front")
    axes[0].axvline(-9, color="#444444", linestyle="--", linewidth=1.0)
    axes[0].axvline(-10, color="#444444", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Docking score (kcal/mol)")
    axes[0].set_ylabel("QED")
    axes[0].set_title("Docking vs QED")

    axes[1].scatter(pop["sa"], pop["qed"], s=12, color="#bdbdbd", alpha=0.35, label="Final population")
    axes[1].scatter(f1["sa"], f1["qed"], s=28, color=MODEL_COLOR["spectralmol"], alpha=0.85, label="F1 Pareto front")
    axes[1].set_xlabel("SA score")
    axes[1].set_ylabel("QED")
    axes[1].set_title("SA vs QED")

    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
    fig.suptitle("Figure 8 update: SATURN final population and first Pareto front", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, out_dir, "figure8_saturn_pareto_front", formats, dpi)


def figure9_saturn_crowding(input_dir: Path, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    plt, np, pd, _ = require_plotting()
    path = input_dir / "saturn" / "figure9_f1_crowding_distances.tsv"
    df = read_table(path, pd)
    ensure_columns(df, ["rank_by_scalar", "crowding_distance", "docking_raw", "qed", "sa"], path)
    for col in ["rank_by_scalar", "crowding_distance", "docking_raw", "qed", "sa"]:
        df[col] = as_num(df[col], pd)
    df = df.replace([math.inf, -math.inf], math.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    finite = df["crowding_distance"].dropna()
    axes[0].hist(finite, bins=35, color=MODEL_COLOR["spectralmol"], alpha=0.72)
    axes[0].set_xlabel("Crowding distance")
    axes[0].set_ylabel("F1 molecule count")
    axes[0].set_title("F1 crowding-distance distribution")
    axes[0].grid(color="#dddddd", linewidth=0.5, alpha=0.7)

    axes[1].scatter(df["rank_by_scalar"], df["crowding_distance"], c=df["docking_raw"], cmap="magma_r", s=26, alpha=0.75)
    axes[1].set_xlabel("Scalar rank within final population")
    axes[1].set_ylabel("Crowding distance")
    axes[1].set_title("Crowding vs scalar rank")
    axes[1].grid(color="#dddddd", linewidth=0.5, alpha=0.7)
    fig.suptitle("Figure 9 update: SATURN F1 Pareto-front crowding distances", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, out_dir, "figure9_saturn_f1_crowding_distances", formats, dpi)


FIGURE_BUILDERS: list[tuple[str, Callable[[Path, Path, list[str], int], list[Path]]]] = [
    ("figure3_guacamol_task_comparison", figure3_guacamol_task_comparison),
    ("figure4_guacamol_best_score_trajectories", figure4_guacamol_trajectories),
    ("figure5_guacamol_top200_score_distributions", figure5_guacamol_top200_distributions),
    ("figure6_saturn_qed_sa_docking_hits", figure6_saturn_qed_sa),
    ("figure7_saturn_initial_final_distributions", figure7_saturn_initial_final),
    ("figure8_saturn_pareto_front", figure8_saturn_pareto),
    ("figure9_saturn_f1_crowding_distances", figure9_saturn_crowding),
]


def parse_formats(raw: str) -> list[str]:
    formats = [x.strip().lower().lstrip(".") for x in raw.split(",") if x.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required.")
    return formats


def main(argv: list[str] | None = None) -> int:
    default_input = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input, help="Path to manuscript_update_20260824 package.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for regenerated figures.")
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,pdf,svg"), help="Comma-separated formats.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs.")
    parser.add_argument(
        "--only",
        default="",
        help="Optional comma-separated figure stems to generate, e.g. figure3_guacamol_task_comparison,figure6_saturn_qed_sa_docking_hits.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    out_dir = (args.output_dir or (input_dir / "figures_updated")).resolve()
    selected = {x.strip() for x in args.only.split(",") if x.strip()}

    manifest_rows: list[dict[str, str]] = []
    for stem, builder in FIGURE_BUILDERS:
        if selected and stem not in selected:
            continue
        print(f"[figures] building {stem}", flush=True)
        try:
            outputs = builder(input_dir, out_dir, args.formats, args.dpi)
        except Exception as exc:
            print(f"[figures][error] {stem}: {exc}", file=sys.stderr)
            raise
        for output in outputs:
            try:
                manifest_path = output.relative_to(out_dir).as_posix()
            except ValueError:
                manifest_path = str(output)
            manifest_rows.append(
                {
                    "figure": stem,
                    "path": manifest_path,
                    "format": output.suffix.lstrip("."),
                    "size_bytes": str(output.stat().st_size),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "figure_generation_manifest.tsv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["figure", "path", "format", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[figures] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
