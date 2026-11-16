#!/usr/bin/env python3
"""G10 - Same destination, sooner (REAL timings).

Four lanes = the four acceleration configurations (baseline CPU, Dask CPU, RAPIDS V100, Dask+RAPIDS V100).
x     = generation (0..30) of the Median_molecules_1 task.
Head  = how far each configuration has got at real wall-clock time t (seed-averaged time-to-generation from
        scores.csv `absolute_time`; animation clock = real clock x constant, so lane speed ratios are the
        measured ones, not exaggerated).
Trunk = the search lineage, drawn with the SAME shape/brightness gradient in every lane: for a given seed the
        best-so-far trajectory is identical across configurations (checked: bit-identical), so the lanes
        differ only in how fast they traverse it. Side twigs are schematic.
Star  = shared destination (same final best score); lights up when a lane arrives.
Dashed vertical line = where the baseline CPU head is, so faster lanes visibly lead it.
Loop  : empty -> race -> all arrived -> fade to empty (frame N == frame 0).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import common as C
import matrix_data as MD

LANES = [  # (cfg, label, colour)
    ("baseline_cpu", "CPU", "#b59f84"),
    ("dask_cpu", "Dask · CPU", "#d9b46a"),
    ("rapids_v100", "RAPIDS · V100", "#4fd1e8"),
    ("dask_rapids_v100", "Dask + RAPIDS · V100", "#a8f6ff"),
]


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Median_molecules_1")
    ap.add_argument("--race-seconds", type=float, default=7.0, help="animation seconds for the baseline to finish")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    curves = MD.race_curves(a.task)
    ng = len(curves["baseline_cpu"][0])
    gens = np.arange(ng, dtype=float)
    t_end = {c: curves[c][0] for c in curves}
    best = curves["baseline_cpu"][1]
    q = (best - best[0]) / max(best[-1] - best[0], 1e-9)                  # 0..1 progress of best-so-far
    print("seed-averaged finish time (s):", {c: round(float(t_end[c][-1]), 1) for c in t_end})
    base_T = t_end["baseline_cpu"][-1]
    k = a.race_seconds / base_T                                           # animation s per real s

    t0, hold, fade_d = 0.9, 1.8, 1.0
    arrive_anim = {c: t0 + t_end[c][-1] * k for c in t_end}
    T = t0 + a.race_seconds + hold + fade_d
    t_fade = T - fade_d
    nf = int(round(T * a.fps))
    print(f"T={T:.1f}s frames={nf}; arrival (anim s):", {c: round(v, 2) for c, v in arrive_anim.items()})

    rng = np.random.RandomState(11)
    twigs = [[(rng.uniform(0.45, 1.0), rng.choice([-1, 1]) * rng.uniform(0.14, 0.34)) for _ in range(3)]
             for _ in range(ng)]
    xs = np.linspace(0, ng - 1, 400)

    def trunk_y(x):
        return 0.10 * np.sin(0.9 * x) + 0.06 * np.sin(2.3 * x + 1.0)

    qx = np.interp(xs, gens, q)
    lane_y = {cfg: 3.0 - i for i, (cfg, _, _) in enumerate(LANES)}
    frames = []
    for f in range(nf):
        time = f / a.fps
        fade = 1.0 - smooth((time - t_fade) / fade_d) if time > t_fade else 1.0
        fin = smooth(time / 0.6)
        A = fade * fin
        t_real = max(0.0, (time - t0) / k)
        fig = C.new_fig()
        ax = fig.add_axes([0.17, 0.08, 0.78, 0.84], facecolor=C.BG)
        ax.set_xlim(-0.6, ng + 2.4); ax.set_ylim(-0.7, 3.7); ax.axis("off")

        gb = float(np.interp(t_real, t_end["baseline_cpu"], gens)) if t_real > 0 else 0.0
        # baseline front marker
        ax.plot([gb, gb], [-0.45, 3.45], color="#b59f84", lw=1.0, ls=(0, (3, 4)), alpha=0.35 * A * (1 if gb < ng - 1 else 0.0), zorder=1)

        for cfg, label, col in LANES:
            y0 = lane_y[cfg]
            gh = float(np.interp(t_real, t_end[cfg], gens)) if t_real > 0 else 0.0
            gh = min(gh, ng - 1.0)
            # lane track (faint) + label
            ax.plot([0, ng - 1], [y0, y0], color="#122234", lw=1.4, alpha=A, zorder=1)
            fig.text(0.015, 0.08 + 0.84 * (y0 + 0.7) / 4.4, label, color=col, fontsize=8.5, va="center", alpha=A)
            # trunk with brightness = best-so-far progress
            m = xs <= gh
            if m.sum() > 2:
                px, py = xs[m], y0 + trunk_y(xs[m])
                segs = np.stack([np.column_stack([px[:-1], py[:-1]]), np.column_stack([px[1:], py[1:]])], axis=1)
                from matplotlib.colors import to_rgba
                al = (0.30 + 0.70 * qx[m][:-1]) * A
                for lw, mult in ((7.0, 0.12), (3.8, 0.22), (1.8, 1.0)):
                    cols = [(*to_rgba(col)[:3], float(x) * mult) for x in al]
                    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round", zorder=3))
            # twigs
            for g in range(int(gh) + 1):
                age = gh - g
                if age < 0:
                    continue
                ax0, ay0 = g, y0 + trunk_y(g)
                grow = smooth(age / 1.2)
                al = 0.55 * np.exp(-age / 7.0) * A
                for (dx, dy) in twigs[g]:
                    ax.plot([ax0, ax0 + dx * grow], [ay0, ay0 + dy * grow], color=col, lw=0.9, alpha=al, zorder=2)
            # head
            hx, hy = gh, y0 + trunk_y(gh)
            if t_real > 0:
                for mult, aa in ((9, 0.07), (5, 0.14), (2.4, 0.3)):
                    ax.scatter([hx], [hy], s=50 * mult, color=col, alpha=aa * A, linewidths=0, zorder=4)
                ax.scatter([hx], [hy], s=40, color="#ffffff", alpha=A, linewidths=0, zorder=5)
            # shared destination star
            arrived = smooth((time - arrive_anim[cfg]) / 0.35) if time >= arrive_anim[cfg] else 0.0
            sx, sy = ng + 0.2, y0
            ax.scatter([sx], [sy], s=210, marker="*", color="#2a3b4c", alpha=0.9 * A, linewidths=0, zorder=3)
            if arrived > 0:
                for mult, aa in ((8, 0.06), (4.5, 0.12), (2.4, 0.22)):
                    ax.scatter([sx], [sy], s=210 * mult, marker="*", color=col, alpha=aa * arrived * A, linewidths=0, zorder=4)
                ax.scatter([sx], [sy], s=230, marker="*", color="#ffffff", alpha=arrived * A, linewidths=0, zorder=5)
                ring = (time - arrive_anim[cfg]) / 0.7
                if 0 < ring < 1:
                    ax.scatter([sx], [sy], s=900 * (0.3 + 2.2 * ring), facecolors="none", edgecolors=col,
                               alpha=0.8 * (1 - ring) * A, linewidths=1.3, zorder=4)
        frames.append(C.fig_to_image(fig))
        plt.close(fig)

    print("wrote", C.save_loop(frames, "G10_same_destination", fps=a.fps))


if __name__ == "__main__":
    main()
