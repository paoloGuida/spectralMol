#!/usr/bin/env python3
"""F10 - Pareto Frontier Sweep (GuacaMol two-objective task, REAL run data).

Task      : Median_molecules_1 by default (objectives: similarity to camphor, similarity to menthol).
Phase A   : seed-7 full-spectrum run. Molecules stream in generation by generation; the white staircase is
            the non-dominated set of everything visited so far; the shaded region is what it dominates.
            Inset: the best-score molecule so far (score = geometric mean of the two similarities).
Phase B   : final fronts of the four frequency-mode conditions, 6 seeds pooled per condition.
CAVEAT    : these runs optimised the geometric mean with the SMILES-operator evolver (conditions are
            operator-space proxies); they did NOT use NSGA-II. The picture is the non-dominated set of
            molecules the runs visited in the two objectives.
Loop      : empty -> sweep -> conditions -> fade to empty (frame N == frame 0).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import common as C
import guac_data as G

COND_STYLE = {"full": ("#7ee8ff", "full"), "high": (C.HIGH, "high only"),
              "low": ("#6f8bff", "low only"), "random": ("#9aa5b1", "random")}
TASKS = {"Median_molecules_1": ["camphor_Sim", "menthol_Sim"], "Median_molecules_2": ["tadalafil_Sim", "sildenafil_Sim"]}


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def staircase(F: np.ndarray):
    """Boundary polygon of the region dominated by front F (maximise both)."""
    F = F[np.argsort(F[:, 0])]                       # x ascending => y descending
    pts = [(0.0, F[0, 1])]
    for k in range(len(F)):
        pts.append((F[k, 0], F[k, 1]))
        pts.append((F[k, 0], F[k + 1, 1] if k + 1 < len(F) else 0.0))
    return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Median_molecules_1", choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    comps = TASKS[a.task]

    run = G.load_run(G.MODELS["full"], a.seed, a.task, comps)
    P = run[comps].values; gen = run.generation.values; score = run.score.values
    smi = run.smiles.values
    ngen = int(gen.max()) + 1
    pooled = {}
    for k, m in G.MODELS.items():
        dfs = [G.load_run(m, s, a.task, comps)[comps].values for s in G.SEEDS]
        U = np.vstack(dfs)
        pooled[k] = (U, U[G.front_idx(U)])
    allmax = np.max([U.max(0) for U, _ in pooled.values()] + [P.max(0)], axis=0)
    xlimB, ylimB = allmax[0] * 1.10, allmax[1] * 1.10
    xlimA, ylimA = P[:, 0].max() * 1.18, P[:, 1].max() * 1.18
    print(f"task={a.task} gens={ngen} points={len(P)}; pooled front sizes:",
          {k: len(f) for k, (_, f) in pooled.items()})
    print("pooled hypervolume:", {k: round(G.hypervolume(U), 4) for k, (U, _) in pooled.items()})

    fps = a.fps
    t0, per_gen = 0.6, 0.232
    t_sweep_end = t0 + ngen * per_gen
    t_hold_end = t_sweep_end + 1.0
    t_cond0 = t_hold_end + 0.6
    cond_order = ["full", "high", "low", "random"]
    t_cond = {k: t_cond0 + 0.8 * i for i, k in enumerate(cond_order)}
    T = t_cond["random"] + 0.6 + 2.0 + 1.0          # last cond fade-in, hold, fade-out
    t_fade = T - 1.0
    nf = int(round(T * fps))
    print(f"T={T:.1f}s frames={nf}")

    # best-so-far molecule per generation boundary (by score)
    frames = []
    for f in range(nf):
        time = f / fps
        fade = 1.0 - smooth((time - t_fade) / 1.0) if time > t_fade else 1.0
        fade_in = smooth(time / 0.5)
        g_alpha = fade * fade_in
        p = np.clip((time - t0) / per_gen, 0, ngen)
        g = min(int(p), ngen - 1)
        frac = p - g if p < ngen else 1.0
        n_rev = int(np.searchsorted(gen, g, side="left") + frac * (gen == g).sum()) if p < ngen else len(P)
        n_rev = max(0, min(len(P), n_rev))
        # phase B progress (dims phase A cloud)
        dimB = smooth((time - t_hold_end) / 0.6)

        fig = C.new_fig()
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.88], facecolor=C.BG)
        zoom = smooth((time - t_hold_end) / 1.4)           # ease axes out as the condition fronts arrive
        xlim = xlimA + (xlimB - xlimA) * zoom; ylim = ylimA + (ylimB - ylimA) * zoom
        ax.set_xlim(0, xlim); ax.set_ylim(0, ylim); ax.axis("off")
        ax.plot([0, xlim], [0, 0], color="#16283a", lw=1.2, alpha=g_alpha, zorder=1)
        ax.plot([0, 0], [0, ylim], color="#16283a", lw=1.2, alpha=g_alpha, zorder=1)

        if n_rev > 0:
            Q = P[:n_rev]
            old = Q[: max(0, n_rev - int(frac * 256) - 1)] if p < ngen else Q
            cloud_a = 0.16 * (1 - 0.65 * dimB) * g_alpha
            ax.scatter(Q[:, 0], Q[:, 1], s=6, color="#3fa7d6", alpha=cloud_a, linewidths=0, zorder=2, rasterized=True)
            if p < ngen:
                cur = P[gen == g][: int(frac * 256) + 1]
                ax.scatter(cur[:, 0], cur[:, 1], s=9, color="#d6f6ff", alpha=0.85 * g_alpha, linewidths=0, zorder=3)
            fi = G.front_idx(Q); F = Q[fi]
            sA = (1 - dimB) * g_alpha
            if sA > 0.01:
                stc = staircase(F)
                ax.add_patch(Polygon(np.vstack([stc, [[0, 0]]]), closed=True, facecolor="#7ee8ff", alpha=0.07 * sA, lw=0, zorder=1.5))
                C.glow(ax, stc[:, 0], stc[:, 1], C.WHITE, lw=2.0, alpha=sA, z=4)
                ax.scatter(F[:, 0], F[:, 1], s=26, color=C.WHITE, alpha=sA, linewidths=0, zorder=5)
            # best molecule inset
            best_i = int(np.argmax(score[:n_rev]))
            if sA > 0.02:
                axm = fig.add_axes([0.05, 0.60, 0.26, 0.34], facecolor=C.BG)
                axm.set_xlim(-1.05, 1.05); axm.set_ylim(-1.05, 1.05); axm.set_aspect("equal"); axm.axis("off")
                C.draw_mol(axm, smi[best_i], alpha=sA, lw=1.4, fontsize=8)

        # phase B: pooled fronts of the four conditions
        for k in cond_order:
            a_k = smooth((time - t_cond[k]) / 0.6) * fade
            if a_k <= 0.01:
                continue
            col = COND_STYLE[k][0]
            U, Fk = pooled[k]
            sub = U[:: max(1, len(U) // 3000)]
            ax.scatter(sub[:, 0], sub[:, 1], s=4, color=col, alpha=0.10 * a_k, linewidths=0, zorder=2, rasterized=True)
            stc = staircase(Fk)
            ax.add_patch(Polygon(np.vstack([stc, [[0, 0]]]), closed=True, facecolor=col, alpha=0.05 * a_k, lw=0, zorder=1.6))
            C.glow(ax, stc[:, 0], stc[:, 1], col, lw=2.0, alpha=a_k, z=4)
            ax.scatter(Fk[:, 0], Fk[:, 1], s=16, color=col, alpha=a_k, linewidths=0, zorder=5)

        # tiny legend (condition colours) + provenance caption
        a_leg = smooth((time - t_cond["full"]) / 0.6) * fade
        if a_leg > 0.01:
            for i, k in enumerate(cond_order):
                col, name = COND_STYLE[k]
                yy = 0.93 - 0.06 * i
                ax.plot([xlim * 0.80, xlim * 0.85], [ylim * yy, ylim * yy], color=col, lw=2.2, alpha=a_leg * smooth((time - t_cond[k]) / 0.6) + 0.0)
                ax.text(xlim * 0.86, ylim * yy, name, color=col, fontsize=8, va="center",
                        alpha=a_leg * smooth((time - t_cond[k]) / 0.6))
            fig.text(0.955, 0.012, "6 seeds pooled · operator-space proxy", color=C.GREY, fontsize=6.5, ha="right", alpha=a_leg)
        frames.append(C.fig_to_image(fig))
        plt.close(fig)

    print("wrote", C.save_loop(frames, "F10_front_sweep", fps=fps))


if __name__ == "__main__":
    main()
