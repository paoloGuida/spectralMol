#!/usr/bin/env python3
"""G7 - Four oceans (ablation scene A1).

Four seas, same energy, different wave content:
  full   = low swells + fine ripples      high = only fine chop
  low    = only slow big swells           random = broadband, uncorrelated, incoherent
A boat sails from the shared starting population toward a lighthouse.

WHAT IS DATA AND WHAT IS NOT
  * Wave content is ILLUSTRATIVE. The GuacaMol ablation (full / high-only / low-only / random-matrix) is an
    operator-space proxy, not Fourier-space, so the seas are a metaphor for the paper's frequency modes.
  * Boat position IS data: seed- and task-averaged best-so-far score per generation, rescaled so every boat
    starts at the shared initial population (progress 0) and the lighthouse is the best any condition reached
    (progress 1). Averaged over the GuacaMol tasks that are not saturated (see `select_tasks`).
  * What the data says (and the picture therefore shows): random matrix is clearly worst (~0.31 of the way);
    high-only (0.97), full (0.90) and low-only (0.83) end up close to one another. The progress rescaling
    stretches differences relative to raw scores (raw task-mean finals: 0.754 high, 0.749 full, 0.737 low,
    0.647 random), and per-seed spread was not tested here, so only 'random is worst' is a robust claim.

Loop: empty -> boats sail -> arrive, lighthouse flares -> fade to empty (frame N == frame 0).
Usage: g7_four_oceans.py [--fps 24] [--data-only]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

import common as C
import guac_data as G

COND = {"full": ("#7ee8ff", "full spectrum"), "high": (C.HIGH, "high frequencies only"),
        "low": ("#6f8bff", "low frequencies only"), "random": ("#9aa5b1", "random matrix")}
ORDER = ["full", "high", "low", "random"]       # panel order: row-major 2x2
CACHE = C.CACHE_DIR / "g7_progress_cache.npz"

T_FADE_IN, T_SAIL0, T_SAIL1, T_FADE_OUT0, T_TOTAL = 0.0, 0.6, 6.8, 8.4, 9.2


# ---------------------------------------------------------------------------
# Data: boat progress
# ---------------------------------------------------------------------------
def all_tasks() -> list[str]:
    pat = str(G.RUN_ROOT / "runs" / "local_evolution" / "seed_7" / "case_*" / "benchmark_runs" / "*" / "*_*")
    return sorted({os.path.basename(p).split("evolution_ga_", 1)[1] for p in glob.glob(pat)
                   if os.path.isdir(p) and "evolution_ga_" in p})


def best_so_far(model: str, seed: int, task: str) -> np.ndarray:
    df = G.load_run(model, seed, task, [])
    ng = int(df.generation.max()) + 1
    b = df.groupby("generation").score.max().reindex(range(ng)).ffill()
    return np.maximum.accumulate(b.values)


def select_tasks(finals: dict[str, dict[str, float]]) -> list[str]:
    """Not saturated: the four conditions do not all sit at ~1.0, and the best condition is above the
    floor (excludes Valsartan_smarts). Chosen without looking at the ordering of conditions."""
    keep = []
    for t, f in finals.items():
        v = np.array(list(f.values()))
        if (v < 0.999).any() and v.max() > 0.1:
            keep.append(t)
    return keep


def progress_curves() -> tuple[dict[str, np.ndarray], list[str]]:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in ORDER}, list(z["tasks"])
    tasks = all_tasks()
    curves: dict[str, dict[str, np.ndarray]] = {}       # task -> model -> (seeds, gens)
    finals: dict[str, dict[str, float]] = {}
    for t in tasks:
        curves[t], finals[t] = {}, {}
        for k in ORDER:
            arr = np.stack([best_so_far(G.MODELS[k], s, t) for s in G.SEEDS])
            curves[t][k] = arr
            finals[t][k] = float(arr[:, -1].mean())
    sel = select_tasks(finals)
    out = {}
    for k in ORDER:
        per_task = []
        for t in sel:
            norm = max(finals[t].values())
            p0 = np.mean([curves[t][c][:, 0].mean() for c in ORDER])
            per_task.append(((curves[t][k] - p0) / max(norm - p0, 1e-9)).mean(0))
        out[k] = np.mean(per_task, axis=0)
    np.savez(CACHE, tasks=np.array(sel), **out)
    return out, sel


# ---------------------------------------------------------------------------
# Sea model (integer temporal harmonics -> exactly periodic)
# ---------------------------------------------------------------------------
def make_waves(kind: str):
    """list of (k, amp, n, phase): y = sum amp*sin(2pi*(k*x + n*t/T) + phase), unit RMS."""
    rng = np.random.RandomState({"full": 1, "high": 2, "low": 3, "random": 4}[kind])
    if kind == "low":
        comps = [(1, 1.0, 1), (2, 0.7, 1), (3, 0.45, 2)]
    elif kind == "high":
        comps = [(9, 0.7, 3), (11, 0.8, 4), (13, 0.7, 5), (16, 0.5, 6)]
    elif kind == "full":
        comps = [(1, 1.0, 1), (2, 0.7, 1), (3, 0.45, 2), (9, 0.5, 3), (11, 0.55, 4), (13, 0.5, 5), (16, 0.35, 6)]
    else:  # random: flat spectrum, random phase and travel direction/speed per component
        comps = []
        for k in range(1, 19):
            n = int(rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]))
            comps.append((k, float(rng.uniform(0.3, 1.0)), n))
    ph = rng.uniform(0, 2 * np.pi, len(comps))
    rms = np.sqrt(sum(a * a for _, a, _ in comps) / 2)
    return [(k, a / rms, n, p) for (k, a, n), p in zip(comps, ph)]


def field(x, t_over_T, waves, dphase=0.0):
    y = np.zeros_like(x)
    for k, a, n, p in waves:
        y += a * np.sin(2 * np.pi * (k * x + n * t_over_T) + p + dphase * k)
    return y


# ---------------------------------------------------------------------------
# Drawing (panel coordinates: x,y in [0,1]; px conversion for boat shape)
# ---------------------------------------------------------------------------
PW, PH = 480.0, 270.0
SEA0 = 0.40           # mean sea level
AMP = 0.060           # wave amplitude (RMS) in panel-y units
X0, XL = 0.08, 0.80   # boat start / arrival x
LIGHT_X = 0.925


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def boat_polys(x, y, ang):
    """Hull + sail polygons in panel coords, centred at (x,y) on the water, rotated by ang (rad, pixel space)."""
    hull = np.array([[-17, 2], [17, 2], [12, -7], [-12, -7]], float)
    sail = np.array([[-1, 4], [-1, 34], [14, 4]], float)
    jib = np.array([[-4, 4], [-4, 26], [-17, 4]], float)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    out = []
    for P in (hull, sail, jib):
        Q = P @ R.T
        out.append(np.c_[x + Q[:, 0] / PW, y + Q[:, 1] / PH])
    return out


def lighthouse(ax, color, level):
    """Tower on a rock; lamp glow and beam scale with `level` in [0,1]."""
    bx = LIGHT_X
    ax.fill([bx - 0.05, bx + 0.05, bx + 0.025, bx - 0.025], [SEA0 - 0.05, SEA0 - 0.05, SEA0 + 0.02, SEA0 + 0.02],
            color="#0c1c28", zorder=3)
    tw0, tw1 = 14 / PW / 2, 8 / PW / 2
    y0, y1 = SEA0 + 0.02, SEA0 + 0.50
    ax.fill([bx - tw0, bx + tw0, bx + tw1, bx - tw1], [y0, y0, y1, y1], color="#16303f", zorder=3)
    for fy in (0.17, 0.30):                                         # bands
        w = tw0 + (tw1 - tw0) * (fy / 0.5)
        ax.plot([bx - w, bx + w], [y0 + fy] * 2, color="#26485c", lw=1.5, zorder=3.1)
    lamp = (bx, y1 + 0.02)
    ax.fill([bx - 0.016, bx + 0.016, bx + 0.013, bx - 0.013], [y1, y1, y1 + 0.04, y1 + 0.04], color="#1d3a4c", zorder=3)
    if level > 0.02:
        # beam wedge toward the sea on the left
        a = 0.34 * level ** 1.5
        ax.fill([lamp[0], 0.12, 0.12], [lamp[1], lamp[1] + 0.14, lamp[1] - 0.20], color=color, alpha=a * 0.45, lw=0, zorder=1.5)
        ax.fill([lamp[0], 0.34, 0.34], [lamp[1], lamp[1] + 0.06, lamp[1] - 0.10], color=color, alpha=a * 0.6, lw=0, zorder=1.6)
        for r, al in ((0.060, 0.07), (0.038, 0.12), (0.022, 0.22)):
            ax.scatter([lamp[0]], [lamp[1]], s=(r * PW * 2) ** 2 * 0.78, color=color, alpha=al * level, lw=0, zorder=4)
        ax.scatter([lamp[0]], [lamp[1]], s=18, color="#ffffff", alpha=0.25 + 0.75 * level, lw=0, zorder=4.2)
    else:
        ax.scatter([lamp[0]], [lamp[1]], s=10, color="#3a5668", lw=0, zorder=4)


def render_frame(fig, axes, t, curves, waves, ngen):
    tT = t / T_TOTAL
    xs = np.linspace(0, 1, 360)
    g_pos = np.clip((t - T_SAIL0) / (T_SAIL1 - T_SAIL0), 0, 1) * (ngen - 1)
    gi = int(np.floor(g_pos))
    for ax, k in zip(axes, ORDER):
        ax.cla()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        col, label = COND[k]
        w = waves[k]
        # progress -> boat x
        c = curves[k]
        p = np.interp(g_pos, np.arange(ngen), c)
        p = float(np.clip(p, 0, 1))
        bx = X0 + (XL - X0) * p
        # sea: three layers, back to front
        for li, (dph, amp, alpha, dy) in enumerate(((1.6, 0.8, 0.10, 0.045), (0.8, 0.9, 0.16, 0.022), (0.0, 1.0, 0.30, 0.0))):
            ys = SEA0 + dy + AMP * amp * field(xs, tT, w, dphase=dph)
            ax.fill_between(xs, 0, ys, color=col, alpha=alpha, lw=0, zorder=1 + li * 0.2)
        ys0 = SEA0 + AMP * field(xs, tT, w)
        C.glow(ax, xs, ys0, col, lw=1.3, alpha=0.95, z=2)
        # boat rides the locally averaged surface
        win = np.linspace(-0.035, 0.035, 9)
        yb = SEA0 + AMP * field(bx + win, tT, w).mean()
        slope = (AMP * (field(np.array([bx + 0.04]), tT, w) - field(np.array([bx - 0.04]), tT, w))[0]) / 0.08
        ang = float(np.arctan(slope * PH / PW)) * 0.8
        for poly, fc, a in zip(boat_polys(bx, yb, ang), ("#d7e6f2", C.WHITE, "#b6cde0"), (0.95, 0.95, 0.80)):
            ax.fill(poly[:, 0], poly[:, 1], color=fc, alpha=a, lw=0, zorder=3.5)
            ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], color=col, lw=0.8, alpha=0.9, zorder=3.6)
        # wake trail: faint dots behind the boat (path so far)
        tr = np.linspace(X0, bx, 40)
        ax.scatter(tr, SEA0 + AMP * field(tr, tT, w) - 0.012, s=1.2, color=col, alpha=0.22, lw=0, zorder=2.2)
        lighthouse(ax, col, smooth((p - 0.55) / 0.45))
        ax.text(0.03, 0.93, label, color=col, fontsize=9.5, va="top", ha="left", alpha=0.95)
    fig.canvas.draw()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--data-only", action="store_true")
    a = ap.parse_args()

    curves, tasks = progress_curves()
    ngen = len(curves["full"])
    print(f"[g7] {len(tasks)} non-saturated tasks: {tasks}")
    for k in ORDER:
        c = curves[k]
        print(f"[g7] progress {k:6s} gen0={c[0]:.3f} gen10={c[min(10, ngen-1)]:.3f} gen25={c[min(25, ngen-1)]:.3f} final={c[-1]:.3f}")
    if a.data_only:
        return

    waves = {k: make_waves(k) for k in ORDER}
    fig = C.new_fig()
    gs = fig.add_gridspec(2, 2, left=0.012, right=0.988, top=0.985, bottom=0.075, wspace=0.025, hspace=0.04)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor(C.BG)
    cap = fig.text(0.5, 0.025, "illustrative wave content   ·   boat = mean best-so-far, start to best reached "
                   f"({len(tasks)} GuacaMol tasks x 6 seeds, operator-space proxy)",
                   color=C.GREY, fontsize=7.5, ha="center", va="center")
    gtxt = fig.text(0.985, 0.025, "", color=C.GREY, fontsize=7.5, ha="right", va="center")

    nfr = int(round(T_TOTAL * a.fps))
    bg = np.array(Image.new("RGB", (960 * C.SCALE, 540 * C.SCALE), C.BG))
    frames = []
    for i in range(nfr):
        t = i / a.fps
        render_frame(fig, axes, t, curves, waves, ngen)
        g = int(round(np.clip((t - T_SAIL0) / (T_SAIL1 - T_SAIL0), 0, 1) * (ngen - 1)))
        gtxt.set_text(f"generation {g}/{ngen - 1}")
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth((t - T_FADE_IN) / 0.8) * (1 - smooth((t - T_FADE_OUT0) / (T_TOTAL - T_FADE_OUT0)))
        im = np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(im))
        if i % 48 == 0:
            print(f"[g7] frame {i}/{nfr}", flush=True)
    d = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(frames[-1]).astype(int)).max()
    print(f"[g7] first-vs-last max pixel diff = {d} (frame N would be frame 0 -> loop seamless if ~0)")
    gif = C.save_loop(frames, "G7_four_oceans", fps=a.fps, gif_colors=64)
    print(f"[g7] wrote {gif}")


if __name__ == "__main__":
    main()
