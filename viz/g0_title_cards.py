#!/usr/bin/env python3
"""G0 - Title and outro cards: slow sine ribbons (integer harmonics) behind a line of text.

Texts are plain arguments so they can be changed without touching code:
  --title/--subtitle (title card)    --outro/--outro-sub (outro card)
Both fade from/to the background colour (frame N == frame 0).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common as C


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def card(name: str, line1: str, line2: str, seconds: float, fps: int, ribbon_alpha: float, size1: float) -> None:
    fig = C.new_fig()
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x = np.linspace(0, 1, 500)
    K = 6
    rng = np.random.RandomState(3)
    ph = rng.uniform(0, 2 * np.pi, K)
    nfr = int(round(seconds * fps))
    bg = np.array(Image.new("RGB", tuple(int(v) for v in fig.canvas.get_width_height()), C.BG))
    frames = []
    for i in range(nfr):
        t = i / nfr
        ax.cla(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for k in range(K):
            col = C.FREQ_CMAP(k / (K - 1))
            y = 0.5 + (0.30 - 0.03 * k) * np.sin(2 * np.pi * ((k + 1) * 0.5 * x * 2 + (1 + k % 2) * t) + ph[k]) * (0.55 + 0.1 * k) \
                + (k - (K - 1) / 2) * 0.035
            C.glow(ax, x, y, col, lw=1.3, alpha=ribbon_alpha, z=1)
        tt = i / fps
        ax.add_patch(plt.Rectangle((0.0, 0.36), 1.0, 0.28, color=C.BG, alpha=0.62 * smooth((tt - 0.4) / 0.9), lw=0, zorder=4))
        a_text = smooth((tt - 0.5) / 0.9) * (1 - smooth((tt - (seconds - 1.2)) / 0.9))
        ax.text(0.5, 0.56, line1, color=C.WHITE, fontsize=size1, ha="center", va="center", alpha=a_text, zorder=5,
                fontweight="bold")
        ax.text(0.5, 0.44, line2, color="#9fb4c6", fontsize=size1 * 0.46, ha="center",
                va="center", alpha=a_text, zorder=5)
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth(tt / 0.7) * (1 - smooth((tt - (seconds - 0.7)) / 0.7))
        frames.append(Image.fromarray(np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)))
    print(f"[g0] {name}: {C.save_loop(frames, name, fps=fps, gif_colors=64)}  ({nfr} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Molecules emerge from frequencies")
    ap.add_argument("--subtitle", default="frequency-controlled evolution, accelerated by GPU computing")
    ap.add_argument("--outro", default="same destination · sooner")
    ap.add_argument("--outro-sub", default="SpectralMol")
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    card("G0_title", a.title, a.subtitle, 5.0, a.fps, 0.28, 24)
    card("G13_outro", a.outro, a.outro_sub, 4.0, a.fps, 0.30, 24)


if __name__ == "__main__":
    main()
