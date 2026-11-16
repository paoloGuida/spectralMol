#!/usr/bin/env python3
"""G1 - Fourier Mode Orchestra.

Each column of the real Phi (core.fourier_theta.build_fourier_basis) is drawn as a
luminous ribbon. Ribbons drift as travelling waves; because every mode k has an integer
number of cycles per loop, frame T == frame 0 exactly (seamless loop).
The white curve is a fixed-weight sum of the modes ("a Z channel").
"""
from __future__ import annotations

import argparse

import numpy as np

import common as C
from core.fourier_theta import build_fourier_basis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    L, K = a.L, a.K
    Phi = build_fourier_basis(L, K)          # (L, 1+2K): real basis, [DC, cos 1..K, sin 1..K]
    M = Phi.shape[1]
    cols = C.freq_colors(K)
    rng = np.random.RandomState(a.seed)
    w = rng.normal(0, 1, M) * np.r_[0.5, np.linspace(1.0, 0.45, K), np.linspace(1.0, 0.45, K)]

    n = int(round(a.seconds * a.fps))
    x = np.linspace(0, L, 600)               # fine grid for smooth drawing
    frames = []
    # ribbons shown: DC + cos k=1..K (sin partners are the same wave shifted; shown as drift)
    for f in range(n):
        t = f / n                            # t in [0,1) -> periodic
        fig = C.new_fig()
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.90], facecolor=C.BG)
        ax.set_xlim(0, L); ax.set_ylim(0, 10.4); ax.axis("off")

        total = np.zeros_like(x)
        for k in range(0, K + 1):
            # travelling wave: k integer cycles per loop => periodic in t
            phase = 2 * np.pi * (k * x / L - k * t) if k else np.zeros_like(x)
            y = np.cos(phase) if k else np.ones_like(x) * 0.6
            amp = 0.42 * (1.0 if k == 0 else 1.0 / (0.6 + 0.35 * k) )
            row = 1.0 + 1.55 * k             # low frequency at bottom, high at top
            col = cols[k]
            C.glow(ax, x, row + amp * y, col, lw=1.6, alpha=0.95)
            # dots = the discrete sequence positions i=0..L-1 (Phi is sampled only there)
            if k:
                idx = np.arange(L)
                ax.scatter(idx, row + amp * np.cos(2*np.pi*(k*idx/L - k*t)), s=5, color=col, alpha=0.55, zorder=3, linewidths=0)
            total += w[k] * amp * y

        # the sum, drawn above, in white-hot
        top = 1.0 + 1.55 * (K + 0.95)
        total = total / (np.abs(total).max() + 1e-9) * 0.9
        C.glow(ax, x, top + total, C.WHITE, lw=2.4, alpha=1.0, z=5)
        frames.append(C.fig_to_image(fig))
        import matplotlib.pyplot as plt; plt.close(fig)

    print("wrote", C.save_loop(frames, "G1_mode_orchestra", fps=a.fps))


if __name__ == "__main__":
    main()
