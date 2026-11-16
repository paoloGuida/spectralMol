#!/usr/bin/env python3
"""G2 - Theta desk -> Z ribbon field.

Left : mixing desk = Theta (one slider per Fourier mode; blue = low freq, amber = high freq).
Right: Z = Phi @ Theta from the REAL core.fourier_theta.theta_to_Z, three latent channels
       drawn as ribbons over the L sequence slots.
Animation: the sliders of one frequency band swing with A*sin(2*pi*t) (periodic => seamless),
alternating low band then high band, so the viewer sees "low = broad reshaping,
high = fine ripple".
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

import common as C
from core.fourier_theta import build_fourier_basis, theta_to_Z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--D", type=int, default=3, help="latent channels drawn")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    L, K, D = a.L, a.K, a.D
    Phi = build_fourier_basis(L, K)
    M = Phi.shape[1]
    rng = np.random.RandomState(a.seed)
    theta0 = rng.normal(0, 0.26, size=(M, D))
    kidx = np.r_[0, np.arange(1, K + 1), np.arange(1, K + 1)]   # frequency of each Theta row
    low_rows = np.where((kidx >= 1) & (kidx <= max(1, K // 3)))[0]
    high_rows = np.where(kidx > K - max(1, K // 3))[0]

    ch_cols = ["#7ee8ff", "#c8a0ff", "#ffd27a"]
    n = int(round(a.seconds * a.fps))
    slots = np.arange(L)
    frames = []
    for f in range(n):
        t = f / n
        # first half of loop: low band swings; second half: high band swings
        phase_low = np.sin(2 * np.pi * min(1, t * 2)) if t < 0.5 else 0.0
        phase_high = np.sin(2 * np.pi * (t * 2 - 1)) if t >= 0.5 else 0.0
        theta = theta0.copy()
        theta[low_rows, 0] += 0.9 * phase_low
        theta[high_rows, 0] += 0.7 * phase_high
        Z = theta_to_Z(theta, Phi)                  # real Phi @ Theta

        fig = C.new_fig()
        # ---- desk (left) ----
        axd = fig.add_axes([0.03, 0.12, 0.34, 0.76], facecolor=C.BG)
        axd.set_xlim(-0.6, M - 0.4); axd.set_ylim(-3, 3); axd.axis("off")
        cols = C.freq_colors(K)
        for m in range(M):
            axd.plot([m, m], [-2.6, 2.6], color="#16283a", lw=5, solid_capstyle="round", zorder=1)
            v = float(np.clip(theta[m, 0], -2.4, 2.4))
            active = (m in low_rows and t < 0.5) or (m in high_rows and t >= 0.5)
            axd.scatter([m], [v], s=120 if active else 70, color=cols[m],
                        edgecolors=C.WHITE if active else "none", linewidths=1.2, zorder=3)
        # ---- ribbon field (right) ----
        axz = fig.add_axes([0.42, 0.12, 0.55, 0.76], facecolor=C.BG)
        axz.set_xlim(0, L - 1); axz.set_ylim(-5.0, 5.0); axz.axis("off")
        xs = np.linspace(0, L - 1, 400)
        for d in range(D):
            y = np.interp(xs, slots, Z[:, d])
            C.glow(axz, xs, y + (d - (D - 1) / 2) * 3.0, ch_cols[d % 3], lw=1.8)
            axz.scatter(slots, Z[:, d] + (d - (D - 1) / 2) * 3.0, s=7, color=ch_cols[d % 3], alpha=0.7, zorder=4, linewidths=0)
        frames.append(C.fig_to_image(fig))
        plt.close(fig)

    print("wrote", C.save_loop(frames, "G2_theta_desk", fps=a.fps))


if __name__ == "__main__":
    main()
