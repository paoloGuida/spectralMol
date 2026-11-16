#!/usr/bin/env python3
"""G3 - Nearest star (v2).

Stars  = the distinct rows of the REAL structured embedding table E (core.embedding.build_structured_E
         on the real vocabulary from core.vocab). Positions: 3D metric MDS of the distinct rows
         (keeps token-to-token distances; PCA collapsed many of them), then a linear map
         W with (E_u - mean) @ W = MDS coords, so any latent point Z_i is placed consistently.
Points = rows of Z = Phi @ Theta (real core.fourier_theta basis) for the token positions of
         aspirin's real SELFIES.
Motion = ILLUSTRATIVE: Theta is blended from the genotype of a RANDOM token sequence (every slot starts
         on some star) to aspirin's genotype (Theta* = pinv(Phi) E[tokens]) and back. NOT an evolution trace.
Camera = slow periodic yaw sway so stars that overlap in one view separate in another.
Each point links to its TRUE nearest star in the full D-dim space. A slot is 'locked' when its target
token ties for nearest (E has exact duplicate rows, e.g. [C]/[C+1], which distance alone cannot split;
the decoder's validity masks/state do that).
Loop: random -> aspirin -> hold -> random (frame N == frame 0).
"""
from __future__ import annotations

import argparse
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import selfies as sf
from matplotlib.colors import to_rgba
from sklearn.manifold import MDS

import common as C
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, theta_to_Z
from core.vocab import build_vocab_from_semantic_robust_alphabet

ELEMS = {"C", "N", "O", "F", "S", "Cl"}
GROUP_COL = {"atom": "#7ee8ff", "arom": "#9dffd0", "bond": "#c8a0ff", "branch": "#8a5cff",
             "ring": C.HIGH, "macro": "#ff9ec7", "pad": "#5a6a7a"}


def group_of(tok: str) -> str:
    if tok == "[PAD]":
        return "pad"
    if tok.startswith("[Ring") or tok.startswith("[=Ring"):
        return "ring"
    if "Branch" in tok:
        return "branch"
    if tok.startswith("[=") or tok.startswith("[#"):
        return "bond"
    core = tok.strip("[]")
    if core[:1].islower():
        return "arom"
    base = core.rstrip("+-0123456789")
    if base in ELEMS or base in {"nH"}:
        return "atom"
    return "macro"


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--yaw-deg", type=float, default=28.0, help="peak camera yaw (periodic sway)")
    a = ap.parse_args()

    vocab = build_vocab_from_semantic_robust_alphabet(allow_elements=set(ELEMS))
    tok2id = {t: i for i, t in enumerate(vocab)}
    E = build_structured_E(vocab=vocab, D=a.D, pad_token="[PAD]", seed=a.seed, target_std=0.4, allowed_elements=set(ELEMS))
    toks = list(sf.split_selfies(sf.encoder(a.smiles)))
    missing = [t for t in toks if t not in tok2id]
    if missing:
        raise SystemExit(f"tokens missing from vocab: {missing}")
    n = len(toks)
    ids = np.array([tok2id[t] for t in toks] + [tok2id["[PAD]"]] * (a.L - n))

    # distinct embedding rows (tolerance-based, robust to float noise)
    Eu, first, uid = np.unique(np.round(E, 5), axis=0, return_index=True, return_inverse=True)
    uid = uid.reshape(-1)
    nu = len(Eu)
    Eu = E[first]                                  # exact rows of the representatives
    groups_u = [group_of(vocab[i]) for i in first]
    members = [[vocab[v] for v in np.where(uid == k)[0]] for k in range(nu)]
    print(f"vocab={len(vocab)} distinct embedding rows={nu} D={a.D} L={a.L} K={a.K} tokens={n}")

    # --- 3D map: metric MDS on distinct rows + exact linear interpolant for off-star points
    M3 = MDS(n_components=3, init="random", n_init=8, max_iter=600, random_state=0).fit_transform(Eu)
    M3 -= M3.mean(0)
    mu = Eu.mean(0)
    W = np.linalg.pinv(Eu - mu) @ M3                # (D, 3); stars reproduced (almost) exactly
    print("map residual at stars (max):", float(np.abs((Eu - mu) @ W - M3).max()))
    star3 = (Eu - mu) @ W

    Phi = build_fourier_basis(a.L, a.K)
    Theta_t = np.linalg.pinv(Phi) @ E[ids]          # exactly encodes aspirin (Phi has full row rank)
    rng = np.random.RandomState(a.seed)
    rand_ids = np.array([first[rng.randint(nu)] for _ in range(a.L)])
    Theta_n = np.linalg.pinv(Phi) @ E[rand_ids]      # genotype of a random token sequence

    target_u = uid[ids[:n]]
    u_is_target = np.zeros(nu, bool); u_is_target[target_u] = True
    ucols = np.array([to_rgba(GROUP_COL[g]) for g in groups_u])
    pos_cols = [colorsys.hsv_to_rgb(i / n * 0.85, 0.55, 1.0) for i in range(n)]

    def s_of(t):
        u = abs(2 * t - 1)
        return float(smooth((u - 0.15) / 0.85))

    def view(P3, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        x = P3[:, 0] * c + P3[:, 2] * s
        z = -P3[:, 0] * s + P3[:, 2] * c
        return x, P3[:, 1], z

    # fixed limits: cover all yaws for stars
    ext = np.abs(star3).max(0)
    R = float(np.sqrt(ext[0] ** 2 + ext[2] ** 2)) * 1.12
    Ry = float(ext[1]) * 1.15
    lim_x, lim_y = R, max(Ry, R * 0.55)

    nf = int(round(a.seconds * a.fps))
    hist: list[np.ndarray] = []
    frames = []
    trail = 5
    for f in range(nf):
        t = f / nf
        yaw = np.deg2rad(a.yaw_deg) * np.sin(2 * np.pi * t)
        s = s_of(t)
        Theta = Theta_t + s * (Theta_n - Theta_t)
        Z = theta_to_Z(Theta, Phi)
        d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
        tied = d2[np.arange(a.L), ids] <= d2.min(1) + 1e-9
        dec = np.where(tied, ids, d2.argmin(1))
        correct = tied[:n]
        P3 = (Z[:n] - mu) @ W
        hist.append(P3.copy()); hist = hist[-trail:]

        fig = C.new_fig()
        ax = fig.add_axes([0.03, 0.14, 0.94, 0.84], facecolor=C.BG)
        ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y); ax.axis("off")

        sx, sy, sz = view(star3, yaw)
        depth = (sz / R + 1) / 2                     # 0 far .. 1 near
        locked = np.bincount(uid[dec[:n][correct]], minlength=nu) if correct.any() else np.zeros(nu, int)
        order = np.argsort(sz)                       # far first
        base = np.where(u_is_target, 150, 80) * (0.55 + 0.9 * depth)
        c0 = ucols.copy(); c0[:, 3] = np.where(u_is_target, 0.55, 0.28) * (0.5 + 0.6 * depth)
        ax.scatter(sx[order], sy[order], s=base[order], c=c0[order], linewidths=0, zorder=2)
        lk = np.where(locked > 0)[0]
        if lk.size:
            for mult, al in ((9, 0.06), (5, 0.12), (2.6, 0.25)):
                g = ucols[lk].copy(); g[:, 3] = al
                ax.scatter(sx[lk], sy[lk], s=base[lk] * mult, c=g, linewidths=0, zorder=2.5)
            ax.scatter(sx[lk], sy[lk], s=base[lk] * 1.2, color=C.WHITE, alpha=0.95, linewidths=0, zorder=3)

        px, py, pz = view(P3, yaw)
        for i in range(n):
            v = uid[dec[i]]
            ax.plot([px[i], sx[v]], [py[i], sy[v]], color=pos_cols[i], lw=0.8,
                    alpha=0.65 if correct[i] else 0.30, zorder=3)
        for k, H in enumerate(hist[:-1]):
            hx, hy, _ = view(H, yaw)
            ax.scatter(hx, hy, s=8, c=pos_cols, alpha=0.10 + 0.25 * (k + 1) / len(hist), linewidths=0, zorder=4)
        ax.scatter(px, py, s=40 * (0.7 + 0.6 * (pz / R + 1) / 2), c=pos_cols, edgecolors=C.WHITE,
                   linewidths=0.5, alpha=0.95, zorder=5)

        axs = fig.add_axes([0.03, 0.03, 0.94, 0.08], facecolor=C.BG)
        axs.set_xlim(-0.5, n - 0.5); axs.set_ylim(-1, 1); axs.axis("off")
        for i in range(n):
            axs.scatter([i], [0], s=190, color=pos_cols[i] if correct[i] else "#1b2b3b",
                        edgecolors=C.WHITE if correct[i] else "#2a3b4c", linewidths=1.0,
                        alpha=1.0 if correct[i] else 0.8, zorder=3)
        frames.append(C.fig_to_image(fig))
        plt.close(fig)
        if f in (0, nf // 2):
            print(f"frame {f}: locked slots {int(correct.sum())}/{n}")

    print("wrote", C.save_loop(frames, "G3_nearest_star", fps=a.fps))


if __name__ == "__main__":
    main()
