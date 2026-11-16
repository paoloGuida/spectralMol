#!/usr/bin/env python3
"""G5 - Frequency brush (scene 4), computed on the REAL pipeline pieces.

Same set-up as G8: aspirin's exact genotype Theta* in the real Fourier basis (L=32, K=16, D=32) and the real
structured embedding; nearest-token decoding (tie-aware) -> SELFIES -> molecule.
A 'brush stroke' = Gaussian noise added to ONE frequency band of Theta, scaled so that the induced change in
Z has RMS = 1.2 x the RMS of Z itself (enough to flip a few tokens):
  blue  brush : low band  k 1-4        amber brush : high band k 9-16
The painted area is the change of the waveform (projection onto the leading direction). Slots that decode to a
different token than aspirin's are red.
Strokes shown are the FIRST three random draws of each brush that edit >= 2 tokens (not hand-picked).
Statistic over 300 draws per brush is printed and written in the caption: how often an edited token has an
edited neighbour. Low-band noise changes neighbouring slots together (smooth change of Z); high-band noise
changes isolated slots. It does NOT mean 'low band = scaffold': the effect is on contiguity, and flips are few.
This is NOT the repository's mutation operator (core.fourier_theta.mutate_Theta adds noise to all rows).
Loop: empty -> 3 blue strokes -> 3 amber strokes -> fade to empty (frame N == frame 0).
"""
from __future__ import annotations

import argparse

import numpy as np
import selfies as sf

import common as C
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, theta_to_Z
from core.vocab import build_vocab_from_semantic_robust_alphabet

ELEMS = {"C", "N", "O", "F", "S", "Cl"}
L, K, D = 32, 16, 32
Z_REL = 1.2
BANDS = {"low": (1, 4, "#6f8bff", "low band  k 1-4"), "high": (9, 16, C.HIGH, "high band  k 9-16")}
BAD, GOOD = "#ff7a7a", "#7ee8ff"
STROKE, T_INTRO, GAP = 1.5, 0.9, 0.5      # seconds per stroke / intro / gap between brushes


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--strokes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    vocab = build_vocab_from_semantic_robust_alphabet(allow_elements=set(ELEMS))
    tok2id = {t: i for i, t in enumerate(vocab)}
    E = build_structured_E(vocab=vocab, D=D, pad_token="[PAD]", seed=a.seed, target_std=0.4, allowed_elements=set(ELEMS))
    toks = list(sf.split_selfies(sf.encoder(a.smiles)))
    n = len(toks)
    ids = np.array([tok2id[t] for t in toks] + [tok2id["[PAD]"]] * (L - n))
    Phi = build_fourier_basis(L, K)
    M = Phi.shape[1]
    Theta = np.linalg.pinv(Phi) @ E[ids]
    kof = np.r_[0, np.arange(1, K + 1), np.arange(1, K + 1)]
    Z0 = theta_to_Z(Theta, Phi)
    zr = float(np.sqrt((Z0[:n] ** 2).mean()))
    w = np.linalg.svd(Z0[:n] - Z0[:n].mean(0), full_matrices=False)[2][0]

    def decode(Th):
        Z = theta_to_Z(Th, Phi)
        d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
        tied = d2[np.arange(L), ids] <= d2.min(1) + 1e-9
        dec = np.where(tied, ids, d2.argmin(1))
        out = [vocab[i] for i in dec if vocab[i] != "[PAD]"]
        try:
            smi = sf.decoder("".join(out))
        except Exception:
            smi = ""
        return smi, tied

    def band_noise(rng, lo, hi):
        dTh = np.zeros_like(Theta)
        sel = (kof >= lo) & (kof <= hi) & (np.arange(M) > 0)
        dTh[sel] = rng.normal(size=(sel.sum(), D))
        dZ = theta_to_Z(dTh, Phi)
        return dTh * (Z_REL * zr / np.sqrt((dZ[:n] ** 2).mean()))

    # ---- statistic over 300 draws + the first `strokes` draws with >=2 edited tokens
    strokes, adj = {}, {}
    for name, (lo, hi, _, _) in BANDS.items():
        rng = np.random.RandomState(100 + lo)
        picked, fracs = [], []
        for _ in range(300):
            dTh = band_noise(rng, lo, hi)
            _, tied = decode(Theta + dTh)
            bad = np.where(~tied[:n])[0]
            if len(bad) >= 2:
                fracs.append(float(np.mean([(b - 1 in bad) or (b + 1 in bad) for b in bad])))
                if len(picked) < a.strokes:
                    picked.append(dTh)
        strokes[name] = picked
        adj[name] = float(np.mean(fracs)) if fracs else float("nan")
        print(f"[g5] {name}: {len(fracs)}/300 draws edit >=2 tokens; edited tokens with an edited neighbour: {adj[name]:.2f}")
    assert all(len(v) == a.strokes for v in strokes.values()), "not enough qualifying draws"

    # ---- timeline: list of (t_start, band, dTh)
    sched, t = [], T_INTRO
    for name in ("low", "high"):
        for dTh in strokes[name]:
            sched.append((t, name, dTh)); t += STROKE
        t += GAP
    T_TOTAL = t - GAP + 1.0 + 0.8

    def state_at(tt):
        for t0, name, dTh in sched:
            if t0 <= tt < t0 + STROKE:
                u = (tt - t0) / STROKE
                r = float(smooth(u / 0.30) * (1 - smooth((u - 0.80) / 0.20)))
                return name, dTh, r
        return None, None, 0.0

    # dense waveform pieces (periodic in the slot index)
    uu = np.linspace(0, L, 480)
    def curve(Th):
        c = np.zeros_like(uu)
        coef = Th @ w
        for m in range(M):
            k = kof[m]
            if m == 0:
                c += coef[0]
            elif m <= K:
                c += coef[m] * np.cos(2 * np.pi * k * uu / L)
            else:
                c += coef[m] * np.sin(2 * np.pi * k * uu / L)
        return c
    base = curve(Theta)
    sc = 2.9 / np.abs(base).max()

    import matplotlib.pyplot as plt
    from PIL import Image
    fig = C.new_fig()
    axW = fig.add_axes([0.04, 0.30, 0.56, 0.60]); axS = fig.add_axes([0.04, 0.085, 0.56, 0.17])
    axR = fig.add_axes([0.64, 0.30, 0.34, 0.62]); axB = fig.add_axes([0.64, 0.085, 0.34, 0.15])
    nfr = int(round(T_TOTAL * a.fps))
    dec_list = []
    for i in range(nfr):
        name, dTh, r = state_at(i / a.fps)
        dec_list.append(decode(Theta + (r * dTh if dTh is not None else 0)))

    bg = np.array(Image.new("RGB", tuple(int(v) for v in fig.canvas.get_width_height()), C.BG))
    frames, last_change, prev_smi, cur_smi = [], -99, "", dec_list[0][0]
    energy = np.array([np.linalg.norm(Theta[kof == k]) for k in range(1, K + 1)])
    for i in range(nfr):
        t = i / a.fps
        name, dTh, r = state_at(t)
        smi, tied = dec_list[i]
        if smi != cur_smi:
            prev_smi, cur_smi, last_change = cur_smi, smi, i
        ramp = float(smooth((i - last_change) / 5.0))
        col = BANDS[name][2] if name else C.GREY
        Th = Theta + (r * dTh if dTh is not None else 0)

        axW.cla(); axW.set_xlim(0, 1); axW.set_ylim(-3.9, 4.1); axW.axis("off")
        x = uu / L
        cur = curve(Th)
        axW.plot(x, sc * base, color=C.GREY, lw=1.0, alpha=0.5, ls=(0, (3, 3)))
        if name:
            axW.fill_between(x, sc * base, sc * cur, color=col, alpha=0.55, lw=0)
        C.glow(axW, x, sc * cur, C.WHITE, lw=1.4, alpha=0.95, z=3)
        slot_y = sc * np.interp(np.arange(L), uu, cur)
        colr = [(GOOD if tied[j] else BAD) if j < n else "#46586a" for j in range(L)]
        axW.scatter(np.arange(L) / L, slot_y, s=[24 if j < n else 9 for j in range(L)], c=colr, zorder=4, lw=0)
        axW.text(0.0, 4.05, "paint = change of the waveform caused by the brush" if name else "", color=C.GREY, fontsize=7, va="top")
        label = BANDS[name][3] + "   (noise on these frequencies only)" if name else ""
        axW.text(0.0, -3.85, label, color=col, fontsize=10, va="bottom")

        axS.cla(); axS.set_xlim(0.4, K + 0.6); axS.set_ylim(0, 1.25); axS.axis("off")
        for k in range(1, K + 1):
            c_k = C.FREQ_CMAP((k - 1) / (K - 1))
            axS.bar(k, energy[k - 1] / energy.max(), width=0.7, color=c_k, alpha=0.35)
        if name:
            lo, hi = BANDS[name][:2]
            axS.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=0.18 + 0.12 * r)
            noise = np.array([np.linalg.norm((r * dTh)[kof == k]) for k in range(1, K + 1)])
            for k in range(lo, hi + 1):
                axS.bar(k, min(1.2, noise[k - 1] / energy.max()), width=0.35, color=C.WHITE, alpha=0.9)
        axS.text(0.45, 1.2, "frequency k  (bars: Θ energy; white: brush noise)", color=C.GREY, fontsize=6.5, va="top")

        axR.cla(); axR.set_xlim(-1.15, 1.15); axR.set_ylim(-1.15, 1.15); axR.set_aspect("equal"); axR.axis("off")
        C.draw_mol(axR, a.smiles, alpha=0.10, lw=0.8, fontsize=7, color=C.GREY)
        if prev_smi and ramp < 1:
            C.draw_mol(axR, prev_smi, alpha=0.85 * (1 - ramp), lw=1.5, fontsize=9)
        if cur_smi:
            C.draw_mol(axR, cur_smi, alpha=0.95 * ramp if prev_smi or i else 0.95, lw=1.5, fontsize=9)

        axB.cla(); axB.set_xlim(-0.5, n - 0.5); axB.set_ylim(-1.8, 1.0); axB.axis("off")
        for j in range(n):
            axB.scatter([j], [0.2], s=48, color=GOOD if tied[j] else "none", edgecolors=GOOD if tied[j] else BAD,
                        linewidths=1.0, zorder=3)
        axB.text((n - 1) / 2, -0.9, f"{int((~tied[:n]).sum())} of {n} tokens edited", color=C.GREY, fontsize=7, ha="center")
        if i == 0:
            fig.text(0.5, 0.025, (f"aspirin's exact genotype · nearest-token decode · noise strength {Z_REL}× signal · over 300 draws, edited tokens with an edited neighbour: "
                                  f"{adj['low']:.0%} (low band) vs {adj['high']:.0%} (high band)"),
                     color=C.GREY, fontsize=6.5, ha="center", va="center")
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth(t / 0.8) * (1 - smooth((t - (T_TOTAL - 0.8)) / 0.8))
        frames.append(Image.fromarray(np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)))
        if i % 48 == 0:
            print(f"[g5] frame {i}/{nfr} t={t:.1f} {name} r={r:.2f} edited={int((~tied[:n]).sum())} {smi}", flush=True)
    d = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(frames[-1]).astype(int)).max()
    print(f"[g5] first-vs-last max pixel diff = {d}")
    print(f"[g5] wrote {C.save_loop(frames, 'G5_frequency_brush', fps=a.fps, gif_colors=64)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
