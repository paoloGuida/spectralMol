#!/usr/bin/env python3
"""G8 - Harmonic peel (ablation scene A2), computed on the REAL pipeline pieces.

Genotype  : Theta* = pinv(Phi) @ E[tokens of aspirin's real SELFIES]  (L=32, K=16, D=32), i.e. the exact
            encoding of aspirin in the real Fourier basis (core.fourier_theta) and the real structured
            embedding table (core.embedding, real vocab).
Curves    : the waveform is the projection of Z = Phi Theta onto its leading principal direction, drawn
            continuously (basis is periodic in the slot index u). Each coloured row is one frequency k
            (its cos+sin pair); the white curve is the sum of the rows that are currently switched on.
Peel      : mask m_k in [0,1] scales Theta's rows for frequency k. Phase A removes the high band (k 9..16),
            phase B the low band (k 1..8); both are restored in between. DC always stays.
Decode    : after every mask change Z' = Phi (m*Theta) is decoded by NEAREST TOKEN in the full D-dim space
            (tie-aware: E has exact duplicate rows, so a slot counts as correct when its target token ties
            for nearest), then SELFIES -> SMILES -> molecule. This is NOT the constraint-aware sampler of
            the full pipeline; it is the geometry of the representation.
WHAT IT SHOWS (and what it does not): this Theta* is redundant (33 modes for 19 tokens + padding), so the
molecule survives removing single bands/modes and degrades gradually; removing a whole half breaks it
(high half removed: 10/19 slots; low half removed: 14/19). It does NOT show 'low = scaffold,
high = substituents'; it shows that both ends of the spectrum are needed to spell this molecule exactly.
Charges from SELFIES (e.g. [C+1]) are not drawn.
Loop: empty -> peel high -> restore -> peel low -> restore -> fade to empty (frame N == frame 0).
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
GOOD, BAD = "#7ee8ff", "#ff7a7a"

# timeline (seconds)
A0, A1 = 1.0, 4.1          # peel high band
A_HOLD, A_RES0, A_RES1 = 5.2, 5.2, 6.0
B0, B1 = 6.6, 9.7          # peel low band
B_HOLD, B_RES0, B_RES1 = 10.8, 10.8, 11.6
F_OUT0, T_TOTAL = 12.0, 12.8
STAG_DUR = 0.5


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def masks_at(t: float) -> np.ndarray:
    """m[k] for k=0..K (k=0 is DC, always 1)."""
    m = np.ones(K + 1)
    ks_hi = list(range(K, K // 2, -1))          # 16..9  (first removed -> last removed)
    ks_lo = list(range(1, K // 2 + 1))          # 1..8
    for ph, (p0, p1, r0, r1, ks) in enumerate(((A0, A1, A_RES0, A_RES1, ks_hi), (B0, B1, B_RES0, B_RES1, ks_lo))):
        step = (p1 - p0 - STAG_DUR) / (len(ks) - 1)
        for j, k in enumerate(ks):
            if t < r0:
                m[k] = min(m[k], 1 - smooth((t - (p0 + j * step)) / STAG_DUR))
            elif t < r1:
                m[k] = min(m[k], smooth((t - r0) / (r1 - r0)))
            # after r1: back to 1 (only for this phase)
    return m


def phase_label(t: float) -> str:
    if A0 - 0.2 <= t < A_RES0:
        return "removing the HIGH band  (k 9-16)"
    if A_RES0 <= t < A_RES1:
        return "restoring"
    if B0 - 0.2 <= t < B_RES0:
        return "removing the LOW band  (k 1-8)"
    if B_RES0 <= t < B_RES1:
        return "restoring"
    return "full spectrum"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe", action="store_true", help="print decode results per phase end and exit")
    a = ap.parse_args()

    vocab = build_vocab_from_semantic_robust_alphabet(allow_elements=set(ELEMS))
    tok2id = {t: i for i, t in enumerate(vocab)}
    E = build_structured_E(vocab=vocab, D=D, pad_token="[PAD]", seed=a.seed, target_std=0.4, allowed_elements=set(ELEMS))
    toks = list(sf.split_selfies(sf.encoder(a.smiles)))
    n = len(toks)
    ids = np.array([tok2id[t] for t in toks] + [tok2id["[PAD]"]] * (L - n))
    Phi = build_fourier_basis(L, K)
    Theta = np.linalg.pinv(Phi) @ E[ids]                      # (M, D)
    kof = np.r_[0, np.arange(1, K + 1), np.arange(1, K + 1)]  # frequency index per Phi column

    Zfull = theta_to_Z(Theta, Phi)
    zc = Zfull[:n] - Zfull[:n].mean(0)
    w = np.linalg.svd(zc, full_matrices=False)[2][0]           # leading direction of the aspirin slots

    # dense, continuous basis (periodic in u)
    u = np.linspace(0, L, 480)
    def phi_dense(k):
        if k == 0:
            return [np.ones_like(u)]
        return [np.cos(2 * np.pi * k * u / L), np.sin(2 * np.pi * k * u / L)]
    coef = Theta @ w                                           # (M,) projection of every mode
    comp = np.zeros((K + 1, len(u)))                           # row k = contribution of frequency k
    for k in range(K + 1):
        cols = [0] if k == 0 else [k, K + k]
        for cidx, ph in zip(cols, phi_dense(k)):
            comp[k] += coef[cidx] * ph
    full_curve = comp.sum(0)

    def decode(m_k: np.ndarray):
        mvec = m_k[kof]
        Z = theta_to_Z(Theta * mvec[:, None], Phi)
        d2 = ((Z[:, None, :] - E[None, :, :]) ** 2).sum(-1)
        tied = d2[np.arange(L), ids] <= d2.min(1) + 1e-9
        dec = np.where(tied, ids, d2.argmin(1))
        out = [vocab[i] for i in dec if vocab[i] != "[PAD]"]
        try:
            smi = sf.decoder("".join(out))
        except Exception:
            smi = ""
        return smi, tied, Z

    if a.probe:
        for name, tt in (("full", 0.5), ("high band removed", A1 + 0.3), ("restored", A_RES1 + 0.2),
                         ("low band removed", B1 + 0.3)):
            smi, tied, _ = decode(masks_at(tt))
            print(f"{name:18s} t={tt:5.2f}  slots correct {int(tied[:n].sum())}/{n}  {smi}")
        return

    import matplotlib.pyplot as plt
    from PIL import Image

    fig = C.new_fig()
    axL = fig.add_axes([0.045, 0.075, 0.555, 0.855]); axR = fig.add_axes([0.635, 0.30, 0.345, 0.63])
    axB = fig.add_axes([0.635, 0.115, 0.345, 0.12])
    for ax in (axL, axR, axB):
        ax.set_facecolor(C.BG)
    scale = 0.40 / np.abs(comp[1:]).max()
    rec_scale = 1.55 / full_curve.std()
    ROW0, ROWSTEP = 6.2, 0.98
    row_y = {k: ROW0 + (k - 1) * ROWSTEP for k in range(1, K + 1)}
    DC_Y = 5.0

    nfr = int(round(T_TOTAL * a.fps))
    # pre-decode every frame (cheap) so the molecule can cross-fade between distinct decodes
    dec_list = []
    for i in range(nfr):
        dec_list.append(decode(masks_at(i / a.fps)))
    smiles_list = [d[0] for d in dec_list]

    bg = np.array(Image.new("RGB", (960 * C.SCALE, 540 * C.SCALE), C.BG))
    frames = []
    last_change, prev_smi, cur_smi = -99, "", smiles_list[0]
    for i in range(nfr):
        t = i / a.fps
        m = masks_at(t)
        smi, tied, Z = dec_list[i]
        if smi != cur_smi:
            prev_smi, cur_smi, last_change = cur_smi, smi, i
        ramp = float(smooth((i - last_change) / 5.0))

        # ---- left: components + reconstruction
        axL.cla(); axL.set_xlim(0, 1); axL.set_ylim(-4.6, ROW0 + K * ROWSTEP); axL.axis("off")
        x = u / L
        for k in range(1, K + 1):
            col = C.FREQ_CMAP((k - 1) / (K - 1))
            y = row_y[k]
            axL.plot(x, y + scale * comp[k], color=col, lw=1.0, alpha=0.10, zorder=1)          # ghost
            if m[k] > 0.02:
                axL.plot(x - (1 - m[k]) * 0.05, y + scale * comp[k] * (0.4 + 0.6 * m[k]), color=col,
                         lw=1.5, alpha=0.95 * m[k] ** 0.8, zorder=2)
            if k in (1, 4, 8, 12, 16):
                axL.text(-0.012, y, f"k={k}" if k == 1 else str(k), color=C.GREY, fontsize=6.5, ha="right",
                         va="center", clip_on=False)
        axL.plot(x, DC_Y + scale * comp[0] * 0 + 0.0 * x, color=C.WHITE, lw=0.8, alpha=0.35, ls=(0, (1, 3)))
        axL.text(-0.012, DC_Y, "DC", color=C.GREY, fontsize=6.5, ha="right", va="center", clip_on=False)
        # brackets for the two bands
        for (k0, k1, name) in ((1, 8, "low"), (9, 16, "high")):
            ya, yb = row_y[k0] - 0.4, row_y[k1] + 0.4
            c0 = C.FREQ_CMAP((k0 - 1) / (K - 1))
            axL.plot([1.012, 1.012], [ya, yb], color=c0, lw=1.2, alpha=0.7, clip_on=False)
            axL.text(1.02, (ya + yb) / 2, name, color=c0, fontsize=7, rotation=270, ha="left", va="center",
                     clip_on=False, alpha=0.9)
        # reconstruction: ghost of the full waveform, current sum, slot dots
        rec = (comp * m[:, None]).sum(0)
        axL.plot(x, rec_scale * full_curve, color=C.GREY, lw=1.0, alpha=0.35, ls=(0, (3, 3)), zorder=1)
        C.glow(axL, x, rec_scale * rec, C.WHITE, lw=1.6, alpha=0.95, z=3)
        slot_x = np.arange(L) / L
        slot_y = rec_scale * np.interp(np.arange(L), u, rec)
        ok = np.zeros(L, bool); ok[:n] = tied[:n]
        cols = [GOOD if ok[j] else (BAD if j < n else "#46586a") for j in range(L)]
        axL.scatter(slot_x, slot_y, s=[22 if j < n else 9 for j in range(L)], c=cols, zorder=4, lw=0)
        axL.text(0.0, -4.25, "sum of the active frequencies  ·  dots = token slots (cyan: decodes to the aspirin token)",
                 color=C.GREY, fontsize=6.5, ha="left", va="center")
        axL.text(0.0, ROW0 + K * ROWSTEP + 0.15, phase_label(t), color=C.WHITE, fontsize=10.5, ha="left", va="bottom")

        # ---- right: decoded molecule
        axR.cla(); axR.set_xlim(-1.15, 1.15); axR.set_ylim(-1.15, 1.15); axR.set_aspect("equal"); axR.axis("off")
        C.draw_mol(axR, a.smiles, alpha=0.10, lw=0.8, fontsize=7, color=C.GREY)
        if prev_smi and ramp < 1:
            C.draw_mol(axR, prev_smi, alpha=0.85 * (1 - ramp), lw=1.5, fontsize=9)
        if cur_smi:
            C.draw_mol(axR, cur_smi, alpha=0.95 * ramp if prev_smi or i else 0.95, lw=1.5, fontsize=9)

        # ---- right-bottom: slots strip
        axB.cla(); axB.set_xlim(-0.5, n - 0.5); axB.set_ylim(-1.6, 1.0); axB.axis("off")
        for j in range(n):
            axB.scatter([j], [0.2], s=60, color=GOOD if tied[j] else "none", edgecolors=GOOD if tied[j] else BAD,
                        linewidths=1.0, alpha=1.0 if tied[j] else 0.8, zorder=3)
        axB.text((n - 1) / 2, -1.0, f"{int(tied[:n].sum())}/{n} slots decode to the aspirin token",
                 color=C.GREY, fontsize=7, ha="center", va="center")

        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth(t / 0.7) * (1 - smooth((t - F_OUT0) / (T_TOTAL - F_OUT0)))
        im = np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(im))
        if i % 48 == 0:
            print(f"[g8] frame {i}/{nfr}  t={t:.1f}  slots {int(tied[:n].sum())}/{n}  {smi}", flush=True)

    for nm, tt in (("A end", A1 + 0.3), ("B end", B1 + 0.3)):
        s, td, _ = dec_list[int(tt * a.fps)]
        print(f"[g8] {nm}: {int(td[:n].sum())}/{n} slots, {s}")
    d = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(frames[-1]).astype(int)).max()
    print(f"[g8] first-vs-last max pixel diff = {d}")
    gif = C.save_loop(frames, "G8_harmonic_peel", fps=a.fps, gif_colors=64)
    print(f"[g8] wrote {gif}")
    plt.close(fig)


if __name__ == "__main__":
    main()
