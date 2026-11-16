#!/usr/bin/env python3
"""G4 - Crystallise: SELFIES beads -> molecule (aspirin).

Tokens come from the REAL selfies encoder. At each step we decode the token PREFIX with the
real selfies decoder and map each atom/bond to the first step at which it exists, so the
molecule grows exactly as the decoder builds it. 2D coordinates are computed once for the
final molecule, so nothing jitters. A half-read [RingN] token decodes to a misleading
intermediate, so while a Ring head awaits its index we keep showing the previous state.

Loop: dim beads -> light up one by one while the molecule grows -> hold (ring-closure flash)
-> fade to the dim start state (frame N == frame 0).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import rdDepictor

import common as C

HETERO = {"O": "#ff7a7a", "N": "#7aa8ff", "S": "#ffe27a", "F": "#9dff9d", "Cl": "#9dff9d"}


def smooth(x: float) -> float:
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def bead_color(tok: str) -> str:
    if tok.startswith("[Ring"):
        return C.HIGH
    if "Branch" in tok:
        return "#8a5cff"
    if tok.startswith("[=") or tok.startswith("[#"):
        return "#c8a0ff"
    return "#7ee8ff"


def prep(smiles: str):
    tokens = list(sf.split_selfies(sf.encoder(smiles)))
    n = len(tokens)
    full_smi = sf.decoder("".join(tokens))
    full = Chem.MolFromSmiles(full_smi, sanitize=False)   # keeps Kekule bond orders + atom order
    san = Chem.MolFromSmiles(full_smi)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(san)
    conf = san.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(san.GetNumAtoms())])

    atom_step = [None] * full.GetNumAtoms()
    bond_step = {}
    for j in range(1, n + 1):
        jj = j - 1 if tokens[j - 1].startswith("[Ring") else j      # hold state while Ring head awaits index
        if jj == 0:
            continue
        pm = Chem.MolFromSmiles(sf.decoder("".join(tokens[:jj])), sanitize=False)
        if pm is None:
            continue
        for i in range(pm.GetNumAtoms()):
            assert pm.GetAtomWithIdx(i).GetSymbol() == full.GetAtomWithIdx(i).GetSymbol(), "atom order mismatch"
            if atom_step[i] is None:
                atom_step[i] = j
        for b in pm.GetBonds():
            key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            bond_step.setdefault(key, j)
    bonds = []
    for b in full.GetBonds():
        key = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        bonds.append((key[0], key[1], b.GetBondTypeAsDouble(), bond_step[key]))
    syms = [a.GetSymbol() for a in full.GetAtoms()]
    ring_step = max(bond_step.values()) if False else None
    # ring-closure bond = the bond that appears with a [Ring*] index token
    closing = [s for (_, _, _, s) in bonds if tokens[s - 1] and any(tokens[k].startswith("[Ring") for k in range(max(0, s - 2), s))]
    return tokens, xy, syms, atom_step, bonds, (min(closing) if closing else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="CC(=O)Oc1ccccc1C(=O)O")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()

    tokens, xy, syms, atom_step, bonds, ring_close = prep(a.smiles)
    n = len(tokens)
    print("tokens:", " ".join(tokens)); print("atom_step:", atom_step, "ring-closure step:", ring_close)

    # fit molecule into axes with equal aspect
    mid = (xy.max(0) + xy.min(0)) / 2
    span = (xy.max(0) - xy.min(0)).max()
    P = (xy - mid) / span * 1.9                       # roughly [-0.95, 0.95]

    T = a.seconds
    t_start, t_end, t_fade = 0.7, 5.6, 1.0
    dt = (t_end - t_start) / n
    nf = int(round(T * a.fps))
    frames = []
    for f in range(nf):
        time = f / a.fps
        p = float(np.clip((time - t_start) / dt, 0, n))                 # continuous step count
        fade = 1.0 - smooth((time - (T - t_fade)) / t_fade) if time > T - t_fade else 1.0
        fig = C.new_fig()

        # --- beads (top) ---
        axb = fig.add_axes([0.03, 0.80, 0.94, 0.17], facecolor=C.BG)
        axb.set_xlim(-0.5, n - 0.5); axb.set_ylim(-1, 1); axb.axis("off")
        axb.plot([0, n - 1], [0, 0], color="#16283a", lw=2, zorder=1)
        for j, tok in enumerate(tokens):
            lit = smooth(p - j) * fade                                    # token j+1 consumed
            alpha = 0.22 + 0.78 * lit
            cur = (p - j) > 0 and (p - j) < 1.0 and fade >= 1.0
            axb.scatter([j], [0], s=(330 if cur else 230), color=bead_color(tok), alpha=alpha,
                        edgecolors=C.WHITE if cur else "none", linewidths=1.3, zorder=3)
            axb.text(j, 0, tok.strip("[]").replace("Branch", "Br").replace("Ring", "Rg"), ha="center", va="center",
                     fontsize=5.2, color=C.BG, alpha=min(1.0, 0.35 + lit), zorder=4, fontweight="bold")

        # --- molecule ---
        axm = fig.add_axes([0.22, 0.03, 0.56, 0.74], facecolor=C.BG)
        axm.set_xlim(-1.0, 1.0); axm.set_ylim(-1.0, 1.0); axm.set_aspect("equal"); axm.axis("off")
        for (i, j, order, s) in bonds:
            al = smooth(p - (s - 1)) * fade
            if al <= 0.01:
                continue
            x1, y1 = P[i]; x2, y2 = P[j]
            d = np.array([x2 - x1, y2 - y1]); L = np.linalg.norm(d); nrm = np.array([-d[1], d[0]]) / L
            flash = 0.0
            if ring_close is not None and s == ring_close:
                flash = max(0.0, 1.0 - (p - (s - 1)) / 3.0) if p >= s - 1 else 0.0
                flash = max(flash, 0.0)
                if fade >= 1.0 and time > t_end:                          # one more pulse during the hold
                    flash = max(flash, 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (time - t_end) / 0.9)))
            col = C.WHITE
            offs = [-0.025, 0.025] if order == 2 else ([-0.04, 0.0, 0.04] if order == 3 else [0.0])
            for o in offs:
                sx, sy = nrm * o
                C.glow(axm, [x1 + sx, x2 + sx], [y1 + sy, y2 + sy], C.HIGH if flash > 0.3 else col,
                       lw=2.0 + 2.2 * flash, alpha=al)
        for i, (sym, (x, y)) in enumerate(zip(syms, P)):
            s = atom_step[i]
            if s is None:
                continue
            al = smooth(p - (s - 1)) * fade
            if al <= 0.01:
                continue
            if sym != "C":
                axm.scatter([x], [y], s=520, color=C.BG, zorder=5)
                axm.text(x, y, sym, ha="center", va="center", fontsize=15, color=HETERO.get(sym, C.WHITE),
                         alpha=al, zorder=6, fontweight="bold")
            else:
                axm.scatter([x], [y], s=26, color=C.WHITE, alpha=al * 0.9, zorder=5, linewidths=0)
            newest = 0 < (p - (s - 1)) < 1 and fade >= 1.0
            if newest:                                                    # birth ripple
                r = (p - (s - 1))
                axm.scatter([x], [y], s=500 * (0.4 + 2.0 * r), facecolors="none", edgecolors=C.HIGH,
                            alpha=0.8 * (1 - r), linewidths=1.4, zorder=4)
        frames.append(C.fig_to_image(fig))
        plt.close(fig)

    print("wrote", C.save_loop(frames, "G4_crystallise", fps=a.fps))


if __name__ == "__main__":
    main()
