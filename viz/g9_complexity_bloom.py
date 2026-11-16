#!/usr/bin/env python3
"""G9 - Molecule complexity bloom (ablation scene A3), REAL run data.

Four panels (full / high only / low only / random matrix), same task, same starting population.
  big molecule : the best-so-far molecule of the seed-7 run at each generation (evolution_log.tsv)
  small cloud  : 24 molecules sampled from that generation's evaluated molecules (seed 7); drawn at one
                 common scale (px per Angstrom), so bigger molecules really take more room
  numbers      : population mean heavy atoms / rings at that generation, mean over the 6 seeds
WHAT THIS IS NOT: the runs start from ChEMBL-derived drug-like molecules (not from nothing) and evolve them
with SMILES operators; there is no parent/lineage file, so molecules are not grown atom by atom. Complexity
moves toward whatever the task rewards: on Median_molecules_2 (large polycyclic targets: tadalafil,
sildenafil) the structured conditions grow; on small-target tasks (Median_molecules_1, C11H24) they shrink.
The random matrix stays near the starting population in both. Default task is the 'growth' one; choose it
knowing that (the caption names the task).
Loop: empty -> generations 0..N -> hold -> fade to empty (frame N == frame 0).
Usage: g9_complexity_bloom.py [--task Median_molecules_2] [--fps 24]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors as rd

import common as C
import guac_data as G

RDLogger.DisableLog("rdApp.*")
COND = {"full": ("#7ee8ff", "full spectrum"), "high": (C.HIGH, "high frequencies only"),
        "low": ("#6f8bff", "low frequencies only"), "random": ("#9aa5b1", "random matrix")}
ORDER = ["full", "high", "low", "random"]
NTHUMB, SEED_VIS = 24, 7
T_G0, T_G1, T_HOLD_END, T_TOTAL = 0.8, 8.4, 9.4, 10.2
HETERO = C.HETERO_COL
SC_BIG, SC_SMALL = 11.0, 3.6          # px per Angstrom (RDKit coord units)


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def stats(smi: str):
    m = Chem.MolFromSmiles(smi)
    return (np.nan, np.nan) if m is None else (m.GetNumHeavyAtoms(), rd.CalcNumRings(m))


def build_data(task: str) -> dict:
    cache = C.CACHE_DIR / f"g9_cache_{task}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out: dict = {"task": task, "cond": {}}
    for k in ORDER:
        m = G.MODELS[k]
        ha_s, rg_s = [], []
        for s in G.SEEDS:
            d = G.task_dir(m, s, task)
            df = pd.read_csv(d / "molecule_scores_by_generation.tsv", sep="\t")
            ngen = int(df.generation.max()) + 1
            ha, rg = np.zeros(ngen), np.zeros(ngen)
            for g in range(ngen):
                st = np.array([stats(x) for x in df.smiles[df.generation == g].values])
                ha[g], rg[g] = np.nanmean(st[:, 0]), np.nanmean(st[:, 1])
            ha_s.append(ha); rg_s.append(rg)
            if s == SEED_VIS:
                ev = pd.read_csv(d / "evolution_log.tsv", sep="\t")
                best = [str(x) for x in ev.sort_values("gen").best_smiles.values]
                thumbs = []
                for g in range(ngen):
                    pool = df.smiles[df.generation == g].values
                    idx = np.random.RandomState(1000 + g).choice(len(pool), min(NTHUMB, len(pool)), replace=False)
                    thumbs.append([str(pool[i]) for i in idx])
        out["cond"][k] = {"ha": np.mean(ha_s, 0).tolist(), "rings": np.mean(rg_s, 0).tolist(),
                          "best": best, "thumbs": thumbs}
    cache.write_text(json.dumps(out))
    return out


_LAY: dict = {}


def layout(smi: str):
    """(xy centred, symbols, bonds[(i,j,order)]) in Angstrom-like units, or None."""
    if smi in _LAY:
        return _LAY[smi]
    res = None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumAtoms() > 0:
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)
            cf = mol.GetConformer()
            xy = np.array([[cf.GetAtomPosition(i).x, cf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
            xy -= (xy.max(0) + xy.min(0)) / 2
            km = Chem.Mol(mol)
            Chem.Kekulize(km, clearAromaticFlags=True)
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in km.GetBonds()]
            res = (xy, [a.GetSymbol() for a in mol.GetAtoms()], bonds)
    except Exception:
        res = None
    _LAY[smi] = res
    return res


def segs_for(smi: str, cx: float, cy: float, scale: float, rot: float = 0.0):
    lay = layout(smi)
    if lay is None:
        return []
    xy, _, bonds = lay
    c, s = np.cos(rot), np.sin(rot)
    P = np.c_[xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c] * scale + [cx, cy]
    return [((P[i, 0], P[i, 1]), (P[j, 0], P[j, 1])) for i, j, _ in bonds]


def draw_big(ax, smi: str, cx: float, cy: float, alpha: float, color: str = C.WHITE):
    lay = layout(smi)
    if lay is None or alpha <= 0.01:
        return
    xy, syms, bonds = lay
    P = xy * SC_BIG + [cx, cy]
    for i, j, order in bonds:
        d = P[j] - P[i]
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        offs = [-1.7, 1.7] if order == 2 else ([-3.0, 0, 3.0] if order == 3 else [0.0])
        for o in offs:
            q = nrm * o
            C.glow(ax, [P[i, 0] + q[0], P[j, 0] + q[0]], [P[i, 1] + q[1], P[j, 1] + q[1]], color, lw=1.1, alpha=alpha)
    for i, sym in enumerate(syms):
        if sym != "C":
            ax.scatter([P[i, 0]], [P[i, 1]], s=70, color=C.BG, zorder=5, linewidths=0, alpha=alpha)
            ax.text(P[i, 0], P[i, 1], sym, ha="center", va="center", fontsize=7, color=HETERO.get(sym, color),
                    alpha=alpha, zorder=6, fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Median_molecules_2")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--data-only", action="store_true")
    a = ap.parse_args()

    data = build_data(a.task)
    ngen = len(data["cond"]["full"]["ha"])
    for k in ORDER:
        c = data["cond"][k]
        print(f"[g9] {k:6s} heavy atoms {c['ha'][0]:.1f} -> {c['ha'][-1]:.1f}   rings {c['rings'][0]:.2f} -> {c['rings'][-1]:.2f}"
              f"   best(seed7) HA/rings {stats(c['best'][0])} -> {stats(c['best'][-1])}")
    if a.data_only:
        return

    fig = C.new_fig()
    gs = fig.add_gridspec(2, 2, left=0.012, right=0.988, top=0.985, bottom=0.085, wspace=0.025, hspace=0.04)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    W_PX, H_PX = 960, 540
    sizes = []
    for ax in axes:
        bb = ax.get_position()
        sizes.append((bb.width * W_PX, bb.height * H_PX))
    fig.text(0.5, 0.040, f"{a.task}  ·  large molecule = best so far (seed 7)  ·  cloud = 24 sampled molecules of the generation, one common scale",
             color=C.GREY, fontsize=7, ha="center", va="center")
    fig.text(0.5, 0.018, "numbers = population mean over 6 seeds  ·  all four start from the same population  ·  operator-space proxy",
             color=C.GREY, fontsize=7, ha="center", va="center")
    gtxt = fig.text(0.985, 0.029, "", color=C.GREY, fontsize=7.5, ha="right", va="center")

    rng = np.random.RandomState(5)
    cells = None
    nfr = int(round(T_TOTAL * a.fps))
    bg = np.array(Image.new("RGB", (W_PX * C.SCALE, H_PX * C.SCALE), C.BG))
    frames = []
    last = {k: (None, None, -99) for k in ORDER}      # prev best smiles, cur best smiles, change frame
    for fi in range(nfr):
        t = fi / a.fps
        gpos = np.clip((t - T_G0) / (T_G1 - T_G0), 0, 1) * (ngen - 1)
        g = int(np.floor(gpos)); frac = gpos - g
        for ax, k, (pw, ph) in zip(axes, ORDER, sizes):
            ax.cla(); ax.set_xlim(0, pw); ax.set_ylim(0, ph); ax.axis("off"); ax.set_facecolor(C.BG)
            col, label = COND[k]
            c = data["cond"][k]
            ax.text(10, ph - 10, label, color=col, fontsize=9.5, va="top", ha="left")
            # --- thumbnails (cross-fade between generations)
            gx0, gx1, gy0, gy1 = pw * 0.44, pw - 8, 22, ph - 28
            ncol, nrow = 6, 4
            cw, chh = (gx1 - gx0) / ncol, (gy1 - gy0) / nrow
            gens_draw = [(g, 1 - smooth(frac * 2.2))]
            if g + 1 < ngen:
                gens_draw.append((g + 1, smooth(frac * 2.2)))
            for gg, al in gens_draw:
                if al < 0.02:
                    continue
                lc = []
                for ti, smi in enumerate(c["thumbs"][gg][:ncol * nrow]):
                    r, cc = divmod(ti, ncol)
                    cx, cy = gx0 + (cc + 0.5) * cw, gy1 - (r + 0.5) * chh
                    rot = ((hash((gg, ti)) % 628) / 100.0) * 0.0           # fixed orientation (RDKit coordgen)
                    lc += segs_for(smi, cx, cy, SC_SMALL, rot)
                ax.add_collection(LineCollection(lc, colors=[col], linewidths=0.9, alpha=0.62 * al, zorder=2))
            # --- big best molecule
            bests = c["best"]
            cur = bests[min(g + (1 if frac > 0.5 else 0), len(bests) - 1)]
            prev, curs, ch = last[k]
            if curs is None:
                last[k] = (cur, cur, -99)
            elif cur != curs:
                last[k] = (curs, cur, fi)
            prev, curs, ch = last[k]
            ramp = smooth((fi - ch) / 6.0)
            bx, by = pw * 0.215, ph * 0.50
            if prev != curs and ramp < 1:
                draw_big(ax, prev, bx, by, 0.9 * (1 - ramp))
            draw_big(ax, curs, bx, by, 0.95 if prev == curs else 0.95 * ramp)
            # --- numbers
            ha = np.interp(gpos, np.arange(ngen), c["ha"]); rg = np.interp(gpos, np.arange(ngen), c["rings"])
            ax.text(10, 8, f"atoms {ha:4.1f}   rings {rg:3.1f}", color=col, fontsize=8, va="bottom", ha="left", alpha=0.9)
        gtxt.set_text(f"generation {int(round(gpos))}/{ngen - 1}")
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth(t / 0.8) * (1 - smooth((t - T_HOLD_END) / (T_TOTAL - T_HOLD_END)))
        frames.append(Image.fromarray(np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)))
        if fi % 48 == 0:
            print(f"[g9] frame {fi}/{nfr}", flush=True)
    d = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(frames[-1]).astype(int)).max()
    print(f"[g9] first-vs-last max pixel diff = {d}")
    print(f"[g9] wrote {C.save_loop(frames, 'G9_complexity_bloom', fps=a.fps, gif_colors=64)}")


if __name__ == "__main__":
    main()
