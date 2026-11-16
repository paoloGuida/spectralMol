#!/usr/bin/env python3
"""G12 - Galaxy dolly (scene 8), REAL run data.

Space   : 2048-bit Morgan count fingerprints (r=2) of molecules the ablation runs evaluated, projected to 3D
          with a TruncatedSVD (PCA-like) fitted on the background sample. It is a linear map of fingerprint
          space, not a faithful chemical-space atlas (no UMAP/t-SNE here); the first 3 components hold only a
          small part of the variance, so distances are indicative only.
Stars   : grey-blue  = random sample of every molecule evaluated across ALL 20 GuacaMol tasks x 4 conditions
                       x 6 seeds (the exact total is printed and put in the caption)
          cyan       = every molecule visited by the 6 full-spectrum seeds on one task (Median_molecules_2)
          amber glow = the top 0.5 % of those by score (the 'target family': near tadalafil/sildenafil)
Arc     : the pooled non-dominated front (tadalafil_Sim vs sildenafil_Sim) of the full-spectrum runs, drawn as
          a faint path with line-art of 10 of its molecules, set on an arc with leader lines. As in F10 these runs optimised the geometric
          mean with the SMILES-operator evolver, not NSGA-II; the 'front' is the non-dominated set visited.
Camera  : perspective dolly from inside the target neighbourhood out to the whole galaxy, slow yaw.
Loop    : empty -> dolly out -> hold -> fade to empty (frame N == frame 0).
Usage   : g12_galaxy_dolly.py [--task Median_molecules_2] [--fps 24] [--data-only]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import TruncatedSVD

import common as C
import guac_data as G
from g7_four_oceans import all_tasks

RDLogger.DisableLog("rdApp.*")
TASKS = {"Median_molecules_1": ["camphor_Sim", "menthol_Sim"], "Median_molecules_2": ["tadalafil_Sim", "sildenafil_Sim"]}
N_BG_PER_RUN = 250
S = C.SCALE
W, H = 960 * S, 540 * S
T_D0, T_D1, T_HOLD, T_TOTAL = 0.7, 8.9, 9.5, 10.3
BG_COL, VIS_COL, TOP_COL = np.array([0.36, 0.52, 0.72]), np.array([0.49, 0.91, 1.0]), np.array([1.0, 0.82, 0.48])


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fps(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    out, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out.append(_GEN.GetCountFingerprintAsNumPy(m).astype(np.float32))
        ok.append(i)
    return np.log1p(np.stack(out)), np.array(ok)


def build(task: str) -> dict:
    cache = C.CACHE_DIR / f"g12_cache_{task}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    comps = TASKS[task]
    rng = np.random.RandomState(0)
    total, bg_smiles = 0, []
    tasks = all_tasks()
    for t in tasks:
        for k, m in G.MODELS.items():
            for s in G.SEEDS:
                try:
                    d = G.task_dir(m, s, t)
                    sm = pd.read_csv(d / "molecule_scores_by_generation.tsv", sep="\t", usecols=["smiles"]).smiles.values
                except Exception:
                    continue
                total += len(sm)
                bg_smiles += list(rng.choice(sm, min(N_BG_PER_RUN, len(sm)), replace=False))
    bg_smiles = sorted(set(map(str, bg_smiles)))
    print(f"[g12] {total} evaluations across {len(tasks)} tasks x 4 conditions x {len(G.SEEDS)} seeds; background sample {len(bg_smiles)} unique")
    # visited by full-spectrum runs on the task
    vis = []
    for s in G.SEEDS:
        r = G.load_run(G.MODELS["full"], s, task, comps)
        vis.append(r)
    vis = pd.concat(vis).drop_duplicates("smiles").reset_index(drop=True)
    print(f"[g12] {len(vis)} unique molecules visited by the full-spectrum runs on {task}")

    Xb, okb = fps(bg_smiles)
    svd = TruncatedSVD(n_components=3, random_state=0).fit(Xb)
    print("[g12] explained variance of the 3 components:", np.round(svd.explained_variance_ratio_, 3))
    bg3 = svd.transform(Xb)
    Xv, okv = fps(vis.smiles.tolist())
    vis = vis.iloc[okv].reset_index(drop=True)
    vis3 = svd.transform(Xv)
    P = vis[comps].values
    fidx = G.front_idx(P)
    fidx = fidx[np.argsort(P[fidx, 1])]
    return_dict = {"bg3": bg3.astype(np.float32), "vis3": vis3.astype(np.float32), "score": vis.score.values.astype(np.float32),
                   "front3": vis3[fidx].astype(np.float32), "front_smiles": np.array(vis.smiles.values[fidx], dtype=object),
                   "front_P": P[fidx].astype(np.float32), "total": np.array(total), "ntasks": np.array(len(tasks)),
                   "var": svd.explained_variance_ratio_}
    np.savez(cache, **return_dict)
    return return_dict


LOOK = [np.zeros(3)]      # current look-at point (set per frame)


def rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry


def project(P, R, d, f):
    q = (P - LOOK[0]) @ R.T
    zc = d - q[:, 2]
    ok = zc > 0.06
    zs = np.where(ok, zc, 1.0)
    return W / 2 + f * q[:, 0] / zs, H / 2 - f * q[:, 1] / zs, zc, ok


def splat(x, y, w, sigma):
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    img = np.bincount(yi[m] * W + xi[m], weights=w[m], minlength=W * H).reshape(H, W)
    return gaussian_filter(img, sigma * S) if sigma > 0 else img


def layer(P, R, d, f, weight, colour, gain, glow=0.0):
    x, y, zc, ok = project(P, R, d, f)
    out = np.zeros((H, W, 3))
    wgt = ok * weight
    # nearer stars are blurrier/bigger (perspective), far ones are crisp
    for lo, hi, sig in ((0.06, 0.7, 2.4), (0.7, 1.6, 1.3), (1.6, 99, 0.65)):
        sel = (zc >= lo) & (zc < hi) & ok
        if sel.any():
            dens = splat(x[sel], y[sel], wgt[sel] if np.ndim(wgt) else np.full(sel.sum(), wgt), sig)
            out += dens[..., None] * colour * gain
    if glow > 0:
        g = splat(x[ok], y[ok], np.ones(ok.sum()), 6.0)
        out += g[..., None] * colour * glow
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Median_molecules_2", choices=list(TASKS))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--data-only", action="store_true")
    a = ap.parse_args()

    D = build(a.task)
    if a.data_only:
        return
    bg3, vis3, score = D["bg3"].astype(np.float64), D["vis3"].astype(np.float64), D["score"]
    front3, fsmi, fP = D["front3"].astype(np.float64), list(D["front_smiles"]), D["front_P"]
    total = int(D["total"])
    top = score >= np.quantile(score, 0.995)
    centre = vis3[top].mean(0)
    bg3, vis3, front3 = bg3 - centre, vis3 - centre, front3 - centre
    Rg = float(np.quantile(np.linalg.norm(bg3, axis=1), 0.95))
    s = 1.0 / Rg
    bg3, vis3, front3 = bg3 * s, vis3 * s, front3 * s
    print(f"[g12] {len(bg3)} background stars, {len(vis3)} visited ({top.sum()} top), {len(front3)} front molecules; R95={Rg:.2f}")

    # 12 front molecules evenly spaced along the front, drawn as line-art
    pick = np.unique(np.linspace(0, len(front3) - 1, 10).round().astype(int))
    gc = bg3.mean(0)                 # galaxy centroid: the camera ends up looking here
    d0, d1 = 0.45, 3.4
    f = 0.80 * H * d1 / 1.6

    fig = plt.figure(figsize=(9.6, 5.4), dpi=100 * S, facecolor=C.BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    cap1 = (f"PCA of Morgan fingerprints  ·  grey: sample of {total / 1e6:.1f} M molecule evaluations (all tasks, conditions, seeds)  ·  "
            f"cyan: visited by full-spectrum runs, {a.task}  ·  amber: top 0.5 % scores")
    nfr = int(round(T_TOTAL * a.fps))
    bg = np.array(Image.new("RGB", (W, H), C.BG)).astype(np.float32)
    bgcol = np.array([5, 16, 26], float) / 255.0
    frames = []
    for fi in range(nfr):
        t = fi / a.fps
        u = smooth((t - T_D0) / (T_D1 - T_D0))
        d = d0 * (d1 / d0) ** u
        R = rot(0.35 + 1.0 * u, 0.40 - 0.15 * u)
        LOOK[0] = gc * smooth(u)
        img = np.zeros((H, W, 3))
        img += layer(bg3, R, d, f, 1.0, BG_COL, 0.55 + 0.45 * (1 - u))
        img += layer(vis3[~top], R, d, f, 1.0, VIS_COL, 0.16 + 0.20 * (1 - u))
        img += layer(vis3[top], R, d, f, 1.0, TOP_COL, 2.4, glow=6.0)
        img = 1 - np.exp(-img)                                        # soft tone map
        img = bgcol + img * (1 - bgcol)
        ax.cla(); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.imshow(img, extent=(0, W, H, 0), interpolation="bilinear", zorder=0, aspect="auto")
        # front: its molecules (white dots) + line-art of 10 of them placed on an arc beside the cluster,
        # each tied to its true position by a leader line (they sit too close together to draw in place)
        al = smooth((u - 0.38) / 0.30)
        if al > 0.02:
            x, y, zc, ok = project(front3, R, d, f)
            if ok.all():
                ax.scatter(x, y, s=9, color=C.WHITE, alpha=0.8 * al, lw=0, zorder=2)
                cx0, cy0 = float(np.mean(x)), float(np.mean(y))
                ang = np.deg2rad(np.linspace(-68, 68, len(pick)))
                for a_, j in zip(ang, pick):
                    tx, ty = cx0 + 205 * S * np.cos(a_), max(44.0 * S, cy0 + 45 * S + 120 * S * np.sin(a_))
                    ax.plot([x[j], tx - 28 * S * np.cos(a_)], [y[j], ty - 28 * S * np.sin(a_)], color=C.WHITE, lw=0.5, alpha=0.28 * al, zorder=2)
                    lay = C.mol_layout(fsmi[j])
                    if lay is None:
                        continue
                    xy, syms, bonds = lay
                    sc = 15.0 * S
                    for (i1, i2, o) in bonds:
                        ax.plot([tx + xy[i1, 0] * sc, tx + xy[i2, 0] * sc],
                                [ty - xy[i1, 1] * sc, ty - xy[i2, 1] * sc], color=C.WHITE, lw=1.0, alpha=0.95 * al, zorder=3)
        ax.add_patch(plt.Rectangle((0, H - 28 * S), W, 28 * S, color=C.BG, alpha=0.7, zorder=4, lw=0))
        ax.text(W / 2, H - 14 * S, cap1, color=C.GREY, fontsize=6.3, ha="center", va="center", zorder=5)
        fig.canvas.draw()
        im = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32)
        e = smooth(t / 0.8) * (1 - smooth((t - T_HOLD) / (T_TOTAL - T_HOLD)))
        frames.append(Image.fromarray(np.clip(bg * (1 - e) + im * e, 0, 255).astype(np.uint8)))
        if fi % 24 == 0:
            print(f"[g12] frame {fi}/{nfr} d={d:.2f}", flush=True)
    dmax = np.abs(np.asarray(frames[0]).astype(int) - np.asarray(frames[-1]).astype(int)).max()
    print(f"[g12] first-vs-last max pixel diff = {dmax}")
    print(f"[g12] wrote {C.save_loop(frames, 'G12_galaxy_dolly', fps=a.fps, gif_colors=64)}")


if __name__ == "__main__":
    main()
