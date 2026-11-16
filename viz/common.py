"""Shared style + frame/GIF helpers for the SpectralMol showcase visuals."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

# Resolution: VIZ_SCALE=1 -> 960x540 (viz/out, PNG + GIF); VIZ_SCALE=2 -> 1920x1080 (viz/out_1080, PNG only).
# The figure size in inches stays 9.6x5.4, only the dpi changes, so every layout/line width scales together.
SCALE = int(os.environ.get("VIZ_SCALE", "1"))
CACHE_DIR = ROOT / "viz" / "out"                       # data caches shared by all resolutions
OUT_DIR = ROOT / "viz" / ("out" if SCALE == 1 else f"out_{540 * SCALE}")

BG = "#05101A"
LOW = "#3b5bff"      # low frequency: deep blue
HIGH = "#ffb020"     # high frequency: amber
WHITE = "#eaf6ff"
GREY = "#6b7a88"

# low -> high frequency colour ramp (blue -> violet -> amber)
FREQ_CMAP = LinearSegmentedColormap.from_list("freq", [LOW, "#8a5cff", HIGH])


def freq_colors(K: int) -> list[tuple[float, float, float, float]]:
    """Colour per Phi column (index 0 = DC, 1..K = cos, K+1..2K = sin)."""
    cols = [FREQ_CMAP(0.0)]
    for k in range(1, K + 1):
        cols.append(FREQ_CMAP((k - 1) / max(1, K - 1)))
    for k in range(1, K + 1):
        cols.append(FREQ_CMAP((k - 1) / max(1, K - 1)))
    return cols


def new_fig(w: int = 960, h: int = 540, dpi: int = 100):
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi * SCALE, facecolor=BG)
    return fig


def glow(ax, x, y, color, lw=2.0, alpha=1.0, z=2):
    """Cheap glow: a few wide faint strokes under a crisp one."""
    for mult, a in ((6, 0.05), (3.5, 0.10), (2, 0.18)):
        ax.plot(x, y, color=color, lw=lw * mult, alpha=alpha * a, solid_capstyle="round", zorder=z)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z + 0.1)


def fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[..., :3].copy())


def save_loop(frames: list[Image.Image], name: str, fps: int = 24, gif_colors: int = 64) -> Path:
    """Write PNG sequence + looping GIF. Frame N is NOT repeated (loop is periodic)."""
    seq_dir = OUT_DIR / name
    seq_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(seq_dir / f"{name}_{i:04d}.png")
    if SCALE > 1:                       # GIFs at 1080p are huge; PNG sequence only
        return seq_dir
    gif = OUT_DIR / f"{name}.gif"
    pal = [im.quantize(colors=gif_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=False, disposal=1)
    return gif


# ---------------------------------------------------------------------------
# Glowing line-art molecule (stable 2D layout, RDKit coordinates)
# ---------------------------------------------------------------------------
_MOL_CACHE: dict = {}
HETERO_COL = {"O": "#ff7a7a", "N": "#7aa8ff", "S": "#ffe27a", "F": "#9dff9d", "Cl": "#9dff9d", "Br": "#d9a07a"}


def mol_layout(smiles: str):
    """(xy, symbols, bonds[(i,j,order)]) with xy normalised to [-1,1]; None if unparsable."""
    if smiles in _MOL_CACHE:
        return _MOL_CACHE[smiles]
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    res = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)
            conf = mol.GetConformer()
            xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
            km = Chem.Mol(mol)
            Chem.Kekulize(km, clearAromaticFlags=True)
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in km.GetBonds()]
            syms = [a.GetSymbol() for a in mol.GetAtoms()]
            mid = (xy.max(0) + xy.min(0)) / 2
            span = max((xy.max(0) - xy.min(0)).max(), 1e-6)
            res = ((xy - mid) / span * 1.8, syms, bonds)
    except Exception:
        res = None
    _MOL_CACHE[smiles] = res
    return res


def draw_mol(ax, smiles: str, alpha: float = 1.0, lw: float = 1.5, fontsize: int = 9, color: str = WHITE):
    """Draw molecule on ax in data coords [-1,1]^2 (caller sets limits/aspect)."""
    lay = mol_layout(smiles)
    if lay is None:
        return False
    P, syms, bonds = lay
    for (i, j, order) in bonds:
        d = P[j] - P[i]
        L = np.linalg.norm(d) + 1e-9
        nrm = np.array([-d[1], d[0]]) / L
        offs = [-0.035, 0.035] if order == 2 else ([-0.06, 0.0, 0.06] if order == 3 else [0.0])
        for o in offs:
            s = nrm * o
            glow(ax, [P[i, 0] + s[0], P[j, 0] + s[0]], [P[i, 1] + s[1], P[j, 1] + s[1]], color, lw=lw, alpha=alpha)
    for i, sym in enumerate(syms):
        if sym != "C":
            ax.scatter([P[i, 0]], [P[i, 1]], s=fontsize * 38, color=BG, zorder=5, linewidths=0)
            ax.text(P[i, 0], P[i, 1], sym, ha="center", va="center", fontsize=fontsize,
                    color=HETERO_COL.get(sym, color), alpha=alpha, zorder=6, fontweight="bold")
    return True
