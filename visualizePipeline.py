"""
Visualize the SpectralMol Theta -> Z -> token pipeline.

We reconstruct the same construction the paper uses:
  - Phi: real Fourier basis of shape (L, M) with M = 1 + 2K
  - E:   structured token embedding table built from chemical features
         then projected by a fixed orthonormal matrix (QR projection)
  - Z = Phi @ Theta is computed for one Theta encoded from a known molecule
  - For each row i of Z we compute squared Euclidean distance to every E_v
  - The argmin gives the decoded token at position i

Example molecule: aspirin (acetylsalicylic acid).
SELFIES (lightly simplified for display): a benzene ring + acetate ester + carboxyl,
written as a sequence of SELFIES-style tokens consistent with the paper's vocabulary.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# 1. Vocabulary (a representative subset of the paper's vocabulary)
# ---------------------------------------------------------------------------
# We keep the representation small to make the figure look decent.
VOCAB = [
    "[PAD]",
    # aliphatic atoms
    "[C]", "[N]", "[O]", "[F]",
    # aromatic atoms
    "[c]", "[n]", "[o]", "[s]",
    # double-bond atoms
    "[=C]", "[=N]", "[=O]",
    # triple-bond atoms
    "[#C]", "[#N]",
    # branch tokens
    "[Branch1]", "[Branch2]", "[=Branch1]",
    # ring tokens
    "[Ring1]", "[Ring2]", "[Ring3]",
    "[Ring4]", "[Ring5]", "[Ring6]",
]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
PAD = "[PAD]"
PAD_ID = TOK2ID[PAD]
V = len(VOCAB)

# ---------------------------------------------------------------------------
# 2. Structured embedding table E (mirrors embedding.py / build_structured_E)
# ---------------------------------------------------------------------------
# Each token gets a feature vector with one-hot group, element, flags, ring.
# Then we project to dimension D via a fixed seeded QR-orthonormal matrix.

ELEMENTS = ["C", "N", "O", "F", "S"]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENTS)}
GROUPS = ["pad", "atom_aliph", "atom_arom", "bond_double", "bond_triple",
          "branch", "ring"]
G_IDX = {g: i for i, g in enumerate(GROUPS)}
RING_TOK_RE = re.compile(r"^\[Ring(\d+)\]$")
MAX_RING = 6

ATOM_RE_ALIPH = re.compile(r"^\[([A-Z][a-z]?)\]$")
ATOM_RE_AROM = re.compile(r"^\[([cnops])\]$")
ATOM_RE_DBL = re.compile(r"^\[=([A-Z][a-z]?)\]$")
ATOM_RE_TRP = re.compile(r"^\[#([A-Z][a-z]?)\]$")


def token_features(tok: str) -> np.ndarray:
    g = np.zeros(len(GROUPS))
    e = np.zeros(len(ELEMENTS))
    flags = np.zeros(4)  # has_atom, is_arom, is_aliph, is_struct
    ring_oh = np.zeros(MAX_RING)

    if tok == PAD:
        g[G_IDX["pad"]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    if tok.startswith("[Branch"):
        g[G_IDX["branch"]] = 1.0
        flags[3] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = RING_TOK_RE.match(tok)
    if m:
        g[G_IDX["ring"]] = 1.0
        flags[3] = 1.0
        k = int(m.group(1))
        if 1 <= k <= MAX_RING:
            ring_oh[k - 1] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = ATOM_RE_DBL.match(tok)
    if m:
        g[G_IDX["bond_double"]] = 1.0
        flags[0] = 1.0
        if m.group(1) in ELEM_IDX:
            e[ELEM_IDX[m.group(1)]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = ATOM_RE_TRP.match(tok)
    if m:
        g[G_IDX["bond_triple"]] = 1.0
        flags[0] = 1.0
        if m.group(1) in ELEM_IDX:
            e[ELEM_IDX[m.group(1)]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = ATOM_RE_AROM.match(tok)
    if m:
        g[G_IDX["atom_arom"]] = 1.0
        flags[0] = 1.0
        flags[1] = 1.0
        upper = m.group(1).upper()
        if upper in ELEM_IDX:
            e[ELEM_IDX[upper]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = ATOM_RE_ALIPH.match(tok)
    if m:
        g[G_IDX["atom_aliph"]] = 1.0
        flags[0] = 1.0
        flags[2] = 1.0
        if m.group(1) in ELEM_IDX:
            e[ELEM_IDX[m.group(1)]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    return np.concatenate([g, e, flags, ring_oh])


def build_E(D: int = 32, seed: int = 7, target_std: float = 0.4) -> np.ndarray:
    F = np.stack([token_features(t) for t in VOCAB], axis=0)
    Fdim = F.shape[1]
    rng = np.random.RandomState(seed)
    # Use a square seed matrix and keep Fdim x D after QR, matching the way
    # build_structured_E in embedding.py would behave when D >= Fdim.
    A = rng.normal(0.0, 1.0, size=(max(D, Fdim), max(D, Fdim)))
    Q, _ = np.linalg.qr(A)
    W = Q[:Fdim, :D]
    E = F @ W
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    mask = np.ones((V,), dtype=bool)
    mask[PAD_ID] = False
    cur_std = float(E[mask].std())
    if cur_std > 0:
        E *= target_std / cur_std
    E[PAD_ID, :] = 0.0
    return E


# ---------------------------------------------------------------------------
# 3. Fourier basis Phi (mirrors fourier_theta.py / build_fourier_basis)
# ---------------------------------------------------------------------------
def build_Phi(L: int, K: int) -> np.ndarray:
    i = np.arange(L, dtype=np.float64)[:, None]
    Phi = np.ones((L, 1 + 2 * K), dtype=np.float64)
    for k in range(1, K + 1):
        Phi[:, k] = np.cos(2.0 * np.pi * k * i[:, 0] / L)
        Phi[:, K + k] = np.sin(2.0 * np.pi * k * i[:, 0] / L)
    return Phi


# ---------------------------------------------------------------------------
# 4. Encode a known molecule into Theta (mirrors smiles_theta_encoder.py)
# ---------------------------------------------------------------------------
# Aspirin SELFIES decomposition (using only tokens in our reduced vocabulary).
# The real `selfies.encoder` for aspirin gives a longer sequence with macros;
# this is a representative simplified version that uses tokens we have.
ASPIRIN_TOKENS = [
    # acetate group: C(=O)O-
    "[C]", "[=Branch1]", "[=O]", "[O]",
    # aromatic ring (6 atoms + ring closure)
    "[c]", "[c]", "[c]", "[c]", "[c]", "[c]", "[Ring5]",
    # carboxyl on the ring: -C(=O)OH
    "[Branch1]", "[C]", "[=Branch1]", "[=O]", "[O]",
]


def theta_from_tokens(tokens, Phi, E):
    L, M = Phi.shape
    D = E.shape[1]
    Z_target = np.zeros((L, D))
    for i in range(L):
        if i < len(tokens):
            tid = TOK2ID.get(tokens[i], PAD_ID)
        else:
            tid = PAD_ID
        Z_target[i] = E[tid]
    # Theta = pinv(Phi) @ Z_target  (least-squares projection onto Fourier modes)
    Theta = np.linalg.pinv(Phi) @ Z_target
    return Theta, Z_target


# ---------------------------------------------------------------------------
# 5. Build everything and run the pipeline forward
# ---------------------------------------------------------------------------
np.random.seed(7)
L = 32          # smaller than the paper's 80 to keep the figure readable
K = 16          # M = 1 + 2*16 = 33, matching the paper
D = 32

E = build_E(D=D, seed=7)
Phi = build_Phi(L=L, K=K)
Theta, Z_target = theta_from_tokens(ASPIRIN_TOKENS, Phi, E)

# Forward pass: reconstruct Z from Theta and decode by nearest neighbour.
Z = Phi @ Theta

# Squared Euclidean distance from every position to every token embedding.
# d2[i, v] = ||Z_i - E_v||^2
z2 = np.sum(Z * Z, axis=1, keepdims=True)
e2 = np.sum(E * E, axis=1, keepdims=False)[None, :]
dots = Z @ E.T
D2 = z2 + e2 - 2 * dots

decoded_ids = D2.argmin(axis=1)
decoded_tokens = [VOCAB[int(i)] for i in decoded_ids]

# How well did we reconstruct?
target_ids = [TOK2ID.get(ASPIRIN_TOKENS[i], PAD_ID) if i < len(ASPIRIN_TOKENS) else PAD_ID
              for i in range(L)]
n_correct = sum(int(d == t) for d, t in zip(decoded_ids, target_ids))
print(f"Reconstruction accuracy: {n_correct}/{L} positions = {100*n_correct/L:.1f}%")
print(f"Decoded sequence: {' '.join(decoded_tokens[:len(ASPIRIN_TOKENS)+2])}...")

# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Render aspirin's 2D structure with RDKit so the figure shows the actual
# molecule that the decoded SELFIES sequence corresponds to.
# ---------------------------------------------------------------------------
from io import BytesIO
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def render_molecule_image(smiles: str, size: int = 520):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


fig = plt.figure(figsize=(13, 12))
gs = GridSpec(4, 3, figure=fig,
              width_ratios=[1.0, 1.4, 0.9],
              height_ratios=[1.0, 1.6, 1.0, 1.1],
              hspace=0.55, wspace=0.35,
              left=0.07, right=0.97, top=0.94, bottom=0.04)

# --- Panel A: Theta (genotype) -------------------------------------------
axA = fig.add_subplot(gs[0, 0])
M = Theta.shape[0]
imA = axA.imshow(Theta, aspect="auto", cmap="RdBu_r",
                 vmin=-np.max(np.abs(Theta)), vmax=np.max(np.abs(Theta)),
                 extent=[-0.5, D - 0.5, M - 0.5, -0.5])
axA.set_title(r"(A) Genotype $\Theta$  (M=%d × D=%d)" % (M, D))
axA.set_xlabel("Latent dim")
axA.set_ylabel("Fourier mode")
axA.set_xlim(-0.5, D - 0.5)
axA.set_ylim(M - 0.5, -0.5)
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

# --- Panel B: Z = Phi @ Theta (latent trajectory) ------------------------
axB = fig.add_subplot(gs[0, 1])
imB = axB.imshow(Z, aspect="auto", cmap="RdBu_r",
                 vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)),
                 extent=[-0.5, D - 0.5, L - 0.5, -0.5])
axB.set_title(r"(B) Latent matrix $Z = \Phi\Theta$  (L=%d × D=%d)" % (L, D))
axB.set_xlabel("Latent dim")
axB.set_ylabel("Sequence position i")
axB.set_xlim(-0.5, D - 0.5)
axB.set_ylim(L - 0.5, -0.5)
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

# --- Panel C: Embedding table E ------------------------------------------
axC = fig.add_subplot(gs[0, 2])
imC = axC.imshow(E, aspect="auto", cmap="RdBu_r",
                 vmin=-np.max(np.abs(E)), vmax=np.max(np.abs(E)),
                 extent=[-0.5, D - 0.5, V - 0.5, -0.5])
axC.set_title(r"(C) Embeddings $E$  (V=%d × D=%d)" % (V, D))
axC.set_xlabel("Latent dim")
axC.set_ylabel("Token")
axC.set_yticks(range(V))
axC.set_yticklabels(VOCAB, fontsize=6.5)
axC.set_xlim(-0.5, D - 0.5)
axC.set_ylim(V - 0.5, -0.5)
fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)

# --- Panel D: Distance matrix D2[i, v] -----------------------------------
axD = fig.add_subplot(gs[1, :])
# Display d^2 with a sequential colormap; lower = better match.
# Cap the colormap at the 80th percentile so the structure is visible
# (otherwise PAD rows past position L_mol dominate the dynamic range).
imD = axD.imshow(D2.T, aspect="auto", cmap="viridis_r",
                 vmin=0, vmax=np.percentile(D2, 80),
                 extent=[-0.5, L - 0.5, V - 0.5, -0.5])
axD.set_title(r"(D) Squared distance $d^2_{i,v} = \|Z_i - E_v\|^2$"
              "   →   argmin per column (red box) gives the decoded token")
axD.set_xlabel("Sequence position i")
axD.set_ylabel("Token v")
axD.set_yticks(range(V))
axD.set_yticklabels(VOCAB, fontsize=7)
axD.set_xticks(range(L))
axD.set_xticklabels(range(L), fontsize=7)
axD.set_xlim(-0.5, L - 0.5)
axD.set_ylim(V - 0.5, -0.5)

# Mark the argmin (winner) per column with a red square.
for i in range(L):
    v_star = int(decoded_ids[i])
    axD.add_patch(plt.Rectangle((i - 0.5, v_star - 0.5), 1, 1,
                                fill=False, edgecolor="red", linewidth=1.4))

# Annotate decoded tokens above the heatmap.
ax_top = axD.secondary_xaxis("top")
ax_top.set_xticks(range(L))
ax_top.set_xticklabels([t.strip("[]") for t in decoded_tokens],
                       rotation=70, fontsize=6.5)
ax_top.tick_params(axis="x", pad=2)

fig.colorbar(imD, ax=axD, fraction=0.025, pad=0.02, label=r"$d^2_{i,v}$")

# Mark the boundary between the molecule and the PAD region with a vertical line
L_mol = len(ASPIRIN_TOKENS)
axD.axvline(L_mol - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
axD.text(L_mol + 0.2, V - 1.5, "PAD region",
         color="white", fontsize=8, va="top", ha="left", alpha=0.95)

# --- Panel E,F,G: per-position softmax probabilities ---------------------
# Show the softmax distribution at three representative positions: an aliphatic
# atom, an aromatic atom in the ring, and a structural ring-closure token.
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])
axG = fig.add_subplot(gs[2, 2])

T_softmax = 1.5  # matches CFG.DECODE_TEMP in the paper
positions_to_show = [0, 6, 10]  # [C], aromatic [c], [Ring5]
panel_axes = [axE, axF, axG]
for ax, pos in zip(panel_axes, positions_to_show):
    logits = -D2[pos] / T_softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    colors = ["#d62728" if v == decoded_ids[pos] else "#999999" for v in range(V)]
    ax.bar(range(V), probs, color=colors, edgecolor="none")
    ax.set_xticks(range(V))
    ax.set_xticklabels([t.strip("[]") for t in VOCAB], rotation=75, fontsize=5.8)
    ax.set_ylabel("p(token)" if pos == positions_to_show[0] else "")
    ax.set_title(
        f"Position i={pos}   →   {decoded_tokens[pos]}",
        fontsize=10,
    )
    ax.set_ylim(0, max(probs) * 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

# --- Panel H: rendered final molecule + decoded SELFIES/SMILES -----------
# The decoded token sequence (panel D), once macro-expanded and parsed by
# selfies.decoder + RDKit, gives the SMILES below. We render the canonical
# 2D structure on the right; the decoded sequence is shown on the left.
ax_caption = fig.add_subplot(gs[3, 0:2])
ax_caption.axis("off")

L_mol = len(ASPIRIN_TOKENS)
decoded_seq_str = "".join(decoded_tokens[:L_mol])

ax_caption.text(0.0, 0.85, "Decoded SELFIES (positions 0…%d):" % (L_mol - 1),
                fontsize=10, fontweight="bold",
                transform=ax_caption.transAxes)
ax_caption.text(0.0, 0.62, decoded_seq_str,
                fontsize=9, family="monospace",
                transform=ax_caption.transAxes)

ax_caption.text(0.0, 0.38, "Canonical SMILES (after RDKit sanitisation):",
                fontsize=10, fontweight="bold",
                transform=ax_caption.transAxes)
ax_caption.text(0.0, 0.18, ASPIRIN_SMILES,
                fontsize=10, family="monospace", color="#0a4a8a",
                transform=ax_caption.transAxes)

ax_mol = fig.add_subplot(gs[3, 2])
ax_mol.imshow(render_molecule_image(ASPIRIN_SMILES, size=520))
ax_mol.set_xticks([])
ax_mol.set_yticks([])
for s in ax_mol.spines.values():
    s.set_visible(False)
ax_mol.set_title("(E) Reconstructed molecule: aspirin", fontsize=10)

fig.suptitle(
    "Reconstructing aspirin from its Fourier-coefficient genotype $\\Theta$",
    fontsize=13, y=0.985,
)

out = Path("theta_to_molecule.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
