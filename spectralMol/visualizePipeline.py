"""
Visualize the SpectralMol Theta -> Z -> SELFIES-token pipeline.

This version keeps the intermediate representation self-consistent:
  - The target molecule is defined by an explicit SELFIES string for aspirin.
  - The target SELFIES tokens are embedded into Z_target.
  - Theta is obtained by least-squares projection onto the Fourier basis.
  - The forward pass Z = Phi @ Theta is decoded by nearest-neighbour token
    selection against the embedding table E.
  - The decoded SELFIES shown in the figure is the actual decoded token
    sequence; [PAD] appears only after the molecule terminates.

RDKit/SMILES are used only internally to render the final 2D chemical structure.
The displayed molecular string is SELFIES.
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    import selfies as sf
except ImportError as exc:  # pragma: no cover - user-environment guard
    raise ImportError(
        "This script requires the `selfies` package. Install it with `pip install selfies`."
    ) from exc

# ---------------------------------------------------------------------------
# 1. Vocabulary: a representative subset containing all SELFIES tokens needed
#    for aspirin plus common tokens from the paper vocabulary.
# ---------------------------------------------------------------------------
VOCAB = [
    "[PAD]",
    # aliphatic atoms
    "[C]", "[N]", "[O]", "[F]",
    # aromatic atoms, kept for compatibility with paper-style figures
    "[c]", "[n]", "[o]", "[s]",
    # double-bond atoms
    "[=C]", "[=N]", "[=O]",
    # triple-bond atoms
    "[#C]", "[#N]",
    # branch tokens, including bond-specific SELFIES branch tokens
    "[Branch1]", "[Branch2]", "[Branch3]",
    "[=Branch1]", "[=Branch2]", "[=Branch3]",
    "[#Branch1]", "[#Branch2]", "[#Branch3]",
    # ring tokens
    "[Ring1]", "[Ring2]", "[Ring3]", "[Ring4]", "[Ring5]", "[Ring6]",
]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
PAD = "[PAD]"
PAD_ID = TOK2ID[PAD]
V = len(VOCAB)

# ---------------------------------------------------------------------------
# 2. Structured embedding table E, mirroring embedding.py / build_structured_E.
# ---------------------------------------------------------------------------
ELEMENTS = ["C", "N", "O", "F", "S"]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENTS)}
GROUPS = [
    "pad", "atom_aliph", "atom_arom", "bond_double", "bond_triple",
    "branch", "ring"
]
G_IDX = {g: i for i, g in enumerate(GROUPS)}
RING_TOK_RE = re.compile(r"^\[Ring(\d+)\]$")
MAX_RING = 6

ATOM_RE_ALIPH = re.compile(r"^\[([A-Z][a-z]?)\]$")
ATOM_RE_AROM = re.compile(r"^\[([cnops])\]$")
ATOM_RE_DBL = re.compile(r"^\[=([A-Z][a-z]?)\]$")
ATOM_RE_TRP = re.compile(r"^\[#([A-Z][a-z]?)\]$")
BRANCH_TOKENS = {
    "[Branch1]", "[Branch2]", "[Branch3]",
    "[=Branch1]", "[=Branch2]", "[=Branch3]",
    "[#Branch1]", "[#Branch2]", "[#Branch3]",
}


def token_features(tok: str) -> np.ndarray:
    """Convert one SELFIES token into a deterministic chemical feature vector."""
    g = np.zeros(len(GROUPS), dtype=np.float64)
    e = np.zeros(len(ELEMENTS), dtype=np.float64)
    flags = np.zeros(4, dtype=np.float64)  # has_atom, is_arom, is_aliph, is_struct
    ring_oh = np.zeros(MAX_RING, dtype=np.float64)

    if tok == PAD:
        g[G_IDX["pad"]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    # Important fix: bond-specific branch tokens such as [=Branch1] are branch
    # tokens too. Treating them as unknown made their embedding the zero vector,
    # indistinguishable from [PAD], which caused [PAD] to appear in the middle
    # of the decoded sequence.
    if tok in BRANCH_TOKENS:
        g[G_IDX["branch"]] = 1.0
        flags[3] = 1.0
        if tok.startswith("[="):
            flags[0] = 0.5  # carries a double-bond branch context
        elif tok.startswith("[#"):
            flags[0] = 0.5  # carries a triple-bond branch context
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
        el = m.group(1)
        if el in ELEM_IDX:
            e[ELEM_IDX[el]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    m = ATOM_RE_TRP.match(tok)
    if m:
        g[G_IDX["bond_triple"]] = 1.0
        flags[0] = 1.0
        el = m.group(1)
        if el in ELEM_IDX:
            e[ELEM_IDX[el]] = 1.0
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
        el = m.group(1)
        if el in ELEM_IDX:
            e[ELEM_IDX[el]] = 1.0
        return np.concatenate([g, e, flags, ring_oh])

    raise ValueError(f"Token {tok!r} is not represented in token_features().")


def build_E(D: int = 32, seed: int = 7, target_std: float = 0.4) -> np.ndarray:
    F = np.stack([token_features(t) for t in VOCAB], axis=0)
    Fdim = F.shape[1]
    rng = np.random.RandomState(seed)

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
    return E.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Fourier basis Phi, mirroring fourier_theta.py / build_fourier_basis.
# ---------------------------------------------------------------------------
def build_Phi(L: int, K: int) -> np.ndarray:
    i = np.arange(L, dtype=np.float64)[:, None]
    Phi = np.ones((L, 1 + 2 * K), dtype=np.float64)
    for k in range(1, K + 1):
        Phi[:, k] = np.cos(2.0 * np.pi * k * i[:, 0] / L)
        Phi[:, K + k] = np.sin(2.0 * np.pi * k * i[:, 0] / L)
    return Phi


# ---------------------------------------------------------------------------
# 4. Define aspirin in SELFIES, then encode it into Theta.
# ---------------------------------------------------------------------------
ASPIRIN_NAME = "aspirin"
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # used internally for validation/rendering only
ASPIRIN_SELFIES = sf.encoder(ASPIRIN_SMILES)
ASPIRIN_TOKENS = list(sf.split_selfies(ASPIRIN_SELFIES))

missing = [tok for tok in ASPIRIN_TOKENS if tok not in TOK2ID]
if missing:
    raise ValueError(f"Vocabulary is missing SELFIES tokens required for aspirin: {missing}")


def theta_from_tokens(tokens: list[str], Phi: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L, _M = Phi.shape
    D = E.shape[1]
    Z_target = np.zeros((L, D), dtype=np.float64)

    for i in range(L):
        tid = TOK2ID[tokens[i]] if i < len(tokens) else PAD_ID
        Z_target[i] = E[tid]

    Theta = np.linalg.pinv(Phi) @ Z_target
    return Theta.astype(np.float64), Z_target


# ---------------------------------------------------------------------------
# 5. Build the pipeline and decode by nearest neighbour.
# ---------------------------------------------------------------------------
np.random.seed(7)
L = 32
K = 16
D = 32

E = build_E(D=D, seed=7)
Phi = build_Phi(L=L, K=K)
Theta, Z_target = theta_from_tokens(ASPIRIN_TOKENS, Phi, E)
Z = Phi @ Theta

z2 = np.sum(Z * Z, axis=1, keepdims=True)
e2 = np.sum(E * E, axis=1, keepdims=False)[None, :]
dots = Z @ E.T
D2 = z2 + e2 - 2.0 * dots

decoded_ids = D2.argmin(axis=1)
decoded_tokens = [VOCAB[int(i)] for i in decoded_ids]

target_ids = [TOK2ID[ASPIRIN_TOKENS[i]] if i < len(ASPIRIN_TOKENS) else PAD_ID for i in range(L)]
n_correct = sum(int(d == t) for d, t in zip(decoded_ids, target_ids))

L_mol = len(ASPIRIN_TOKENS)
first_pad = next((i for i, tok in enumerate(decoded_tokens) if tok == PAD), L)
if first_pad < L_mol:
    raise RuntimeError(
        f"Inconsistent decode: [PAD] appears at position {first_pad} before the molecule ends at {L_mol}."
    )

DECODED_SELFIES = "".join(decoded_tokens[:first_pad])
if DECODED_SELFIES != ASPIRIN_SELFIES:
    print("Warning: decoded SELFIES differs from target SELFIES.")
    print("Target :", ASPIRIN_SELFIES)
    print("Decoded:", DECODED_SELFIES)

# Validate that the displayed SELFIES decodes to aspirin. SMILES is not shown as
# the primary sequence in the figure; it is used only for validation/rendering.
decoded_smiles = sf.decoder(DECODED_SELFIES)
decoded_mol = Chem.MolFromSmiles(decoded_smiles)
if decoded_mol is None:
    raise RuntimeError("Decoded SELFIES did not produce a valid RDKit molecule.")
canonical_smiles = Chem.MolToSmiles(decoded_mol, canonical=True, isomericSmiles=False)

print(f"Reconstruction accuracy: {n_correct}/{L} positions = {100*n_correct/L:.1f}%")
print("Target SELFIES :", ASPIRIN_SELFIES)
print("Decoded SELFIES:", DECODED_SELFIES)
print("Internal canonical SMILES used only for rendering:", canonical_smiles)

# ---------------------------------------------------------------------------
# 6. Plotting.
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


def render_molecule_image_from_selfies(selfies: str, size: int = 520) -> Image.Image:
    smiles = sf.decoder(selfies)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Cannot render: SELFIES decoded to an invalid molecule.")
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


fig = plt.figure(figsize=(13, 10.8))
gs = GridSpec(
    3, 3,
    figure=fig,
    width_ratios=[1.0, 1.4, 0.9],
    height_ratios=[1.45, 1.50, 1.0],
    hspace=0.78,
    wspace=0.35,
    left=0.07,
    right=0.97,
    top=0.94,
    bottom=0.07,
)

# Panel A: Theta.
axA = fig.add_subplot(gs[0, 0])
M = Theta.shape[0]
mx_theta = np.max(np.abs(Theta))
imA = axA.imshow(
    Theta,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-mx_theta,
    vmax=mx_theta,
    extent=[-0.5, D - 0.5, M - 0.5, -0.5],
)
axA.set_title(r"(A) Genotype $\Theta$  (M=%d x D=%d)" % (M, D))
axA.set_xlabel("Latent dim")
axA.set_ylabel("Fourier mode")
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

# Panel B: Z = Phi @ Theta.
axB = fig.add_subplot(gs[0, 1])
mx_z = np.max(np.abs(Z))
imB = axB.imshow(
    Z,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-mx_z,
    vmax=mx_z,
    extent=[-0.5, D - 0.5, L - 0.5, -0.5],
)
axB.set_title(r"(B) Latent matrix $Z = \Phi\Theta$  (L=%d x D=%d)" % (L, D))
axB.set_xlabel("Latent dim")
axB.set_ylabel("Sequence position i")
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

# Panel C: Embedding table E.
axC = fig.add_subplot(gs[0, 2])
mx_e = np.max(np.abs(E))
imC = axC.imshow(
    E,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-mx_e,
    vmax=mx_e,
    extent=[-0.5, D - 0.5, V - 0.5, -0.5],
)
axC.set_title(r"(C) Embeddings $E$  (V=%d x D=%d)" % (V, D))
axC.set_xlabel("Latent dim")
axC.set_ylabel("SELFIES token")
axC.set_yticks(range(V))
axC.set_yticklabels(VOCAB, fontsize=6.2)
fig.colorbar(imC, ax=axC, fraction=0.046, pad=0.04)

# Panel D: distance matrix.
axD = fig.add_subplot(gs[1, :])
imD = axD.imshow(
    D2.T,
    aspect="auto",
    cmap="viridis_r",
    vmin=0,
    vmax=np.percentile(D2, 80),
    extent=[-0.5, L - 0.5, V - 0.5, -0.5],
)
axD.set_title(
    r"(D) Squared distance $d^2_{i,v}=\|Z_i-E_v\|^2$; "
    r"nearest SELFIES token is boxed in red"
)
axD.set_xlabel("Sequence position i")
axD.set_ylabel("SELFIES token v")
axD.set_yticks(range(V))
axD.set_yticklabels(VOCAB, fontsize=7)
axD.set_xticks(range(L))
axD.set_xticklabels(range(L), fontsize=7)

for i in range(L):
    v_star = int(decoded_ids[i])
    axD.add_patch(
        plt.Rectangle(
            (i - 0.5, v_star - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=1.4,
        )
    )

ax_top = axD.secondary_xaxis("top")
ax_top.set_xticks(range(L))
ax_top.set_xticklabels([t.strip("[]") for t in decoded_tokens], rotation=70, fontsize=6.5)
ax_top.tick_params(axis="x", pad=2)
fig.colorbar(imD, ax=axD, fraction=0.025, pad=0.02, label=r"$d^2_{i,v}$")

axD.axvline(L_mol - 0.5, color="white", linestyle="--", linewidth=1.2, alpha=0.85)
axD.text(
    L_mol + 0.2,
    V - 1.5,
    "PAD region",
    color="white",
    fontsize=8,
    va="top",
    ha="left",
    alpha=0.95,
)

# Panels E/F/G: token probabilities at representative positions.
axE = fig.add_subplot(gs[2, 0])
axF = fig.add_subplot(gs[2, 1])
axG = fig.add_subplot(gs[2, 2])

T_softmax = 1.5
positions_to_show = [0, 6, 12]  # [C], ring [C], [Ring1]
for ax, pos in zip([axE, axF, axG], positions_to_show):
    logits = -D2[pos] / T_softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    colors = ["#d62728" if v == decoded_ids[pos] else "#999999" for v in range(V)]
    ax.bar(range(V), probs, color=colors, edgecolor="none")
    ax.set_xticks(range(V))
    ax.set_xticklabels([t.strip("[]") for t in VOCAB], rotation=75, fontsize=5.6)
    ax.set_ylabel("p(token)" if pos == positions_to_show[0] else "")
    ax.set_title(f"Position i={pos} -> {decoded_tokens[pos]}", fontsize=10)
    ax.set_ylim(0, max(probs) * 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

fig.suptitle(
    "Reconstructing aspirin from its Fourier-coefficient genotype $\\Theta$",
    fontsize=13,
    y=0.985,
)

out = Path("theta_to_molecule_selfies_stretched_first_row.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved: {out}")
