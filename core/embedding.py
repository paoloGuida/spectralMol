from __future__ import annotations
import re
import numpy as np
from typing import List, Set
from .vocab import MACRO_EXPANSIONS, MACRO_IS_AROM

def build_structured_E(
    vocab: List[str],
    D: int,
    pad_token: str,
    seed: int,
    target_std: float,
    allowed_elements: Set[str],
) -> np.ndarray:
    elements = sorted([e for e in allowed_elements if e != "H"])
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    elem_pat_up = "|".join(re.escape(e) for e in elements)
    arom_letters = sorted({e.lower() for e in elements if e in {"C", "N", "O", "S"}})
    elem_pat_ar = "".join(re.escape(a) for a in arom_letters)

    groups = ["pad", "atom_aliph", "atom_arom", "bond_double", "bond_triple", "branch", "ring", "macro"]
    g_to_idx = {g: i for i, g in enumerate(groups)}

    ring_idx_max = 0
    ring_tok_pat = re.compile(r"^\[Ring(\d+)\]$")
    for t in vocab:
        m = ring_tok_pat.match(t)
        if m:
            ring_idx_max = max(ring_idx_max, int(m.group(1)))
    max_ring = max(1, ring_idx_max)

    def token_features(tok: str) -> np.ndarray:
        g = np.zeros(len(groups), dtype=np.float64)
        e = np.zeros(len(elements), dtype=np.float64)
        flags = np.zeros(6, dtype=np.float64)
        ring_oh = np.zeros(max_ring, dtype=np.float64)

        if tok == pad_token:
            g[g_to_idx["pad"]] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        if tok in MACRO_EXPANSIONS:
            g[g_to_idx["macro"]] = 1.0
            flags[5] = 1.0
            flags[0] = 1.0
            if MACRO_IS_AROM.get(tok, False):
                flags[1] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        if tok in ("[Branch1]", "[Branch2]", "[Branch3]", "[=Branch1]", "[=Branch2]", "[=Branch3]", "[#Branch1]", "[#Branch2]", "[#Branch3]"):
            g[g_to_idx["branch"]] = 1.0
            flags[5] = 1.0
            flags[0] = 0.5
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        m = re.fullmatch(r"\[Ring(\d+)\]", tok)
        if m:
            g[g_to_idx["ring"]] = 1.0
            flags[5] = 1.0
            k = int(m.group(1))
            if 1 <= k <= max_ring:
                ring_oh[k - 1] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        m = re.fullmatch(rf"\[=({elem_pat_up})([+-]1)?\]", tok)
        if m:
            g[g_to_idx["bond_double"]] = 1.0
            flags[0] = 1.0
            flags[3] = 1.0
            el = m.group(1)
            if el in elem_to_idx:
                e[elem_to_idx[el]] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        m = re.fullmatch(rf"\[#({elem_pat_up})([+-]1)?\]", tok)
        if m:
            g[g_to_idx["bond_triple"]] = 1.0
            flags[0] = 1.0
            flags[4] = 1.0
            el = m.group(1)
            if el in elem_to_idx:
                e[elem_to_idx[el]] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        if elem_pat_ar:
            m = re.fullmatch(rf"\[([{elem_pat_ar}])\]$", tok)
            if m:
                g[g_to_idx["atom_arom"]] = 1.0
                flags[0] = 1.0
                flags[1] = 1.0
                flags[2] = 1.0
                el = m.group(1).upper()
                if el in elem_to_idx:
                    e[elem_to_idx[el]] = 1.0
                return np.concatenate([g, e, flags, ring_oh], axis=0)

        m = re.fullmatch(rf"\[({elem_pat_up})([+-]1)?\]", tok)
        if m:
            g[g_to_idx["atom_aliph"]] = 1.0
            flags[0] = 1.0
            flags[2] = 1.0
            el = m.group(1)
            if el in elem_to_idx:
                e[elem_to_idx[el]] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        flags[5] = 1.0
        return np.concatenate([g, e, flags, ring_oh], axis=0)

    F = np.stack([token_features(t) for t in vocab], axis=0)
    V, Fdim = F.shape

    rng = np.random.RandomState(seed)
    A = rng.normal(0.0, 1.0, size=(Fdim, max(D, Fdim))).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    W = Q[:, :D] if Q.shape[1] >= D else np.pad(Q, ((0, 0), (0, D - Q.shape[1])), mode="constant")

    E = (F @ W).copy()

    pad_id = vocab.index(pad_token)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms

    mask = np.ones((V,), dtype=bool)
    mask[pad_id] = False
    cur_std = float(E[mask].std())
    if cur_std > 0:
        E *= (target_std / cur_std)

    E[pad_id, :] = 0.0
    return E.astype(np.float64)
