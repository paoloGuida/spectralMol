from __future__ import annotations
import re
import numpy as np
import selfies as sf
from typing import Dict, List, Optional, Set
from .config import CFG
from .vocab import MACRO_EXPANSIONS, merge_macro_expansions


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(getattr(CFG, name, default))
    except Exception:
        return float(default)


def _token_is_atom_like(tok: str) -> bool:
    return bool(re.fullmatch(r"\[(?:=|#)?[A-Za-z][a-z]?(?:H)?(?:[+\-]\d+)?\]", tok))


def _token_element(tok: str) -> str:
    core = tok.strip("[]")
    if core.startswith("=") or core.startswith("#"):
        core = core[1:]
    core = core.replace("+1", "").replace("-1", "")
    return core[:2] if core[:2] in {"Cl", "Br"} else core[:1]


def _medchem_motif_features(tok: str, expansion: str) -> np.ndarray:
    """Small deterministic motif axes used to smooth medchem theta neighborhoods."""
    try:
        tokens = list(sf.split_selfies(expansion)) if expansion else [tok]
    except Exception:
        tokens = [tok]
    if not tokens:
        tokens = [tok]

    atom_tokens = [t for t in tokens if _token_is_atom_like(t)]
    atom_count = max(1, len(atom_tokens))
    elements = [_token_element(t) for t in atom_tokens]

    arom = sum(1 for t in atom_tokens if _token_element(t) in {"c", "n", "o", "s"}) / atom_count
    hetero = sum(1 for e in elements if e not in {"C", "c", "H", "F", "Cl", "Br", "I"}) / atom_count
    halogen = sum(1 for e in elements if e in {"F", "Cl", "Br", "I"}) / atom_count
    carbonyl = 1.0 if "[=O]" in tokens else 0.0
    amide = 1.0 if carbonyl and any(t in {"[N]", "[NH1]", "[NH2]", "[n]"} for t in tokens) else 0.0
    ring = min(1.0, sum(1 for t in tokens if "Ring" in t) / 2.0)
    ether = 1.0 if "[O]" in tokens and any(_token_element(t) == "c" for t in atom_tokens) else 0.0
    size = min(1.0, atom_count / 14.0)

    return np.asarray([arom, amide, carbonyl, halogen, hetero, ring, ether, size], dtype=np.float64)

def build_structured_E(
    vocab: List[str],
    D: int,
    pad_token: str,
    seed: int,
    target_std: float,
    allowed_elements: Set[str],
    macro_expansions: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    elements = sorted([e for e in allowed_elements if e != "H"])
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    all_macro_expansions = merge_macro_expansions(macro_expansions)

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

    def empty_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g = np.zeros(len(groups), dtype=np.float64)
        e = np.zeros(len(elements), dtype=np.float64)
        flags = np.zeros(6, dtype=np.float64)
        ring_oh = np.zeros(max_ring, dtype=np.float64)
        return g, e, flags, ring_oh

    def token_features(tok: str, *, expand_macro: bool = True) -> np.ndarray:
        g, e, flags, ring_oh = empty_features()

        if tok == pad_token:
            g[g_to_idx["pad"]] = 1.0
            return np.concatenate([g, e, flags, ring_oh], axis=0)

        if expand_macro and tok in all_macro_expansions:
            g[g_to_idx["macro"]] = 1.0
            flags[5] = 1.0
            expansion = all_macro_expansions.get(tok, "")
            child_features: list[np.ndarray] = []
            try:
                expansion_tokens = list(sf.split_selfies(expansion))
            except Exception:
                expansion_tokens = []
            for child in expansion_tokens:
                if child == tok:
                    continue
                child_features.append(token_features(child, expand_macro=False))
            if child_features:
                child_mean = np.mean(np.stack(child_features, axis=0), axis=0)
                g += 0.45 * child_mean[: len(groups)]
                e += child_mean[len(groups) : len(groups) + len(elements)]
                flags += child_mean[len(groups) + len(elements) : len(groups) + len(elements) + len(flags)]
                ring_oh += child_mean[-max_ring:]
                flags[0] = max(flags[0], 1.0)
            else:
                flags[0] = 1.0
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
    tok2id = {tok: i for i, tok in enumerate(vocab)}

    rng = np.random.RandomState(seed)
    A = rng.normal(0.0, 1.0, size=(Fdim, max(D, Fdim))).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    W = Q[:, :D] if Q.shape[1] >= D else np.pad(Q, ((0, 0), (0, D - Q.shape[1])), mode="constant")

    E = (F @ W).copy()

    medchem_bias = max(0.0, _cfg_float("SPECTRAL_EMBED_MEDCHEM_BIAS", 0.0))
    if medchem_bias > 0.0 and D > 0:
        motif_axes = min(8, D)
        motif_features = np.zeros((V, motif_axes), dtype=np.float64)
        for i, tok in enumerate(vocab):
            if tok == pad_token:
                continue
            expansion = all_macro_expansions.get(tok, tok)
            motif_features[i, :motif_axes] = _medchem_motif_features(tok, expansion)[:motif_axes]
        motif_features -= motif_features.mean(axis=0, keepdims=True)
        E[:, :motif_axes] += float(medchem_bias) * motif_features

    macro_centroid_blend = min(1.0, max(0.0, _cfg_float("SPECTRAL_EMBED_MACRO_EXPANSION_BLEND", 0.0)))
    if macro_centroid_blend > 0.0:
        base_E = E.copy()
        for i, tok in enumerate(vocab):
            expansion = all_macro_expansions.get(tok)
            if not expansion:
                continue
            try:
                child_tokens = list(sf.split_selfies(expansion))
            except Exception:
                child_tokens = []
            child_ids = [
                tok2id[child]
                for child in child_tokens
                if child in tok2id and child != tok and tok2id[child] != i
            ]
            if not child_ids:
                continue
            centroid = np.mean(base_E[child_ids, :], axis=0)
            E[i, :] = (1.0 - macro_centroid_blend) * base_E[i, :] + macro_centroid_blend * centroid

    identity_scale = max(0.0, _cfg_float("SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE", 0.0))
    if identity_scale > 0.0 and D > 0:
        identity_rng = np.random.RandomState(int(seed) + 7919)
        identity = identity_rng.normal(0.0, 1.0, size=(V, D)).astype(np.float64)
        identity /= np.linalg.norm(identity, axis=1, keepdims=True) + 1e-12
        identity[vocab.index(pad_token), :] = 0.0
        E += float(identity_scale) * identity

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
