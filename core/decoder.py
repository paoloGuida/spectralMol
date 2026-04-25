from __future__ import annotations
from dataclasses import dataclass
import random
import re
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from .mol_cache import get_molstuff

from .config import CFG
from .vocab import MACRO_EXPANSIONS, MACRO_ATOM_COUNTS
from .xp import backend_name, get_array_module, to_numpy


@dataclass
class DecodeAccelerator:
    xp_mod: Any
    xp_ctx: Any
    phi_dev: Any
    e_dev: Any
    e_t_dev: Any
    e2_dev: Any
    dtype_np: np.dtype


@dataclass(frozen=True)
class DecodeTokenState:
    ring_id_to_k: Dict[int, int]
    k_to_ring_id: Dict[int, int]
    struct_tokens: Set[int]
    macro_ids: Set[int]
    macro_allowed_ids: Set[int]
    macro_blocked_ids: Set[int]
    macro_tokens: Set[str]
    atom_tokens: Set[str]
    forbidden_atom_ids: Set[int]
    hetero_atom_ids: Set[int]
    ring_arom_allowed: Set[str]
    ring_aliph_allowed: Set[str]


def build_decode_accelerator(Phi: np.ndarray, E: np.ndarray) -> Optional[DecodeAccelerator]:
    """Prepare GPU-resident decode tensors for repeated distance computations."""
    xp_mod, xp_ctx = get_array_module(
        bool(getattr(CFG, "USE_CUDA", False)),
        int(getattr(CFG, "CUDA_DEVICE", 0)),
        use_mps=bool(getattr(CFG, "USE_MPS", False)),
    )

    if xp_mod is np:
        return None

    dtype_np = np.float32 if bool(getattr(CFG, "DECODE_FP32", True)) else np.float64

    with xp_ctx:
        phi_dev = xp_mod.asarray(Phi.astype(dtype_np, copy=False))
        e_dev = xp_mod.asarray(E.astype(dtype_np, copy=False))
        e_t_dev = e_dev.T
        e2_dev = xp_mod.sum(e_dev * e_dev, axis=1, keepdims=False)[None, :]

    return DecodeAccelerator(
        xp_mod=xp_mod,
        xp_ctx=xp_ctx,
        phi_dev=phi_dev,
        e_dev=e_dev,
        e_t_dev=e_t_dev,
        e2_dev=e2_dev,
        dtype_np=np.dtype(dtype_np),
    )


_TOKEN_STATE_CACHE: Dict[Tuple[int, Tuple[str, ...]], DecodeTokenState] = {}


def build_decode_token_state(id2tok: dict[int, str]) -> DecodeTokenState:
    """Build/cache token-indexed decode metadata reused across all decode attempts."""
    allowed_elems = tuple(getattr(CFG, "ALLOWED_ELEMENTS", ("C", "N", "O", "S", "F", "Cl", "Br", "I", "H")))
    macro_allow_cfg = tuple(getattr(CFG, "MACRO_WHITELIST", ()) or ())
    macro_deny_cfg = tuple(getattr(CFG, "MACRO_BLACKLIST", ()) or ())
    cache_key = (id(id2tok), allowed_elems, macro_allow_cfg, macro_deny_cfg)
    cached = _TOKEN_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ring_token_pat = re.compile(r"^\[Ring(\d+)\]$")
    atom_any_re = re.compile(r"^\[(?:=|#)?([A-Za-z][a-z]?)(?:H)?(?:[+-]\d+)?\]$")
    ring_arom_re = re.compile(r"^\[(?:=|#)?(?:c|n|o|s)(?:H)?(?:[+-]1)?\]$")

    branch_tokens = {"[Branch1]", "[Branch2]", "[Branch3]"}
    macro_tokens = set(MACRO_EXPANSIONS.keys())
    atom_tokens = {t for t in set(id2tok.values()) if atom_any_re.match(t)}

    allowed_elem_set = set(allowed_elems) | {e.lower() for e in allowed_elems}
    forbidden_atom_ids: Set[int] = set()
    hetero_atom_ids: Set[int] = set()

    ring_id_to_k: Dict[int, int] = {}
    k_to_ring_id: Dict[int, int] = {}
    macro_ids: Set[int] = set()
    macro_allowed_ids: Set[int] = set()
    macro_blocked_ids: Set[int] = set()
    branch_ids: Set[int] = set()
    macro_allow_set = set(macro_allow_cfg)
    macro_deny_set = set(macro_deny_cfg)

    for tid, tok in id2tok.items():
        i = int(tid)
        m_ring = ring_token_pat.match(tok)
        if m_ring:
            kk = int(m_ring.group(1))
            ring_id_to_k[i] = kk
            k_to_ring_id[kk] = i

        if tok in macro_tokens:
            macro_ids.add(i)
            if (macro_allow_set and tok not in macro_allow_set) or (tok in macro_deny_set):
                macro_blocked_ids.add(i)
            else:
                macro_allowed_ids.add(i)
        if tok in branch_tokens:
            branch_ids.add(i)

        m_atom = atom_any_re.match(tok)
        if m_atom:
            elem = m_atom.group(1)
            if elem not in allowed_elem_set:
                forbidden_atom_ids.add(i)
            if elem.lower() not in {"c", "h"}:
                hetero_atom_ids.add(i)

    struct_tokens: Set[int] = set(ring_id_to_k.keys()) | branch_ids | macro_ids

    ring_arom_allowed = {t for t in set(id2tok.values()) if ring_arom_re.match(t)}
    ring_aliph_allowed = {t for t in atom_tokens if t.startswith("[") and not t.startswith("[#")}

    state = DecodeTokenState(
        ring_id_to_k=ring_id_to_k,
        k_to_ring_id=k_to_ring_id,
        struct_tokens=struct_tokens,
        macro_ids=macro_ids,
        macro_allowed_ids=macro_allowed_ids,
        macro_blocked_ids=macro_blocked_ids,
        macro_tokens=macro_tokens,
        atom_tokens=atom_tokens,
        forbidden_atom_ids=forbidden_atom_ids,
        hetero_atom_ids=hetero_atom_ids,
        ring_arom_allowed=ring_arom_allowed,
        ring_aliph_allowed=ring_aliph_allowed,
    )
    _TOKEN_STATE_CACHE[cache_key] = state
    return state


def compute_dist2_batch(Theta_batch: np.ndarray, accel: DecodeAccelerator) -> np.ndarray:
    """Compute batched dist2 for Thetas of shape (B, M, D) -> (B, L, V)."""
    xp_mod = accel.xp_mod
    with accel.xp_ctx:
        theta_dev = xp_mod.asarray(Theta_batch.astype(accel.dtype_np, copy=False))
        Z_dev = accel.phi_dev @ theta_dev
        z2 = xp_mod.sum(Z_dev * Z_dev, axis=2, keepdims=True)
        dots = Z_dev @ accel.e_t_dev
        dist2 = z2 + accel.e2_dev - 2.0 * dots
    return to_numpy(xp_mod, dist2)


def compute_decode_topk_batch(
    Theta_batch: np.ndarray,
    accel: DecodeAccelerator,
    *,
    topk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-position top-k token candidates for a Theta batch.

    Returns:
      - cand_ids: (B, L, K) token ids
      - cand_dist2: (B, L, K) corresponding squared distances

    CUDA path keeps top-k selection on GPU and transfers only compact top-k tensors.
    Non-CUDA path falls back to CPU-side top-k extraction from full dist2.
    """
    k_req = int(CFG.DECODE_TOPK if topk is None else topk)
    k_req = max(1, k_req)

    if backend_name(accel.xp_mod) != "cuda":
        dist2 = compute_dist2_batch(Theta_batch, accel)
        k = min(k_req, dist2.shape[2])
        cand_ids = np.argpartition(dist2, kth=k - 1, axis=2)[:, :, :k].astype(np.int32, copy=False)
        cand_dist2 = np.take_along_axis(dist2, cand_ids.astype(np.int64, copy=False), axis=2)
        return cand_ids, cand_dist2

    xp_mod = accel.xp_mod
    with accel.xp_ctx:
        theta_dev = xp_mod.asarray(Theta_batch.astype(accel.dtype_np, copy=False))
        Z_dev = accel.phi_dev @ theta_dev
        z2 = xp_mod.sum(Z_dev * Z_dev, axis=2, keepdims=True)
        dots = Z_dev @ accel.e_t_dev
        dist2 = z2 + accel.e2_dev - 2.0 * dots  # (B,L,V)
        k = min(k_req, int(dist2.shape[2]))
        cand_ids_dev = xp_mod.argpartition(dist2, kth=k - 1, axis=2)[:, :, :k]
        cand_dist2_dev = xp_mod.take_along_axis(dist2, cand_ids_dev, axis=2)

    return to_numpy(xp_mod, cand_ids_dev).astype(np.int32, copy=False), to_numpy(xp_mod, cand_dist2_dev)

def ids_to_selfies(ids: np.ndarray, id2tok: dict[int, str]) -> str:
    toks: list[str] = []
    for tid in ids.tolist():
        tok = id2tok.get(int(tid), "[PAD]")
        if tok == "[PAD]":
            break
        if tok in MACRO_EXPANSIONS:
            toks.append(MACRO_EXPANSIONS[tok])
        else:
            toks.append(tok)
    return "".join(toks)

def compute_dist2(Z: np.ndarray, E: np.ndarray, *, xp=None) -> np.ndarray:
    """Squared distance matrix between rows of Z and embeddings E.

    Uses CuPy/MPS when requested; always returns a NumPy array so downstream
    code stays unchanged.
    """
    xp_mod = xp if xp is not None else np

    Z_dev = xp_mod.asarray(Z)
    E_dev = xp_mod.asarray(E)

    z2 = xp_mod.sum(Z_dev * Z_dev, axis=1, keepdims=True)
    e2 = xp_mod.sum(E_dev * E_dev, axis=1, keepdims=False)[None, :]
    dots = Z_dev @ E_dev.T
    dist2 = z2 + e2 - 2.0 * dots

    if xp_mod is np:
        return dist2

    return to_numpy(xp_mod, dist2)

def effective_min_atoms_before_pad(gen: int) -> int:
    start = int(CFG.MIN_ATOMS_WARMUP_START)
    end = int(CFG.MIN_ATOMS_BEFORE_PAD)
    warm = max(1, int(CFG.MIN_MW_WARMUP_GENS))
    if gen >= warm:
        return end
    t = gen / float(warm)
    return int(round((1.0 - t) * start + t * end))

def murcko_scaffold_smiles(canon_smiles: str) -> Optional[str]:
    """Return canonical Murcko scaffold SMILES for a molecule SMILES.

    Returns None if scaffold cannot be computed or scaffold is empty (e.g., acyclic).
    Uses a global cache to avoid repeated RDKit parsing/scaffold extraction.
    """
    return get_molstuff(canon_smiles).scaffold

def dist2_to_ids_sample(
    dist2: Optional[np.ndarray],
    id2tok: dict[int, str],
    pad_id: int,
    gen: int,
    token_state: Optional[DecodeTokenState] = None,
    strict_no_struct: bool = False,
    force_ring_plan: bool = False,
    force_ring_type: str = "arom",
    dist2_topk_ids: Optional[np.ndarray] = None,
    dist2_topk_vals: Optional[np.ndarray] = None,
) -> np.ndarray:
    if dist2_topk_ids is not None and dist2_topk_vals is not None:
        Lloc = int(dist2_topk_ids.shape[0])
    elif dist2 is not None:
        Lloc = int(dist2.shape[0])
    else:
        raise ValueError("dist2_to_ids_sample requires either full dist2 or top-k precomputed tensors.")
    state = token_state if token_state is not None else build_decode_token_state(id2tok)
    ring_id_to_k = state.ring_id_to_k
    k_to_ring_id = state.k_to_ring_id
    struct_tokens = state.struct_tokens
    macro_ids = state.macro_ids
    macro_allowed_ids = state.macro_allowed_ids
    macro_blocked_ids = state.macro_blocked_ids
    macro_tokens = state.macro_tokens
    atom_tokens = state.atom_tokens
    forbidden_atom_ids = state.forbidden_atom_ids
    hetero_atom_ids = state.hetero_atom_ids
    branch_tokens = {"[Branch1]", "[Branch2]", "[Branch3]"}

    ids = np.empty((Lloc,), dtype=np.int64)

    atom_count = 0
    consecutive_hetero_atoms = 0
    last_was_struct = False

    ring_active = False
    ring_target_len = 0
    ring_type = None
    ring_atoms_emitted = 0

    ring_arom_allowed = state.ring_arom_allowed
    ring_aliph_allowed = state.ring_aliph_allowed

    min_atoms_now = effective_min_atoms_before_pad(gen)
    enable_macros = bool(getattr(CFG, "ENABLE_MACROS", True))

    for pos in range(Lloc):
        if dist2_topk_ids is not None and dist2_topk_vals is not None:
            cand = dist2_topk_ids[pos].astype(np.int64, copy=False)
            row_vals = dist2_topk_vals[pos]
            if cand.size == 0:
                cand = np.array([int(pad_id)], dtype=np.int64)
                row_vals = np.array([0.0], dtype=np.float64)
        else:
            assert dist2 is not None
            row = dist2[pos]
            k = min(CFG.DECODE_TOPK, row.shape[0])
            cand = np.argpartition(row, k - 1)[:k].astype(np.int64)
            row_vals = row[cand]

        logits = -row_vals / max(1e-12, float(CFG.DECODE_TEMP))
        logits -= np.max(logits)
        probs = np.exp(logits)

        if strict_no_struct:
            for j, tid in enumerate(cand):
                if int(tid) in struct_tokens:
                    probs[j] = 0.0
        # Element mask: drop atom tokens whose element is not allowed for this benchmark.
        if forbidden_atom_ids:
            for j, tid in enumerate(cand):
                if int(tid) in forbidden_atom_ids:
                    probs[j] = 0.0
        if not enable_macros:
            for j, tid in enumerate(cand):
                if int(tid) in macro_ids:
                    probs[j] = 0.0
        elif macro_blocked_ids:
            for j, tid in enumerate(cand):
                if int(tid) in macro_blocked_ids:
                    probs[j] = 0.0
        if macro_allowed_ids:
            for j, tid in enumerate(cand):
                tid_i = int(tid)
                if (tid_i in macro_ids) and (tid_i not in macro_allowed_ids):
                    probs[j] = 0.0

        max_consec_hetero = int(getattr(CFG, "MAX_CONSEC_HETERO_TOKENS", 3))
        if max_consec_hetero > 0 and consecutive_hetero_atoms >= max_consec_hetero:
            # Softly downweight long hetero streaks in token space (hard chemistry gate runs later).
            streak_penalty = float(getattr(CFG, "HETERO_STREAK_PENALTY", 0.15))
            for j, tid in enumerate(cand):
                if int(tid) in hetero_atom_ids:
                    probs[j] *= streak_penalty


        if (not strict_no_struct) and (not ring_active):
            # Force a ring plan at the start of decoding (used to guarantee
            # some aromatic candidates get evaluated each generation).
            if force_ring_plan and pos == 0:
                ring_active = True
                ring_target_len = 6  # force 6-member ring (phenyl-like)
                ring_type = force_ring_type
                ring_atoms_emitted = 0
            elif pos < (Lloc - CFG.RING_PLAN_MIN_REMAIN) and random.random() < CFG.P_START_RING_PLAN:
                ring_active = True
                ring_target_len = 6 if random.random() < CFG.RING_PLAN_LEN_BIAS_6 else 5
                ring_type = "arom" if random.random() < CFG.RING_PLAN_AROM_BIAS else "aliph"
                ring_atoms_emitted = 0

        if atom_count < min_atoms_now:
            for j, tid in enumerate(cand):
                if int(tid) == int(pad_id):
                    probs[j] = 0.0
        else:
            if pos < max(CFG.EARLY_PAD_CUTOFF, CFG.MIN_TOKENS_BEFORE_PAD):
                for j, tid in enumerate(cand):
                    if int(tid) == int(pad_id):
                        probs[j] *= CFG.PAD_PENALTY

        if (not strict_no_struct) and (not ring_active):
            if pos >= (Lloc - CFG.STRUCT_FORBID_LAST_N):
                for j, tid in enumerate(cand):
                    if int(tid) in struct_tokens:
                        probs[j] = 0.0

            if CFG.STRUCT_FORBID_CONSECUTIVE and last_was_struct:
                for j, tid in enumerate(cand):
                    if int(tid) in struct_tokens:
                        probs[j] = 0.0

            for j, tid in enumerate(cand):
                if int(tid) in struct_tokens:
                    probs[j] *= CFG.STRUCT_SUPPRESS_BASE

        prev_tok = id2tok.get(int(ids[pos - 1]), None) if pos > 0 else None
        prev_was_atom = (prev_tok in atom_tokens) if prev_tok is not None else False
        prev_was_macro = (prev_tok in macro_tokens) if prev_tok is not None else False

        # --- Macro placement constraints (attachment-point safety) ---
        # Macros expand into SELFIES fragments that assume they are emitted
        # at an attachment-capable position (start or after an atom/macro).
        # This avoids invalid contexts like emitting a macro right after a branch token.
        if not strict_no_struct:
            if pos > 0 and not (prev_was_atom or prev_was_macro):
                for j, tid in enumerate(cand):
                    if int(tid) in macro_ids:
                        probs[j] = 0.0
            if getattr(CFG, 'MACRO_FORBID_CONSECUTIVE', True) and prev_was_macro:
                for j, tid in enumerate(cand):
                    if int(tid) in macro_ids:
                        probs[j] = 0.0
            if getattr(CFG, 'MACRO_FORBID_RING_AFTER_MACRO', True) and prev_was_macro:
                for j, tid in enumerate(cand):
                    if int(tid) in ring_id_to_k:
                        probs[j] = 0.0

            for j, tid in enumerate(cand):
                if int(tid) in ring_id_to_k and not (prev_was_atom or prev_was_macro):
                    probs[j] = 0.0

        if ring_active and (not strict_no_struct):
            if ring_atoms_emitted < ring_target_len:
                for j, tid in enumerate(cand):
                    tok = id2tok.get(int(tid), "[PAD]")

                    if int(tid) == int(pad_id):
                        probs[j] = 0.0
                        continue

                    if int(tid) in ring_id_to_k:
                        probs[j] = 0.0
                        continue

                    if tok in branch_tokens:
                        probs[j] = 0.0
                        continue

                    if int(tid) in macro_ids:
                        if not CFG.RING_PLAN_ALLOW_MACROS:
                            probs[j] = 0.0
                            continue
                        continue

                    if ring_type == "arom":
                        if tok not in ring_arom_allowed:
                            probs[j] = 0.0
                    else:
                        if tok not in ring_aliph_allowed:
                            probs[j] = 0.0

            if ring_atoms_emitted == ring_target_len and prev_was_atom:
                needed_k = ring_target_len - 1
                needed_id = k_to_ring_id.get(needed_k, None)
                for j, tid in enumerate(cand):
                    probs[j] = 1.0 if (needed_id is not None and int(tid) == int(needed_id)) else 0.0

        s = float(np.sum(probs))
        probs = probs / s if s > 0 else np.ones_like(probs) / len(probs)

        chosen = (
            int(np.random.choice(cand, p=probs))
            if random.random() < CFG.P_SAMPLE
            else int(cand[np.argmax(probs)])
        )

        ids[pos] = chosen
        tok = id2tok.get(chosen, "[PAD]")

        if tok in atom_tokens:
            atom_count += 1
        elif tok in macro_tokens:
            atom_count += int(MACRO_ATOM_COUNTS.get(tok, 1))

        if chosen in hetero_atom_ids:
            consecutive_hetero_atoms += 1
        else:
            consecutive_hetero_atoms = 0

        last_was_struct = (chosen in struct_tokens)

        if ring_active and (not strict_no_struct):
            if tok in macro_tokens:
                ring_active = False
                ring_target_len = 0
                ring_type = None
                ring_atoms_emitted = 0
            else:
                if ring_type == "arom":
                    if tok in ring_arom_allowed:
                        ring_atoms_emitted += 1
                else:
                    if tok in ring_aliph_allowed:
                        ring_atoms_emitted += 1

                if chosen in ring_id_to_k:
                    ring_active = False
                    ring_target_len = 0
                    ring_type = None
                    ring_atoms_emitted = 0

    return ids

def decode_theta_best_of_k(
    Theta: np.ndarray,
    Phi: np.ndarray,
    E: np.ndarray,
    id2tok: dict[int, str],
    pad_id: int,
    k: int,
    gen: int,
    selfies_to_smiles_fn,
    objectives_from_smiles_fn,
    *,
    seen_smiles: Optional[set[str]] = None,
    seen_scaffolds: Optional[set[str]] = None,
    novelty_mode: str = "both",  # "smiles" | "scaffold" | "both" | "off"
    accel: Optional[DecodeAccelerator] = None,
    dist2_precomputed: Optional[np.ndarray] = None,
    dist2_topk_ids_precomputed: Optional[np.ndarray] = None,
    dist2_topk_vals_precomputed: Optional[np.ndarray] = None,
    token_state: Optional[DecodeTokenState] = None,
) -> tuple[Optional[str], tuple[float, float], str, int]:
    """
    Decode a Theta into a molecule by sampling k candidates and taking the best feasible.

    Novelty gating modes:
      - "smiles"   : reject if canonical SMILES is already in seen_smiles
      - "scaffold" : reject if Murcko scaffold is already in seen_scaffolds
      - "both"     : reject if either SMILES or scaffold is already seen
      - "off"      : no novelty rejection
    """
    best_smi: Optional[str] = None
    best_f: tuple[float, float] = (1e9, 1e9)
    best_macro_count: int = 0
    reason_counts: Dict[str, int] = {}

    dist2: Optional[np.ndarray] = None
    if dist2_topk_ids_precomputed is None or dist2_topk_vals_precomputed is None:
        if dist2_precomputed is not None:
            dist2 = dist2_precomputed
        elif accel is not None:
            dist2 = compute_dist2_batch(np.expand_dims(Theta, axis=0), accel)[0]
        else:
            xp_mod, xp_ctx = get_array_module(
                bool(getattr(CFG, "USE_CUDA", False)),
                int(getattr(CFG, "CUDA_DEVICE", 0)),
                use_mps=bool(getattr(CFG, "USE_MPS", False)),
            )
            with xp_ctx:
                Z_dev = xp_mod.asarray(Phi) @ xp_mod.asarray(Theta)
                dist2 = compute_dist2(Z_dev, E=xp_mod.asarray(E), xp=xp_mod)

    decoder_fail_count = 0
    strict_mode = False
    macro_tokens = set(MACRO_EXPANSIONS.keys())

    def _is_duplicate(canon: str) -> bool:
        if novelty_mode == "off":
            return False

        dup_smiles = (seen_smiles is not None) and (canon in seen_smiles)
        if novelty_mode == "smiles":
            return dup_smiles

        scaf = murcko_scaffold_smiles(canon)
        dup_scaf = (scaf is not None) and (seen_scaffolds is not None) and (scaf in seen_scaffolds)

        if novelty_mode == "scaffold":
            return dup_scaf

        # novelty_mode == "both"
        return dup_smiles or dup_scaf

    # ------------------------------------------------------------
    # Attempt 0: forced aromatic ring-plan (guarantees aromatics)
    # ------------------------------------------------------------
    if k >= 1:
        ids = dist2_to_ids_sample(
            dist2,
            id2tok=id2tok,
            pad_id=pad_id,
            gen=gen,
            token_state=token_state,
            strict_no_struct=False,
            force_ring_plan=True,
            force_ring_type="arom",
            dist2_topk_ids=dist2_topk_ids_precomputed,
            dist2_topk_vals=dist2_topk_vals_precomputed,
        )

        cand_macro_count = 0
        for tid in ids.tolist():
            tok = id2tok.get(int(tid), "[PAD]")
            if tok == "[PAD]":
                break
            if tok in macro_tokens:
                cand_macro_count += 1

        selfies = ids_to_selfies(ids, id2tok=id2tok)
        if selfies:
            canon, r = selfies_to_smiles_fn(selfies)
            if canon is not None:
                if _is_duplicate(canon):
                    reason_counts["DUPLICATE_NOVELTY"] = reason_counts.get("DUPLICATE_NOVELTY", 0) + 1
                else:
                    f1, f2, _r2 = objectives_from_smiles_fn(canon, gen=gen)
                    if f1 < 1e8:
                        f = (f1, f2)
                        if f < best_f:
                            best_f = f
                            best_smi = canon
                            best_macro_count = cand_macro_count
            else:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        else:
            reason_counts["EMPTY_SELFIES"] = reason_counts.get("EMPTY_SELFIES", 0) + 1

    # ------------------------------------------------------------
    # Remaining attempts: stochastic decoding with adaptive strict mode
    # ------------------------------------------------------------
    for _ in range(max(0, k - 1)):
        if (not strict_mode) and decoder_fail_count >= 2:
            strict_mode = True

        ids = dist2_to_ids_sample(
            dist2,
            id2tok=id2tok,
            pad_id=pad_id,
            gen=gen,
            token_state=token_state,
            strict_no_struct=strict_mode,
            dist2_topk_ids=dist2_topk_ids_precomputed,
            dist2_topk_vals=dist2_topk_vals_precomputed,
        )

        cand_macro_count = 0
        for tid in ids.tolist():
            tok = id2tok.get(int(tid), "[PAD]")
            if tok == "[PAD]":
                break
            if tok in macro_tokens:
                cand_macro_count += 1

        selfies = ids_to_selfies(ids, id2tok=id2tok)
        if not selfies:
            reason_counts["EMPTY_SELFIES"] = reason_counts.get("EMPTY_SELFIES", 0) + 1
            continue

        canon, r = selfies_to_smiles_fn(selfies)
        if canon is None:
            reason_counts[r] = reason_counts.get(r, 0) + 1
            if r == "SELFIES2SMILES_FAIL":
                decoder_fail_count += 1
            continue

        if _is_duplicate(canon):
            reason_counts["DUPLICATE_NOVELTY"] = reason_counts.get("DUPLICATE_NOVELTY", 0) + 1
            continue

        f1, f2, r = objectives_from_smiles_fn(canon, gen=gen)
        if f1 >= 1e8:
            reason_counts[r] = reason_counts.get(r, 0) + 1
            continue

        f = (f1, f2)
        if f < best_f:
            best_f = f
            best_smi = canon
            best_macro_count = cand_macro_count

    if best_smi is not None:
        return best_smi, best_f, "OK", best_macro_count

    if reason_counts:
        worst = max(reason_counts.items(), key=lambda kv: kv[1])[0]
        return None, (1e9, 1e9), worst, 0

    return None, (1e9, 1e9), "FILTER_FAIL", 0
