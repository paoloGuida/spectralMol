from __future__ import annotations

import re
from typing import Dict, Set, List, Optional, Iterable

import selfies as sf

# =============================================================================
# Comprehensive motif macros for GuacaMol v2 goal-directed suite
# =============================================================================
#
# These macros do NOT change the GuacaMol scoring function.
# They reshape the search space so evolution can reach high-scoring chemotypes
# more reliably (especially important for rediscovery and scaffold/decoration hop).
#
# Safety mechanism:
# - Every macro is filtered at runtime against the installed SELFIES semantic-robust
#   alphabet via sf.split_selfies(). Incompatible macros are silently excluded.
# =============================================================================

MACRO_EXPANSIONS: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Aromatic 6-member cores
    # -------------------------------------------------------------------------
    "[PHENYL]":   "[c][c][c][c][c][c][Ring5]",
    "[BENZ]":     "[c][c][c][c][c][c][Ring5]",

    "[PYRID]":    "[c][n][c][c][c][c][Ring5]",
    "[PYRIM]":    "[n][c][n][c][c][c][Ring5]",
    "[PYRAZ6]":   "[n][n][c][c][c][c][Ring5]",
    "[TRIAZ6]":   "[n][n][n][c][c][c][Ring5]",

    # -------------------------------------------------------------------------
    # Aromatic 5-member heteroaromatics
    # -------------------------------------------------------------------------
    "[FUR]":      "[o][c][c][c][c][Ring4]",
    "[THIO]":     "[s][c][c][c][c][Ring4]",
    "[IMID]":     "[n][c][n][c][c][Ring4]",
    "[OXAZ]":     "[o][c][n][c][c][Ring4]",
    "[THIAZ]":    "[s][c][n][c][c][Ring4]",
    "[TRIAZ5]":   "[n][n][n][c][c][Ring4]",
    "[PYRAZ5]":   "[n][n][c][c][c][Ring4]",

    # -------------------------------------------------------------------------
    # Saturated carbocycles
    # -------------------------------------------------------------------------
    "[CYC3]":     "[C][C][C][Ring2]",
    "[CYC4]":     "[C][C][C][C][Ring3]",
    "[CYC5]":     "[C][C][C][C][C][Ring4]",
    "[CYC6]":     "[C][C][C][C][C][C][Ring5]",
    "[CYC7]":     "[C][C][C][C][C][C][C][Ring6]",

    # -------------------------------------------------------------------------
    # Saturated heterocycles (medchem staples)
    # -------------------------------------------------------------------------
    "[PIPER]":    "[N][C][C][C][C][C][Ring5]",          # piperidine
    "[PIPAZ]":    "[N][C][C][N][C][C][Ring5]",          # piperazine
    "[MORPH]":    "[O][C][C][N][C][C][Ring5]",          # morpholine
    "[DIOX6]":    "[O][C][C][O][C][C][Ring5]",          # 1,4-dioxane
    "[THF]":      "[O][C][C][C][C][Ring4]",

    # -------------------------------------------------------------------------
    # Small substituents & halogens (useful for similarity/MPO tuning)
    # -------------------------------------------------------------------------
    "[Me]":       "[C]",
    "[Et]":       "[C][C]",
    "[iPr]":      "[C][Branch1][C][C]",

    "[F]":        "[F]",
    "[Cl]":       "[Cl]",
    "[Br]":       "[Br]",
    "[I]":        "[I]",

    # Fluoroalkyl motifs
    "[CF3]":      "[C][Branch1][F][Branch1][F][F]",
    "[CF2]":      "[C][Branch1][F][F]",

    # -------------------------------------------------------------------------
    # Oxygen functional groups
    # -------------------------------------------------------------------------
    "[OH]":       "[O]",
    "[OMe]":      "[O][C]",
    "[OEt]":      "[O][C][C]",

    "[C=O]":      "[C][Branch1][=O]",                    # carbonyl handle
    "[COOH]":     "[C][Branch1][=O][O]",                 # acid handle
    "[COO]":      "[C][Branch1][=O][O]",                 # ester/carboxylate handle
    "[COOMe]":    "[C][Branch1][=O][O][C]",
    "[COOEt]":    "[C][Branch1][=O][O][C][C]",

    "[CARBAM]":   "[O][C][Branch1][=O][N]",              # carbamate handle
    "[CARBON]":   "[O][C][Branch1][=O][O]",              # carbonate handle

    # -------------------------------------------------------------------------
    # Nitrogen functional groups
    # -------------------------------------------------------------------------
    "[NH2]":      "[N]",
    "[NHMe]":     "[N][C]",
    "[NMe2]":     "[N][Branch1][C][C]",                  # approximation

    "[AMIDE]":    "[C][Branch1][=O][N]",
    "[UREA]":     "[N][C][Branch1][=O][N]",

    "[IMINE]":    "[C][Branch1][=N]",                    # imine handle (if supported)

    "[CN]":       "[C][#N]",
    "[NO2]":      "[N][Branch1][=O][=O]",

    # -------------------------------------------------------------------------
    # Sulfur functional groups
    # -------------------------------------------------------------------------
    "[SMe]":      "[S][C]",
    "[SO]":       "[S][Branch1][=O]",                    # sulfoxide handle
    "[SO2]":      "[S][Branch1][=O][=O]",                # sulfone handle
    "[SO2N]":     "[S][Branch1][=O][=O][N]",             # sulfonamide handle

    # -------------------------------------------------------------------------
    # Phosphorus (for formula task if supported)
    # -------------------------------------------------------------------------
    "[PO]":       "[P][Branch1][=O]",

    # -------------------------------------------------------------------------
    # Aggressive scaffold jumps (filtered if incompatible)
    # -------------------------------------------------------------------------
    "[PHENYL_CF3]":  "[c][c][c][c][c][c][Ring5][Branch1][C][Branch1][F][Branch1][F][F]",
    "[PHENYL_SO2N]": "[c][c][c][c][c][c][Ring5][S][Branch1][=O][=O][N]",
    "[PYRAZ5_PHENYL]": "[n][n][c][c][c][Ring4][Branch1][c][c][c][c][c][c][Ring5]",

    # -------------------------------------------------------------------------
    # Drug-like scaffold & motif macros (medchem-biased search space)
    #
    # These are intentionally *motifs* rather than exact fused polycycles.
    # In SELFIES, fused polycycles require careful multi-ring closures; to keep
    # semantic-robustness and avoid brittle encodings, we represent many
    # polycyclic "drug scaffolds" as biaryl / linked-ring motifs plus common
    # drug-like linkers.
    # -------------------------------------------------------------------------

    # --- Additional heteroaryl rings (single-ring, very common in drugs) ---
    "[PZIN]":     "[n][c][c][n][c][c][Ring5]",          # pyrazine (1,4-diazine)
    "[TRIAZ3]":   "[n][c][n][n][c][c][Ring5]",          # 1,2,4-triazine-like (approx)
    "[TRIAZ2]":   "[n][n][c][n][c][c][Ring5]",          # triazine-like (approx)
    "[IMID2]":    "[n][c][n][c][n][Ring4]",             # more N-rich 5-ring (approx)
    "[OXAD5]":    "[o][n][c][n][c][Ring4]",             # oxadiazole-like (approx)
    "[THIAD5]":   "[s][n][c][n][c][Ring4]",             # thiadiazole-like (approx)

    # --- Biaryl / heteroaryl motifs (scaffold-hopping without fused rings) ---
    "[BIPHENYL]":        "[c][c][c][c][c][c][Ring5][Branch1][c][c][c][c][c][c][Ring5]",
    "[PHENYL_PYRID]":    "[c][c][c][c][c][c][Ring5][Branch1][c][n][c][c][c][c][Ring5]",
    "[PHENYL_PYRIM]":    "[c][c][c][c][c][c][Ring5][Branch1][n][c][n][c][c][c][Ring5]",
    "[PHENYL_IMID]":     "[c][c][c][c][c][c][Ring5][Branch1][n][c][n][c][c][Ring4]",
    "[PHENYL_THIAZ]":    "[c][c][c][c][c][c][Ring5][Branch1][s][c][n][c][c][Ring4]",

    # --- Real-drug linkers & handles ---
    "[ANILIDE]":         "[c][c][c][c][c][c][Ring5][C][Branch1][=O][N]",               # phenyl-amide
    "[BENZYL]":          "[c][c][c][c][c][c][Ring5][C]",                              # phenyl-CH3 handle
    "[BENZYLO]":         "[c][c][c][c][c][c][Ring5][O]",                              # phenoxy handle
    "[ARYL_ETHER]":      "[c][c][c][c][c][c][Ring5][O][c][c][c][c][c][c][Ring5]",      # diaryl ether motif
    "[ARYL_SULFON]":     "[c][c][c][c][c][c][Ring5][S][Branch1][=O][=O]",              # aryl-sulfone handle
    "[ARYL_SULFONAM]":   "[c][c][c][c][c][c][Ring5][S][Branch1][=O][=O][N]",           # aryl-sulfonamide
    "[ARYL_UREA]":       "[c][c][c][c][c][c][Ring5][N][C][Branch1][=O][N]",            # aryl-urea
    "[ARYL_CARBAM]":     "[c][c][c][c][c][c][Ring5][O][C][Branch1][=O][N]",            # aryl-carbamate

    # --- Common solubilizing tails (seen in many marketed drugs) ---
    "[PHENYL_MORPH]":    "[c][c][c][c][c][c][Ring5][Branch1][O][C][C][N][C][C][Ring5]",
    "[PHENYL_PIPER]":    "[c][c][c][c][c][c][Ring5][Branch1][N][C][C][C][C][C][Ring5]",
    "[PHENYL_PIPAZ]":    "[c][c][c][c][c][c][Ring5][Branch1][N][C][C][N][C][C][Ring5]",

    # --- Additional ring systems and heterocycle-rich motifs ---
    "[CYC8]":            "[C][C][C][C][C][C][C][C][Ring7]",
    "[CYC9]":            "[C][C][C][C][C][C][C][C][C][Ring8]",
    "[AZEP]":            "[N][C][C][C][C][C][C][Ring6]",          # azepane
    "[OXAZEP]":          "[O][C][C][N][C][C][C][Ring6]",          # oxazepane-like

    "[PYRROL]":          "[n][c][c][c][c][Ring4]",
    "[TRIAZOL]":         "[n][n][c][n][c][Ring4]",
    "[TETRAZ]":          "[n][n][n][n][c][Ring4]",
    "[ISOX]":            "[o][n][c][c][c][Ring4]",
    "[ISOTHZ]":          "[s][n][c][c][c][Ring4]",

    "[QUINOLINE_MOTIF]": "[c][c][c][c][c][c][Ring5][Branch1][n][c][c][c][c][c][Ring5]",
    "[ISOQUIN_MOTIF]":   "[c][c][c][c][c][c][Ring5][Branch1][c][c][c][n][c][c][Ring5]",
    "[INDOLE_MOTIF]":    "[c][c][c][c][c][c][Ring5][Branch1][n][c][c][c][c][Ring4]",
    "[BENZOFUR_MOTIF]":  "[c][c][c][c][c][c][Ring5][Branch1][o][c][c][c][c][Ring4]",
    "[BENZOTHI_MOTIF]":  "[c][c][c][c][c][c][Ring5][Branch1][s][c][c][c][c][Ring4]",
    "[BENZIMID_MOTIF]":  "[c][c][c][c][c][c][Ring5][Branch1][n][c][n][c][c][Ring4]",
    "[BENZOXAZ_MOTIF]":  "[c][c][c][c][c][c][Ring5][Branch1][o][c][n][c][c][Ring4]",
    "[BENZOTHZ_MOTIF]":  "[c][c][c][c][c][c][Ring5][Branch1][s][c][n][c][c][Ring4]",

    # --- Linkers / side chains commonly needed for MPO and similarity tasks ---
    "[VINYL]":           "[C][=C]",
    "[ALLYL]":           "[C][C][=C]",
    "[ETHYNYL]":         "[C][#C]",
    "[PROPYNYL]":        "[C][C][#C]",
    "[ETHER_LINK]":      "[O][C][C]",
    "[AMIDE_LINK]":      "[N][C][Branch1][=O][C]",
    "[UREA_LINK]":       "[N][C][Branch1][=O][N][C]",
    "[SULFON_LINK]":     "[S][Branch1][=O][=O][C]",
    "[SULFONAMIDE]":     "[S][Branch1][=O][=O][N]",
    "[PHOSPHONATE]":     "[P][Branch1][=O][O][C]",

    # --- Ionizable / solubilizing tails ---
    "[CH2OH]":           "[C][O]",
    "[CH2NH2]":          "[C][N]",
    "[DIMETHYLAMINO]":   "[N][Branch1][C][C]",
    "[DIETHYLAMINO]":    "[N][Branch1][C][C][C][C]",
    "[PIPERIDINYL]":     "[N][C][C][C][C][C][Ring5]",
    "[PIPERAZINYL]":     "[N][C][C][N][C][C][Ring5]",
    "[MORPHOLINYL]":     "[O][C][C][N][C][C][Ring5]",

    # --- Task-oriented scaffold hopping motifs ---
    "[TETRAZ_PHENYL]":   "[n][n][n][n][c][Ring4][Branch1][c][c][c][c][c][c][Ring5]",
    "[BIPHENYL_OH]":     "[c][c][c][c][c][c][Ring5][Branch1][c][c][c][c][c][c][Ring5][O]",
    "[BIPHENYL_COOH]":   "[c][c][c][c][c][c][Ring5][Branch1][c][c][c][c][c][c][Ring5][C][Branch1][=O][O]",
    "[ARYL_CYANO]":      "[c][c][c][c][c][c][Ring5][C][#N]",
    "[ARYL_TRIFLUOROMETHYL]": "[c][c][c][c][c][c][Ring5][C][Branch1][F][Branch1][F][F]",
}


# =============================================================================
# Macro validation + extended alphabet
# =============================================================================
# The project assumes macros are always available. We therefore validate that every
# macro expansion is parseable SELFIES, and we *extend* the base semantic-robust
# alphabet with any additional symbols required by these macros.
#
# If a macro is not compatible with the installed SELFIES version, we fail fast
# with a clear error rather than silently filtering it out.
# =============================================================================

# Tokens required by all macro expansions (computed once at import time).
_MACRO_REQUIRED_TOKENS: Set[str] = set()
for _name, _exp in MACRO_EXPANSIONS.items():
    try:
        _toks = sf.split_selfies(_exp)
    except Exception as _e:
        raise ValueError(
            f"Invalid SELFIES macro expansion for {_name}: {_exp!r}. "
            f"sf.split_selfies failed with: {_e}"
        )
    _MACRO_REQUIRED_TOKENS.update(_toks)

# Base semantic-robust alphabet extended with all tokens used by macros.
SEMANTIC_ROBUST_ALPHABET_EXTENDED: Set[str] = set(sf.get_semantic_robust_alphabet()) | _MACRO_REQUIRED_TOKENS

# -------------------------
# Macro accounting utilities expected by the repo
# -------------------------

# Used by decoder.py to estimate size contribution of macro tokens.
_MACRO_ATOM_PAT = re.compile(r"\[([A-Za-z=#0-9+\-]+)\]")

def _macro_atom_count(expansion: str) -> int:
    """Count atom-like tokens in an expansion (excluding Branch/Ring control tokens)."""
    n = 0
    for m in _MACRO_ATOM_PAT.finditer(expansion):
        core = m.group(1)
        if core.startswith("Ring") or core.startswith("Branch"):
            continue
        n += 1
    return n

def _macro_is_aromatic(expansion: str) -> bool:
    """Heuristic: expansion contains aromatic atom tokens."""
    return any(t in expansion for t in ("[c]", "[n]", "[o]", "[s]"))

# These names are imported by other modules (embedding.py, decoder.py).
MACRO_ATOM_COUNTS: Dict[str, int] = {k: _macro_atom_count(v) for k, v in MACRO_EXPANSIONS.items()}
MACRO_IS_AROM: Dict[str, bool] = {k: _macro_is_aromatic(v) for k, v in MACRO_EXPANSIONS.items()}

# =============================================================================
# Vocabulary builder (semantic-robust alphabet)
# =============================================================================

_AROM_ATOM_RE = re.compile(r"^\[(?:=|#)?(?:c|n|o|s)(?:H)?(?:[+\-]1)?\]$")
_AROM_SPECIAL = {"[nH]", "[nH+1]", "[nH-1]"}  # kept only if present in alphabet

def _collect_aromatic_tokens(alphabet: Set[str]) -> Set[str]:
    out = {t for t in alphabet if _AROM_ATOM_RE.match(t)}
    out |= (_AROM_SPECIAL & alphabet)
    return out

def _collect_atom_tokens(
    alphabet: Set[str],
    allow_elements: Optional[Set[str]],
    allow_charged: bool,
    include_aromatic: bool,
) -> Set[str]:
    atom_any_re = re.compile(r"^\[(?:=|#)?([A-Za-z][a-z]?)(?:[+\-]1)?\]$")
    allow_set: Optional[Set[str]] = None
    if allow_elements:
        allow_set = set(allow_elements) | {e.lower() for e in allow_elements}

    atom_keep: Set[str] = set()
    for tok in alphabet:
        m = atom_any_re.match(tok)
        if not m:
            continue
        elem = m.group(1)
        if (not allow_charged) and ("+" in tok or "-" in tok):
            continue
        if allow_set is not None and elem not in allow_set:
            continue
        atom_keep.add(tok)

    if not include_aromatic:
        atom_keep -= _collect_aromatic_tokens(alphabet)

    return atom_keep

def _collect_branch_tokens(alphabet: Set[str]) -> Set[str]:
    return {
        "[Branch1]", "[Branch2]", "[Branch3]",
        "[=Branch1]", "[=Branch2]", "[=Branch3]",
        "[#Branch1]", "[#Branch2]", "[#Branch3]",
    } & alphabet

def _collect_ring_tokens(alphabet: Set[str], max_ring_index: int) -> Set[str]:
    return (
        {f"[Ring{i}]" for i in range(1, max_ring_index + 1)}
        | {f"[=Ring{i}]" for i in range(1, max_ring_index + 1)}
        | {f"[#Ring{i}]" for i in range(1, max_ring_index + 1)}
    ) & alphabet


def build_vocab_from_semantic_robust_alphabet(
    *,
    # Drug-like chemistry frequently uses medium rings/macrocycles; allowing a
    # slightly larger ring index increases expressivity without forcing it.
    max_ring_index: int = 8,
    allow_elements: Optional[Set[str]] = None,
    allow_charged: bool = True,
    pad_token: str = "[PAD]",
    include_aromatic_atoms: bool = True,
    include_macros: bool = True,
    include_full_semantic_tokens: bool = False,
    macro_allowlist: Optional[Iterable[str]] = None,
    macro_denylist: Optional[Iterable[str]] = None,
    extra_tokens: Optional[Iterable[str]] = None,
    # Backward-compatible aliases:
    try_add_aromatic: Optional[bool] = None,
    try_add_macros: Optional[bool] = None,
) -> List[str]:
    # Map legacy kwargs
    if try_add_aromatic is not None:
        include_aromatic_atoms = bool(try_add_aromatic)
    if try_add_macros is not None:
        include_macros = bool(try_add_macros)

    alphabet = set(SEMANTIC_ROBUST_ALPHABET_EXTENDED)

    atom_keep = _collect_atom_tokens(
        alphabet=alphabet,
        allow_elements=allow_elements,
        allow_charged=allow_charged,
        include_aromatic=include_aromatic_atoms,
    )
    branch_keep = _collect_branch_tokens(alphabet)
    ring_keep = _collect_ring_tokens(alphabet, max_ring_index=max_ring_index)

    kept = atom_keep | branch_keep | ring_keep
    if include_full_semantic_tokens:
        kept |= (alphabet - {pad_token})
    if extra_tokens:
        for tok in extra_tokens:
            if not tok or tok == pad_token:
                continue
            if tok in MACRO_EXPANSIONS and not include_macros:
                continue
            kept.add(tok)

    vocab: List[str] = [pad_token]
    vocab += sorted(t for t in kept if t != pad_token)

    if include_macros:
        allow = set(macro_allowlist) if macro_allowlist else None
        deny = set(macro_denylist) if macro_denylist else set()
        for name in sorted(MACRO_EXPANSIONS.keys()):
            if allow is not None and name not in allow:
                continue
            if name in deny:
                continue
            vocab.append(name)

    return vocab
