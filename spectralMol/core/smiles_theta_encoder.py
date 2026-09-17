from __future__ import annotations

import csv
import heapq
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import selfies as sf
from rdkit import Chem

from .vocab import MACRO_EXPANSIONS, merge_macro_expansions


def _build_macro_token_seqs(macro_expansions: Optional[Dict[str, str]] = None) -> List[Tuple[str, Tuple[str, ...]]]:
    macro_token_seqs: List[Tuple[str, Tuple[str, ...]]] = []
    for _name, _exp in merge_macro_expansions(macro_expansions).items():
        try:
            _seq = tuple(sf.split_selfies(_exp))
        except Exception:
            _seq = ()
        if _seq:
            macro_token_seqs.append((_name, _seq))
    macro_token_seqs.sort(key=lambda x: len(x[1]), reverse=True)
    return macro_token_seqs


_MACRO_TOKEN_SEQS: List[Tuple[str, Tuple[str, ...]]] = []
for _name, _exp in MACRO_EXPANSIONS.items():
    try:
        _seq = tuple(sf.split_selfies(_exp))
    except Exception:
        _seq = ()
    if _seq:
        _MACRO_TOKEN_SEQS.append((_name, _seq))
_MACRO_TOKEN_SEQS.sort(key=lambda x: len(x[1]), reverse=True)


def _iter_sanitized_csv_lines(path: Path):
    """Yield CSV lines while stripping NUL bytes and UTF-8 BOM if present."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        first = True
        for line in f:
            if "\x00" in line:
                line = line.replace("\x00", "")
            if first:
                first = False
                if line.startswith("\ufeff"):
                    line = line.lstrip("\ufeff")
            yield line


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None

    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    except Exception:
        pass

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def smiles_to_selfies_tokens(smiles: str) -> Optional[List[str]]:
    canon = canonicalize_smiles(smiles)
    if not canon:
        return None

    try:
        selfies = sf.encoder(canon)
        toks = list(sf.split_selfies(selfies))
    except Exception:
        return None

    return toks if toks else None


_ATOM_TOKEN_RE = re.compile(r"^\[(?:=|#)?[A-Za-z][a-z]?(?:H)?(?:[+\-]\d+)?\]$")
_CONTROL_TOKEN_RE = re.compile(r"^\[(?:=|#)?(?:Branch|Ring)\d+\]$")


def macroize_selfies_tokens(
    tokens: Sequence[str],
    macro_expansions: Optional[Dict[str, str]] = None,
) -> List[str]:
    macro_token_seqs = _MACRO_TOKEN_SEQS if macro_expansions is None else _build_macro_token_seqs(macro_expansions)
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        for name, seq in macro_token_seqs:
            m = len(seq)
            if m == 0 or i + m > n:
                continue
            if tuple(tokens[i : i + m]) == seq:
                out.append(name)
                i += m
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return out


def _is_atom_like_token(token: str) -> bool:
    return bool(_ATOM_TOKEN_RE.match(token))


def _seed_macro_candidate_ok(tokens: Tuple[str, ...], min_atoms: int) -> bool:
    if not tokens:
        return False
    if not _is_atom_like_token(tokens[0]):
        return False
    if tokens[-1].startswith("[Branch") or tokens[-1].startswith("[=Branch") or tokens[-1].startswith("[#Branch"):
        return False
    atom_count = sum(1 for tok in tokens if _is_atom_like_token(tok))
    if atom_count < int(min_atoms):
        return False
    control_count = sum(1 for tok in tokens if _CONTROL_TOKEN_RE.match(tok))
    if control_count >= len(tokens):
        return False
    return True


def build_seed_macro_expansions(
    smiles_list: Sequence[str],
    *,
    max_macros: int = 256,
    min_n: int = 3,
    max_n: int = 10,
    min_atoms: int = 3,
    min_frequency: int = 1,
    prefix: str = "SEEDM",
) -> Dict[str, str]:
    """Build deterministic SELFIES n-gram macros from the seed population."""
    max_macros = max(0, int(max_macros))
    if max_macros <= 0:
        return {}

    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))
    min_atoms = max(1, int(min_atoms))
    min_frequency = max(1, int(min_frequency))

    counts: Dict[Tuple[str, ...], int] = {}
    molecule_counts: Dict[Tuple[str, ...], int] = {}

    for smiles in smiles_list:
        toks = smiles_to_selfies_tokens(smiles)
        if not toks:
            continue
        seen_here: Set[Tuple[str, ...]] = set()
        n_tokens = len(toks)
        for width in range(min_n, min(max_n, n_tokens) + 1):
            for i in range(0, n_tokens - width + 1):
                seq = tuple(toks[i : i + width])
                if not _seed_macro_candidate_ok(seq, min_atoms=min_atoms):
                    continue
                counts[seq] = counts.get(seq, 0) + 1
                seen_here.add(seq)
        for seq in seen_here:
            molecule_counts[seq] = molecule_counts.get(seq, 0) + 1

    if not counts:
        return {}

    def score_item(item: Tuple[Tuple[str, ...], int]) -> Tuple[float, int, int, Tuple[str, ...]]:
        seq, count = item
        mol_count = molecule_counts.get(seq, 0)
        atom_count = sum(1 for tok in seq if _is_atom_like_token(tok))
        arom_bonus = 1 if any(tok.startswith("[c") or tok.startswith("[n") or tok in {"[o]", "[s]"} for tok in seq) else 0
        hetero_bonus = 1 if any(tok not in {"[C]", "[=C]", "[#C]", "[c]"} and _is_atom_like_token(tok) for tok in seq) else 0
        control_bonus = 1 if any(_CONTROL_TOKEN_RE.match(tok) for tok in seq) else 0
        score = (
            6.0 * mol_count
            + 1.5 * count
            + 0.35 * atom_count
            + 1.25 * arom_bonus
            + 0.75 * hetero_bonus
            + 0.50 * control_bonus
        )
        return (score, atom_count, len(seq), seq)

    ranked = [
        (seq, count)
        for seq, count in counts.items()
        if count >= min_frequency
    ]
    ranked.sort(key=score_item, reverse=True)

    out: Dict[str, str] = {}
    seen_expansions: Set[str] = set(MACRO_EXPANSIONS.values())
    for seq, _count in ranked:
        expansion = "".join(seq)
        if expansion in seen_expansions:
            continue
        name = f"[{prefix}{len(out):04d}]"
        out[name] = expansion
        seen_expansions.add(expansion)
        if len(out) >= max_macros:
            break

    return out


def build_target_window_macro_expansions(
    smiles_list: Sequence[str],
    *,
    max_macros: int = 128,
    min_n: int = 2,
    max_n: int = 12,
    min_atoms: int = 2,
    prefix: str = "TASKW",
) -> Dict[str, str]:
    """Build deterministic target-window macros for task-conditioned search.

    Seed macros are frequency ranked. Target molecules are usually one-off
    references, so this helper preserves diverse local windows while still
    letting repeated/rank-weighted target lists emphasize the best anchors.
    """
    max_macros = max(0, int(max_macros))
    if max_macros <= 0:
        return {}

    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))
    min_atoms = max(1, int(min_atoms))

    candidates: Dict[Tuple[str, ...], Tuple[float, int, int, int, int, Tuple[str, ...]]] = {}

    for target_rank, smiles in enumerate(smiles_list):
        toks = smiles_to_selfies_tokens(smiles)
        if not toks:
            continue
        n_tokens = len(toks)
        rank_weight = 1.0 / (1.0 + 0.02 * float(target_rank))
        for width in range(min_n, min(max_n, n_tokens) + 1):
            stride = 1 if width <= 6 else 2
            for start in range(0, n_tokens - width + 1, stride):
                seq = tuple(toks[start : start + width])
                if not _seed_macro_candidate_ok(seq, min_atoms=min_atoms):
                    continue
                atom_count = sum(1 for tok in seq if _is_atom_like_token(tok))
                hetero_count = sum(
                    1
                    for tok in seq
                    if _is_atom_like_token(tok)
                    and not tok.lstrip("[=#").startswith(("C", "c", "H"))
                )
                branch_count = sum(1 for tok in seq if "Branch" in tok)
                ring_count = sum(1 for tok in seq if "Ring" in tok)
                terminal_bonus = 0.25 if start == 0 or start + width == n_tokens else 0.0
                hetero_bonus = 0.20 * min(4, hetero_count)
                control_bonus = 0.08 * min(3, branch_count + ring_count)
                size_score = 1.0 / (1.0 + abs(atom_count - 5.0))
                score = rank_weight * (size_score + hetero_bonus + control_bonus + terminal_bonus)
                item = (score, 1, atom_count, -target_rank, -start, seq)
                prev = candidates.get(seq)
                if prev is None:
                    candidates[seq] = item
                else:
                    candidates[seq] = (
                        prev[0] + score,
                        prev[1] + 1,
                        max(prev[2], atom_count),
                        max(prev[3], -target_rank),
                        max(prev[4], -start),
                        seq,
                    )

    if not candidates:
        return {}

    ranked = heapq.nlargest(max_macros, candidates.values(), key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
    ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3], -x[4], x[5]))

    out: Dict[str, str] = {}
    seen_expansions: Set[str] = set(MACRO_EXPANSIONS.values())
    for _score, _support, _atom_count, _neg_rank, _neg_start, seq in ranked:
        expansion = "".join(seq)
        if expansion in seen_expansions:
            continue
        name = f"[{prefix}{len(out):04d}]"
        out[name] = expansion
        seen_expansions.add(expansion)
        if len(out) >= max_macros:
            break
    return out


def collect_tokens_from_smiles_list(
    smiles_list: Sequence[str],
    *,
    prefer_macros: bool = True,
    macro_expansions: Optional[Dict[str, str]] = None,
) -> Set[str]:
    out: Set[str] = set()
    for smiles in smiles_list:
        raw = smiles_to_selfies_tokens(smiles)
        if not raw:
            continue
        seq = macroize_selfies_tokens(raw, macro_expansions=macro_expansions) if prefer_macros else raw
        for tok in seq:
            if tok:
                out.add(tok)
    return out


def _resolve_csv_columns(fieldnames: Sequence[str], benchmark_name: str) -> Tuple[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header.")

    smiles_col = None
    score_col = None

    lower_to_original: Dict[str, str] = {f.strip().lower(): f for f in fieldnames}

    for candidate in ("smiles", "canonical_smiles"):
        if candidate in lower_to_original:
            smiles_col = lower_to_original[candidate]
            break
    if smiles_col is None:
        smiles_col = fieldnames[0]

    target = benchmark_name.strip().lower()
    if target in lower_to_original:
        score_col = lower_to_original[target]
    else:
        # Secondary match: normalize repeated spaces, punctuation, case.
        def norm(s: str) -> str:
            return " ".join(s.strip().lower().split())

        target_n = norm(benchmark_name)
        for f in fieldnames:
            if norm(f) == target_n:
                score_col = f
                break

    if score_col is None:
        raise KeyError(
            f"Benchmark column '{benchmark_name}' was not found in CSV. "
            f"Columns: {list(fieldnames)}"
        )

    return smiles_col, score_col


def select_top_smiles_from_score_csv(
    csv_path: str | Path,
    benchmark_name: str,
    n_samples: int,
    *,
    rng_seed: int = 7,
    top_pool_multiplier: int = 8,
    min_score: float = 0.0,
) -> List[str]:
    pairs = select_top_scored_smiles_from_score_csv(
        csv_path=csv_path,
        benchmark_name=benchmark_name,
        n_samples=n_samples,
        rng_seed=rng_seed,
        top_pool_multiplier=top_pool_multiplier,
        min_score=min_score,
    )
    return [smi for smi, _score in pairs]


def select_top_scored_smiles_from_score_csv(
    csv_path: str | Path,
    benchmark_name: str,
    n_samples: int,
    *,
    rng_seed: int = 7,
    top_pool_multiplier: int = 8,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """Select task-specific high-scoring SMILES from a score CSV.

    Strategy:
      1) keep a top-score pool with a bounded min-heap,
      2) canonicalize + deduplicate,
      3) return the top `n_samples` molecules by score.
    """
    _ = rng_seed  # kept for backward-compatible call sites
    if n_samples <= 0:
        return []

    p = Path(csv_path)
    if not p.exists():
        return []

    pool_size = max(int(n_samples), int(n_samples) * max(1, int(top_pool_multiplier)))
    heap: List[Tuple[float, int, str]] = []

    reader = csv.DictReader(_iter_sanitized_csv_lines(p))
    try:
        if reader.fieldnames is None:
            return []

        smiles_col, score_col = _resolve_csv_columns(reader.fieldnames, benchmark_name)

        for idx, row in enumerate(reader):
            smiles = str(row.get(smiles_col, "") or "").strip()
            if not smiles:
                continue

            try:
                score = float(row.get(score_col, "0") or 0.0)
            except Exception:
                continue

            if score < float(min_score):
                continue

            item = (float(score), idx, smiles)
            if len(heap) < pool_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    except csv.Error as e:
        raise ValueError(f"Failed to parse score CSV '{p}': {e}") from e

    if not heap:
        return []

    ranked = sorted(heap, key=lambda x: (x[0], -x[1]), reverse=True)

    unique_scored: List[Tuple[str, float]] = []
    seen: Set[str] = set()
    for score, _idx, smiles in ranked:
        canon = canonicalize_smiles(smiles)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        unique_scored.append((canon, float(score)))

    if not unique_scored:
        return []

    return unique_scored[:n_samples]


def theta_from_token_sequence(
    tokens: Sequence[str],
    *,
    Phi: np.ndarray,
    E: np.ndarray,
    tok2id: Dict[str, int],
    pad_id: int,
    clip_theta_norm: float,
) -> np.ndarray:
    L = int(Phi.shape[0])
    Z = np.zeros((L, int(E.shape[1])), dtype=np.float64)

    for i in range(min(L, len(tokens))):
        tid = int(tok2id.get(tokens[i], pad_id))
        Z[i] = E[tid]

    for i in range(min(L, len(tokens)), L):
        Z[i] = E[int(pad_id)]

    theta = (np.linalg.pinv(Phi) @ Z).astype(np.float64)
    nrm = float(np.linalg.norm(theta))
    clip = float(clip_theta_norm)
    if clip > 0.0 and nrm > clip:
        theta *= clip / max(1e-12, nrm)
    return theta


def _pick_vocab_covered_sequence(
    raw_tokens: Sequence[str],
    macro_tokens: Sequence[str],
    tok2id: Dict[str, int],
    *,
    prefer_macros: bool,
) -> List[str]:
    if prefer_macros:
        first = list(macro_tokens)
        second = list(raw_tokens)
    else:
        first = list(raw_tokens)
        second = list(macro_tokens)

    cov_first = sum(1 for t in first if t in tok2id)
    cov_second = sum(1 for t in second if t in tok2id)

    if cov_first > cov_second:
        return first
    if cov_second > cov_first:
        return second
    if cov_first > 0:
        return first
    return second


def encode_smiles_list_to_thetas(
    smiles_list: Sequence[str],
    *,
    Phi: np.ndarray,
    E: np.ndarray,
    tok2id: Dict[str, int],
    pad_id: int,
    clip_theta_norm: float,
    prefer_macros: bool = True,
    macro_expansions: Optional[Dict[str, str]] = None,
) -> List[np.ndarray]:
    """Encode a list of SMILES into Theta tensors for GA initialization."""
    pairs = encode_smiles_list_to_theta_pairs(
        smiles_list,
        Phi=Phi,
        E=E,
        tok2id=tok2id,
        pad_id=pad_id,
        clip_theta_norm=clip_theta_norm,
        prefer_macros=prefer_macros,
        macro_expansions=macro_expansions,
    )
    return [theta for theta, _smiles in pairs]


def encode_smiles_list_to_theta_pairs(
    smiles_list: Sequence[str],
    *,
    Phi: np.ndarray,
    E: np.ndarray,
    tok2id: Dict[str, int],
    pad_id: int,
    clip_theta_norm: float,
    prefer_macros: bool = True,
    macro_expansions: Optional[Dict[str, str]] = None,
) -> List[Tuple[np.ndarray, str]]:
    """Encode SMILES into Theta tensors, preserving canonical SMILES mapping."""
    out: List[np.ndarray] = []
    out_smiles: List[str] = []
    seen: Set[str] = set()

    for smiles in smiles_list:
        canon = canonicalize_smiles(smiles)
        if not canon or canon in seen:
            continue
        seen.add(canon)

        raw = smiles_to_selfies_tokens(canon)
        if not raw:
            continue

        macro = macroize_selfies_tokens(raw, macro_expansions=macro_expansions)
        seq = _pick_vocab_covered_sequence(raw, macro, tok2id, prefer_macros=prefer_macros)
        if not seq or not any(t in tok2id for t in seq):
            continue

        try:
            theta = theta_from_token_sequence(
                seq,
                Phi=Phi,
                E=E,
                tok2id=tok2id,
                pad_id=pad_id,
                clip_theta_norm=clip_theta_norm,
            )
        except Exception:
            continue
        out.append(theta)
        out_smiles.append(canon)

    return list(zip(out, out_smiles))
