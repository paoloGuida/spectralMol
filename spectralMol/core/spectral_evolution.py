from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
from typing import Callable, Sequence

import numpy as np
import selfies as sf
from rdkit import Chem

from .config import CFG
from .decoder import build_decode_accelerator, build_decode_token_state, decode_theta_best_of_k
from .embedding import build_structured_E
from .fourier_theta import build_fourier_basis
from .smiles_theta_encoder import (
    build_seed_macro_expansions,
    build_target_window_macro_expansions,
    canonicalize_smiles,
    encode_smiles_list_to_theta_pairs,
    smiles_to_selfies_tokens,
)
from .vocab import build_vocab_from_semantic_robust_alphabet, merge_macro_expansions


FREQUENCY_MODES = ("full-spectrum", "high-only", "low-only", "random-matrix")


def _static_target_smiles_from_env() -> list[str]:
    raw = os.environ.get("MOLSCORE_SPECTRAL_STATIC_TARGET_SMILES", "").strip()
    if not raw:
        return []
    values: list[str] = []
    for chunk in raw.replace("\n", ";").split(";"):
        smiles = canonicalize_smiles(chunk.strip())
        if smiles and smiles not in values:
            values.append(smiles)
    return values


@dataclass
class SpectralIndividual:
    theta: np.ndarray
    smiles: str
    score: float = float("nan")
    decode_reason: str = "OK"
    macro_count: int = 0


@dataclass(frozen=True)
class SpectralSettings:
    L: int = int(getattr(CFG, "L", 32))
    K: int = int(getattr(CFG, "K", 16))
    D: int = int(getattr(CFG, "D", 32))
    frequency_mode: str = "full-spectrum"
    decode_attempts: int = int(getattr(CFG, "DECODE_ATTEMPTS", 8))
    clip_theta_norm: float = float(getattr(CFG, "CLIP_THETA_NORM", 4.0))
    embed_seed: int = int(getattr(CFG, "EMBED_SEED", 13))
    embed_target_std: float = float(getattr(CFG, "EMBED_TARGET_STD", 1.0))
    low_cutoff_fraction: float = float(getattr(CFG, "FREQUENCY_LOW_CUTOFF_FRACTION", 0.5))
    zero_inactive_rows: bool = bool(getattr(CFG, "FREQUENCY_ZERO_INACTIVE_ROWS", True))


def _frequency_index_for_rows(K: int) -> np.ndarray:
    freqs = np.zeros((1 + 2 * K,), dtype=np.int64)
    for k in range(1, K + 1):
        freqs[k] = k
        freqs[K + k] = k
    return freqs


def active_frequency_mask(K: int, mode: str, low_cutoff_fraction: float) -> np.ndarray:
    if mode not in FREQUENCY_MODES:
        raise ValueError(f"Unknown frequency mode {mode!r}; expected one of {', '.join(FREQUENCY_MODES)}")
    mask = np.ones((1 + 2 * K,), dtype=bool)
    if mode in {"full-spectrum", "random-matrix"}:
        return mask

    freqs = _frequency_index_for_rows(K)
    cutoff = max(1, int(math.floor(K * float(low_cutoff_fraction))))
    if mode == "low-only":
        return freqs <= cutoff
    if mode == "high-only":
        return freqs > cutoff
    return mask


def canonical_smiles_or_none(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


class SpectralGenerator:
    """Fourier-genotype generator used by the local evolution benchmarks."""

    def __init__(self, settings: SpectralSettings, seed: int):
        if settings.frequency_mode not in FREQUENCY_MODES:
            raise ValueError(
                f"Unknown frequency mode {settings.frequency_mode!r}; expected one of {', '.join(FREQUENCY_MODES)}"
            )
        self.settings = settings
        self.rng = np.random.RandomState(int(seed))
        self.py_rng = random.Random(int(seed))

        self.Phi = self._build_basis()
        self.Phi_pinv = np.linalg.pinv(self.Phi).astype(np.float64)
        pad_token = str(getattr(CFG, "PAD_TOKEN", "[PAD]"))
        self.pad_token = pad_token
        allowed_elements = set(getattr(CFG, "ALLOWED_ELEMENTS", ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H")))
        self.allowed_elements = allowed_elements
        self.dynamic_macro_expansions: dict[str, str] = {}
        self.seed_macro_expansions: dict[str, str] = {}
        self.task_macro_expansions: dict[str, str] = {}
        self.task_target_smiles: list[str] = []
        self.static_target_smiles: list[str] = _static_target_smiles_from_env()
        self.task_target_theta_pairs: list[tuple[np.ndarray, str]] = []
        self.macro_expansions = merge_macro_expansions()
        self._configure_vocabulary()

        self.active_rows = active_frequency_mask(
            K=int(settings.K),
            mode=str(settings.frequency_mode),
            low_cutoff_fraction=float(settings.low_cutoff_fraction),
        )

    def _configure_vocabulary(self) -> None:
        self.vocab = build_vocab_from_semantic_robust_alphabet(
            max_ring_index=8,
            allow_elements=self.allowed_elements,
            allow_charged=bool(getattr(CFG, "ALLOW_CHARGED_TOKENS", True)),
            pad_token=self.pad_token,
            include_aromatic_atoms=True,
            include_macros=bool(getattr(CFG, "ENABLE_MACROS", True)),
            dynamic_macro_expansions=self.dynamic_macro_expansions,
        )
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2tok = {i: tok for tok, i in self.tok2id.items()}
        self.pad_id = int(self.tok2id[self.pad_token])

        self.E = build_structured_E(
            vocab=self.vocab,
            D=int(self.settings.D),
            pad_token=self.pad_token,
            seed=int(self.settings.embed_seed),
            target_std=float(self.settings.embed_target_std),
            allowed_elements=self.allowed_elements,
            macro_expansions=self.dynamic_macro_expansions,
        )
        self.macro_expansions = merge_macro_expansions(self.dynamic_macro_expansions)
        self.token_state = build_decode_token_state(self.id2tok, macro_expansions=self.dynamic_macro_expansions)
        self.accel = build_decode_accelerator(self.Phi, self.E)
        self._configure_token_neighborhoods()

    def _refresh_dynamic_macro_expansions(self) -> int:
        combined: dict[str, str] = {}
        seen_expansions = set(merge_macro_expansions().values())

        for source in (self.task_macro_expansions, self.seed_macro_expansions):
            for name, expansion in source.items():
                if not name or not expansion or expansion in seen_expansions:
                    continue
                combined[name] = expansion
                seen_expansions.add(expansion)

        self.dynamic_macro_expansions = combined
        self._configure_vocabulary()
        self._refresh_task_target_theta_cache()
        return len(combined)

    def _refresh_task_target_theta_cache(self) -> int:
        targets = list(getattr(self, "static_target_smiles", []) or [])
        for smiles in list(getattr(self, "task_target_smiles", []) or []):
            if smiles and smiles not in targets:
                targets.append(smiles)
        if not targets:
            self.task_target_theta_pairs = []
            return 0
        pairs = encode_smiles_list_to_theta_pairs(
            targets,
            Phi=self.Phi,
            E=self.E,
            tok2id=self.tok2id,
            pad_id=self.pad_id,
            clip_theta_norm=float(self.settings.clip_theta_norm),
            prefer_macros=bool(getattr(CFG, "ENABLE_MACROS", True)),
            macro_expansions=self.dynamic_macro_expansions,
        )
        self.task_target_theta_pairs = [(self.apply_frequency_mode(theta), smiles) for theta, smiles in pairs]
        return len(self.task_target_theta_pairs)

    def _token_neighborhood_key(self, tok: str) -> str | None:
        if tok == self.pad_token:
            return None
        if tok in self.token_state.macro_tokens:
            expansion = self.macro_expansions.get(tok, "")
            atom_count = self.token_state.macro_atom_counts.get(tok, 1)
            arom = any(t in expansion for t in ("[c]", "[n]", "[o]", "[s]"))
            if atom_count <= 4:
                size = "small"
            elif atom_count <= 8:
                size = "medium"
            else:
                size = "large"
            return f"macro_{'arom' if arom else 'aliph'}_{size}"
        if tok.startswith("[Ring") or tok.startswith("[=Ring") or tok.startswith("[#Ring"):
            return "ring"
        if tok.startswith("[Branch") or tok.startswith("[=Branch") or tok.startswith("[#Branch"):
            return "branch"
        if tok not in self.token_state.atom_tokens:
            return None

        core = tok[1:-1]
        bond = "single"
        if core.startswith("="):
            bond = "double"
            core = core[1:]
        elif core.startswith("#"):
            bond = "triple"
            core = core[1:]
        core = core.replace("+1", "").replace("-1", "")
        elem = core[:2] if core[:2] in {"Cl", "Br"} else core[:1]

        if elem in {"F", "Cl", "Br", "I"}:
            return "halogen"
        if elem in {"c", "n", "o", "s"}:
            return "aromatic_atom"
        if bond == "triple":
            return "triple_atom"
        if bond == "double":
            return "double_atom"
        if elem == "C":
            return "aliphatic_carbon"
        return "aliphatic_hetero"

    def _configure_token_neighborhoods(self) -> None:
        groups: dict[str, list[str]] = {}
        atom_insert: list[str] = []
        macro_insert: list[str] = []
        target_macro_insert: list[str] = []

        for tok in self.vocab:
            key = self._token_neighborhood_key(tok)
            if key is not None:
                groups.setdefault(key, []).append(tok)
            if tok in self.token_state.atom_tokens and not tok.startswith("[#"):
                atom_insert.append(tok)
            if tok in self.token_state.macro_tokens:
                tid = int(self.tok2id.get(tok, -1))
                if tid not in self.token_state.macro_blocked_ids:
                    macro_insert.append(tok)
                    if tok in self.task_macro_expansions:
                        target_macro_insert.append(tok)

        preferred_atoms = [
            tok
            for tok in ("[C]", "[N]", "[O]", "[S]", "[c]", "[n]", "[F]", "[Cl]", "[Br]")
            if tok in self.tok2id
        ]
        self._token_neighborhood_groups = {
            key: sorted(set(vals))
            for key, vals in groups.items()
            if vals
        }
        self._atom_insert_tokens = preferred_atoms or sorted(set(atom_insert))
        self._macro_insert_tokens = sorted(set(macro_insert))
        self._target_macro_insert_tokens = sorted(set(target_macro_insert))

    def _theta_nearest_tokens(self, theta: np.ndarray) -> list[str]:
        z = self.Phi @ np.asarray(theta, dtype=np.float64)
        z2 = np.sum(z * z, axis=1, keepdims=True)
        e2 = np.sum(self.E * self.E, axis=1, keepdims=False)[None, :]
        dist2 = z2 + e2 - 2.0 * (z @ self.E.T)
        ids = np.argmin(dist2, axis=1)

        toks: list[str] = []
        for tid in ids.tolist():
            tok = self.id2tok.get(int(tid), self.pad_token)
            if tok == self.pad_token:
                break
            toks.append(tok)
        return toks

    def _theta_from_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        L = int(self.Phi.shape[0])
        Z = np.zeros((L, int(self.E.shape[1])), dtype=np.float64)
        n = min(L, len(tokens))
        for i in range(n):
            tid = int(self.tok2id.get(tokens[i], self.pad_id))
            Z[i] = self.E[tid]
        if n < L:
            Z[n:, :] = self.E[int(self.pad_id)]
        return self.apply_frequency_mode(self.Phi_pinv @ Z)

    def _expand_macro_tokens_for_edit(self, tokens: Sequence[str]) -> list[str]:
        """Expand macro tokens before local token edits when the sequence still fits."""
        expanded: list[str] = []
        changed = False
        max_len = int(self.settings.L)

        for tok in tokens:
            expansion = self.macro_expansions.get(tok)
            if not expansion:
                expanded.append(tok)
                continue
            try:
                exp_tokens = [t for t in sf.split_selfies(expansion) if t]
            except Exception:
                exp_tokens = []
            if not exp_tokens or any(t not in self.tok2id for t in exp_tokens):
                expanded.append(tok)
                continue
            changed = True
            expanded.extend(exp_tokens)
            if len(expanded) > max_len:
                return list(tokens)

        if not changed or len(expanded) > max_len:
            return list(tokens)
        return expanded

    def _sample_insert_token(self, macro_insert_probability: float) -> str | None:
        macro_p = max(0.0, min(1.0, float(macro_insert_probability)))
        if self._macro_insert_tokens and self.py_rng.random() < macro_p:
            target_macro_p = max(
                0.0,
                min(1.0, float(getattr(CFG, "SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION", 0.70))),
            )
            if self._target_macro_insert_tokens and self.py_rng.random() < target_macro_p:
                return self.py_rng.choice(self._target_macro_insert_tokens)
            return self.py_rng.choice(self._macro_insert_tokens)
        if self._atom_insert_tokens:
            return self.py_rng.choice(self._atom_insert_tokens)
        if self._macro_insert_tokens:
            return self.py_rng.choice(self._macro_insert_tokens)
        return None

    def mutate_token_neighborhood(
        self,
        theta: np.ndarray,
        *,
        max_edits: int = 2,
        insert_probability: float = 0.20,
        delete_probability: float = 0.05,
        macro_insert_probability: float = 0.25,
        blend: float = 0.80,
        expand_macros_before_edit: bool = False,
        edge_position_probability: float = 0.0,
        insert_tokens: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Make a class-aware SELFIES-token edit and project it back to Theta.

        This is still a Theta mutation operator: callers receive only the edited
        Fourier genotype. No phenotype proposal is scored or stored here.
        """
        base_theta = np.asarray(theta, dtype=np.float64)
        tokens = self._theta_nearest_tokens(base_theta)
        if bool(expand_macros_before_edit):
            tokens = self._expand_macro_tokens_for_edit(tokens)
        valid_insert_tokens = [
            tok
            for tok in (insert_tokens or [])
            if tok in self.tok2id and tok != self.pad_token
        ]

        def sample_insert_token() -> str | None:
            if valid_insert_tokens:
                return self.py_rng.choice(valid_insert_tokens)
            return self._sample_insert_token(macro_insert_probability)

        target_jump_p = max(
            0.0,
            min(1.0, float(getattr(CFG, "SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION", 0.03))),
        )
        if (
            not bool(expand_macros_before_edit)
            and self._target_macro_insert_tokens
            and self.py_rng.random() < target_jump_p
        ):
            return self._theta_from_tokens([self.py_rng.choice(self._target_macro_insert_tokens)])
        if not tokens:
            tok = sample_insert_token()
            if tok is None:
                return self.apply_frequency_mode(base_theta)
            tokens = [tok]

        max_len = int(self.settings.L)
        edits = max(1, int(max_edits))
        insert_p = max(0.0, min(1.0, float(insert_probability)))
        delete_p = max(0.0, min(1.0, float(delete_probability)))
        edge_p = max(0.0, min(1.0, float(edge_position_probability)))

        def choose_position(n: int) -> int:
            if n <= 1:
                return 0
            if self.py_rng.random() >= edge_p:
                return self.py_rng.randrange(0, n)
            edge_width = max(1, min(n, int(math.ceil(0.20 * n))))
            if n <= 2 * edge_width:
                return self.py_rng.randrange(0, n)
            if self.py_rng.random() < 0.5:
                return self.py_rng.randrange(0, edge_width)
            return self.py_rng.randrange(n - edge_width, n)

        for _ in range(edits):
            if not tokens:
                op = "insert"
            else:
                r = self.py_rng.random()
                if len(tokens) < max_len and r < insert_p:
                    op = "insert"
                elif len(tokens) > 1 and r < (insert_p + delete_p):
                    op = "delete"
                else:
                    op = "replace"

            if op == "insert":
                if len(tokens) >= max_len:
                    continue
                tok = sample_insert_token()
                if tok is None:
                    continue
                pos = choose_position(len(tokens) + 1)
                tokens.insert(pos, tok)
            elif op == "delete":
                pos = choose_position(len(tokens))
                tokens.pop(pos)
            else:
                pos = choose_position(len(tokens))
                old = tokens[pos]
                key = self._token_neighborhood_key(old)
                choices = list(self._token_neighborhood_groups.get(key or "", []))
                if len(choices) > 1:
                    replacement = self.py_rng.choice(choices)
                    if replacement == old:
                        alt = [tok for tok in choices if tok != old]
                        if alt:
                            replacement = self.py_rng.choice(alt)
                    tokens[pos] = replacement
                else:
                    tok = sample_insert_token()
                    if tok is not None:
                        tokens[pos] = tok

            tokens = tokens[:max_len]

        edited_theta = self._theta_from_tokens(tokens)
        blend = max(0.0, min(1.0, float(blend)))
        if blend < 1.0:
            edited_theta = (blend * edited_theta) + ((1.0 - blend) * base_theta)
        return self.apply_frequency_mode(edited_theta)

    def mutate_token_site_scan(
        self,
        theta: np.ndarray,
        *,
        max_edits: int = 2,
        blend: float = 1.0,
        expand_macros_before_edit: bool = True,
        edge_position_probability: float = 0.75,
        motif_choice_probability: float = 0.0,
        scan_tokens: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Make motif/site substitutions in token space and project back to theta."""
        base_theta = np.asarray(theta, dtype=np.float64)
        tokens = self._theta_nearest_tokens(base_theta)
        if bool(expand_macros_before_edit):
            tokens = self._expand_macro_tokens_for_edit(tokens)
        if not tokens:
            return self.apply_frequency_mode(base_theta)

        max_len = int(self.settings.L)
        motif_tokens = [
            tok
            for tok in (
                scan_tokens
                or (
                    "[F]",
                    "[Cl]",
                    "[Br]",
                    "[C]",
                    "[=C]",
                    "[N]",
                    "[O]",
                    "[S]",
                    "[AMIDE]",
                    "[BENZAMIDE]",
                    "[PHENETHYL]",
                    "[PHENETHYL_AMIDE]",
                    "[PHENOXY_ETHYL]",
                )
            )
            if tok in self.tok2id and tok != self.pad_token
        ]
        if not motif_tokens:
            motif_tokens = [tok for tok in ("[F]", "[Cl]", "[C]", "[N]", "[O]") if tok in self.tok2id]
        if not motif_tokens:
            return self.apply_frequency_mode(base_theta)

        editable = [
            i
            for i, tok in enumerate(tokens)
            if tok != self.pad_token
            and not tok.startswith("[Ring")
            and not tok.startswith("[=Ring")
            and not tok.startswith("[#Ring")
            and not tok.startswith("[Branch")
            and not tok.startswith("[=Branch")
            and not tok.startswith("[#Branch")
        ]
        if not editable:
            return self.apply_frequency_mode(base_theta)

        edge_p = max(0.0, min(1.0, float(edge_position_probability)))
        motif_p = max(0.0, min(1.0, float(motif_choice_probability)))

        def choose_position() -> int:
            if len(editable) <= 1 or self.py_rng.random() >= edge_p:
                return self.py_rng.choice(editable)
            n = len(tokens)
            edge_width = max(1, min(n, int(math.ceil(0.25 * n))))
            edge_positions = [idx for idx in editable if idx < edge_width or idx >= n - edge_width]
            return self.py_rng.choice(edge_positions or editable)

        for _ in range(max(1, int(max_edits))):
            pos = choose_position()
            old = tokens[pos]
            op = self.py_rng.random()
            if op < 0.55:
                if motif_p > 0.0 and self.py_rng.random() < motif_p:
                    choices = motif_tokens
                else:
                    choices = list(self._token_neighborhood_groups.get(self._token_neighborhood_key(old) or "", []))
                    if not choices:
                        choices = motif_tokens
                choices = [tok for tok in choices if tok != old and tok in self.tok2id]
                if choices:
                    tokens[pos] = self.py_rng.choice(choices)
            elif op < 0.90 and len(tokens) < max_len:
                insert_pos = pos + (1 if self.py_rng.random() < 0.65 else 0)
                insert_pos = max(0, min(len(tokens), insert_pos))
                tokens.insert(insert_pos, self.py_rng.choice(motif_tokens))
                editable = [idx if idx < insert_pos else idx + 1 for idx in editable]
                editable.append(insert_pos)
                editable = sorted(set(i for i in editable if i < len(tokens)))
            elif len(tokens) > 1:
                tokens.pop(pos)
                editable = [idx if idx < pos else idx - 1 for idx in editable if idx != pos]
                editable = sorted(set(i for i in editable if 0 <= i < len(tokens)))
                if not editable:
                    break
            tokens = tokens[:max_len]

        edited_theta = self._theta_from_tokens(tokens)
        blend = max(0.0, min(1.0, float(blend)))
        if blend < 1.0:
            edited_theta = (blend * edited_theta) + ((1.0 - blend) * base_theta)
        return self.apply_frequency_mode(edited_theta)

    def adapt_vocabulary_from_seed_pool(self, seed_pool: Sequence[str]) -> int:
        if not bool(getattr(CFG, "SPECTRAL_ENABLE_SEED_MACROS", True)):
            if self.seed_macro_expansions:
                self.seed_macro_expansions = {}
                self._refresh_dynamic_macro_expansions()
            return 0
        dynamic_macros = build_seed_macro_expansions(
            seed_pool,
            max_macros=int(getattr(CFG, "SPECTRAL_SEED_MACRO_MAX", 256)),
            min_n=int(getattr(CFG, "SPECTRAL_SEED_MACRO_MIN_N", 3)),
            max_n=int(getattr(CFG, "SPECTRAL_SEED_MACRO_MAX_N", 10)),
            min_atoms=int(getattr(CFG, "SPECTRAL_SEED_MACRO_MIN_ATOMS", 3)),
            min_frequency=int(getattr(CFG, "SPECTRAL_SEED_MACRO_MIN_FREQUENCY", 1)),
            prefix=str(getattr(CFG, "SPECTRAL_SEED_MACRO_PREFIX", "SEEDM")),
        )
        self.seed_macro_expansions = dynamic_macros
        self._refresh_dynamic_macro_expansions()
        return len(dynamic_macros)

    def adapt_vocabulary_from_task_targets(self, target_smiles: Sequence[str]) -> int:
        if not bool(getattr(CFG, "SPECTRAL_ENABLE_TASK_TARGET_MACROS", True)):
            if self.task_macro_expansions:
                self.task_macro_expansions = {}
                self.task_target_smiles = []
                self.task_target_theta_pairs = []
                self._refresh_dynamic_macro_expansions()
            return 0

        targets = [canonicalize_smiles(s) for s in target_smiles]
        unique_targets = [s for i, s in enumerate(targets) if s and s not in targets[:i]]
        self.task_target_smiles = list(unique_targets)
        if not unique_targets:
            if self.task_macro_expansions:
                self.task_macro_expansions = {}
                self.task_target_theta_pairs = []
                self._refresh_dynamic_macro_expansions()
            return 0

        weight = max(1, int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_WEIGHT", 8)))
        weighted_targets: list[str] = []
        rank_bias = max(0.0, float(getattr(CFG, "SPECTRAL_TASK_TARGET_SAMPLE_BIAS", 0.0)))
        denom = max(1, len(unique_targets) - 1)
        for idx, smiles in enumerate(unique_targets):
            repeat = weight
            if rank_bias > 0.0 and len(unique_targets) > 1:
                repeat = max(1, int(round(float(weight) * math.exp(-rank_bias * (idx / denom)))))
            weighted_targets.extend([smiles] * repeat)

        full_task_macros: dict[str, str] = {}
        seen_expansions = set(merge_macro_expansions().values())
        if bool(getattr(CFG, "SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS", True)):
            full_macro_max = max(0, int(getattr(CFG, "SPECTRAL_TASK_TARGET_FULL_MACRO_MAX", 16)))
            full_prefix = str(getattr(CFG, "SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX", "TASKT"))
            for smiles in unique_targets[:full_macro_max]:
                toks = smiles_to_selfies_tokens(smiles)
                if not toks:
                    continue
                expansion = "".join(toks)
                if expansion in seen_expansions:
                    continue
                name = f"[{full_prefix}{len(full_task_macros):04d}]"
                full_task_macros[name] = expansion
                seen_expansions.add(expansion)

        window_task_macros: dict[str, str] = {}
        if bool(getattr(CFG, "SPECTRAL_ENABLE_TASK_TARGET_WINDOW_MACROS", True)):
            window_task_macros = build_target_window_macro_expansions(
                weighted_targets,
                max_macros=int(getattr(CFG, "SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX", 128)),
                min_n=int(getattr(CFG, "SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_N", 2)),
                max_n=int(getattr(CFG, "SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N", 12)),
                min_atoms=int(getattr(CFG, "SPECTRAL_TASK_TARGET_WINDOW_MACRO_MIN_ATOMS", 2)),
                prefix=str(getattr(CFG, "SPECTRAL_TASK_TARGET_WINDOW_MACRO_PREFIX", "TASKW")),
            )

        ngram_task_macros = build_seed_macro_expansions(
            weighted_targets,
            max_macros=int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_MAX", 192)),
            min_n=int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_MIN_N", 2)),
            max_n=int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_MAX_N", 16)),
            min_atoms=int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS", 2)),
            min_frequency=int(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY", 1)),
            prefix=str(getattr(CFG, "SPECTRAL_TASK_TARGET_MACRO_PREFIX", "TASKM")),
        )

        task_macros = dict(full_task_macros)
        for name, expansion in window_task_macros.items():
            if expansion in seen_expansions:
                continue
            final_name = name
            if final_name in task_macros:
                final_name = final_name.replace("]", f"_{len(task_macros)}]")
            task_macros[final_name] = expansion
            seen_expansions.add(expansion)

        for name, expansion in ngram_task_macros.items():
            if expansion in seen_expansions:
                continue
            final_name = name
            if final_name in task_macros:
                final_name = final_name.replace("]", f"_{len(task_macros)}]")
            task_macros[final_name] = expansion
            seen_expansions.add(expansion)

        self.task_macro_expansions = task_macros
        self._refresh_dynamic_macro_expansions()
        return len(task_macros)

    def sample_task_target_theta(self) -> np.ndarray | None:
        if not self.task_target_theta_pairs:
            self._refresh_task_target_theta_cache()
        if not self.task_target_theta_pairs:
            return None
        pairs = list(self.task_target_theta_pairs)
        bias = max(0.0, float(getattr(CFG, "SPECTRAL_TASK_TARGET_SAMPLE_BIAS", 0.0)))
        if bias > 0.0 and len(pairs) > 1:
            denom = max(1, len(pairs) - 1)
            weights = [math.exp(-bias * (idx / denom)) for idx in range(len(pairs))]
            theta, _smiles = self.py_rng.choices(pairs, weights=weights, k=1)[0]
        else:
            theta, _smiles = self.py_rng.choice(pairs)
        return np.asarray(theta, dtype=np.float64).copy()

    @property
    def M(self) -> int:
        return 1 + 2 * int(self.settings.K)

    def _build_basis(self) -> np.ndarray:
        if self.settings.frequency_mode != "random-matrix":
            return build_fourier_basis(int(self.settings.L), int(self.settings.K))
        phi = self.rng.normal(0.0, 1.0, size=(int(self.settings.L), self.M)).astype(np.float64)
        norms = np.linalg.norm(phi, axis=0, keepdims=True) + 1e-12
        return phi / norms

    def apply_frequency_mode(self, theta: np.ndarray) -> np.ndarray:
        out = np.asarray(theta, dtype=np.float64).copy()
        if bool(self.settings.zero_inactive_rows):
            out[~self.active_rows, :] = 0.0
        return self._clip_theta(out)

    def random_theta(self) -> np.ndarray:
        theta = self.rng.normal(
            0.0,
            float(getattr(CFG, "THETA_INIT_STD", 0.75)),
            size=(self.M, int(self.settings.D)),
        ).astype(np.float64)
        return self.apply_frequency_mode(theta)

    def encode_seed_thetas(self, seed_pool: Sequence[str], limit: int) -> list[tuple[np.ndarray, str]]:
        pairs = encode_smiles_list_to_theta_pairs(
            list(seed_pool)[: max(0, int(limit))],
            Phi=self.Phi,
            E=self.E,
            tok2id=self.tok2id,
            pad_id=self.pad_id,
            clip_theta_norm=float(self.settings.clip_theta_norm),
            prefer_macros=bool(getattr(CFG, "ENABLE_MACROS", True)),
            macro_expansions=self.dynamic_macro_expansions,
        )
        return [(self.apply_frequency_mode(theta), smiles) for theta, smiles in pairs]

    def encode_smiles_to_individual(
        self,
        smiles: str,
        *,
        decode_reason: str = "ENCODED_SMILES",
    ) -> SpectralIndividual | None:
        pairs = encode_smiles_list_to_theta_pairs(
            [smiles],
            Phi=self.Phi,
            E=self.E,
            tok2id=self.tok2id,
            pad_id=self.pad_id,
            clip_theta_norm=float(self.settings.clip_theta_norm),
            prefer_macros=bool(getattr(CFG, "ENABLE_MACROS", True)),
            macro_expansions=self.dynamic_macro_expansions,
        )
        if not pairs:
            return None
        theta, canon = pairs[0]
        return SpectralIndividual(
            theta=self.apply_frequency_mode(theta),
            smiles=canon,
            decode_reason=decode_reason,
            macro_count=0,
        )

    def _clip_theta(self, theta: np.ndarray) -> np.ndarray:
        clip = float(self.settings.clip_theta_norm)
        if clip <= 0:
            return theta
        norms = np.linalg.norm(theta, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, clip / norms)
        return theta * scale

    def mutate_theta(
        self,
        theta: np.ndarray,
        gen: int,
        generations: int,
        *,
        sigma_scale: float = 1.0,
        param_noise_scale: float = 1.0,
        row_reset_scale: float = 1.0,
    ) -> np.ndarray:
        y = np.asarray(theta, dtype=np.float64).copy()
        active = self.active_rows
        anneal = 1.0 - 0.20 * (int(gen) / max(1, int(generations) - 1))
        sigma = float(getattr(CFG, "GAUSS_STD_THETA", 0.18)) * anneal * max(0.0, float(sigma_scale))

        p_noise = float(getattr(CFG, "P_PARAM_NOISE", 0.10)) * max(0.0, float(param_noise_scale))
        p_noise = min(1.0, max(0.0, p_noise))
        noise_mask = self.rng.rand(*y.shape) < p_noise
        noise_mask &= active[:, None]
        if sigma > 0.0 and np.any(noise_mask):
            y[noise_mask] += self.rng.normal(0.0, sigma, size=int(noise_mask.sum()))

        p_row_reset = float(getattr(CFG, "P_ROW_RESET", 0.015)) * max(0.0, float(row_reset_scale))
        p_row_reset = min(1.0, max(0.0, p_row_reset))
        row_reset = self.rng.rand(self.M) < p_row_reset
        row_reset &= active
        if np.any(row_reset):
            y[row_reset, :] = self.rng.normal(
                0.0,
                float(getattr(CFG, "THETA_INIT_STD", 0.75)),
                size=(int(row_reset.sum()), int(self.settings.D)),
            )
        return self.apply_frequency_mode(y)

    def mutate_repeated(
        self,
        theta: np.ndarray,
        gen: int,
        generations: int,
        depth: int,
        *,
        sigma_scale: float = 1.0,
        param_noise_scale: float = 1.0,
        row_reset_scale: float = 1.0,
    ) -> np.ndarray:
        out = np.asarray(theta, dtype=np.float64)
        for _ in range(max(1, int(depth))):
            out = self.mutate_theta(
                out,
                gen=gen,
                generations=generations,
                sigma_scale=sigma_scale,
                param_noise_scale=param_noise_scale,
                row_reset_scale=row_reset_scale,
            )
        return out

    def crossover_theta(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        mask = self.rng.rand(self.M) < 0.5
        child = np.asarray(left, dtype=np.float64).copy()
        child[mask, :] = np.asarray(right, dtype=np.float64)[mask, :]
        return self.apply_frequency_mode(child)

    def blend_crossover_theta(self, left: np.ndarray, right: np.ndarray, *, alpha: float | None = None) -> np.ndarray:
        """Convex arithmetic crossover in Fourier-genotype space."""
        a = self.py_rng.random() if alpha is None else float(alpha)
        a = max(0.0, min(1.0, a))
        child = (a * np.asarray(left, dtype=np.float64)) + ((1.0 - a) * np.asarray(right, dtype=np.float64))
        return self.apply_frequency_mode(child)

    def differential_theta(
        self,
        base: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        *,
        scale: float = 0.45,
    ) -> np.ndarray:
        """Differential-evolution style proposal in Fourier-genotype space."""
        f = max(0.0, float(scale))
        child = np.asarray(base, dtype=np.float64) + f * (
            np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
        )
        return self.apply_frequency_mode(child)

    def selfies_to_smiles(self, selfies: str) -> tuple[str | None, str]:
        try:
            smiles = sf.decoder(selfies)
        except Exception:
            return None, "SELFIES2SMILES_FAIL"
        canon = canonical_smiles_or_none(smiles)
        if canon is None:
            return None, "RDKIT_SANITIZE_FAIL"
        return canon, "OK"

    def dummy_objectives(self, smiles: str, gen: int) -> tuple[float, float, str]:
        _ = smiles, gen
        return 0.0, 0.0, "OK"

    def decode_theta(
        self,
        theta: np.ndarray,
        gen: int,
        *,
        seen_smiles: set[str] | None = None,
        novelty_mode: str = "smiles",
        objectives_from_smiles_fn: Callable[[str, int], tuple[float, float, str]] | None = None,
    ) -> tuple[str | None, str, int]:
        objectives_fn = objectives_from_smiles_fn or self.dummy_objectives
        smiles, _fitness, reason, macro_count = decode_theta_best_of_k(
            theta,
            self.Phi,
            self.E,
            self.id2tok,
            self.pad_id,
            k=int(self.settings.decode_attempts),
            gen=int(gen),
            selfies_to_smiles_fn=self.selfies_to_smiles,
            objectives_from_smiles_fn=objectives_fn,
            seen_smiles=seen_smiles,
            seen_scaffolds=None,
            novelty_mode=novelty_mode,
            accel=self.accel,
            token_state=self.token_state,
        )
        return smiles, reason, int(macro_count)

    def build_initial_population(self, seed_pool: Sequence[str], pop_size: int) -> list[SpectralIndividual]:
        individuals: list[SpectralIndividual] = []
        seen: set[str] = set()
        encoded = self.encode_seed_thetas(seed_pool, limit=max(len(seed_pool), pop_size))
        self.py_rng.shuffle(encoded)

        for theta, seed_smiles in encoded:
            if len(individuals) >= int(pop_size):
                break
            smiles = canonicalize_smiles(seed_smiles)
            if smiles is None or smiles in seen:
                continue
            seen.add(smiles)
            individuals.append(SpectralIndividual(theta=theta, smiles=smiles, decode_reason="SEED_ENCODED", macro_count=0))

        attempts = 0
        max_attempts = max(200, int(pop_size) * 80)
        while len(individuals) < int(pop_size) and attempts < max_attempts:
            attempts += 1
            theta = self.random_theta()
            smiles, reason, macro_count = self.decode_theta(theta, gen=0, seen_smiles=seen, novelty_mode="smiles")
            if smiles is None or smiles in seen:
                continue
            seen.add(smiles)
            individuals.append(SpectralIndividual(theta=theta, smiles=smiles, decode_reason=reason, macro_count=macro_count))

        if len(individuals) < int(pop_size):
            for raw in seed_pool:
                smiles = canonicalize_smiles(raw)
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    ind = self.encode_smiles_to_individual(smiles, decode_reason="SEED_FILL_ENCODED")
                    if ind is None:
                        ind = SpectralIndividual(theta=self.random_theta(), smiles=smiles, decode_reason="SEED_FILL")
                    individuals.append(ind)
                if len(individuals) >= int(pop_size):
                    break

        return individuals[: int(pop_size)]

    def propose_child(
        self,
        parents: Sequence[SpectralIndividual],
        gen: int,
        generations: int,
        *,
        crossover_probability: float = 0.35,
        mutation_depth: int = 1,
        sigma_scale: float = 1.0,
        param_noise_scale: float = 1.0,
        row_reset_scale: float = 1.0,
    ) -> np.ndarray:
        if not parents:
            return self.mutate_repeated(
                self.random_theta(),
                gen=gen,
                generations=generations,
                depth=mutation_depth,
                sigma_scale=sigma_scale,
                param_noise_scale=param_noise_scale,
                row_reset_scale=row_reset_scale,
            )
        left = self.py_rng.choice(list(parents))
        theta = left.theta
        if len(parents) > 1 and self.py_rng.random() < float(crossover_probability):
            right = self.py_rng.choice(list(parents))
            theta = self.crossover_theta(left.theta, right.theta)
        return self.mutate_repeated(
            theta,
            gen=gen,
            generations=generations,
            depth=mutation_depth,
            sigma_scale=sigma_scale,
            param_noise_scale=param_noise_scale,
            row_reset_scale=row_reset_scale,
        )
