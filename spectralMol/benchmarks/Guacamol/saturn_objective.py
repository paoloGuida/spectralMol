#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors


@dataclass(frozen=True)
class SaturnObjectiveConfig:
    saturn_repo_root: str
    oracle_config_path: str
    oracle_config_key: str = "oracle"
    run_dir: str = ""
    apply_diversity_penalty: bool = False
    diversity_bucket_size: int = 10
    invalid_score: float = 0.0
    budget_override: int | None = None
    allow_oracle_repeats: bool | None = None


class _NoDiversityFilter:
    def penalize_reward(self, smiles, rewards):  # noqa: ANN001 - match Saturn's loose API.
        _ = smiles
        return np.asarray(rewards, dtype=float)

    def update(self, smiles) -> None:  # noqa: ANN001 - match Saturn's loose API.
        _ = smiles


_LOCAL_ORACLE_COMPONENTS = {
    "qed",
    "mw",
    "molecular_weight",
    "tpsa",
    "slogp",
    "heavy_atoms",
    "num_hba",
    "num_hbd",
    "num_rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "num_aliphatic_rings",
    "tanimoto_similarity",
    "matching_substructure",
    "smarts_alerts",
}


def _canonical_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _read_oracle_config(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key and isinstance(payload, dict) and key in payload:
        payload = payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"SATURN oracle config must be a JSON object: {path}")
    if "components" not in payload:
        raise ValueError(
            f"SATURN oracle config does not contain 'components'. "
            f"Use --saturn-oracle-config-key if {path} is a full SATURN config."
        )
    return dict(payload)


def _replace_placeholders(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for token, repl in mapping.items():
            out = out.replace(token, repl)
        return out
    if isinstance(value, list):
        return [_replace_placeholders(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _replace_placeholders(v, mapping) for k, v in value.items()}
    return value


def _quickvina_binary_default(saturn_root: Path) -> Path:
    return (
        saturn_root
        / "experimental_reproduction"
        / "synthesizability"
        / "QuickVina2-GPU-2.1"
        / "QuickVina2-GPU-2-1"
    )


def _prepend_env_path(name: str, value: Path) -> None:
    raw = str(value)
    current = os.environ.get(name, "")
    if not current:
        os.environ[name] = raw
        return
    parts = current.split(os.pathsep)
    if raw not in parts:
        os.environ[name] = raw + os.pathsep + current


def _latest_child_dir(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        return None
    return sorted(children, key=lambda p: p.name)[-1]


def _resolve_oracle_placeholders(oracle_cfg: dict, *, saturn_root: Path, run_dir: Path) -> dict:
    obabel_binary_raw = os.environ.get("MOLSCORE_SATURN_OBABEL_BINARY", "").strip()
    if obabel_binary_raw:
        obabel_binary = Path(obabel_binary_raw).expanduser()
        obabel_prefix = obabel_binary.parent.parent
        _prepend_env_path("PATH", obabel_prefix / "bin")
        _prepend_env_path("LD_LIBRARY_PATH", obabel_prefix / "lib")
        data_dir = _latest_child_dir(obabel_prefix / "share" / "openbabel")
        if data_dir is not None:
            os.environ.setdefault("BABEL_DATADIR", str(data_dir))
        lib_dir = _latest_child_dir(obabel_prefix / "lib" / "openbabel")
        if lib_dir is not None:
            os.environ.setdefault("BABEL_LIBDIR", str(lib_dir))

    quickvina_binary = Path(
        os.environ.get("MOLSCORE_SATURN_QUICKVINA_BINARY", str(_quickvina_binary_default(saturn_root)))
    ).expanduser()
    receptor_file = Path(
        os.environ.get(
            "MOLSCORE_SATURN_RECEPTOR_FILE",
            str(saturn_root / "experimental_reproduction" / "synthesizability" / "7uvu-2-monomers-pdbfixer.pdbqt"),
        )
    ).expanduser()
    reference_ligand = Path(
        os.environ.get(
            "MOLSCORE_SATURN_REFERENCE_LIGAND_FILE",
            str(saturn_root / "experimental_reproduction" / "synthesizability" / "7uvu-reference.pdb"),
        )
    ).expanduser()

    resolved = _replace_placeholders(
        oracle_cfg,
        {
            "__SATURN_REPO_ROOT__": str(saturn_root),
            "__QUICKVINA_BINARY__": str(quickvina_binary),
            "__RUN_DIR__": str(run_dir),
        },
    )
    if not isinstance(resolved, dict):
        raise ValueError("Resolved SATURN oracle config is not a JSON object.")

    for component in resolved.get("components", []) or []:
        if not isinstance(component, dict):
            continue
        if _component_name(component) != "quickvina2_gpu":
            continue
        spec = component.setdefault("specific_parameters", {})
        if not isinstance(spec, dict):
            continue
        spec["binary"] = str(quickvina_binary)
        spec["receptor"] = str(receptor_file)
        spec["reference_ligand"] = str(reference_ligand)
        spec["results_dir"] = str(run_dir / "docking_results")

    return resolved


def _component_name(component: dict) -> str:
    return str(component.get("name", "")).strip().lower()


def _reward_shaping_params(component: dict) -> tuple[str, dict]:
    raw = component.get("reward_shaping_function_parameters", {}) or {}
    if not isinstance(raw, dict):
        return "no_transformation", {}
    transform = str(raw.get("transformation_function", "no_transformation")).strip().lower()
    params = raw.get("parameters", {}) or {}
    return transform, dict(params) if isinstance(params, dict) else {}


def _apply_reward_shaping(values: np.ndarray, component: dict) -> np.ndarray:
    transform, params = _reward_shaping_params(component)
    x = np.asarray(values, dtype=float)
    if transform == "no_transformation":
        return x
    if transform == "binary":
        return (x >= 1.0).astype(float)
    if transform == "step":
        low = float(params["low"])
        high = float(params["high"])
        return ((x >= low) & (x <= high)).astype(float)
    if transform == "sigmoid":
        low = float(params["low"])
        high = float(params["high"])
        k = float(params["k"])
        with np.errstate(over="ignore", invalid="ignore"):
            denom = 1.0 + np.power(10.0, 10.0 * k * (x - (low + high) * 0.5) / (low - high))
            out = 1.0 / denom
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if transform == "reverse_sigmoid":
        low = float(params["low"])
        high = float(params["high"])
        k = float(params["k"])
        with np.errstate(over="ignore", invalid="ignore"):
            denom = 1.0 + np.power(10.0, k * (x - (high + low) * 0.5) * 10.0 / (high - low))
            out = 1.0 / denom
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if transform == "double_sigmoid":
        low = float(params["low"])
        high = float(params["high"])
        coef_div = float(params["coef_div"])
        coef_si = float(params["coef_si"])
        coef_se = float(params["coef_se"])
        with np.errstate(over="ignore", invalid="ignore"):
            a = np.power(10.0, coef_se * (x / coef_div))
            b = a + np.power(10.0, coef_se * (low / coef_div))
            c = np.power(10.0, coef_si * (x / coef_div))
            d = c + np.power(10.0, coef_si * (high / coef_div))
            out = (a / b) - (c / d)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"Unsupported SATURN reward shaping function for local fallback: {transform}")


def _local_raw_values(name: str, mols: list[Chem.Mol], component: dict) -> np.ndarray:
    params = component.get("specific_parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    if name == "qed":
        return np.asarray([QED.qed(mol) for mol in mols], dtype=float)
    if name in {"mw", "molecular_weight"}:
        return np.asarray([Descriptors.MolWt(mol) for mol in mols], dtype=float)
    if name == "tpsa":
        return np.asarray([rdMolDescriptors.CalcTPSA(mol) for mol in mols], dtype=float)
    if name == "slogp":
        return np.asarray([Descriptors.MolLogP(mol) for mol in mols], dtype=float)
    if name == "heavy_atoms":
        return np.asarray([mol.GetNumHeavyAtoms() for mol in mols], dtype=float)
    if name == "num_hba":
        return np.asarray([rdMolDescriptors.CalcNumHBA(mol) for mol in mols], dtype=float)
    if name == "num_hbd":
        return np.asarray([rdMolDescriptors.CalcNumHBD(mol) for mol in mols], dtype=float)
    if name == "num_rotatable_bonds":
        return np.asarray([rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in mols], dtype=float)
    if name == "num_rings":
        return np.asarray([mol.GetRingInfo().NumRings() for mol in mols], dtype=float)
    if name == "num_aromatic_rings":
        return np.asarray([rdMolDescriptors.CalcNumAromaticRings(mol) for mol in mols], dtype=float)
    if name == "num_aliphatic_rings":
        return np.asarray([rdMolDescriptors.CalcNumAliphaticRings(mol) for mol in mols], dtype=float)
    if name == "tanimoto_similarity":
        refs = params.get("reference_smiles", []) or []
        if isinstance(refs, str):
            refs = [refs]
        ref_mols = [Chem.MolFromSmiles(str(s)) for s in refs]
        ref_mols = [mol for mol in ref_mols if mol is not None]
        if not ref_mols:
            raise ValueError("Local tanimoto_similarity requires specific_parameters.reference_smiles")
        radius = int(params.get("radius", 3))
        use_counts = bool(params.get("use_counts", True))
        use_features = bool(params.get("use_features", True))
        ref_fps = [
            AllChem.GetMorganFingerprint(mol, radius=radius, useCounts=use_counts, useFeatures=use_features)
            for mol in ref_mols
        ]
        vals = []
        for mol in mols:
            fp = AllChem.GetMorganFingerprint(mol, radius=radius, useCounts=use_counts, useFeatures=use_features)
            vals.append(max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)))
        return np.asarray(vals, dtype=float)
    if name == "matching_substructure":
        smarts = params.get("smarts") or params.get("smarts_pattern") or params.get("SMARTS")
        if isinstance(smarts, list):
            patterns = [Chem.MolFromSmarts(str(s)) for s in smarts]
        else:
            patterns = [Chem.MolFromSmarts(str(smarts))]
        patterns = [p for p in patterns if p is not None]
        if not patterns:
            raise ValueError("Local matching_substructure requires SMARTS in specific_parameters")
        return np.asarray([1.0 if any(mol.HasSubstructMatch(p) for p in patterns) else 0.0 for mol in mols])
    if name == "smarts_alerts":
        smarts = params.get("smarts") or params.get("smarts_alerts") or params.get("alerts") or []
        if isinstance(smarts, str):
            smarts = [smarts]
        patterns = [Chem.MolFromSmarts(str(s)) for s in smarts]
        patterns = [p for p in patterns if p is not None]
        if not patterns:
            return np.ones((len(mols),), dtype=float)
        return np.asarray([0.0 if any(mol.HasSubstructMatch(p) for p in patterns) else 1.0 for mol in mols])
    raise NotImplementedError(f"Local SATURN fallback does not support component {name!r}")


class _LocalSaturnOracle:
    def __init__(self, oracle_cfg: dict, invalid_score: float):
        self.oracle_configuration = dict(oracle_cfg)
        self.components = list(oracle_cfg.get("components", []))
        unsupported = sorted({_component_name(c) for c in self.components} - _LOCAL_ORACLE_COMPONENTS)
        if unsupported:
            raise RuntimeError(
                "SATURN optional dependencies are unavailable, and the local fallback "
                f"does not support components: {','.join(unsupported)}"
            )
        self.budget = int(oracle_cfg.get("budget", 0) or 0)
        self.allow_oracle_repeats = bool(oracle_cfg.get("allow_oracle_repeats", False))
        self.aggregator = str(oracle_cfg.get("aggregator", "product")).strip().lower()
        if self.aggregator not in {"sum", "product"}:
            raise ValueError(f"Unsupported SATURN aggregator for local fallback: {self.aggregator}")
        self.invalid_score = float(invalid_score)
        self.calls = 0
        self.cache: dict[str, list[float]] = {}
        self.repeated_sampled_smiles: dict[str, tuple[int, float]] = {}
        self.repeated_hallucinated_smiles: dict[str, tuple[int, float]] = {}
        self.oracle_history = pd.DataFrame(
            {"oracle_calls": [], "scaffold": [], "smiles": [], "reward": [], "penalized_reward": []}
        )

    def _aggregate(self, rewards: np.ndarray, weights: np.ndarray) -> np.ndarray:
        total_weight = float(np.sum(weights)) or 1.0
        if self.aggregator == "sum":
            return np.sum(rewards.T * weights, axis=1) / total_weight
        product = np.ones((rewards.shape[1],), dtype=float)
        for r, w in zip(rewards, weights):
            product *= np.power(np.clip(r, 0.0, None), float(w) / total_weight)
        return product

    def __call__(self, smiles: np.ndarray, diversity_filter, is_hallucinated_batch: bool = False):
        _ = diversity_filter, is_hallucinated_batch
        input_smiles = [str(s) for s in smiles]
        unique_smiles: list[str] = []
        seen: set[str] = set()
        mols: list[Chem.Mol] = []
        for smi in input_smiles:
            canon = _canonical_smiles(smi)
            if canon is None:
                continue
            if canon in seen:
                continue
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                continue
            seen.add(canon)
            unique_smiles.append(canon)
            mols.append(mol)

        if not unique_smiles:
            return np.asarray([], dtype=str), np.asarray([], dtype=float)

        component_rewards: list[np.ndarray] = []
        raw_cols: dict[str, np.ndarray] = {}
        reward_cols: dict[str, np.ndarray] = {}
        weights: list[float] = []
        for component in self.components:
            name = _component_name(component)
            raw = _local_raw_values(name, mols, component)
            reward = _apply_reward_shaping(raw, component)
            component_rewards.append(np.asarray(reward, dtype=float))
            raw_cols[f"{name}_raw_values"] = np.asarray(raw, dtype=float)
            reward_cols[f"{name}_reward"] = np.asarray(reward, dtype=float)
            weights.append(float(component.get("weight", 1.0)))

        rewards = self._aggregate(np.vstack(component_rewards), np.asarray(weights, dtype=float))
        self.calls += len(unique_smiles)
        self.oracle_history = pd.concat(
            [
                self.oracle_history,
                pd.DataFrame(
                    {
                        "oracle_calls": np.full((len(unique_smiles),), self.calls),
                        "scaffold": ["" for _ in unique_smiles],
                        "smiles": unique_smiles,
                        "reward": rewards,
                        "penalized_reward": rewards,
                        **raw_cols,
                        **reward_cols,
                    }
                ),
            ],
            ignore_index=True,
        )
        for smi, reward in zip(unique_smiles, rewards.tolist()):
            self.cache.setdefault(smi, []).append(float(reward))
        return np.asarray(unique_smiles, dtype=str), np.asarray(rewards, dtype=float)

    def write_out_oracle_history(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.oracle_history.to_csv(Path(path) / "oracle_history.csv", index=False)

    def write_out_repeat_history(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "repeated_sampled_smiles_history.json").write_text(
            json.dumps(self.repeated_sampled_smiles, indent=2),
            encoding="utf-8",
        )
        (Path(path) / "repeated_hallucinated_smiles_history.json").write_text(
            json.dumps(self.repeated_hallucinated_smiles, indent=2),
            encoding="utf-8",
        )


class SaturnObjective:
    """Batch scorer adapter around Schwallergroup SATURN's Oracle.

    SpectralMol calls this only as an objective function. Molecule generation
    and all evolutionary operators remain in SpectralMol's theta genotype space.
    """

    def __init__(self, config: SaturnObjectiveConfig):
        self.config = config
        saturn_root = Path(config.saturn_repo_root).expanduser().resolve()
        if not saturn_root.exists():
            raise FileNotFoundError(f"SATURN repo root not found: {saturn_root}")
        if str(saturn_root) not in sys.path:
            sys.path.insert(0, str(saturn_root))

        run_dir = Path(config.run_dir).expanduser().resolve() if config.run_dir else Path.cwd().resolve()
        oracle_cfg = _read_oracle_config(Path(config.oracle_config_path).expanduser().resolve(), config.oracle_config_key)
        oracle_cfg = _resolve_oracle_placeholders(oracle_cfg, saturn_root=saturn_root, run_dir=run_dir)
        if config.budget_override is not None:
            oracle_cfg["budget"] = int(config.budget_override)
        if config.allow_oracle_repeats is not None:
            oracle_cfg["allow_oracle_repeats"] = bool(config.allow_oracle_repeats)

        self.invalid_score = float(config.invalid_score)
        try:
            from oracles.dataclass import OracleConfiguration  # type: ignore
            from oracles.oracle import Oracle  # type: ignore

            self.oracle = Oracle(OracleConfiguration(**oracle_cfg))
            self.using_local_fallback = False
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or str(exc)
            print(
                "[saturn] warning: SATURN Oracle import failed because optional dependency "
                f"{missing!r} is unavailable; trying local RDKit fallback.",
                file=sys.stderr,
                flush=True,
            )
            self.oracle = _LocalSaturnOracle(oracle_cfg, invalid_score=self.invalid_score)
            self.using_local_fallback = True

        if bool(config.apply_diversity_penalty):
            try:
                from diversity_filter.dataclass import DiversityFilterParameters  # type: ignore
                from diversity_filter.diversity_filter import DiversityFilter  # type: ignore

                self.diversity_filter = DiversityFilter(
                    DiversityFilterParameters(bucket_size=int(config.diversity_bucket_size))
                )
            except ModuleNotFoundError as exc:
                missing = getattr(exc, "name", "") or str(exc)
                print(
                    "[saturn] warning: SATURN diversity filter import failed because optional dependency "
                    f"{missing!r} is unavailable; diversity penalty disabled.",
                    file=sys.stderr,
                    flush=True,
                )
                self.diversity_filter = _NoDiversityFilter()
        else:
            self.diversity_filter = _NoDiversityFilter()

    @property
    def calls(self) -> int:
        return int(getattr(self.oracle, "calls", 0))

    @property
    def budget(self) -> int:
        return int(getattr(self.oracle, "budget", 0))

    def score_population(self, smiles: Sequence[str]) -> list[float]:
        originals = [str(s) for s in smiles]
        out = [self.invalid_score for _ in originals]
        valid_by_canon: dict[str, list[int]] = {}
        valid_smiles: list[str] = []

        for idx, smi in enumerate(originals):
            canon = _canonical_smiles(smi)
            if canon is None:
                continue
            if canon not in valid_by_canon:
                valid_smiles.append(canon)
            valid_by_canon.setdefault(canon, []).append(idx)

        if not valid_smiles:
            return out

        returned_smiles, rewards = self.oracle(
            smiles=np.asarray(valid_smiles, dtype=str),
            diversity_filter=self.diversity_filter,
            is_hallucinated_batch=False,
        )
        rewards = np.asarray(rewards, dtype=float)
        for smi, reward in zip(list(returned_smiles), rewards.tolist()):
            canon = _canonical_smiles(str(smi))
            if canon is None:
                continue
            for idx in valid_by_canon.get(canon, []):
                out[idx] = float(reward)
        return out

    def write_oracle_history(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        writer = getattr(self.oracle, "write_out_oracle_history", None)
        if callable(writer):
            writer(str(out))
            return out / "oracle_history.csv"
        history = getattr(self.oracle, "oracle_history", None)
        if isinstance(history, pd.DataFrame):
            path = out / "oracle_history.csv"
            history.to_csv(path, index=False)
            return path
        raise RuntimeError("SATURN oracle does not expose oracle history.")

    def write_repeat_history(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        writer = getattr(self.oracle, "write_out_repeat_history", None)
        if callable(writer):
            writer(str(out))
            return

        for attr, filename in (
            ("repeated_sampled_smiles", "repeated_sampled_smiles_history.json"),
            ("repeated_hallucinated_smiles", "repeated_hallucinated_smiles_history.json"),
        ):
            data = getattr(self.oracle, attr, {})
            (out / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
