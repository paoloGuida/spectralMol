from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.diversity_filter.dataclass import DiversityFilterParameters
from core.diversity_filter.diversity_filter import DiversityFilter
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, mutate_Theta
from core.oracles.dataclass import OracleConfiguration
from core.oracles.oracle import Oracle
from core.oracles.reward_aggregator.reward_aggregator import RewardAggregator
from core.reports import write_guacamol_like_reports
from core.utils.chemistry_utils import canonicalize_smiles, get_bemis_murcko_scaffold


@pytest.mark.unit
def test_fourier_basis_shape() -> None:
    phi = build_fourier_basis(L=16, K=4)
    assert phi.shape == (16, 9)
    assert np.isfinite(phi).all()


@pytest.mark.unit
def test_theta_mutation_shape_and_finiteness() -> None:
    rng = np.random.RandomState(7)
    theta = rng.normal(0.0, 1.0, size=(6, 8)).astype(np.float64)
    mutated = mutate_Theta(theta, gen=1, generations=10, rng=rng)
    assert mutated.shape == theta.shape
    assert np.isfinite(mutated).all()


@pytest.mark.unit
def test_structured_embedding_shape() -> None:
    vocab = ["[PAD]", "[C]", "[O]", "[N]", "[Ring1]", "[Branch1]"]
    e = build_structured_E(
        vocab=vocab,
        D=12,
        pad_token="[PAD]",
        seed=11,
        target_std=1.0,
        allowed_elements={"C", "N", "O"},
    )
    assert e.shape == (len(vocab), 12)
    assert np.isfinite(e).all()


@pytest.mark.unit
def test_reward_aggregator_sum_and_product() -> None:
    rewards = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    weights = np.array([1.0, 1.0], dtype=np.float32)

    sum_agg = RewardAggregator("sum")
    prod_agg = RewardAggregator("product")

    s = sum_agg(rewards, weights)
    p = prod_agg(rewards, weights)

    assert s.shape == (2,)
    assert p.shape == (2,)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all((p >= 0.0) & (p <= 1.0))


@pytest.mark.unit
def test_diversity_filter_penalizes_over_bucket() -> None:
    df = DiversityFilter(DiversityFilterParameters(bucket_size=0))
    smiles = np.array(["c1ccccc1", "c1ccccc1"], dtype=object)
    rewards = np.array([0.7, 0.6], dtype=np.float32)

    df.update(smiles)
    penalized = df.penalize_reward(smiles, rewards)

    assert penalized.shape == rewards.shape
    assert np.all(penalized == 0.0)


@pytest.mark.unit
def test_chemistry_utils_canonical_and_scaffold() -> None:
    smi = "C1=CC=CC=C1"
    canon = canonicalize_smiles(smi)
    scaffold = get_bemis_murcko_scaffold(smi)

    assert isinstance(canon, str)
    assert len(canon) > 0
    assert isinstance(scaffold, str)


@pytest.mark.unit
def test_oracle_qed_sa_minimal_fixture(repo_root: Path) -> None:
    cfg_path = repo_root / "tests" / "test_oracle_qed_sa.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))["oracle"]
    oracle = Oracle(OracleConfiguration(**raw))

    assert len(oracle.oracle) == 2
    assert oracle.aggregator.aggregator in {"sum", "product"}


@pytest.mark.unit
def test_reports_writer_on_synthetic_scores(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_x"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Minimal MolScore-like scores table.
    import pandas as pd

    pd.DataFrame(
        [
            {"step": 0, "smiles": "CCO", "valid": True, "single": 0.2, "filter": 1.0, "valid_score": 1.0},
            {"step": 1, "smiles": "CCN", "valid": True, "single": 0.4, "filter": 1.0, "valid_score": 1.0},
        ]
    ).to_csv(task_dir / "scores.csv", index=False)

    ok = write_guacamol_like_reports(task_dir)
    assert ok is True
    assert (task_dir / "evolution_log.tsv").exists()
    assert (task_dir / "gen_metrics.tsv").exists()
