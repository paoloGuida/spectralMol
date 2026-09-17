from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import config as cfg
from core.diversity_filter.dataclass import DiversityFilterParameters
from core.diversity_filter.diversity_filter import DiversityFilter
from core.embedding import build_structured_E
from core.fourier_theta import build_fourier_basis, mutate_Theta
from core.oracles.dataclass import OracleConfiguration
from core.oracles.oracle import Oracle
from core.oracles.reward_aggregator.reward_aggregator import RewardAggregator
from core.reports import write_guacamol_like_reports
from core.spectral_evolution import SpectralGenerator, SpectralSettings
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
def test_spectral_generator_carries_and_mutates_theta() -> None:
    settings = SpectralSettings(L=12, K=3, D=12, decode_attempts=2)
    generator = SpectralGenerator(settings=settings, seed=7)

    population = generator.build_initial_population(["CCO", "CCN", "c1ccccc1"], pop_size=2)

    assert len(population) == 2
    assert {ind.smiles for ind in population}.issubset({"CCO", "CCN", "c1ccccc1"})
    assert all(ind.decode_reason == "SEED_ENCODED" for ind in population)
    assert population[0].theta.shape == (1 + 2 * settings.K, settings.D)
    assert np.isfinite(population[0].theta).all()
    mutated = generator.mutate_theta(population[0].theta, gen=1, generations=4)
    assert mutated.shape == population[0].theta.shape
    assert np.isfinite(mutated).all()
    assert not np.allclose(mutated, population[0].theta)


@pytest.mark.unit
def test_spectral_scaled_theta_mutation_can_be_local_noop() -> None:
    settings = SpectralSettings(L=12, K=3, D=12, decode_attempts=2)
    generator = SpectralGenerator(settings=settings, seed=7)

    theta = generator.random_theta()
    mutated = generator.mutate_theta(
        theta,
        gen=1,
        generations=4,
        sigma_scale=0.0,
        param_noise_scale=0.0,
        row_reset_scale=0.0,
    )

    assert mutated.shape == theta.shape
    assert np.isfinite(mutated).all()
    assert np.allclose(mutated, theta)


@pytest.mark.unit
def test_spectral_defaults_are_theta_only() -> None:
    assert cfg.SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION == 0.0
    assert cfg.SPECTRAL_BRICS_CROSSOVER_FRACTION == 0.0
    assert cfg.SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION == 0.0


@pytest.mark.unit
def test_spectral_frequency_mode_masks_inactive_theta_rows() -> None:
    settings = SpectralSettings(L=12, K=4, D=8, frequency_mode="low-only")
    generator = SpectralGenerator(settings=settings, seed=11)

    theta = generator.random_theta()

    assert theta.shape == (1 + 2 * settings.K, settings.D)
    assert np.any(generator.active_rows)
    assert np.any(~generator.active_rows)
    assert np.allclose(theta[~generator.active_rows], 0.0)


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
