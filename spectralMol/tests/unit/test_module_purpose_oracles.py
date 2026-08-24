from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from core.oracles.dataclass import OracleComponentParameters
from core.oracles.physchem.qed import QED
from core.oracles.physchem.mw import MolecularWeight
from core.oracles.similarity.tanimoto_similarity import TanimotoSimilarity
from core.oracles.structural.matching_substructure import MatchingSubstructure


def _params(name: str, specific: dict | None = None) -> OracleComponentParameters:
    return OracleComponentParameters(
        name=name,
        weight=1.0,
        preliminary_check=False,
        specific_parameters=specific or {},
        reward_shaping_function_parameters={
            "transformation_function": "no_transformation",
            "parameters": {},
        },
    )


@pytest.mark.unit
def test_qed_module_purpose_scores_in_range() -> None:
    mols = np.array([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")], dtype=object)
    comp = QED(_params("qed"))
    scores = comp(mols)
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


@pytest.mark.unit
def test_mw_module_purpose_detects_relative_size() -> None:
    mols = np.array([Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCCCCCCC")], dtype=object)
    comp = MolecularWeight(_params("mw"))
    vals = comp(mols)
    assert vals.shape == (2,)
    assert float(vals[1]) > float(vals[0])


@pytest.mark.unit
def test_tanimoto_similarity_module_purpose() -> None:
    query = np.array([Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")], dtype=object)
    comp = TanimotoSimilarity(
        _params(
            "tanimoto_similarity",
            specific={"smiles": ["CCO", "c1ccccc1"], "radius": 2, "use_counts": True, "use_features": True},
        )
    )
    sims = comp(query)
    assert sims.shape == (2,)
    assert np.all((sims >= 0.0) & (sims <= 1.0))
    assert np.max(sims) > 0.95


@pytest.mark.unit
def test_structural_matching_module_purpose() -> None:
    mols = np.array([Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("CCO")], dtype=object)
    comp = MatchingSubstructure(
        _params("matching_substructure", specific={"smiles": "c1ccccc1"})
    )
    out = comp(mols)
    assert out.shape == (2,)
    assert float(out[0]) >= float(out[1])
