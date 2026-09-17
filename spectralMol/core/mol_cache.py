from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem


@dataclass(frozen=True)
class MolStuff:
    smiles: str
    scaffold: str | None


@lru_cache(maxsize=200000)
def get_molstuff(canon_smiles: str) -> MolStuff:
    mol = Chem.MolFromSmiles(canon_smiles)
    if mol is None:
        return MolStuff(smiles=canon_smiles, scaffold=None)
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffold = scaffold or None
    except Exception:
        scaffold = None
    return MolStuff(smiles=canon_smiles, scaffold=scaffold)
