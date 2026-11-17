#!/usr/bin/env python3
"""Generate a timeline of a molecule across successive evolution rounds.

The script writes:
- one input file named like input_molecule_001.smi
- one output file per round named like output_moleculoe_001_round_000.smi

Each round applies a simple deterministic structural evolution by adding a
single methyl substituent to the current molecule when possible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem

from core.utils.chemistry_utils import canonicalize_smiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a molecule evolution timeline")
    parser.add_argument(
        "--input-smiles",
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="Starting SMILES string (default: aspirin)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of evolution rounds to generate",
    )
    parser.add_argument(
        "--output-dir",
        default="molecule_evolution",
        help="Directory where the input/output molecule files will be written",
    )
    parser.add_argument(
        "--molecule-id",
        type=int,
        default=1,
        help="Numeric ID used in file names (e.g. 001)",
    )
    return parser.parse_args()


def evolve_molecule(smiles: str, round_index: int) -> str:
    """Apply a simple deterministic mutation to a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Prefer a small, valid structural change: add a methyl substituent.
    for atom_index, atom in enumerate(mol.GetAtoms()):
        try:
            editable = Chem.EditableMol(mol)
            new_atom_idx = editable.AddAtom(Chem.Atom(6))
            editable.AddBond(atom_index, new_atom_idx, Chem.BondType.SINGLE)
            candidate = editable.GetMol()
            Chem.SanitizeMol(candidate)
            return Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
        except Exception:
            continue

    # Fallback for molecules where a methyl extension is not possible.
    if round_index % 2 == 0:
        return canonicalize_smiles(smiles)

    # As a last resort, create a canonicalized representation of the current molecule.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def write_smiles_file(path: Path, smiles: str) -> None:
    path.write_text(smiles + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_smiles = canonicalize_smiles(args.input_smiles)
    if not input_smiles:
        raise ValueError("The input SMILES could not be canonicalized")

    write_smiles_file(output_dir / f"input_molecule_{args.molecule_id:03d}.smi", input_smiles)

    current_smiles = input_smiles
    for round_index in range(args.rounds):
        current_smiles = evolve_molecule(current_smiles, round_index)
        output_path = output_dir / f"output_moleculoe_{args.molecule_id:03d}_round_{round_index:03d}.smi"
        write_smiles_file(output_path, current_smiles)

    print(f"Wrote {args.rounds} evolution rounds to {output_dir}")
    print(f"Input file: {output_dir / f'input_molecule_{args.molecule_id:03d}.smi'}")
    for round_index in range(args.rounds):
        print(f"Round {round_index:03d}: {output_dir / f'output_moleculoe_{args.molecule_id:03d}_round_{round_index:03d}.smi'}")


if __name__ == "__main__":
    main()
