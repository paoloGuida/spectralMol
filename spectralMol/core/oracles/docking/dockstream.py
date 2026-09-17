"""
Adapted from https://github.com/MolecularAI/reinvent-scoring/blob/main/reinvent_scoring/scoring/score_components/structural/dockstream.py.
"""
import subprocess
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol

class DockStream(OracleComponent):
    """
    DockStream is a wrapper around various ligand enumerators/3D conformation generators and docking algorithms.
    The interface can take as input SMILES strings and return docking scores.
    Based on: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00563-7.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.docking_configuration_path = parameters.specific_parameters["configuration_path"]
        self.docker_script_path = parameters.specific_parameters["docker_script_path"]
        self.environment_path = parameters.specific_parameters["environment_path"]

    def __call__(self, mols: np.ndarray[Mol], oracle_calls: int) -> np.ndarray[float]:
        # FIXME: Bad practice as the function signature is not the same as the parent class abstract method
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles, oracle_calls)
    
    def _compute_property(self, smiles: np.ndarray[str], oracle_calls: int) -> np.ndarray[float]:
        """
        Run DockStream and return the docking scores.
        """
        command = self._create_command(smiles, oracle_calls)
        dockstream_results = self._get_docking_scores(command, len(smiles))
        docking_scores = []
        for result in dockstream_results:
            try:
                docking_scores.append(float(result))
            except ValueError:
                docking_scores.append(0.0)

        return np.array(docking_scores)
        
    def _create_command(self, smiles: np.ndarray[str], oracle_calls: int) -> list[str]:
        """
        Create the CLI command to run DockStream.
        """
        # pass entire batch to DockStream - parallelization is handled by DockStream
        return [
            self.environment_path,
            self.docker_script_path,
            "-conf", self.docking_configuration_path,
            # Tags output poses and scores with the oracle calls so far.
            "-output_prefix", f"oracle_calls_{oracle_calls}_",
            "-smiles", ";".join(smiles),
            "-print_scores",
            "-debug",
        ]
    
    def _get_docking_scores(self, command: list[str], num_scores: int) -> np.ndarray[str]:
        """Execute DockStream and return its score lines."""
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.splitlines()[:num_scores]
