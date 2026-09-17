"""
Workflow based on: https://arxiv.org/abs/2406.08506 Appendix C

Workflow Overview:
* Input canonical SMILES
* Convert to RDKit Mol
* Protonate
* Generate 1 (lowest energy) conformer using RDKit ETKDG and minimize using RDKit UFF
* Convert conformer to PDBQT file
* Dock using QuickVina2-GPU version 2.1

# ETKDG: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654
"""
import os
import re
import subprocess
import tempfile
import shutil
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import AllChem, Mol


class QuickVina2_GPU(OracleComponent):
    """
    Executes QuickVina2-GPU version 2.1.

    References:
    1. https://www.biorxiv.org/content/early/2023/11/05/2023.11.04.565429
    2. https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # QuickVina2-GPU-2.1 binary
        self.binary = self.parameters.specific_parameters.get("binary", None)
        assert self.binary is not None, "Please provide the path to the QuickVina2-GPU binary."
        self.binary = str(self.binary)
        # Ensure the binary is executable for the user
        subprocess.run(["chmod", "u+x", self.binary])
        self.failure_score = float(self.parameters.specific_parameters.get("failure_score", 99.9))

        # Force-field for ligand energy minimization
        force_field_id = self.parameters.specific_parameters.get("force_field", "uff").lower()
        assert force_field_id in ["uff", "mmff94"], "force_field must be either 'uff' or 'mmff94'."
        self.force_field = AllChem.UFFOptimizeMolecule if force_field_id == "uff" else AllChem.MMFFOptimizeMolecule

        # Receptor path
        self.receptor = self.parameters.specific_parameters.get("receptor", None)
        assert self.receptor is not None and self.receptor.endswith(".pdbqt"), "Please provide the path to the receptor PDBQT file."

        # Reference ligand path
        self.reference_ligand = self.parameters.specific_parameters.get("reference_ligand", None)
        assert self.reference_ligand is not None and self.reference_ligand.endswith(".pdb"), "Please provide the path to the reference ligand PDB file."

        # Exhaustiveness (known as "Thread" in QuickVina2-GPU)
        self.thread = self.parameters.specific_parameters.get("thread", 5000)  # Default to 5000 (same as source code default)

        # Setup docking box
        self._setup_docking_box()

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int = 0
    ) -> np.ndarray[float]:
        return self._compute_property(mols, oracle_calls)
    
    def _compute_property(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int = 0
    ) -> np.ndarray[float]:
        """
        Execute QuickVina2-GPU-2.1 as a subprocess.
        """
        # 1. Make temporary files to store the input and output
        temp_input_sdf_dir = tempfile.mkdtemp()
        temp_input_pdbqt_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()
        debug_dir = os.path.join(self.output_dir, f"results_{oracle_calls}")
        os.makedirs(debug_dir, exist_ok=True)
        warnings_path = os.path.join(debug_dir, "quickvina_failures.log")
        docking_scores = np.full(len(mols), self.failure_score, dtype=np.float64)

        def log_warning(message: str) -> None:
            clean = message.strip()
            if not clean:
                return
            with open(warnings_path, "a", encoding="utf-8") as f:
                f.write(clean + "\n")
            print(f"[quickvina2_gpu] {clean}", flush=True)

        try:
            # 2. Convert RDKit Mols to *canonical* SMILES
            canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]

            # 3. Convert back to RDKit Mols
            # NOTE: This is likely redundant but is done to match the original workflow
            mols = [Chem.MolFromSmiles(smiles) for smiles in canonical_smiles]

            # 4. Protonate Mols
            mols = [Chem.AddHs(mol) if mol is not None else None for mol in mols]

            # 5. Generate 1 (lowest energy) conformer and convert to PDBQT with OpenBabel
            prepared_indices: list[int] = []
            obabel_failures: list[str] = []
            for idx, mol in enumerate(mols):
                if mol is None:
                    continue
                try:
                    AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)
                    self.force_field(mol)
                except Exception:
                    continue

                sdf_file = os.path.join(temp_input_sdf_dir, f"ligand_{idx + 1}.sdf")
                writer = Chem.SDWriter(sdf_file)
                writer.write(mol)
                writer.flush()
                writer.close()

                pdbqt_file = os.path.join(temp_input_pdbqt_dir, f"ligand_{idx + 1}.pdbqt")
                obabel = subprocess.run(
                    ["obabel", sdf_file, "-opdbqt", "-O", pdbqt_file],
                    capture_output=True,
                    text=True,
                )
                if obabel.returncode == 0 and os.path.exists(pdbqt_file):
                    prepared_indices.append(idx)
                else:
                    stderr = (obabel.stderr or "").strip().replace("\n", " | ")
                    obabel_failures.append(f"ligand_{idx + 1}: rc={obabel.returncode} stderr={stderr}")

            if not prepared_indices:
                if obabel_failures:
                    with open(os.path.join(debug_dir, "obabel_failures.log"), "w", encoding="utf-8") as f:
                        f.write("\n".join(obabel_failures) + "\n")
                log_warning(
                    "QuickVina2-GPU preflight failed: no ligand PDBQT files were generated. "
                    "Returning failure scores for this batch."
                )
                return docking_scores

            # 6. Run QuickVina2-GPU-2.1
            run_cmd = [
                os.path.abspath(self.binary),
                "--receptor", self.receptor,
                "--ligand_directory", temp_input_pdbqt_dir,
                "--output_directory", temp_output_dir,
                "--thread", str(self.thread),
                "--center_x", str(self.box_center[0]),
                "--center_y", str(self.box_center[1]),
                "--center_z", str(self.box_center[2]),
                "--size_x", str(self.box_size[0]),
                "--size_y", str(self.box_size[1]),
                "--size_z", str(self.box_size[2]),
                "--num_modes", "1",
                "--seed", "0",
            ]

            # Binary needs to be launched from its directory so it can load OpenCL kernel assets.
            current_dir = os.getcwd()
            try:
                os.chdir(os.path.dirname(os.path.abspath(self.binary)))
                qvina = subprocess.run(run_cmd, capture_output=True, text=True)
            finally:
                os.chdir(current_dir)

            with open(os.path.join(debug_dir, "quickvina_stdout.log"), "w", encoding="utf-8") as f:
                f.write(qvina.stdout or "")
            with open(os.path.join(debug_dir, "quickvina_stderr.log"), "w", encoding="utf-8") as f:
                f.write(qvina.stderr or "")
            with open(os.path.join(debug_dir, "quickvina_cmd.txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(run_cmd) + "\n")

            # 7. Persist raw docking output for debugging/inspection even when QuickVina exits non-zero.
            shutil.copytree(temp_output_dir, debug_dir, dirs_exist_ok=True)

            if qvina.returncode != 0:
                tail = (qvina.stderr or qvina.stdout or "").strip().splitlines()[-20:]
                message = "\n".join(tail)
                log_warning(
                    f"QuickVina2-GPU exited with return code {qvina.returncode}. "
                    "Will parse any available outputs and assign failure scores for missing ligands.\n"
                    f"Tail:\n{message}"
                )

            # 8. Parse docking scores.
            parsed_count = 0
            for output_file in sorted(os.listdir(temp_output_dir)):
                full = os.path.join(temp_output_dir, output_file)
                if not os.path.isfile(full):
                    continue
                match = re.search(r"ligand[_-]?(\d+)", output_file)
                if match is None:
                    match = re.search(r"(\d+)", output_file)
                if match is None:
                    continue
                ligand_idx = int(match.group(1)) - 1
                if not (0 <= ligand_idx < len(docking_scores)):
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "REMARK VINA RESULT" not in line:
                            continue
                        score_match = re.search(r"REMARK VINA RESULT:\s*([-+]?\d*\.?\d+)", line)
                        if score_match is None:
                            continue
                        docking_scores[ligand_idx] = float(score_match.group(1))
                        parsed_count += 1
                        break

            if parsed_count == 0:
                log_warning(
                    "QuickVina2-GPU produced no parseable docking scores. "
                    "Returning failure scores for this batch."
                )
                return docking_scores

            return docking_scores
        finally:
            shutil.rmtree(temp_input_sdf_dir, ignore_errors=True)
            shutil.rmtree(temp_input_pdbqt_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    def _setup_docking_box(self):
        """
        Setup the docking box for the target receptor based on the reference ligand.

        Follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C:

        * Centroids are the average position of the reference ligand atoms

        * "Box sizes individually determined to encompass each target binding"
           Unclear how this is done - box size is set to 20 Å x 20 Å x 20 Å instead which is a common default
        """
        # Get the average coordinates of the reference ligand atoms
        ref_mol = Chem.MolFromPDBFile(self.reference_ligand)
        ref_conformer = ref_mol.GetConformer()
        ref_coords = ref_conformer.GetPositions()
        ref_center = tuple(np.mean(ref_coords, axis=0)) 

        # Set the box center and size
        self.box_center = ref_center  # Tuple[float, float, float]
        self.box_size = (20.0, 20.0, 20.0)
