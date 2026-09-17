"""
Adapted from GEAM: https://openreview.net/forum?id=sLGliHckR8
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/docking.py
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/utils.py
"""
from typing import Tuple
import os
import sys
from shutil import rmtree, which
import subprocess
import tempfile
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from openbabel import pybel


class DockingVina(object):
    def __init__(self, target, *, vina_program=None, receptor_file=None, obabel_program=None):
        super().__init__()
        
        if target == 'fa7':
            self.box_center = (10.131, 41.879, 32.097)
            self.box_size = (20.673, 20.198, 21.362)
        elif target == 'parp1':
            self.box_center = (26.413, 11.282, 27.238)
            self.box_size = (18.521, 17.479, 19.995)
        elif target == '5ht1b':
            self.box_center = (-26.602, 5.277, 17.898)
            self.box_size = (22.5, 22.5, 22.5)
        elif target == 'jak2':
            self.box_center = (114.758,65.496,11.345)
            self.box_size= (19.033,17.929,20.283)
        elif target == 'braf':
            self.box_center = (84.194,6.949,-7.081)
            self.box_size = (22.032,19.211,14.106)
        grid_dir = os.environ.get("SPECTRALMOL_DOCKING_GRID_DIR", "").strip()
        self.vina_program = str(
            vina_program
            or os.environ.get("SPECTRALMOL_QVINA_BINARY", "").strip()
            or which("qvina02")
            or which("qvina2")
            or ""
        )
        self.receptor_file = str(
            receptor_file
            or os.environ.get("SPECTRALMOL_RECEPTOR_FILE", "").strip()
            or (os.path.join(grid_dir, f"{target}.pdbqt") if grid_dir else "")
        )
        self.obabel_program = str(
            obabel_program
            or os.environ.get("SPECTRALMOL_OBABEL_BINARY", "").strip()
            or which("obabel")
            or ""
        )
        if not self.vina_program:
            raise FileNotFoundError("QuickVina2 executable not configured; set vina_program or SPECTRALMOL_QVINA_BINARY.")
        if not self.receptor_file or not os.path.isfile(self.receptor_file):
            raise FileNotFoundError("Docking receptor not configured; set receptor_file or SPECTRALMOL_RECEPTOR_FILE.")
        if not self.obabel_program:
            raise FileNotFoundError("OpenBabel executable not configured; set obabel_program or SPECTRALMOL_OBABEL_BINARY.")
        self.exhaustiveness = 1
        self.num_sub_proc = 10
        self.num_cpu_dock = 5
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100

        self.temp_dir = tempfile.mkdtemp(prefix="spectralmol-docking-")
        print(f"Docking tmp dir: {self.temp_dir}")

    def gen_3d(self, smi, ligand_mol_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        subprocess.run(
            [self.obabel_program, f"-:{smi}", "--gen3D", "-O", ligand_mol_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            text=True,
        )

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_mol_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        command = [
            self.vina_program,
            "--receptor", receptor_file,
            "--ligand", ligand_pdbqt_file,
            "--out", docking_pdbqt_file,
            "--center_x", str(self.box_center[0]),
            "--center_y", str(self.box_center[1]),
            "--center_z", str(self.box_center[2]),
            "--size_x", str(self.box_size[0]),
            "--size_y", str(self.box_size[1]),
            "--size_z", str(self.box_size[2]),
            "--cpu", str(self.num_cpu_dock),
            "--num_modes", str(self.num_modes),
            "--exhaustiveness", str(self.exhaustiveness),
        ]
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            text=True,
        )
        result_lines = result.stdout.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print(e)
                print("gen_3d unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            if len(affinity_list)==0:
                affinity_list.append(99.9)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        """
            input SMILES list
            output affinity list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list
    
    def __del__(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed')


def run_vina(
    smis: np.ndarray[str], 
    predictor: DockingVina
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    raw_docking_scores = np.array(predictor.predict(smis))
    rewards = np.clip(raw_docking_scores, 0, None)
    return raw_docking_scores, rewards


class QuickVina2(OracleComponent):
    """
    QuickVina2 docking. 
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        specific = parameters.specific_parameters
        self.vina_oracle = DockingVina(
            specific["target"],
            vina_program=specific.get("vina_program"),
            receptor_file=specific.get("receptor_file"),
            obabel_program=specific.get("obabel_program"),
        )
        
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles)
    
    def _compute_property(
        self,
        smiles: np.ndarray[Mol],
    ) -> np.ndarray[float]:
        """
        Returns the QuickVina2 docking scores.
        """
        raw_vina = np.array(self.vina_oracle.predict(smiles))
        return raw_vina
