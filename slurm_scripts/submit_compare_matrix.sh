#!/bin/bash
set -euo pipefail

REPO=/ibex/user/${USER}/molevoDrugDiscovery
cd "${REPO}"
mkdir -p logs

jid1=$(sbatch --parsable slurm_scripts/compare_matrix_baseline_cpu.sbatch)
echo "baseline_cpu: ${jid1}"
jid2=$(sbatch --parsable slurm_scripts/compare_matrix_dask_cpu.sbatch)
echo "dask_cpu: ${jid2}"
jid3=$(sbatch --parsable slurm_scripts/compare_matrix_rapids_v100.sbatch)
echo "rapids_v100: ${jid3}"
jid4=$(sbatch --parsable slurm_scripts/compare_matrix_dask_rapids_v100.sbatch)
echo "dask_rapids_v100: ${jid4}"
