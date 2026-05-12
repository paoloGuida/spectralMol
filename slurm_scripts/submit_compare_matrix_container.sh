#!/bin/bash
set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
cd "${REPO}"
mkdir -p logs

jid1=$(sbatch --parsable slurm_scripts/compare_matrix_rapids_container_v100.sbatch)
echo "rapids_v100_container: ${jid1}"
jid2=$(sbatch --parsable slurm_scripts/compare_matrix_dask_rapids_container_v100.sbatch)
echo "dask_rapids_v100_container: ${jid2}"
