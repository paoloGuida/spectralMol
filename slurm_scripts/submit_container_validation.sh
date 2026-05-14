#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
cd "${REPO}"
mkdir -p logs

saturn_jid=$(sbatch --parsable slurm_scripts/test_saturn_container_v100.sbatch)
echo "saturn_container_validation: ${saturn_jid}"

rapids_jid=$(sbatch --parsable --dependency=afterok:${saturn_jid} slurm_scripts/test_rapids_container_v100.sbatch)
echo "rapids_container_validation (afterok:${saturn_jid}): ${rapids_jid}"

echo "Submitted sequential validation pipeline."
