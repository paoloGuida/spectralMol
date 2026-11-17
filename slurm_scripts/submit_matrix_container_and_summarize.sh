#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
cd "${REPO}"
mkdir -p logs

BASELINE_SCRIPT=${BASELINE_SCRIPT:-slurm_scripts/compare_matrix_baseline_container_cpu.sbatch}
DASK_CPU_SCRIPT=${DASK_CPU_SCRIPT:-slurm_scripts/compare_matrix_dask_container_cpu.sbatch}
RAPIDS_SCRIPT=${RAPIDS_SCRIPT:-slurm_scripts/compare_matrix_rapids_container_v100.sbatch}
DASK_RAPIDS_SCRIPT=${DASK_RAPIDS_SCRIPT:-slurm_scripts/compare_matrix_dask_rapids_container_v100.sbatch}

SUMMARY_PARTITION=${SUMMARY_PARTITION:-batch}
SUMMARY_CPUS=${SUMMARY_CPUS:-4}
SUMMARY_MEM=${SUMMARY_MEM:-8G}
SUMMARY_TIME=${SUMMARY_TIME:-00:20:00}
SUMMARY_JOB_NAME=${SUMMARY_JOB_NAME:-cmp-matrix-summarize}
SUMMARY_PREFIX=${SUMMARY_PREFIX:-matrix_summary_container}
SUMMARY_CONFIGS=${SUMMARY_CONFIGS:-baseline_cpu_container,dask_cpu_container,rapids_v100_container,dask_rapids_v100_container}

PYTHON_BIN=${PYTHON_BIN:-/ibex/user/${USER}/molevoDrugDiscovery/env/bin/python}
SUMMARY_SCRIPT=${SUMMARY_SCRIPT:-slurm_scripts/summarize_matrix_results.py}

jid1=$(sbatch --parsable "${BASELINE_SCRIPT}")
echo "baseline_cpu_container: ${jid1}"

jid2=$(sbatch --parsable "${DASK_CPU_SCRIPT}")
echo "dask_cpu_container: ${jid2}"

jid3=$(sbatch --parsable "${RAPIDS_SCRIPT}")
echo "rapids_v100_container: ${jid3}"

jid4=$(sbatch --parsable "${DASK_RAPIDS_SCRIPT}")
echo "dask_rapids_v100_container: ${jid4}"

deps="afterok:${jid1}:${jid2}:${jid3}:${jid4}"

summary_wrap="cd ${REPO} && ${PYTHON_BIN} ${SUMMARY_SCRIPT} --configs ${SUMMARY_CONFIGS} --output-prefix ${SUMMARY_PREFIX}"
summary_jid=$(sbatch --parsable \
  --dependency="${deps}" \
  --job-name="${SUMMARY_JOB_NAME}" \
  --partition="${SUMMARY_PARTITION}" \
  --cpus-per-task="${SUMMARY_CPUS}" \
  --mem="${SUMMARY_MEM}" \
  --time="${SUMMARY_TIME}" \
  --output="logs/%x-%j.out" \
  --error="logs/%x-%j.err" \
  --wrap "${summary_wrap}")

echo "summary_job (${deps}): ${summary_jid}"
echo "Submitted matrix + auto-summary pipeline."
