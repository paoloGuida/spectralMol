#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_frequency_ablation.env

BACKEND="${EXECUTION_BACKEND:-local}"
START_TASK_ID="${START_TASK_ID:-0}"
END_TASK_ID="${END_TASK_ID:-479}"

if [[ ! -f "${SEED_SMILES_FILE}" && -f "${SEED_SMILES_FILE}.gz" ]]; then
  echo "[frequency-ablation] decompressing ${SEED_SMILES_FILE}.gz"
  gzip -dk "${SEED_SMILES_FILE}.gz"
fi

case "${BACKEND}" in
  local)
    for ((task_id = START_TASK_ID; task_id <= END_TASK_ID; task_id++)); do
      echo "[frequency-ablation] running task ${task_id}/${END_TASK_ID} locally"
      SLURM_ARRAY_TASK_ID="${task_id}" bash sbatch_guacamol_frequency_ablation_slurm.sh
    done
    ;;
  slurm)
    command -v sbatch >/dev/null 2>&1 || {
      echo "[frequency-ablation] sbatch not found; use EXECUTION_BACKEND=local." >&2
      exit 2
    }
    sbatch --array="${TASK_ARRAY:-0-479%40}" --export=ALL \
      sbatch_guacamol_frequency_ablation_slurm.sh
    ;;
  *)
    echo "[frequency-ablation] unknown EXECUTION_BACKEND=${BACKEND}" >&2
    exit 2
    ;;
esac
