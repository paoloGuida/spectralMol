#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_mpo_v3.env
export OUTPUT_ROOT="${OUTPUT_ROOT:-${GUACAMOL_REPRO_OUTPUT_ROOT}}"

BACKEND="${EXECUTION_BACKEND:-local}"
case "${BACKEND}" in
  local)
    START_TASK_ID="${START_TASK_ID:-0}"
    END_TASK_ID="${END_TASK_ID:-199}"
    for ((task_id = START_TASK_ID; task_id <= END_TASK_ID; task_id++)); do
      echo "[guacamol-repro] running task ${task_id}/${END_TASK_ID} locally"
      SLURM_ARRAY_TASK_ID="${task_id}" \
        bash sbatch_guacamol_spectralmol_graphga_10seed_stats_slurm.sh
    done
    ;;
  slurm)
    command -v sbatch >/dev/null 2>&1 || {
      echo "[guacamol-repro] sbatch not found; use EXECUTION_BACKEND=local." >&2
      exit 2
    }
    sbatch --array="${TASK_ARRAY:-0-199%40}" --cpus-per-task=1 --mem=16G --time=24:00:00 \
      --export=ALL \
      sbatch_guacamol_spectralmol_graphga_10seed_stats_slurm.sh
    ;;
  *)
    echo "[guacamol-repro] unknown EXECUTION_BACKEND=${BACKEND}" >&2
    exit 2
    ;;
esac
