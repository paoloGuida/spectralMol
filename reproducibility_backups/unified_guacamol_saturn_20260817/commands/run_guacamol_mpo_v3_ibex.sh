#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_mpo_v3.env
export OUTPUT_ROOT="${OUTPUT_ROOT:-${GUACAMOL_REPRO_OUTPUT_ROOT}}"

sbatch --array=0-199%40 --cpus-per-task=1 --mem=16G --time=24:00:00 \
  --export=ALL \
  sbatch_guacamol_spectralmol_graphga_10seed_stats_ibex.sh
