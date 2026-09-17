#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_mpo_v3.env

VERIFY_DIR="${VERIFY_DIR:-reproducibility_backups/unified_guacamol_saturn_20260817/verification}"
mkdir -p "${VERIFY_DIR}"

OUTPUT_ROOT="${GUACAMOL_ANALYSIS_OUTPUT_ROOT:-${OUTPUT_ROOT_EXISTING}}" \
SEED_LIST="${SEED_LIST}" \
SEED_LIST_CSV="${SEED_LIST_CSV}" \
bash aggregate_guacamol_10seed_stats_ibex.sh | tee "${VERIFY_DIR}/guacamol_mpo_v3_existing_aggregate.txt"
