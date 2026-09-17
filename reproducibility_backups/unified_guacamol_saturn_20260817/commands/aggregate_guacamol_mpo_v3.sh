#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-reproducibility/manuscript_2026/results/guacamol}"
for result in \
  guacamol_statistics_table.tsv \
  guacamol_per_seed_aggregate.tsv \
  guacamol_per_task_mean_scores.tsv; do
  if [[ ! -f "${RESULTS_DIR}/${result}" ]]; then
    echo "[guacamol-results] missing ${RESULTS_DIR}/${result}" >&2
    exit 2
  fi
  echo "===== ${result}"
  cat "${RESULTS_DIR}/${result}"
done
