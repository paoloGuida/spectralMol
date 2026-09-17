#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/saturn_table8_v119.env

if [[ -n "${SATURN_RAW_OUTPUT:-}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-reproducibility_runs/saturn_table8_analysis}"
  "${SPECTRALMOL_PYTHON:-python3}" \
    spectralMol/benchmarks/Saturn/analyze_theta_nsga2_saturn.py \
    --input-root "${SATURN_RAW_OUTPUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --expected-seeds "${SEED_LIST}" \
    --budget "${BUDGET}"
  cat "${OUTPUT_DIR}/table8_spectralmol.tsv"
else
  cat reproducibility/manuscript_2026/results/saturn/table8_spectralmol.tsv
fi
