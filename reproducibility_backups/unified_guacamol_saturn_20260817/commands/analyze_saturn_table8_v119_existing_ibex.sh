#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/saturn_table8_v119.env

VERIFY_DIR="${VERIFY_DIR:-reproducibility_backups/unified_guacamol_saturn_20260817/verification/saturn_table8_v119_existing}"
mkdir -p "${VERIFY_DIR}"

"${SPECTRALMOL_PYTHON:-/ibex/user/${USER}/conda-environments/molscore/bin/python}" \
  spectralMol/benchmarks/Saturn/analyze_theta_nsga2_saturn.py \
  --input-root "${SATURN_RAW_OUTPUT_EXISTING}" \
  --output-dir "${VERIFY_DIR}" \
  --expected-seeds "${SEED_LIST}" \
  --budget "${BUDGET}"

cat "${VERIFY_DIR}/table8_spectralmol.tsv"
