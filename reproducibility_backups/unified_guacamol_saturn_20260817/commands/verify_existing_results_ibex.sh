#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VERIFY_ROOT="${VERIFY_ROOT:-reproducibility_backups/unified_guacamol_saturn_20260817/verification}"
mkdir -p "${VERIFY_ROOT}"

echo "[unified-verify] checking GuacaMol final existing outputs"
VERIFY_DIR="${VERIFY_ROOT}" \
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/aggregate_guacamol_mpo_v3_existing_ibex.sh

echo
echo "[unified-verify] checking SATURN Table 8 v119 existing outputs"
VERIFY_DIR="${VERIFY_ROOT}/saturn_table8_v119_existing" \
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/analyze_saturn_table8_v119_existing_ibex.sh

echo
echo "[unified-verify] done"
