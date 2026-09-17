#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "[unified-verify] validating bundled code, inputs, and results"
bash reproducibility/manuscript_2026/validate_release.sh

echo
echo "[unified-verify] GuacaMol manuscript values"
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/aggregate_guacamol_mpo_v3.sh

echo
echo "[unified-verify] SATURN Table 8 manuscript values"
bash reproducibility_backups/unified_guacamol_saturn_20260817/commands/analyze_saturn_table8_v119.sh

echo
echo "[unified-verify] done"
