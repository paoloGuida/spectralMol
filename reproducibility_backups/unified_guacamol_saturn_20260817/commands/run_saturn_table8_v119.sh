#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source reproducibility_backups/unified_guacamol_saturn_20260817/settings/saturn_table8_v119.env

export EXECUTION_BACKEND="${EXECUTION_BACKEND:-local}"
bash auto_saturn_theta_tune.sh
