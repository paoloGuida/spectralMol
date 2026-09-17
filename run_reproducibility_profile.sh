#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-${PROFILE:-all}}"
ACTION="${2:-${ACTION:-verify}}"
ROOT="reproducibility_backups/unified_guacamol_saturn_20260817"

usage() {
  cat <<'EOF'
Run or analyze pinned reproducibility profiles.

Usage:
  bash run_reproducibility_profile.sh all verify
  bash run_reproducibility_profile.sh guacamol_mpo_v3 run
  bash run_reproducibility_profile.sh guacamol_mpo_v3 analyze
  bash run_reproducibility_profile.sh saturn_table8_v119 run
  bash run_reproducibility_profile.sh saturn_table8_v119 analyze
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${ROOT}" ]]; then
  echo "[repro] run from the SpectralMol repository root." >&2
  exit 2
fi

case "${PROFILE}:${ACTION}" in
  all:verify|all:analyze)
    exec bash "${ROOT}/commands/verify_bundled_results.sh"
    ;;
  guacamol_mpo_v3:run)
    exec bash "${ROOT}/commands/run_guacamol_mpo_v3.sh"
    ;;
  guacamol_mpo_v3:verify|guacamol_mpo_v3:analyze)
    exec bash "${ROOT}/commands/aggregate_guacamol_mpo_v3.sh"
    ;;
  saturn_table8_v119:run)
    exec bash "${ROOT}/commands/run_saturn_table8_v119.sh"
    ;;
  saturn_table8_v119:verify|saturn_table8_v119:analyze)
    exec bash "${ROOT}/commands/analyze_saturn_table8_v119.sh"
    ;;
  *)
    echo "[repro] unsupported profile/action: ${PROFILE}/${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac

