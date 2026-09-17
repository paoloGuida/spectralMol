#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a custom SpectralMol benchmark locally or with Slurm.

Usage:
  CONFIG_FILE=examples/custom_guacamol.env bash run_custom_spectralmol.sh
  CONFIG_FILE=examples/custom_saturn.env  bash run_custom_spectralmol.sh

Local execution is the default. Set EXECUTION_BACKEND=slurm to print the
sbatch command and add SUBMIT=1 to submit it.

Key variables:
  BENCHMARK_KIND=guacamol|saturn
  CONFIG_FILE=/path/to/sourceable.env
  EXECUTION_BACKEND=local|slurm
  SUBMIT=0|1 (Slurm only)
  WORK_DIR=/path/to/SpectralMol

GuacaMol variables:
  MODELS=spectralmol
  BENCHMARK=GuacaMol
  CUSTOM_BENCHMARK=/path/to/custom/task_json_dir
  TASK_INDEXES_CSV=10
  INCLUDE_CSV=TaskName
  SEEDS=7
  GENERATIONS=50
  POPULATION_SIZE=256
  BATCH_SIZE=64
  SEED_SMILES_FILE=/path/to/seeds.smi
  OUTPUT_DIR=/path/to/results/custom_guacamol

SATURN variables:
  SEED_LIST=0
  TASK_ARRAY=0
  BUDGET=1000
  SATURN_ORACLE_TEMPLATE=/path/to/oracle_template.json
  SEED_SMILES_FILE=/path/to/seeds.smi
  USE_PER_SEED_SEED_SMILES=0
  OUTPUT_ROOT=/path/to/results/custom_saturn
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -n "${CONFIG_FILE:-}" ]]; then
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[custom] CONFIG_FILE not found: ${CONFIG_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
  set +a
fi

set_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name+x}" || -z "${!name}" ]]; then
    export "${name}=${value}"
  fi
}

quote_command() {
  local arg
  for arg in "$@"; do
    printf "%q " "${arg}"
  done
  printf "\n"
}

run_or_print() {
  local -a cmd=("$@")
  echo "[custom] command:"
  quote_command "${cmd[@]}"
  if [[ "${SUBMIT}" == "1" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "[custom] sbatch not found. Use EXECUTION_BACKEND=local or install Slurm." >&2
      exit 2
    fi
    "${cmd[@]}"
  else
    echo "[custom] dry run only. Re-run with SUBMIT=1 to submit."
  fi
}

USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"
if [[ -f "sbatch_guacamol_graphga_vs_spectralmol_slurm.sh" && -d "spectralMol" ]]; then
  DEFAULT_WORK_DIR="$(pwd)"
else
  DEFAULT_WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

set_default WORK_DIR "${DEFAULT_WORK_DIR}"
set_default SUBMIT "0"
set_default EXECUTION_BACKEND "local"
set_default LOCAL_TASK_INDEX "0"
BENCHMARK_KIND="${BENCHMARK_KIND:-${1:-guacamol}}"
BENCHMARK_KIND="$(printf "%s" "${BENCHMARK_KIND}" | tr "[:upper:]" "[:lower:]")"

cd "${WORK_DIR}"

case "${BENCHMARK_KIND}" in
  guacamol)
    set_default CPUS_PER_TASK "32"
    set_default MEM "96G"
    set_default TIME "72:00:00"
    set_default PARTITION ""
    set_default GUACAMOL_USE_TASK_ARRAY "0"
    set_default TASK_ARRAY "0-19%20"
    set_default BENCHMARK "GuacaMol"
    set_default CUSTOM_BENCHMARK ""
    set_default MODELS "spectralmol"
    set_default SEEDS "7"
    set_default GENERATIONS "50"
    set_default POPULATION_SIZE "256"
    set_default BATCH_SIZE "64"
    set_default BUDGET "10000"
    set_default SEED_POOL_SIZE "2000"
    set_default OUTPUT_DIR "${WORK_DIR}/reproducibility_runs/custom_guacamol_$(date +%Y%m%d_%H%M%S)"
    set_default OUTPUT_BASE_DIR "${OUTPUT_DIR}"
    set_default FREQUENCY_MODE "full-spectrum"
    set_default SPECTRAL_L "32"
    set_default SPECTRAL_K "16"
    set_default SPECTRAL_D "32"
    set_default SPECTRAL_DECODE_ATTEMPTS "8"
    set_default SPECTRAL_TASK_PROFILE_MODE "manual"
    set_default SPECTRAL_TASK_PROFILE "manual"
    set_default MOLSCORE_PARALLEL_JOBS "2"
    set_default GRAPHGA_N_JOBS "1"
    set_default GRAPHGA_PATIENCE "20"
    set_default SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION "0"
    set_default SPECTRAL_BRICS_CROSSOVER_FRACTION "0"
    set_default SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION "0"
    export BENCHMARK CUSTOM_BENCHMARK MODELS SEEDS GENERATIONS POPULATION_SIZE BATCH_SIZE BUDGET
    export SEED_POOL_SIZE OUTPUT_DIR OUTPUT_BASE_DIR FREQUENCY_MODE SPECTRAL_L SPECTRAL_K SPECTRAL_D
    export SPECTRAL_DECODE_ATTEMPTS SPECTRAL_TASK_PROFILE_MODE SPECTRAL_TASK_PROFILE
    export MOLSCORE_PARALLEL_JOBS GRAPHGA_N_JOBS GRAPHGA_PATIENCE
    export SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION SPECTRAL_BRICS_CROSSOVER_FRACTION
    export SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION
    export SEED_SMILES_FILE="${SEED_SMILES_FILE:-}"
    export EXAMPLES_ROOT="${EXAMPLES_ROOT:-}"
    export INCLUDE_CSV="${INCLUDE_CSV:-}"
    export EXCLUDE_CSV="${EXCLUDE_CSV:-}"
    export TASK_INDEXES_CSV="${TASK_INDEXES_CSV:-}"
    if [[ "${EXECUTION_BACKEND}" == "local" ]]; then
      if [[ "${GUACAMOL_USE_TASK_ARRAY}" == "1" ]]; then
        SLURM_ARRAY_TASK_ID="${LOCAL_TASK_INDEX}" bash sbatch_guacamol_spectralmol_task_array_slurm.sh
      else
        bash sbatch_guacamol_graphga_vs_spectralmol_slurm.sh
      fi
      exit $?
    elif [[ "${EXECUTION_BACKEND}" != "slurm" ]]; then
      echo "[custom] unknown EXECUTION_BACKEND=${EXECUTION_BACKEND}" >&2
      exit 2
    fi
    cmd=(sbatch --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEM}" --time "${TIME}")
    if [[ -n "${PARTITION}" ]]; then
      cmd+=(--partition "${PARTITION}")
    fi
    cmd+=(--export=ALL)
    if [[ "${GUACAMOL_USE_TASK_ARRAY}" == "1" ]]; then
      cmd+=(--array "${TASK_ARRAY}" sbatch_guacamol_spectralmol_task_array_slurm.sh)
    else
      cmd+=(sbatch_guacamol_graphga_vs_spectralmol_slurm.sh)
    fi
    run_or_print "${cmd[@]}"
    ;;

  saturn)
    set_default CPUS_PER_TASK "1"
    set_default MEM "32G"
    set_default TIME "24:00:00"
    set_default PARTITION ""
    set_default GRES "gpu:1"
    set_default TASK_ARRAY "0"
    set_default SEED_LIST "0"
    set_default BUDGET "1000"
    set_default POPULATION_SIZE "256"
    set_default BATCH_SIZE "16"
    set_default GENERATIONS "0"
    set_default OUTPUT_ROOT "${WORK_DIR}/reproducibility_runs/custom_saturn_theta_$(date +%Y%m%d_%H%M%S)"
    set_default FREQUENCY_MODE "full-spectrum"
    set_default SPECTRAL_L "48"
    set_default SPECTRAL_K "24"
    set_default SPECTRAL_D "32"
    set_default SPECTRAL_DECODE_ATTEMPTS "16"
    set_default USE_PER_SEED_SEED_SMILES "0"
    set_default SATURN_MODULES "cuda/11.8"
    set_default MOLSCORE_SATURN_OPENCL_LIB_DIR "/lib64"
    set_default MOLSCORE_SATURN_OPENCL_PRELOAD "1"
    set_default OCL_ICD_VENDORS "/etc/OpenCL/vendors/nvidia.icd"
    if [[ -n "${ORACLE_TEMPLATE:-}" && -z "${SATURN_ORACLE_TEMPLATE:-}" ]]; then
      export SATURN_ORACLE_TEMPLATE="${ORACLE_TEMPLATE}"
    fi
    if [[ -n "${ORACLE_CONFIG_KEY:-}" && -z "${SATURN_ORACLE_CONFIG_KEY:-}" ]]; then
      export SATURN_ORACLE_CONFIG_KEY="${ORACLE_CONFIG_KEY}"
    fi
    export SEED_LIST BUDGET POPULATION_SIZE BATCH_SIZE GENERATIONS OUTPUT_ROOT
    export FREQUENCY_MODE SPECTRAL_L SPECTRAL_K SPECTRAL_D SPECTRAL_DECODE_ATTEMPTS
    export USE_PER_SEED_SEED_SMILES SATURN_MODULES MOLSCORE_SATURN_OPENCL_LIB_DIR
    export MOLSCORE_SATURN_OPENCL_PRELOAD OCL_ICD_VENDORS
    export SATURN_REPO_ROOT="${SATURN_REPO_ROOT:-}"
    export SATURN_ORACLE_TEMPLATE="${SATURN_ORACLE_TEMPLATE:-}"
    export SATURN_ORACLE_CONFIG_KEY="${SATURN_ORACLE_CONFIG_KEY:-oracle}"
    export ASSETS_ROOT="${ASSETS_ROOT:-}"
    export QUICKVINA_BINARY="${QUICKVINA_BINARY:-}"
    export RECEPTOR_FILE="${RECEPTOR_FILE:-}"
    export REFERENCE_LIGAND_FILE="${REFERENCE_LIGAND_FILE:-}"
    export MOLSCORE_SATURN_OBABEL_BINARY="${MOLSCORE_SATURN_OBABEL_BINARY:-}"
    export SEED_SMILES_FILE="${SEED_SMILES_FILE:-}"
    export PER_SEED_SEED_SMILES_DIR="${PER_SEED_SEED_SMILES_DIR:-}"
    if [[ "${EXECUTION_BACKEND}" == "local" ]]; then
      SLURM_ARRAY_TASK_ID="${LOCAL_TASK_INDEX}" bash sbatch_saturn_theta_nsga2_slurm.sh
      exit $?
    elif [[ "${EXECUTION_BACKEND}" != "slurm" ]]; then
      echo "[custom] unknown EXECUTION_BACKEND=${EXECUTION_BACKEND}" >&2
      exit 2
    fi
    cmd=(sbatch --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEM}" --time "${TIME}")
    if [[ -n "${PARTITION}" ]]; then
      cmd+=(--partition "${PARTITION}")
    fi
    if [[ -n "${GRES}" && "${GRES}" != "none" ]]; then
      cmd+=(--gres "${GRES}")
    fi
    cmd+=(--array "${TASK_ARRAY}" --export=ALL sbatch_saturn_theta_nsga2_slurm.sh)
    run_or_print "${cmd[@]}"
    ;;

  *)
    echo "[custom] unknown BENCHMARK_KIND=${BENCHMARK_KIND}" >&2
    usage >&2
    exit 2
    ;;
esac
