#!/usr/bin/env bash
#SBATCH --job-name=guacamol_freq_ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-479%40
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# GuacaMol frequency-mode ablation described in the manuscript.
#
# Default layout:
#   conditions: full-spectrum, high-only, low-only, random-matrix
#   seeds:      0,1,2,3,4,5
#   tasks:      0..19
#   array id = condition_pos * n_seeds * n_tasks + seed_pos * n_tasks + task_idx
#
# The delegated task launcher runs theta-only SpectralMol. This wrapper fixes
# GENERATIONS=0 so BUDGET remains the manuscript oracle-call limit.

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SOURCE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${SOURCE_DIR}}"
USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"

CONDITION_LIST_RAW="${CONDITION_LIST:-full-spectrum:high-only:low-only:random-matrix}"
CONDITION_LIST_RAW="${CONDITION_LIST_RAW//:/,}"
CONDITION_LIST_RAW="${CONDITION_LIST_RAW//;/,}"
SEED_LIST_RAW="${SEED_LIST:-0:1:2:3:4:5}"
SEED_LIST_RAW="${SEED_LIST_RAW//:/,}"
SEED_LIST_RAW="${SEED_LIST_RAW//;/,}"
N_TASKS="${N_TASKS:-20}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"

IFS=',' read -r -a CONDITION_ARRAY <<< "${CONDITION_LIST_RAW}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEED_LIST_RAW}"
N_CONDITIONS="${#CONDITION_ARRAY[@]}"
N_SEEDS="${#SEED_ARRAY[@]}"
TASKS_PER_CONDITION=$((N_SEEDS * N_TASKS))
CONDITION_POS=$((ARRAY_ID / TASKS_PER_CONDITION))
REMAINDER=$((ARRAY_ID % TASKS_PER_CONDITION))
SEED_POS=$((REMAINDER / N_TASKS))
TASK_INDEX=$((REMAINDER % N_TASKS))

if (( CONDITION_POS < 0 || CONDITION_POS >= N_CONDITIONS )); then
  echo "[ablation] array id ${ARRAY_ID} maps outside condition list ${CONDITION_LIST_RAW}; exiting."
  exit 0
fi
if (( SEED_POS < 0 || SEED_POS >= N_SEEDS )); then
  echo "[ablation] array id ${ARRAY_ID} maps outside seed list ${SEED_LIST_RAW}; exiting."
  exit 0
fi

FREQUENCY_CONDITION="${CONDITION_ARRAY[${CONDITION_POS}]}"
FREQUENCY_CONDITION="${FREQUENCY_CONDITION//[[:space:]]/}"
SEED="${SEED_ARRAY[${SEED_POS}]}"
SEED="${SEED//[[:space:]]/}"

case "${FREQUENCY_CONDITION}" in
  full-spectrum|high-only|low-only|random-matrix) ;;
  *)
    echo "[ablation] invalid frequency condition: ${FREQUENCY_CONDITION}" >&2
    exit 1
    ;;
esac

export MODELS=spectralmol
export SEEDS="${SEED}"
export FREQUENCY_MODE="${FREQUENCY_CONDITION}"
export GENERATIONS="${GENERATIONS:-0}"
export POPULATION_SIZE="${POPULATION_SIZE:-256}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export BUDGET="${BUDGET:-20000}"
export SEED_POOL_SIZE="${SEED_POOL_SIZE:-2000}"
export MOLSCORE_PARALLEL_JOBS="${MOLSCORE_PARALLEL_JOBS:-1}"
export SPECTRAL_TASK_PROFILE_MODE="${SPECTRAL_TASK_PROFILE_MODE:-auto}"

# Theta-only constraint.
export MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION:-0}"
export SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION:-0}"
export MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION="${MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION:-0}"
export SPECTRAL_BRICS_CROSSOVER_FRACTION="${SPECTRAL_BRICS_CROSSOVER_FRACTION:-0}"
export MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION:-0}"
export SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/ibex/scratch/${USER_NAME}/spectralMol/guacamol_frequency_ablation_merged_20260824}"
export OUTPUT_ROOT
export OUTPUT_BASE_DIR="${OUTPUT_ROOT}/${FREQUENCY_CONDITION}/seed_${SEED}"

if [[ -z "${EXAMPLES_ROOT:-}" && -d "/home/${USER_NAME}/molevoDrugDiscovery_2/MolScore_examples" ]]; then
  export EXAMPLES_ROOT="/home/${USER_NAME}/molevoDrugDiscovery_2/MolScore_examples"
fi

if [[ -z "${SEED_SMILES_FILE:-}" ]]; then
  for candidate in \
    "/home/${USER_NAME}/ReinventCommunity/notebooks/data/chembl.filtered.smi" \
    "/home/${USER_NAME}/molevoDrugDiscovery_2/MolScore_examples/GraphGA/ZINC_250k.smi" \
    "/home/${USER_NAME}/molevoDrugDiscovery_2/Saturn/data/zinc250k/zinc250k.smi"
  do
    if [[ -f "${candidate}" ]]; then
      export SEED_SMILES_FILE="${candidate}"
      break
    fi
  done
fi

if [[ -z "${SEED_SMILES_FILE:-}" || ! -f "${SEED_SMILES_FILE}" ]]; then
  echo "[ablation] seed source file not found." >&2
  echo "[ablation] Set SEED_SMILES_FILE=/path/to/chembl_or_seed_pool.smi" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cat > "${OUTPUT_ROOT}/ablation_run_settings.tsv" <<EOF
field	value
condition_list	${CONDITION_LIST_RAW}
seed_list	${SEED_LIST_RAW}
n_tasks	${N_TASKS}
budget	${BUDGET}
generations	${GENERATIONS}
population_size	${POPULATION_SIZE}
batch_size	${BATCH_SIZE}
seed_pool_size	${SEED_POOL_SIZE}
seed_smiles_file	${SEED_SMILES_FILE}
models	${MODELS}
source_script	${SCRIPT_DIR}/sbatch_guacamol_frequency_ablation_ibex.sh
delegated_script	${SCRIPT_DIR}/sbatch_guacamol_spectralmol_task_array_ibex.sh
EOF

echo "[ablation] array_id=${ARRAY_ID} condition_pos=${CONDITION_POS} condition=${FREQUENCY_CONDITION} seed_pos=${SEED_POS} seed=${SEED} task_index=${TASK_INDEX}"
echo "[ablation] output_root=${OUTPUT_ROOT}"
echo "[ablation] output_base_dir=${OUTPUT_BASE_DIR}"
echo "[ablation] seed_smiles_file=${SEED_SMILES_FILE}"
echo "[ablation] budget=${BUDGET} generations=${GENERATIONS} population=${POPULATION_SIZE} batch=${BATCH_SIZE}"

export SLURM_ARRAY_TASK_ID="${TASK_INDEX}"
exec bash "${SCRIPT_DIR}/sbatch_guacamol_spectralmol_task_array_ibex.sh"
