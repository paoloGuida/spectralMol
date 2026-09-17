#!/usr/bin/env bash
#SBATCH --job-name=guacamol_10seed_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-199%40
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# 10-seed statistical GuacaMol run for the final theta-only SpectralMol setup
# versus the fair GraphGA benchmark. This wrapper maps the Slurm array index to
# (seed, GuacaMol task) and delegates execution to the existing task-array
# launcher with a one-seed SEEDS value.
#
# Default array layout:
#   seeds: 7,8,9,10,11,12,13,14,15,16
#   tasks: 0..19
#   array index = seed_position * 20 + task_index
#
# This gives 200 jobs. The default %40 limit runs up to 40 jobs at once.

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SOURCE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${SOURCE_DIR}}"
USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"

# Do not pass comma-separated values through sbatch --export; Slurm also uses
# commas to separate exported variables. Prefer SEED_LIST=7:8:... in sbatch.
SEED_LIST_RAW="${SEED_LIST:-${SEED_LIST_CSV:-7,8,9,10,11,12,13,14,15,16}}"
SEED_LIST_RAW="${SEED_LIST_RAW//:/,}"
SEED_LIST_RAW="${SEED_LIST_RAW//;/,}"
N_TASKS="${N_TASKS:-20}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEED_LIST_RAW}"
N_SEEDS="${#SEED_ARRAY[@]}"
SEED_POS=$((ARRAY_ID / N_TASKS))
TASK_INDEX=$((ARRAY_ID % N_TASKS))

if (( SEED_POS < 0 || SEED_POS >= N_SEEDS )); then
  echo "[10seed] array id ${ARRAY_ID} maps outside seed list ${SEED_LIST_RAW}; exiting."
  exit 0
fi

SEED="${SEED_ARRAY[${SEED_POS}]}"
SEED="${SEED//[[:space:]]/}"
if [[ -z "${SEED}" ]]; then
  echo "[10seed] empty seed at position ${SEED_POS}" >&2
  exit 1
fi

export MODELS="${MODELS:-spectralmol,graphga}"
export SEEDS="${SEED}"
export GENERATIONS="${GENERATIONS:-500}"
export POPULATION_SIZE="${POPULATION_SIZE:-256}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEED_POOL_SIZE="${SEED_POOL_SIZE:-256}"
export BUDGET="${BUDGET:-128256}"
export MOLSCORE_PARALLEL_JOBS="${MOLSCORE_PARALLEL_JOBS:-1}"
export GRAPHGA_N_JOBS="${GRAPHGA_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export GRAPHGA_PATIENCE="${GRAPHGA_PATIENCE:-500}"
export SPECTRAL_TASK_PROFILE_MODE="${SPECTRAL_TASK_PROFILE_MODE:-auto}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE_DIR:-${SCRIPT_DIR}/reproducibility_runs/guacamol_10seed_spectralmol_graphga_stats}}"
export OUTPUT_ROOT
export OUTPUT_BASE_DIR="${OUTPUT_ROOT}/seed_${SEED}"

if [[ -z "${EXAMPLES_ROOT:-}" && -d "${SCRIPT_DIR}/../MolScore_examples" ]]; then
  export EXAMPLES_ROOT="${SCRIPT_DIR}/../MolScore_examples"
fi

if [[ -z "${SEED_SMILES_FILE:-}" ]]; then
  export SEED_SMILES_FILE="${SCRIPT_DIR}/reproducibility/manuscript_2026/inputs/guacamol/shared_initial_population_seed_7.smi"
fi

if [[ -z "${SEED_SMILES_FILE:-}" || ! -f "${SEED_SMILES_FILE}" ]]; then
  echo "[10seed] seed source file not found." >&2
  echo "[10seed] Set SEED_SMILES_FILE=/path/to/chembl_or_seed_pool.smi" >&2
  exit 1
fi

if [[ "${SEED_SMILES_FILE}" == *"/seed_7.smi" ]]; then
  echo "[10seed][warning] SEED_SMILES_FILE points to seed_7.smi." >&2
  echo "[10seed][warning] This is fair across models, but distinct seeds may share the same 256-molecule initial pool." >&2
  echo "[10seed][warning] For fully independent initial populations, use the full ChEMBL/GuacaMol training SMILES file." >&2
fi

# Theta-only SpectralMol settings from the successful target-full/jump run.
export MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION:-0}"
export SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION:-0}"
export MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION="${MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION:-0}"
export SPECTRAL_BRICS_CROSSOVER_FRACTION="${SPECTRAL_BRICS_CROSSOVER_FRACTION:-0}"
export MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION:-0}"
export SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION:-0}"

export FREQUENCY_MODE="${FREQUENCY_MODE:-full-spectrum}"
export SPECTRAL_L="${SPECTRAL_L:-32}"
export SPECTRAL_K="${SPECTRAL_K:-16}"
export SPECTRAL_D="${SPECTRAL_D:-32}"
export SPECTRAL_DECODE_ATTEMPTS="${SPECTRAL_DECODE_ATTEMPTS:-4}"
export MOLSCORE_GAUSS_STD_THETA="${MOLSCORE_GAUSS_STD_THETA:-0.24}"
export MOLSCORE_P_PARAM_NOISE="${MOLSCORE_P_PARAM_NOISE:-0.14}"
export MOLSCORE_P_ROW_RESET="${MOLSCORE_P_ROW_RESET:-0.025}"
export MOLSCORE_THETA_INIT_STD="${MOLSCORE_THETA_INIT_STD:-0.85}"
export MOLSCORE_CLIP_THETA_NORM="${MOLSCORE_CLIP_THETA_NORM:-5.0}"
export MOLSCORE_SPECTRAL_PARENT_MUTATION_STEPS="${MOLSCORE_SPECTRAL_PARENT_MUTATION_STEPS:-3}"
export MOLSCORE_SPECTRAL_IMMIGRANT_MUTATION_STEPS="${MOLSCORE_SPECTRAL_IMMIGRANT_MUTATION_STEPS:-3}"
export MOLSCORE_LOCAL_EVO_MUTATION_STEP_CAP="${MOLSCORE_LOCAL_EVO_MUTATION_STEP_CAP:-12}"

export MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS="${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS:-1}"
export MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS="${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS:-1}"
export MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS="${MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS:-1}"
export MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX="${MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX:-64}"
export MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX="${MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_PREFIX:-TASKFULL}"
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_PREFIX="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_PREFIX:-TASKT}"
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX:-256}"
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT:-3.0}"
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION="${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION:-0.35}"
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION="${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION:-0.20}"

echo "[10seed] array_id=${ARRAY_ID} seed_pos=${SEED_POS} seed=${SEED} task_index=${TASK_INDEX}"
echo "[10seed] models=${MODELS}"
echo "[10seed] output_root=${OUTPUT_ROOT}"
echo "[10seed] output_base_dir=${OUTPUT_BASE_DIR}"
echo "[10seed] seed_smiles_file=${SEED_SMILES_FILE}"
echo "[10seed] examples_root=${EXAMPLES_ROOT:-<unset>}"
echo "[10seed] generations=${GENERATIONS} population=${POPULATION_SIZE} batch=${BATCH_SIZE} budget=${BUDGET}"

export SLURM_ARRAY_TASK_ID="${TASK_INDEX}"
exec bash "${SCRIPT_DIR}/sbatch_guacamol_spectralmol_task_array_slurm.sh"
