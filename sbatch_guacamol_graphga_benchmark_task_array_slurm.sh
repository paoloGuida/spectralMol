#!/usr/bin/env bash
#SBATCH --job-name=guacamol_graphga_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%20
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Fair GraphGA benchmark for the final SpectralMol theta run.
#
# Reuses sbatch_guacamol_spectralmol_task_array_slurm.sh only as a task/budget
# launcher. MODELS=graphga ensures SpectralMol is not run.
# Fairness knobs:
#   - same 20 GuacaMol task indexes
#   - same seed 7
#   - same starting population file
#   - same population size and 500-generation cap
#   - same per-task budget computed from the final SpectralMol task profiles

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SOURCE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${SOURCE_DIR}}"
USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"

export MODELS="${MODELS:-graphga}"
export SPECTRAL_TASK_PROFILE_MODE="${SPECTRAL_TASK_PROFILE_MODE:-auto}"
export SEEDS="${SEEDS:-7}"
export GENERATIONS="${GENERATIONS:-500}"
export POPULATION_SIZE="${POPULATION_SIZE:-256}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEED_POOL_SIZE="${SEED_POOL_SIZE:-256}"
export MOLSCORE_PARALLEL_JOBS="${MOLSCORE_PARALLEL_JOBS:-1}"
export GRAPHGA_N_JOBS="${GRAPHGA_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
# Keep GraphGA from stopping early before the matched 500-generation opportunity.
export GRAPHGA_PATIENCE="${GRAPHGA_PATIENCE:-500}"

if [[ -z "${OUTPUT_BASE_DIR:-}" ]]; then
  export OUTPUT_BASE_DIR="${SCRIPT_DIR}/reproducibility_runs/guacamol_graphga_seed7"
fi

if [[ -z "${SEED_SMILES_FILE:-}" ]]; then
  export SEED_SMILES_FILE="${SCRIPT_DIR}/reproducibility/manuscript_2026/inputs/guacamol/shared_initial_population_seed_7.smi"
fi

echo "[graphga-array] MODELS=${MODELS}"
echo "[graphga-array] SEEDS=${SEEDS}"
echo "[graphga-array] GENERATIONS=${GENERATIONS}"
echo "[graphga-array] POPULATION_SIZE=${POPULATION_SIZE}"
echo "[graphga-array] BATCH_SIZE=${BATCH_SIZE}"
echo "[graphga-array] GRAPHGA_N_JOBS=${GRAPHGA_N_JOBS}"
echo "[graphga-array] GRAPHGA_PATIENCE=${GRAPHGA_PATIENCE}"
echo "[graphga-array] SEED_SMILES_FILE=${SEED_SMILES_FILE:-<auto>}"
echo "[graphga-array] OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR}"
echo "[graphga-array] SCRIPT_DIR=${SCRIPT_DIR}"

exec bash "${SCRIPT_DIR}/sbatch_guacamol_spectralmol_task_array_slurm.sh"
