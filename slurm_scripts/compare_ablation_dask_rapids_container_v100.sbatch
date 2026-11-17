#!/bin/bash
#SBATCH --job-name=cmp-ablation-dask-rapids-v100
#SBATCH --partition=batch
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
MODEL_SPEC=${MODEL_SPEC:-benchmarks/Guacamol/model_specs.json}
SEED_FILE=${SEED_FILE:-env/lib/python3.12/site-packages/guacamol/data/holdout_set_gcm_v1.smiles}
OUTPUT_DIR=${OUTPUT_DIR:-molscore/outputs/comparison_matrix/ablation_dask_rapids_v100_container}
MODELS=${MODELS:-local_evolution,local_evolution_low_only,local_evolution_high_only,local_evolution_random_matrix}
SEEDS=${SEEDS:-7,8,9,10,11,12}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-}
BUDGET=${BUDGET:-20000}
GENERATIONS=${GENERATIONS:-30}
POP=${POP:-256}
BATCH=${BATCH:-256}
FOURIER_MODE=${FOURIER_MODE:-full-spectrum}

if [[ "${OUTPUT_DIR}" = /* ]]; then
  HOST_OUTPUT_DIR="${OUTPUT_DIR}"
else
  HOST_OUTPUT_DIR="${REPO}/${OUTPUT_DIR}"
fi

mkdir -p "${REPO}/logs" "${HOST_OUTPUT_DIR}"
cd "${REPO}"
PYTHON_BIN=${PYTHON_BIN:-$PWD/env/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_PARALLEL_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_GRAPHGA_N_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_CREM_NCPU=${SLURM_CPUS_PER_TASK}
export MOLEVO_DIVERSITY_GPU_ENABLED=${MOLEVO_DIVERSITY_GPU_ENABLED:-1}
export MOLEVO_DIVERSITY_GPU_MIN_N=${MOLEVO_DIVERSITY_GPU_MIN_N:-512}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${EXAMPLES_ROOT}" ]]; then
  EXTRA_ARGS+=(--examples-root "${EXAMPLES_ROOT}")
fi

"${PYTHON_BIN}" benchmarks/Guacamol/benchmark_compare_models.py \
    --executor dask \
    --dataframe-backend auto \
    --model-spec-file "${MODEL_SPEC}" \
    --seed-smiles-file "${SEED_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --models "${MODELS}" \
    --seeds "${SEEDS}" \
    --budget "${BUDGET}" \
    --generations "${GENERATIONS}" \
    --population-size "${POP}" \
    --batch-size "${BATCH}" \
    --fourier-mode "${FOURIER_MODE}" \
    --graphga-n-jobs "${SLURM_CPUS_PER_TASK}" \
    --crem-ncpu "${SLURM_CPUS_PER_TASK}" \
    --no-bootstrap-scoring-envs \
    --continue-on-error \
    "${EXTRA_ARGS[@]}"
