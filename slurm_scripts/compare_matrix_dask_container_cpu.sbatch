#!/bin/bash
#SBATCH --job-name=cmp-dask-ctr-cpu
#SBATCH --partition=batch
#SBATCH --constraint=intel&cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
MODEL_SPEC=${MODEL_SPEC:-benchmarks/Guacamol/model_specs.json}
SEED_FILE=${SEED_FILE:-env/lib/python3.12/site-packages/guacamol/data/holdout_set_gcm_v1.smiles}
OUTPUT_DIR=${OUTPUT_DIR:-molscore/outputs/comparison_matrix/dask_cpu_container}
MODELS=${MODELS:-local_evolution}
SEEDS=${SEEDS:-7,8,9,10,11,12}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-}
BUDGET=${BUDGET:-20000}
GENERATIONS=${GENERATIONS:-30}
POP=${POP:-256}
BATCH=${BATCH:-256}

if [[ "${OUTPUT_DIR}" = /* ]]; then
  HOST_OUTPUT_DIR="${OUTPUT_DIR}"
else
  HOST_OUTPUT_DIR="${REPO}/${OUTPUT_DIR}"
fi

mkdir -p "${REPO}/logs" "${HOST_OUTPUT_DIR}"
cd "${REPO}"
PYTHON_BIN=${PYTHON_BIN:-$PWD/env/bin/python}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_PARALLEL_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_GRAPHGA_N_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_CREM_NCPU=${SLURM_CPUS_PER_TASK}

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
    --dataframe-backend pandas \
    --model-spec-file "${MODEL_SPEC}" \
    --seed-smiles-file "${SEED_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --models "${MODELS}" \
    --seeds "${SEEDS}" \
    --budget "${BUDGET}" \
    --generations "${GENERATIONS}" \
    --population-size "${POP}" \
    --batch-size "${BATCH}" \
    --graphga-n-jobs "${SLURM_CPUS_PER_TASK}" \
    --crem-ncpu "${SLURM_CPUS_PER_TASK}" \
    --no-bootstrap-scoring-envs \
    --continue-on-error \
    "${EXTRA_ARGS[@]}"
