#!/bin/bash
#SBATCH --job-name=cmp-dask-rapids-v100
#SBATCH --partition=batch
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=/ibex/user/${USER}/molevoDrugDiscovery
CONDA=/ibex/user/${USER}/miniconda3/bin/conda
ENV=/ibex/user/${USER}/molevoDrugDiscovery/env
MODEL_SPEC=benchmarks/Guacamol/model_specs.json
SEED_FILE=env/lib/python3.12/site-packages/guacamol/data/holdout_set_gcm_v1.smiles
OUTPUT_DIR=${OUTPUT_DIR:-molscore/outputs/comparison_matrix/dask_rapids_v100}
MODELS=${MODELS:-local_evolution}
SEEDS=${SEEDS:-7,8,9,10,11,12}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-}
BUDGET=${BUDGET:-20000}
GENERATIONS=${GENERATIONS:-30}
POP=${POP:-256}
BATCH=${BATCH:-256}

mkdir -p "${REPO}/logs"
cd "${REPO}"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MOLSCORE_PARALLEL_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_GRAPHGA_N_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_CREM_NCPU=${SLURM_CPUS_PER_TASK}

CMD=(
  "${CONDA}" run -p "${ENV}" --no-capture-output python
  benchmarks/Guacamol/benchmark_compare_models.py
  --executor dask
  --dataframe-backend auto
  --model-spec-file "${MODEL_SPEC}"
  --seed-smiles-file "${SEED_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --models "${MODELS}"
  --seeds "${SEEDS}"
  --budget "${BUDGET}"
  --generations "${GENERATIONS}"
  --population-size "${POP}"
  --batch-size "${BATCH}"
  --graphga-n-jobs "${SLURM_CPUS_PER_TASK}"
  --crem-ncpu "${SLURM_CPUS_PER_TASK}"
  --no-bootstrap-scoring-envs
  --continue-on-error
)

if [[ -n "${EXAMPLES_ROOT}" ]]; then
  CMD+=(--examples-root "${EXAMPLES_ROOT}")
fi

"${CMD[@]}"
