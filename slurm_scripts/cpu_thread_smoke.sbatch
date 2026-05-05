#!/bin/bash
#SBATCH --job-name=molevo-cpu-thread-smoke
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

REPO=/ibex/user/${USER}/molevoDrugDiscovery
CONDA=/ibex/user/${USER}/miniconda3/bin/conda
ENV=/ibex/user/${USER}/molevoDrugDiscovery/env

mkdir -p "${REPO}/logs"
cd "${REPO}"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_PARALLEL_JOBS=${SLURM_CPUS_PER_TASK}

${CONDA} run -p "${ENV}" --no-capture-output python benchmarks/Guacamol/benchmark_compare_models.py \
  --executor thread \
  --budget 100 \
  --generations 5 \
  --population-size 32 \
  --batch-size 16 \
  --graphga-n-jobs ${SLURM_CPUS_PER_TASK} \
  --crem-ncpu ${SLURM_CPUS_PER_TASK} \
  --continue-on-error
