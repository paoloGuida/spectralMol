#!/bin/bash
#SBATCH --job-name=molevo-gpu-evolve-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

${CONDA} run -p "${ENV}" --no-capture-output python benchmarks/Guacamol/evolve_vs_molscore_benchmark.py \
  --objective-backend molscore \
  --skip-random-baseline \
  --budget 100 \
  --population-size 32 \
  --batch-size 16 \
  --max-generations 5
