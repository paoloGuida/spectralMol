#!/bin/bash
#SBATCH --job-name=diversity-speed-gpu
#SBATCH --partition=batch
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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

${CONDA} run -p "${ENV}" --no-capture-output python slurm_scripts/diversity_speed_sweep.py \
  --sizes 32,64,128,256,512,1024,2048 \
  --repeats 10 \
  --warmups 2
