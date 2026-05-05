#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --job-name=create-conda-env
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j-slurm.out
#SBATCH --error=logs/%x-%j-slurm.err

# create the conda environment

export CONDA_PKGS_DIRS=/ibex/user/${USER}/conda_cache
source /ibex/user/${USER}/miniconda3/bin/activate

# entire script fails if a single command fails
set -e

# create the conda environment
PROJECT_DIR="$PWD"
ENV_PREFIX="$PROJECT_DIR"/env
mamba env create --prefix $ENV_PREFIX --file "$PROJECT_DIR"/environment.yml --force