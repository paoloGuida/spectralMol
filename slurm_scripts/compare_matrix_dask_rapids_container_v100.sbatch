#!/bin/bash
#SBATCH --job-name=cmp-dask-rapids-ctr-v100
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

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
RUNTIME=${RUNTIME:-podman}
IMAGE_REF=${IMAGE_REF:-molevo-rapids:cuda12.2}
CONTAINER_REPO=${CONTAINER_REPO:-/workspace}
CONTAINER_MODEL_SPEC=${CONTAINER_MODEL_SPEC:-benchmarks/Guacamol/model_specs.json}
CONTAINER_SEED_FILE=${CONTAINER_SEED_FILE:-/opt/conda/lib/python3.12/site-packages/guacamol/data/holdout_set_gcm_v1.smiles}
OUTPUT_DIR=${OUTPUT_DIR:-molscore/outputs/comparison_matrix/dask_rapids_v100_container}
MODELS=${MODELS:-local_evolution}
SEEDS=${SEEDS:-7,8,9,10,11,12}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-}
BUDGET=${BUDGET:-20000}
GENERATIONS=${GENERATIONS:-30}
POP=${POP:-256}
BATCH=${BATCH:-256}
PODMAN_STORAGE_DRIVER=${PODMAN_STORAGE_DRIVER:-vfs}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images_${PODMAN_STORAGE_DRIVER}}
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${REPO}/logs" "${PODMAN_ROOT}" "${XDG_RUNTIME_DIR}/bus"
cd "${REPO}"

export XDG_RUNTIME_DIR
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_PARALLEL_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_GRAPHGA_N_JOBS=${SLURM_CPUS_PER_TASK}
export MOLSCORE_CREM_NCPU=${SLURM_CPUS_PER_TASK}
export MOLEVO_DIVERSITY_GPU_ENABLED=${MOLEVO_DIVERSITY_GPU_ENABLED:-1}
export MOLEVO_DIVERSITY_GPU_MIN_N=${MOLEVO_DIVERSITY_GPU_MIN_N:-512}

if [[ "${RUNTIME}" != "podman" ]]; then
  echo "Unsupported runtime for this launcher: ${RUNTIME}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${EXAMPLES_ROOT}" ]]; then
  EXTRA_ARGS+=(--examples-root "${EXAMPLES_ROOT}")
fi

podman --root="${PODMAN_ROOT}" run --rm \
  --storage-driver="${PODMAN_STORAGE_DRIVER}" \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -e NVIDIA_VISIBLE_DEVICES='' \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  -e MOLSCORE_PARALLEL_JOBS="${MOLSCORE_PARALLEL_JOBS}" \
  -e MOLSCORE_GRAPHGA_N_JOBS="${MOLSCORE_GRAPHGA_N_JOBS}" \
  -e MOLSCORE_CREM_NCPU="${MOLSCORE_CREM_NCPU}" \
  -e MOLEVO_DIVERSITY_GPU_ENABLED="${MOLEVO_DIVERSITY_GPU_ENABLED}" \
  -e MOLEVO_DIVERSITY_GPU_MIN_N="${MOLEVO_DIVERSITY_GPU_MIN_N}" \
  -v "${REPO}:${CONTAINER_REPO}:Z" \
  -w "${CONTAINER_REPO}" \
  "${IMAGE_REF}" \
  python benchmarks/Guacamol/benchmark_compare_models.py \
    --executor dask \
    --dataframe-backend auto \
    --model-spec-file "${CONTAINER_MODEL_SPEC}" \
    --seed-smiles-file "${CONTAINER_SEED_FILE}" \
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
