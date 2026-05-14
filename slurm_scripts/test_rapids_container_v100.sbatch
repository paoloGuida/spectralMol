#!/bin/bash
#SBATCH --job-name=test-rapids-ctr-v100
#SBATCH --partition=debug
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
RAPIDS_IMAGE=${RAPIDS_IMAGE:-docker.io/dxbarradas/spectralmol-rapids:wolfi-conda}
CONTAINER_REPO=${CONTAINER_REPO:-/workspace}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-molevo-rapids-dask-draft}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images}
TEST_ROOT=${TEST_ROOT:-/ibex/user/${USER}/container-testing/rapids}
MINI_OUTPUT_DIR=${MINI_OUTPUT_DIR:-${TEST_ROOT}/data/minimal_test}

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${REPO}/logs" "${TEST_ROOT}/logs" "${MINI_OUTPUT_DIR}" "/run/user/${UID}/bus" "${PODMAN_ROOT}"
cd "${REPO}"

if ! command -v podman >/dev/null 2>&1; then
  echo "Podman is not available on PATH." >&2
  exit 1
fi

echo "[preflight] nvidia-smi"
podman --root="${PODMAN_ROOT}" pull "${RAPIDS_IMAGE}"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${RAPIDS_IMAGE}" nvidia-smi || true

echo "[smoke] python imports + RAPIDS GPU"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v "${REPO}:${CONTAINER_REPO}" \
  "${RAPIDS_IMAGE}" \
  bash -lc "source /etc/profile.d/conda.sh 2>/dev/null || true; \
_py=''; \
if command -v conda >/dev/null 2>&1; then \
  _req='${CONDA_ENV_NAME}'; _env_path=''; \
  if [[ -n \"$_req\" ]]; then \
    _env_path=$(conda env list | awk -v req=\"$_req\" '!/^#/ && NF {n=$1; p=$NF; if (n==req || p==req) {print p; exit}}'); \
  fi; \
  if [[ -z \"$_env_path\" ]]; then \
    _env_path=$(conda env list | awk '!/^#/ && NF {n=$1; p=$NF; if (n==\"base\") next; print p; exit}'); \
  fi; \
  if [[ -n \"$_env_path\" && -x \"$_env_path/bin/python\" ]]; then \
    _py=\"$_env_path/bin/python\"; \
    echo \"[conda] using python: $_py\"; \
  fi; \
fi; \
if [[ -z \"$_py\" ]]; then _py=$(command -v python || true); fi; \
if [[ -z \"$_py\" ]]; then echo \"No python interpreter found in container\" >&2; exit 1; fi; \
\"$_py\" - <<'PY'
import rdkit
import dask
import cudf
import cuml
import cupy as cp
print('rapids-smoke-imports-ok')
print('cupy-device-count:', cp.cuda.runtime.getDeviceCount())
PY"

echo "[minimal] single-seed reduced GuacaMol run"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v "${REPO}:${CONTAINER_REPO}" \
  "${RAPIDS_IMAGE}" \
  bash -lc "source /etc/profile.d/conda.sh 2>/dev/null || true; \
_py=''; \
if command -v conda >/dev/null 2>&1; then \
  _req='${CONDA_ENV_NAME}'; _env_path=''; \
  if [[ -n \"$_req\" ]]; then \
    _env_path=$(conda env list | awk -v req=\"$_req\" '!/^#/ && NF {n=$1; p=$NF; if (n==req || p==req) {print p; exit}}'); \
  fi; \
  if [[ -z \"$_env_path\" ]]; then \
    _env_path=$(conda env list | awk '!/^#/ && NF {n=$1; p=$NF; if (n==\"base\") next; print p; exit}'); \
  fi; \
  if [[ -n \"$_env_path\" && -x \"$_env_path/bin/python\" ]]; then \
    _py=\"$_env_path/bin/python\"; \
    echo \"[conda] using python: $_py\"; \
  fi; \
fi; \
if [[ -z \"$_py\" ]]; then _py=$(command -v python || true); fi; \
if [[ -z \"$_py\" ]]; then echo \"No python interpreter found in container\" >&2; exit 1; fi; \
cd ${CONTAINER_REPO} && \"$_py\" benchmarks/Guacamol/evolve_vs_molscore_benchmark.py \
    --objective-backend molscore \
    --benchmark GuacaMol \
    --seed 0 \
    --budget 100 \
    --population-size 64 \
    --batch-size 16 \
    --max-generations 5 \
    --skip-random-baseline \
    --output-dir ${MINI_OUTPUT_DIR}"

echo "[done] RAPIDS container quick validation complete"
