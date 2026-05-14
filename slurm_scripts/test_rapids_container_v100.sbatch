#!/bin/bash
#SBATCH --job-name=test-rapids-ctr-v100
#SBATCH --partition=batch
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
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images}
TEST_ROOT=${TEST_ROOT:-/ibex/user/${USER}/container-testing/rapids}

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${REPO}/logs" "${TEST_ROOT}/logs" "/run/user/${UID}/bus" "${PODMAN_ROOT}"
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

echo "[smoke] RAPIDS import test"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${RAPIDS_IMAGE}" \
  python -c "import rdkit; import dask; import cudf; import cuml; print('wolfi-conda-rapids-smoke-ok')"

echo "[done] RAPIDS container smoke validation complete"
