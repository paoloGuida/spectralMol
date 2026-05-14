#!/bin/bash
#SBATCH --job-name=test-saturn-ctr-v100
#SBATCH --partition=batch
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
SATURN_IMAGE=${SATURN_IMAGE:-docker.io/dxbarradas/spectralmol-saturn:wolfi-quickvina-src}
QUICKVINA_BINARY=${QUICKVINA_BINARY:-/vina/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images}
TEST_ROOT=${TEST_ROOT:-/ibex/user/${USER}/container-testing/saturn}

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${REPO}/logs" "${TEST_ROOT}/logs" "/run/user/${UID}/bus" "${PODMAN_ROOT}"
cd "${REPO}"

if ! command -v podman >/dev/null 2>&1; then
  echo "Podman is not available on PATH." >&2
  exit 1
fi

echo "[preflight] nvidia-smi"
podman --root="${PODMAN_ROOT}" pull "${SATURN_IMAGE}"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${SATURN_IMAGE}" nvidia-smi || true

echo "[smoke] Saturn import test"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${SATURN_IMAGE}" \
  python -c "import rdkit; print('saturn-smoke-ok')"

echo "[smoke] QuickVina binary check"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${SATURN_IMAGE}" \
  bash -lc "test -x '${QUICKVINA_BINARY}' && echo quickvina-ok:${QUICKVINA_BINARY}"

echo "[done] Saturn container smoke validation complete"
