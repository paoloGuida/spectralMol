#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE_SOURCE=${IMAGE_SOURCE:-}
IMAGE_NAME=${IMAGE_NAME:-molevo-saturn:cuda12.2-quickvina2}
ARCHIVE_PATH=${ARCHIVE_PATH:-${REPO_ROOT}/molscore/outputs/containers/molevo-saturn-cuda12.2-quickvina2.tar}
PODMAN_STORAGE_DRIVER=${PODMAN_STORAGE_DRIVER:-vfs}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images_${PODMAN_STORAGE_DRIVER}}
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

if [[ -z "${IMAGE_SOURCE}" ]]; then
    echo "Set IMAGE_SOURCE to your pushed image, e.g.:" >&2
    echo "  IMAGE_SOURCE=docker.io/<dockerhub-user>/molevo-saturn:cuda12.2-quickvina2 bash slurm_scripts/pull_saturn_container_podman.sh" >&2
    exit 2
fi

mkdir -p "${XDG_RUNTIME_DIR}/bus" "${PODMAN_ROOT}" "$(dirname "${ARCHIVE_PATH}")"

export XDG_RUNTIME_DIR

echo "Pulling image: ${IMAGE_SOURCE}"
echo "Using podman root: ${PODMAN_ROOT}"
echo "Using storage driver: ${PODMAN_STORAGE_DRIVER}"

podman --root="${PODMAN_ROOT}" --storage-driver="${PODMAN_STORAGE_DRIVER}" pull "${IMAGE_SOURCE}"
podman --root="${PODMAN_ROOT}" --storage-driver="${PODMAN_STORAGE_DRIVER}" tag "${IMAGE_SOURCE}" "${IMAGE_NAME}"
podman --root="${PODMAN_ROOT}" --storage-driver="${PODMAN_STORAGE_DRIVER}" save -o "${ARCHIVE_PATH}" "${IMAGE_NAME}"

echo "Pulled image: ${IMAGE_SOURCE}"
echo "Tagged as: ${IMAGE_NAME}"
echo "Saved archive: ${ARCHIVE_PATH}"