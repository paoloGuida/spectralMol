#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${RUNTIME:-podman}
IMAGE_REF=${IMAGE_REF:-molevo-saturn:cuda12.2-quickvina2}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONTAINER_REPO=${CONTAINER_REPO:-/workspace}
CONTAINER_SATURN_ROOT=${CONTAINER_SATURN_ROOT:-/opt/saturn_core}
QUICKVINA_BIN=${QUICKVINA_BIN:-${CONTAINER_SATURN_ROOT}/experimental_reproduction/synthesizability/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1}
PODMAN_STORAGE_DRIVER=${PODMAN_STORAGE_DRIVER:-vfs}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images_${PODMAN_STORAGE_DRIVER}}
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${XDG_RUNTIME_DIR}/bus" "${PODMAN_ROOT}"

export XDG_RUNTIME_DIR

if [[ "${RUNTIME}" == "podman" ]]; then
    podman --root="${PODMAN_ROOT}" run --rm \
        --storage-driver="${PODMAN_STORAGE_DRIVER}" \
        -e NVIDIA_VISIBLE_DEVICES='' \
        -v "${REPO_ROOT}:${CONTAINER_REPO}:Z" \
        -w "${CONTAINER_REPO}" \
        --device=nvidia.com/gpu=all \
        --security-opt=label=disable \
        "${IMAGE_REF}" \
        bash -lc "python -c 'import rdkit, openbabel, selfies, molscore; print(\"python-imports-ok\")' && ${QUICKVINA_BIN} --help >/tmp/quickvina_help.txt 2>&1 || true && test -f /tmp/quickvina_help.txt && echo quickvina-help-ran"
    exit 0
fi

if [[ "${RUNTIME}" == "singularity" || "${RUNTIME}" == "apptainer" ]]; then
    "${RUNTIME}" exec --nv \
        --bind "${REPO_ROOT}:${CONTAINER_REPO}" \
        "${IMAGE_REF}" \
        bash -lc "cd ${CONTAINER_REPO} && python -c 'import rdkit, openbabel, selfies, molscore; print(\"python-imports-ok\")' && ${QUICKVINA_BIN} --help >/tmp/quickvina_help.txt 2>&1 || true && test -f /tmp/quickvina_help.txt && echo quickvina-help-ran"
    exit 0
fi

echo "Unsupported runtime: ${RUNTIME}" >&2
exit 1