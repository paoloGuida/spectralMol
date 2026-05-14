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
CONTAINER_REPO=${CONTAINER_REPO:-/workspace}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-saturn-runtime}
SATURN_REPO_ROOT=${SATURN_REPO_ROOT:-/opt/saturn_core}
QUICKVINA_BINARY=${QUICKVINA_BINARY:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1}
RECEPTOR_FILE=${RECEPTOR_FILE:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/7uvu-2-monomers-pdbfixer.pdbqt}
REFERENCE_LIGAND_FILE=${REFERENCE_LIGAND_FILE:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/7uvu-reference.pdb}
SEED_SMILES_FILE=${SEED_SMILES_FILE:-${REPO}/chembl24_canon_train.smiles}
ORACLE_TEMPLATE=${ORACLE_TEMPLATE:-${REPO}/benchmarks/Saturn/table2_r_sa_qed_oracle_template.json}
PODMAN_ROOT=${PODMAN_ROOT:-/ibex/user/${USER}/podman_images}
TEST_ROOT=${TEST_ROOT:-/ibex/user/${USER}/container-testing/saturn}
MINI_OUTPUT_DIR=${MINI_OUTPUT_DIR:-${TEST_ROOT}/data/minimal_test}

export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/${UID}}

mkdir -p "${REPO}/logs" "${TEST_ROOT}/logs" "${MINI_OUTPUT_DIR}" "/run/user/${UID}/bus" "${PODMAN_ROOT}"
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

echo "[smoke] python imports + CuPy GPU + QuickVina path"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v "${REPO}:${CONTAINER_REPO}" \
  "${SATURN_IMAGE}" \
  bash -lc "source /etc/profile.d/conda.sh 2>/dev/null || true; \
if command -v conda >/dev/null 2>&1; then \
  _req='${CONDA_ENV_NAME}'; _env=''; \
  if [[ -n \"$_req\" ]]; then \
    if conda env list | awk -v req=\"$_req\" '!/^#/ && NF {if ($1==req || $NF==req) found=1} END {exit found?0:1}'; then _env=\"$_req\"; fi; \
  fi; \
  if [[ -z \"$_env\" ]]; then \
    _env=$(conda env list | awk '!/^#/ && NF {n=$1; p=$NF; if (n==\"base\") next; if (substr(n,1,1)==\"/\") {print p; exit} print n; exit}'); \
  fi; \
  if [[ -n \"$_env\" ]]; then echo \"[conda] activating $_env\"; conda activate \"$_env\" 2>/dev/null || true; fi; \
fi; \
python - <<'PY'
import os
from pathlib import Path
import rdkit
import openbabel
import selfies
import molscore
import cupy as cp
print('saturn-smoke-imports-ok')
print('cupy-device:', cp.cuda.runtime.getDeviceCount())
qv = Path('${QUICKVINA_BINARY}')
print('quickvina-exists:', qv.exists())
if not qv.exists():
    raise SystemExit('QuickVina binary missing at expected path')
PY"

echo "[minimal] scalar-only reduced run"
podman --root="${PODMAN_ROOT}" run --rm \
  -e NVIDIA_VISIBLE_DEVICES='' \
  --device=nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v "${REPO}:${CONTAINER_REPO}" \
  "${SATURN_IMAGE}" \
  bash -lc "source /etc/profile.d/conda.sh 2>/dev/null || true; \
if command -v conda >/dev/null 2>&1; then \
  _req='${CONDA_ENV_NAME}'; _env=''; \
  if [[ -n \"$_req\" ]]; then \
    if conda env list | awk -v req=\"$_req\" '!/^#/ && NF {if ($1==req || $NF==req) found=1} END {exit found?0:1}'; then _env=\"$_req\"; fi; \
  fi; \
  if [[ -z \"$_env\" ]]; then \
    _env=$(conda env list | awk '!/^#/ && NF {n=$1; p=$NF; if (n==\"base\") next; if (substr(n,1,1)==\"/\") {print p; exit} print n; exit}'); \
  fi; \
  if [[ -n \"$_env\" ]]; then echo \"[conda] activating $_env\"; conda activate \"$_env\" 2>/dev/null || true; fi; \
fi; \
cd ${CONTAINER_REPO}/benchmarks/Saturn && python compare_scalar_vs_nsga2_saturn.py \
    --seeds 0 \
    --budgets 50 \
    --population-size 8 \
    --batch-size 8 \
    --max-generations 2 \
    --top-k 10 \
    --tournament-k 4 \
    --skip-nsga2 \
    --seed-smiles-file ${SEED_SMILES_FILE} \
    --seed-pool-size 64 \
    --oracle-template ${ORACLE_TEMPLATE} \
    --saturn-repo-root ${SATURN_REPO_ROOT} \
    --quickvina-binary ${QUICKVINA_BINARY} \
    --receptor-file ${RECEPTOR_FILE} \
    --reference-ligand-file ${REFERENCE_LIGAND_FILE} \
    --output-dir ${MINI_OUTPUT_DIR}"

echo "[done] Saturn container quick validation complete"
