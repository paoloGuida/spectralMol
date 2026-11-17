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
TEST_ROOT=${TEST_ROOT:-/ibex/user/${USER}/container-testing/rapids}
TEST_SENTINEL=${TEST_SENTINEL:-.env_write_test}
PYTHON_BIN=${PYTHON_BIN:-$PWD/env/bin/python}

mkdir -p "${REPO}/logs" "${TEST_ROOT}/logs"
cd "${REPO}"
PYTHON_BIN=${PYTHON_BIN:-$PWD/env/bin/python}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[preflight] nvidia-smi"
nvidia-smi || true

echo "[preflight] workspace mount write test"
id
touch "${REPO}/${TEST_SENTINEL}"
ls -l "${REPO}/${TEST_SENTINEL}"
rm -f "${REPO}/${TEST_SENTINEL}"

echo "[smoke] local env RAPIDS import test"
"${PYTHON_BIN}" -c "import rdkit; import dask; import cudf; import cuml; print('local-conda-rapids-smoke-ok')"

echo "[done] Local conda RAPIDS smoke validation complete"
