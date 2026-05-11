#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/ibex/user/${USER}/molevoDrugDiscovery}
IMAGE=${IMAGE:-${REPO}/molscore/outputs/containers/molevo-saturn-cuda12.2-quickvina2.sif}
SEED_SMILES_FILE=${SEED_SMILES_FILE:-${REPO}/chembl24_canon_train.smiles}
SATURN_REPO_ROOT=${SATURN_REPO_ROOT:-/opt/saturn_core}
QUICKVINA_BINARY=${QUICKVINA_BINARY:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1}
RECEPTOR_FILE=${RECEPTOR_FILE:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/7uvu-2-monomers-pdbfixer.pdbqt}
REFERENCE_LIGAND_FILE=${REFERENCE_LIGAND_FILE:-${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/7uvu-reference.pdb}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/molscore/outputs/saturn_container_smoke}
RUNTIME=${RUNTIME:-singularity}

mkdir -p "${REPO}/logs" "${OUTPUT_DIR}"
cd "${REPO}"

"${RUNTIME}" exec --nv \
    --bind "${REPO}:/workspace" \
    "${IMAGE}" \
    bash -lc "cd /workspace/benchmarks/Saturn && python compare_scalar_vs_nsga2_saturn.py \
        --seeds 0 \
        --budgets 40 \
        --population-size 8 \
        --batch-size 8 \
        --max-generations 2 \
        --skip-nsga2 \
        --seed-smiles-file ${SEED_SMILES_FILE} \
        --seed-pool-size 64 \
        --saturn-repo-root ${SATURN_REPO_ROOT} \
        --quickvina-binary ${QUICKVINA_BINARY} \
        --receptor-file ${RECEPTOR_FILE} \
        --reference-ligand-file ${REFERENCE_LIGAND_FILE} \
        --output-dir ${OUTPUT_DIR}"