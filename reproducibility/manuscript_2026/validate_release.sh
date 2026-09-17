#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

required=(
  spectralMol/benchmarks/Guacamol/model_specs.json
  spectralMol/benchmarks/Saturn/table2_r_sa_qed_oracle_template.json
  reproducibility/manuscript_2026/inputs/guacamol/shared_initial_population_seed_7.smi
  reproducibility/manuscript_2026/inputs/guacamol/chembl.filtered.smi.gz
  reproducibility/manuscript_2026/inputs/saturn/seed_sets/seed_0.smi
  reproducibility/manuscript_2026/inputs/saturn/seed_sets/seed_9.smi
  reproducibility/manuscript_2026/inputs/saturn/docking/7uvu-reference.pdb
  reproducibility/manuscript_2026/inputs/saturn/docking/7uvu-2-monomers-pdbfixer.pdbqt
  reproducibility/manuscript_2026/results/guacamol/guacamol_per_seed_aggregate.tsv
  reproducibility/manuscript_2026/results/saturn/table8_spectralmol.tsv
)
for path in "${required[@]}"; do
  test -s "${path}" || { echo "missing required file: ${path}" >&2; exit 1; }
done

bash -n \
  run_reproducibility_profile_ibex.sh \
  sbatch_guacamol_spectralmol_graphga_10seed_stats_ibex.sh \
  sbatch_guacamol_frequency_ablation_ibex.sh \
  sbatch_saturn_theta_nsga2_ibex.sh \
  reproducibility_backups/unified_guacamol_saturn_20260817/commands/*.sh

python_bin="${SPECTRALMOL_PYTHON:-python3}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/spectralmol_pycache}"
"${python_bin}" -m py_compile \
  spectralMol/core/*.py \
  spectralMol/benchmarks/Guacamol/*.py \
  spectralMol/benchmarks/Saturn/*.py

gzip -t reproducibility/manuscript_2026/inputs/guacamol/chembl.filtered.smi.gz
gzip -t reproducibility/manuscript_2026/results/saturn/all_molecules_standardized.tsv.gz

grep -q '^export MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION=0$' \
  reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_mpo_v3.env
grep -q '^export MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION=0$' \
  reproducibility_backups/unified_guacamol_saturn_20260817/settings/guacamol_mpo_v3.env
grep -q -- '--nsga2-genotype theta' sbatch_saturn_theta_nsga2_ibex.sh

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c reproducibility/manuscript_2026/provenance/SHA256SUMS
else
  shasum -a 256 -c reproducibility/manuscript_2026/provenance/SHA256SUMS
fi

echo "Manuscript reproducibility package validation passed."
