#!/usr/bin/env bash
#SBATCH --job-name=guacamol_graphga_spectralmol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# GuacaMol comparison for local or Slurm execution:
#   - SpectralMol uses the Fourier-genotype local evolution path.
#   - GraphGA uses MolScore_examples/GraphGA/molscore_GB_GA.py.
#   - benchmark_compare_models.py writes one shared per-seed initial SMILES file,
#     then both models consume that same file.
#
# Typical submission:
#   sbatch --export=ALL,EXAMPLES_ROOT=/path/to/MolScore_examples sbatch_guacamol_graphga_vs_spectralmol_slurm.sh
#
# Useful overrides:
#   SEEDS=7,8,9 BUDGET=10000 GENERATIONS=50 POPULATION_SIZE=256 BATCH_SIZE=64
#   SEED_SMILES_FILE=/path/to/chembl24_canon_train.smiles
#   OUTPUT_DIR=/path/to/results/guacamol_graphga_vs_spectralmol
#   CONDA_ENV=molscore

info() { echo "[info] $*"; }
warn() { echo "[warn] $*" >&2; }
err() { echo "[error] $*" >&2; }

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

quote_command() {
  local arg
  for arg in "$@"; do
    printf "%q " "${arg}"
  done
  printf "\n"
}

load_conda_if_needed() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    local conda_sh=""
    for conda_sh in \
      "/home/${USER:-}/miniforge/etc/profile.d/conda.sh" \
      "/home/${USER:-}/miniconda3/etc/profile.d/conda.sh" \
      "/home/${USER:-}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/anaconda3/etc/profile.d/conda.sh"; do
      if [[ -f "${conda_sh}" ]]; then
        # shellcheck source=/dev/null
        source "${conda_sh}"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    err "conda was not found. Set PYTHON_BIN=/path/to/python or load/activate conda before sbatch."
    exit 1
  fi

  if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
    conda activate "${CONDA_ENV_PREFIX}"
  else
    conda activate "${CONDA_ENV:-molscore}"
  fi
  PYTHON_BIN="$(command -v python)"
}

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 2>/dev/null || true)}"

looks_like_spectralmol_root() {
  local candidate="$1"
  [[ -f "${candidate}/core/config.py" && -f "${candidate}/benchmarks/Guacamol/benchmark_compare_models.py" ]]
}

resolve_repo_root() {
  local base candidate

  if [[ -n "${REPO_ROOT:-}" ]]; then
    if looks_like_spectralmol_root "${REPO_ROOT}"; then
      printf "%s\n" "${REPO_ROOT}"
      return 0
    fi
    if looks_like_spectralmol_root "${REPO_ROOT}/spectralMol"; then
      printf "%s\n" "${REPO_ROOT}/spectralMol"
      return 0
    fi
    err "REPO_ROOT is set but does not contain the spectralMol package: ${REPO_ROOT}"
    return 1
  fi

  for base in "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "${SCRIPT_DIR}"; do
    [[ -n "${base}" ]] || continue
    for candidate in \
      "${base}" \
      "${base}/spectralMol" \
      "${base}/SpectralMol/spectralMol" \
      "${base}/../spectralMol" \
      "${base}/../SpectralMol/spectralMol"; do
      if looks_like_spectralmol_root "${candidate}"; then
        cd "${candidate}"
        pwd
        return 0
      fi
    done
  done

  return 1
}

REPO_ROOT="$(resolve_repo_root || true)"

if type module >/dev/null 2>&1; then
  if [[ -n "${MODULES:-}" ]]; then
    module purge || true
    for module_name in ${MODULES}; do
      module load "${module_name}"
    done
  fi
fi

if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/core/config.py" || ! -f "${REPO_ROOT}/benchmarks/Guacamol/benchmark_compare_models.py" ]]; then
  err "REPO_ROOT does not look like the spectralMol package root: ${REPO_ROOT}"
  err "Submit from the directory containing spectralMol/, or set REPO_ROOT=/path/to/spectralMol."
  exit 1
fi

if [[ -z "${SEED_SMILES_FILE:-}" ]]; then
  SEED_SMILES_FILE="${REPO_ROOT}/../reproducibility/manuscript_2026/inputs/guacamol/shared_initial_population_seed_7.smi"
fi

if [[ -z "${EXAMPLES_ROOT:-}" ]]; then
  EXAMPLES_ROOT="$(first_existing_dir \
    "${REPO_ROOT}/MolScore_examples" \
    "${REPO_ROOT}/../MolScore_examples" \
    "${REPO_ROOT}/../../MolScore_examples" \
    || true)"
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/../reproducibility_runs/guacamol_graphga_vs_spectralmol"
fi

BENCHMARK="${BENCHMARK:-GuacaMol}"
CUSTOM_BENCHMARK="${CUSTOM_BENCHMARK:-}"
INCLUDE_CSV="${INCLUDE_CSV:-}"
EXCLUDE_CSV="${EXCLUDE_CSV:-}"
TASK_INDEXES_CSV="${TASK_INDEXES_CSV:-}"
BUDGET="${BUDGET:-10000}"
GENERATIONS="${GENERATIONS:-50}"
POPULATION_SIZE="${POPULATION_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS="${SEEDS:-7,8,9}"
MODELS="${MODELS:-spectralmol,graphga}"
SEED_POOL_SIZE="${SEED_POOL_SIZE:-2000}"
FREQUENCY_MODE="${FREQUENCY_MODE:-full-spectrum}"
SPECTRAL_L="${SPECTRAL_L:-32}"
SPECTRAL_K="${SPECTRAL_K:-16}"
SPECTRAL_D="${SPECTRAL_D:-32}"
SPECTRAL_DECODE_ATTEMPTS="${SPECTRAL_DECODE_ATTEMPTS:-8}"
SPECTRAL_TASK_PROFILE_MODE="${SPECTRAL_TASK_PROFILE_MODE:-manual}"
SPECTRAL_TASK_PROFILE="${SPECTRAL_TASK_PROFILE:-manual}"
MOLSCORE_SPECTRAL_ENABLE_SEED_MACROS="${MOLSCORE_SPECTRAL_ENABLE_SEED_MACROS:-1}"
MOLSCORE_SPECTRAL_SEED_MACRO_MAX="${MOLSCORE_SPECTRAL_SEED_MACRO_MAX:-384}"
MOLSCORE_SPECTRAL_SEED_MACRO_MIN_N="${MOLSCORE_SPECTRAL_SEED_MACRO_MIN_N:-3}"
MOLSCORE_SPECTRAL_SEED_MACRO_MAX_N="${MOLSCORE_SPECTRAL_SEED_MACRO_MAX_N:-12}"
MOLSCORE_SPECTRAL_SEED_MACRO_MIN_ATOMS="${MOLSCORE_SPECTRAL_SEED_MACRO_MIN_ATOMS:-3}"
MOLSCORE_SPECTRAL_SEED_MACRO_MIN_FREQUENCY="${MOLSCORE_SPECTRAL_SEED_MACRO_MIN_FREQUENCY:-1}"
MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS="${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS:-1}"
MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS="${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS:-1}"
MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX="${MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX:-16}"
MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS="${MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS:-1}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX:-256}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N:-2}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N:-18}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS:-2}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY:-1}"
MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT="${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT:-12}"
MOLSCORE_SPECTRAL_TASK_AWARE_DECODE="${MOLSCORE_SPECTRAL_TASK_AWARE_DECODE:-1}"
MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES="${MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES:-1}"
MOLSCORE_SPECTRAL_ELITE_MACRO_REFRESH_EVERY="${MOLSCORE_SPECTRAL_ELITE_MACRO_REFRESH_EVERY:-25}"
MOLSCORE_SPECTRAL_ELITE_MACRO_TOP_N="${MOLSCORE_SPECTRAL_ELITE_MACRO_TOP_N:-96}"
MOLSCORE_SPECTRAL_ELITE_MACRO_SEED_KEEP="${MOLSCORE_SPECTRAL_ELITE_MACRO_SEED_KEEP:-256}"
MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION="${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION:-0.35}"
MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION="${MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION:-0.10}"
MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS="${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS:-2}"
MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB="${MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB:-0.20}"
MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB="${MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB:-0.05}"
MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB="${MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB:-0.25}"
MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION="${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION:-0.75}"
MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION="${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION:-0.03}"
MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND="${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND:-0.80}"
MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION="${MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION:-0.25}"
MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION="${MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION:-0.15}"
MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE="${MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE:-0.40}"
MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT="${MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT:-4}"
MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH="${MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH:-1}"
SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION:-0}"
SPECTRAL_BRICS_CROSSOVER_FRACTION="${SPECTRAL_BRICS_CROSSOVER_FRACTION:-0}"
SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION:-0}"
GRAPHGA_N_JOBS="${GRAPHGA_N_JOBS:-1}"
GRAPHGA_PATIENCE="${GRAPHGA_PATIENCE:-20}"
MOLSCORE_PARALLEL_JOBS="${MOLSCORE_PARALLEL_JOBS:-2}"
EXECUTOR="${EXECUTOR:-thread}"
DATAFRAME_BACKEND="${DATAFRAME_BACKEND:-pandas}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SEED_SMILES_FILE}" || ! -f "${SEED_SMILES_FILE}" ]]; then
  err "seed SMILES file not found."
  err "Set SEED_SMILES_FILE=/path/to/chembl24_canon_train.smiles"
  exit 1
fi

if [[ "${MODELS}" == *graphga* && ( -z "${EXAMPLES_ROOT}" || ! -d "${EXAMPLES_ROOT}" ) ]]; then
  err "MolScore_examples checkout not found."
  err "Set EXAMPLES_ROOT=/path/to/MolScore_examples"
  exit 1
fi

if [[ "${MODELS}" == *graphga* && ! -f "${EXAMPLES_ROOT}/GraphGA/molscore_GB_GA.py" ]]; then
  err "GraphGA script not found: ${EXAMPLES_ROOT}/GraphGA/molscore_GB_GA.py"
  exit 1
fi

load_conda_if_needed

mkdir -p "${OUTPUT_DIR}"
RUN_ROOT="${OUTPUT_DIR}/launcher_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_ROOT}"
ACTIVE_MODEL_SPEC_FILE="${RUN_ROOT}/model_specs_graphga_spectralmol.json"
SETTINGS_TXT="${RUN_ROOT}/run_settings.txt"

cat > "${ACTIVE_MODEL_SPEC_FILE}" <<JSON
{
  "spectralmol": {
    "type": "builtin_local_evolution",
    "enabled": true,
    "description": "SpectralMol Fourier-genotype local evolution runner."
  },
  "graphga": {
    "type": "command",
    "enabled": true,
    "description": "MolScore_examples GraphGA runner.",
    "command": "{python} {repo_root}/benchmarks/Guacamol/run_graphga_example_wrapper.py --graphga-script {examples_root}/GraphGA/molscore_GB_GA.py --molscore {molscore_target} --smiles_file {shared_init_file} --starting_population_file {shared_init_file} --seed {seed} --budget {budget} --output_dir {model_output_dir} --population_size {population_size} --offspring_size {population_size} --generations {generations} --n_jobs {graphga_n_jobs} --patience ${GRAPHGA_PATIENCE} {include_args} {exclude_args} {task_indexes_args}"
  }
}
JSON

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MOLSCORE_PARALLEL_JOBS
export MOLSCORE_LOCAL_EVO_GENERATOR="spectral"
export MOLSCORE_FREQUENCY_MODE="${FREQUENCY_MODE}"
export MOLSCORE_SPECTRAL_L="${SPECTRAL_L}"
export MOLSCORE_SPECTRAL_K="${SPECTRAL_K}"
export MOLSCORE_SPECTRAL_D="${SPECTRAL_D}"
export MOLSCORE_DECODE_ATTEMPTS="${SPECTRAL_DECODE_ATTEMPTS}"
export SPECTRAL_TASK_PROFILE_MODE
export SPECTRAL_TASK_PROFILE
export MOLSCORE_SPECTRAL_ENABLE_SEED_MACROS
export MOLSCORE_SPECTRAL_SEED_MACRO_MAX
export MOLSCORE_SPECTRAL_SEED_MACRO_MIN_N
export MOLSCORE_SPECTRAL_SEED_MACRO_MAX_N
export MOLSCORE_SPECTRAL_SEED_MACRO_MIN_ATOMS
export MOLSCORE_SPECTRAL_SEED_MACRO_MIN_FREQUENCY
export MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS
export MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS
export MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX
export MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_ATOMS
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_FREQUENCY
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT
export MOLSCORE_SPECTRAL_TASK_AWARE_DECODE
export MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES
export MOLSCORE_SPECTRAL_ELITE_MACRO_REFRESH_EVERY
export MOLSCORE_SPECTRAL_ELITE_MACRO_TOP_N
export MOLSCORE_SPECTRAL_ELITE_MACRO_SEED_KEEP
export MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION
export MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION
export MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS
export MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB
export MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB
export MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION
export MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND
export MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION
export MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION
export MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE
export MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT
export MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH
export MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION="${SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION}"
export MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION="${SPECTRAL_BRICS_CROSSOVER_FRACTION}"
export MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION="${SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/benchmarks/Guacamol/benchmark_compare_models.py"
  "--benchmark" "${BENCHMARK}"
  "--budget" "${BUDGET}"
  "--generations" "${GENERATIONS}"
  "--population-size" "${POPULATION_SIZE}"
  "--batch-size" "${BATCH_SIZE}"
  "--seeds" "${SEEDS}"
  "--models" "${MODELS}"
  "--model-spec-file" "${ACTIVE_MODEL_SPEC_FILE}"
  "--examples-root" "${EXAMPLES_ROOT}"
  "--python-bin" "${PYTHON_BIN}"
  "--seed-smiles-file" "${SEED_SMILES_FILE}"
  "--seed-pool-size" "${SEED_POOL_SIZE}"
  "--local-evo-generator" "spectral"
  "--local-evo-frequency-mode" "${FREQUENCY_MODE}"
  "--equal-initial-population"
  "--graphga-n-jobs" "${GRAPHGA_N_JOBS}"
  "--executor" "${EXECUTOR}"
  "--dataframe-backend" "${DATAFRAME_BACKEND}"
  "--no-bootstrap-scoring-envs"
  "--output-dir" "${OUTPUT_DIR}"
)

if [[ -n "${TASK_INDEXES_CSV}" ]]; then
  CMD+=("--task-indexes" "${TASK_INDEXES_CSV}")
elif [[ -n "${INCLUDE_CSV}" ]]; then
  CMD+=("--include" "${INCLUDE_CSV}")
fi
if [[ -n "${CUSTOM_BENCHMARK}" ]]; then
  CMD+=("--custom-benchmark" "${CUSTOM_BENCHMARK}")
fi
if [[ -n "${EXCLUDE_CSV}" ]]; then
  CMD+=("--exclude" "${EXCLUDE_CSV}")
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  CMD+=("--continue-on-error")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

{
  echo "Job ID:                 ${SLURM_JOB_ID:-N/A}"
  echo "Node:                   $(hostname)"
  echo "Repository:             ${REPO_ROOT}"
  echo "Python:                 ${PYTHON_BIN}"
  echo "Benchmark:              ${BENCHMARK}"
  echo "Custom benchmark:       ${CUSTOM_BENCHMARK:-<none>}"
  echo "Models:                 ${MODELS}"
  echo "Task indexes:           ${TASK_INDEXES_CSV:-<none>}"
  echo "Seeds:                  ${SEEDS}"
  echo "Budget:                 ${BUDGET}"
  echo "Generations:            ${GENERATIONS}"
  echo "Population size:        ${POPULATION_SIZE}"
  echo "Batch size:             ${BATCH_SIZE}"
  echo "Seed SMILES file:       ${SEED_SMILES_FILE}"
  echo "MolScore_examples:      ${EXAMPLES_ROOT}"
  echo "Output directory:       ${OUTPUT_DIR}"
  echo "Launcher directory:     ${RUN_ROOT}"
  echo "Active model spec:      ${ACTIVE_MODEL_SPEC_FILE}"
  echo "Frequency mode:         ${FREQUENCY_MODE}"
  echo "Spectral L/K/D:         ${SPECTRAL_L}/${SPECTRAL_K}/${SPECTRAL_D}"
  echo "Spectral decode tries:  ${SPECTRAL_DECODE_ATTEMPTS}"
  echo "Task profile mode:      ${SPECTRAL_TASK_PROFILE_MODE}"
  echo "Task profile:           ${SPECTRAL_TASK_PROFILE}"
  echo "Seed macro enabled:     ${MOLSCORE_SPECTRAL_ENABLE_SEED_MACROS}"
  echo "Seed macro max:         ${MOLSCORE_SPECTRAL_SEED_MACRO_MAX}"
  echo "Seed macro n range:     ${MOLSCORE_SPECTRAL_SEED_MACRO_MIN_N}-${MOLSCORE_SPECTRAL_SEED_MACRO_MAX_N}"
  echo "Task target macros:     ${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_MACROS} max ${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX}"
  echo "Task target full:       ${MOLSCORE_SPECTRAL_ENABLE_TASK_TARGET_FULL_MACROS} max ${MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX}"
  echo "Std GuacaMol targets:   ${MOLSCORE_SPECTRAL_USE_STANDARD_GUACAMOL_TARGETS}"
  echo "Task target n range:    ${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MIN_N}-${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N}"
  echo "Task target weight:     ${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT}"
  echo "Task-aware decode:      ${MOLSCORE_SPECTRAL_TASK_AWARE_DECODE}"
  echo "Decode candidates:      ${MOLSCORE_SPECTRAL_TASK_AWARE_DECODE_CANDIDATES}"
  echo "Elite macro refresh:    every ${MOLSCORE_SPECTRAL_ELITE_MACRO_REFRESH_EVERY} gen, top ${MOLSCORE_SPECTRAL_ELITE_MACRO_TOP_N}"
  echo "Spectral phenotype pf:  ${SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION}"
  echo "Spectral BRICS cross:   ${SPECTRAL_BRICS_CROSSOVER_FRACTION}"
  echo "Spectral BRICS replace: ${SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION}"
  echo "Theta crossover prob:   ${MOLSCORE_SPECTRAL_THETA_CROSSOVER_PROBABILITY:-<default>}"
  echo "Theta local fraction:   ${MOLSCORE_SPECTRAL_THETA_LOCAL_SEARCH_FRACTION:-<default>}"
  echo "Theta local top frac:   ${MOLSCORE_SPECTRAL_THETA_LOCAL_TOP_FRACTION:-<default>}"
  echo "Theta local sigma:      ${MOLSCORE_SPECTRAL_THETA_LOCAL_SIGMA_SCALE:-<default>}"
  echo "Theta sigma schedule:   ${MOLSCORE_SPECTRAL_THETA_LOCAL_SIGMA_SCALES:-<default>}"
  echo "Theta noise schedule:   ${MOLSCORE_SPECTRAL_THETA_LOCAL_PARAM_NOISE_SCALES:-<default>}"
  echo "Theta reset schedule:   ${MOLSCORE_SPECTRAL_THETA_LOCAL_ROW_RESET_SCALES:-<default>}"
  echo "Theta step schedule:    ${MOLSCORE_SPECTRAL_THETA_LOCAL_MUTATION_STEP_SCHEDULE:-<default>}"
  echo "Theta token local:      ${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_FRACTION}"
  echo "Theta token child:      ${MOLSCORE_SPECTRAL_THETA_CHILD_TOKEN_MUTATION_FRACTION}"
  echo "Theta token max edits:  ${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_MAX_EDITS}"
  echo "Theta token ins/del:    ${MOLSCORE_SPECTRAL_THETA_TOKEN_INSERT_PROB}/${MOLSCORE_SPECTRAL_THETA_TOKEN_DELETE_PROB}"
  echo "Theta token macro p:    ${MOLSCORE_SPECTRAL_THETA_TOKEN_MACRO_INSERT_PROB}"
  echo "Theta target macro p:   ${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION}"
  echo "Theta target jump p:    ${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION}"
  echo "Theta token blend:      ${MOLSCORE_SPECTRAL_THETA_TOKEN_MUTATION_BLEND}"
  echo "Theta blend crossover:  ${MOLSCORE_SPECTRAL_THETA_BLEND_CROSSOVER_FRACTION}"
  echo "Theta differential:     ${MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_FRACTION} scale ${MOLSCORE_SPECTRAL_THETA_DIFFERENTIAL_SCALE}"
  echo "Elite macro weight:     ${MOLSCORE_SPECTRAL_ELITE_MACRO_WEIGHT}"
  echo "Reencode after vocab:   ${MOLSCORE_SPECTRAL_REENCODE_POPULATION_AFTER_VOCAB_REFRESH}"
  echo "GraphGA n_jobs:         ${GRAPHGA_N_JOBS}"
  echo "GraphGA patience:       ${GRAPHGA_PATIENCE}"
  echo "Model parallel jobs:    ${MOLSCORE_PARALLEL_JOBS}"
  echo
  echo "Command:"
  quote_command "${CMD[@]}"
} | tee "${SETTINGS_TXT}"

info "Starting GuacaMol GraphGA vs SpectralMol comparison."
"${CMD[@]}"
info "Finished. Launcher metadata is in ${RUN_ROOT}"
