#!/usr/bin/env bash
#SBATCH --job-name=saturn_theta_nsga2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

# Manuscript-style SATURN comparison for SpectralMol:
# - NSGA-II optimizes docking, QED, and SA as separate objectives.
# - The genotype evolved by crossover/mutation is SpectralMol Fourier theta.

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SOURCE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${SOURCE_DIR}}"
USER_NAME="${USER:-$(id -un 2>/dev/null || printf user)}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"

set_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name+x}" || -z "${!name}" ]]; then
    export "${name}=${value}"
  fi
}

SEED_LIST_RAW="${SEED_LIST:-${SEED_LIST_CSV:-0,1,2,3,4,5,6,7,8,9}}"
SEED_LIST_RAW="${SEED_LIST_RAW//:/,}"
SEED_LIST_RAW="${SEED_LIST_RAW//;/,}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEED_LIST_RAW}"
if (( ARRAY_ID < 0 || ARRAY_ID >= ${#SEED_ARRAY[@]} )); then
  echo "[saturn-theta-nsga2] array id ${ARRAY_ID} maps outside seed list ${SEED_LIST_RAW}; exiting."
  exit 0
fi
SEED="${SEED_ARRAY[${ARRAY_ID}]}"
SEED="${SEED//[[:space:]]/}"

set_default PYTHON_BIN "/ibex/user/${USER_NAME}/conda-environments/molscore/bin/python"
set_default REPO_ROOT "${SCRIPT_DIR}/spectralMol"
set_default OUTPUT_ROOT "/ibex/scratch/${USER_NAME}/spectralMol/saturn_theta_nsga2"
set_default RUN_ID_PREFIX "saturn_theta_nsga2"
set_default SATURN_REPO_ROOT "/home/${USER_NAME}/molevoDrugDiscovery_2/Saturn/saturn_repo2"
set_default SATURN_ORACLE_TEMPLATE "/home/${USER_NAME}/molevoDrugDiscovery_2/Saturn/local_evolution/table2_r_sa_qed_oracle_template.json"
set_default SATURN_ORACLE_CONFIG_KEY "oracle"
set_default ASSETS_ROOT "/home/${USER_NAME}/molevoDrugDiscovery_2/Saturn/saturn_repo_clean/experimental_reproduction/synthesizability"
set_default QUICKVINA_BINARY "${SATURN_REPO_ROOT}/experimental_reproduction/synthesizability/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1"
set_default RECEPTOR_FILE "${ASSETS_ROOT}/7uvu-2-monomers-pdbfixer.pdbqt"
set_default REFERENCE_LIGAND_FILE "${ASSETS_ROOT}/7uvu-reference.pdb"
set_default PER_SEED_SEED_SMILES_DIR "/home/${USER_NAME}/molevoDrugDiscovery_2/Saturn/data/zinc250k/saturn1000_overperforming_init_seedsets_10seeds_qsa_diverse_v1"
set_default USE_PER_SEED_SEED_SMILES "1"
set_default SEED_SMILES_FILE ""
set_default SEED_POOL_SIZE "256"
set_default BUDGET "1000"
set_default POPULATION_SIZE "256"
set_default BATCH_SIZE "16"
set_default GENERATIONS "0"
set_default FREQUENCY_MODE "full-spectrum"
set_default SPECTRAL_L "48"
set_default SPECTRAL_K "24"
set_default SPECTRAL_D "32"
set_default SPECTRAL_DECODE_ATTEMPTS "16"
set_default MOLSCORE_DECODE_INCLUDE_GREEDY_NEAREST "0"
set_default TOP_K "100"
set_default TOURNAMENT_K "8"
set_default IMMIGRANT_FRACTION "0.02"
set_default PARENT_POOL_FRACTION "0.50"
set_default STAGNATION_PATIENCE "12"
set_default STAGNATION_MUTATION_BOOST "2"
set_default SATURN_MODULES "cuda/11.8"
set_default MOLSCORE_SATURN_OBABEL_BINARY "/ibex/user/${USER_NAME}/conda-environments/openbabel-cli/bin/obabel"
set_default MOLSCORE_SATURN_OPENCL_LIB_DIR "/lib64"
set_default MOLSCORE_SATURN_OPENCL_PRELOAD "1"
set_default SATURN_THETA_USE_RAW_NSGA_OBJECTIVES "1"
set_default SATURN_THETA_NSGA_DOCKING_OBJECTIVE_CAP "0.0"
set_default SATURN_THETA_DOCKING_FOCUS_FRACTION "0.95"
set_default SATURN_THETA_DOCKING_FOCUS_TOP_FRACTION "0.15"
set_default SATURN_THETA_DOCKING_ELITE_FRACTION "0.15"
set_default SATURN_THETA_DOCKING_FOCUS_MUTATION_STEPS "2"
set_default SATURN_THETA_DOCKING_FOCUS_TARGET_SAMPLE_FRACTION "0.95"
set_default SATURN_THETA_GENERATED_DOCKING_ELITE_FRACTION "0.35"
set_default SATURN_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING "-8.0"
set_default SATURN_THETA_DOCKING_FOCUS_SIGMA_SCALE "0.06"
set_default SATURN_THETA_DOCKING_FOCUS_PARAM_NOISE_SCALE "0.10"
set_default SATURN_THETA_DOCKING_FOCUS_ROW_RESET_SCALE "0.0"
set_default SATURN_THETA_DOCKING_FOCUS_TOKEN_MUTATION_FRACTION "0.35"
set_default SATURN_THETA_DOCKING_FOCUS_TOKEN_BLEND "0.35"
set_default SATURN_THETA_DOCKING_FOCUS_TOKEN_MACRO_INSERT_PROB "0.35"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_FRACTION "0.95"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_MAX_EDITS "1"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_INSERT_PROB "0.15"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_DELETE_PROB "0.0"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_MACRO_INSERT_PROB "0.20"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_BLEND "1.0"
set_default SATURN_THETA_TARGET_TOKEN_ANALOG_EXPAND_MACROS "0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION "0.92"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT "14"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_MAX_EDITS "1"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_PROB "0.85"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_DELETE_PROB "0.0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB "0.20"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_BLEND "1.0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS "0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION "1.0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT "16"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS "12.0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB "0.90"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS "[F],[Cl],[Br],[C],[O],[N],[S]"
set_default SATURN_THETA_INCLUDE_TASK_TARGET_ANCHORS "0"
set_default SATURN_THETA_TASK_TARGET_ANCHOR_LIMIT "0"
set_default SATURN_THETA_HIT_SITE_SCAN_FRACTION "0.35"
set_default SATURN_THETA_HIT_SITE_SCAN_MAX_EDITS "2"
set_default SATURN_THETA_HIT_SITE_SCAN_BLEND "1.0"
set_default SATURN_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY "0.0"
set_default SATURN_THETA_HIT_SITE_SCAN_TOKENS "[F],[Cl],[Br],[C],[=C],[N],[O],[S],[AMIDE],[BENZAMIDE],[PHENETHYL],[PHENETHYL_AMIDE],[PHENOXY_ETHYL],[BENZYL_ETHER]"
set_default SATURN_THETA_LATE_DEEP_SPIKE_START_FRACTION "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS ""
set_default SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS ""
set_default SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW "nan"
set_default SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED "0.0"
set_default SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA "99.0"
set_default MOLSCORE_SPECTRAL_STATIC_TARGET_SMILES ""
set_default SATURN_THETA_HIT_MICROJITTER_FRACTION "1.0"
set_default SATURN_THETA_HIT_MICROJITTER_SIGMA_MIN "0.001"
set_default SATURN_THETA_HIT_MICROJITTER_SIGMA_MAX "0.020"
set_default SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION "0.85"
set_default SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING "-8.0"
set_default SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_SA "99.0"
set_default SATURN_THETA_GENERATED_HIT_ANCHOR_MIN_QED "0.0"
set_default SATURN_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE "0.25"
set_default SATURN_THETA_SCAFFOLD_DIVERSE_ANCHORS "0"
set_default SATURN_THETA_ANCHOR_MAX_PER_SCAFFOLD "1"
set_default SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP "0"
set_default SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION "0.0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT "1"
set_default SATURN_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY "0"
set_default SATURN_THETA_PRIORITY_MIN_QED "0.0"
set_default SATURN_THETA_PRIORITY_MAX_SA "99.0"
set_default SATURN_THETA_PRIORITY_MIN_MW "0.0"
set_default SATURN_THETA_PRIORITY_MAX_MW "9999.0"
set_default SATURN_THETA_GENERATED_SITE_SCAN_MIN_QED "0.0"
set_default SATURN_THETA_GENERATED_SITE_SCAN_MAX_SA "99.0"
set_default SATURN_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS "0"
set_default SATURN_THETA_HIT_MICROJITTER_MAX_ATTEMPTS "0"
set_default SATURN_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS "0"
set_default SATURN_THETA_HIT_POOL_USE_CANDIDATE_BATCH "0"
set_default SATURN_THETA_CANDIDATE_POOL_MULTIPLIER "8"
set_default SATURN_THETA_PRESELECT_BY_CHEAP_PROXY "1"
set_default SATURN_THETA_PRESELECT_SIM_WEIGHT "3.0"
set_default SATURN_THETA_PRESELECT_QED_WEIGHT "0.5"
set_default SATURN_THETA_PRESELECT_SA_WEIGHT "0.8"
set_default SATURN_THETA_PRESELECT_MW_WEIGHT "0.2"
set_default SATURN_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT "0.0"
set_default SATURN_THETA_PRESELECT_QED_FLOOR "0.0"
set_default SATURN_THETA_PRESELECT_QED_FLOOR_WEIGHT "0.0"
set_default SATURN_THETA_PRESELECT_SA_CEILING "99.0"
set_default SATURN_THETA_PRESELECT_SA_CEILING_WEIGHT "0.0"
set_default SATURN_THETA_PRESELECT_BR_PENALTY_WEIGHT "0.0"
set_default SATURN_THETA_PRESELECT_TARGET_MW "340.0"
set_default SATURN_THETA_PRESELECT_MW_SCALE "140.0"
set_default SATURN_THETA_PRESELECT_MIN_QED "0.0"
set_default SATURN_THETA_PRESELECT_MAX_SA "99.0"
set_default SATURN_THETA_PRESELECT_MIN_MW "0.0"
set_default SATURN_THETA_PRESELECT_MAX_MW "9999.0"
set_default SATURN_THETA_REFRESH_HIT_TARGETS_EVERY "3"
set_default SATURN_THETA_HIT_TARGET_MAX_SA "3.5"
set_default SATURN_THETA_HIT_TARGET_MIN_QED "0.0"
set_default SATURN_THETA_ADAPT_HIT_TARGET_MACROS "1"
set_default SATURN_THETA_HIT_TARGET_COUNT "48"
set_default SATURN_DOCKING_CHUNK_SIZE "32"
set_default SATURN_DOCKING_RETRY_CHUNK_SIZE "8"
set_default MOLSCORE_SPECTRAL_ALLOW_CHARGED_TOKENS "0"
set_default MOLSCORE_SPECTRAL_ALLOWED_ELEMENTS "C,N,O,S,F,Cl,Br,H"
set_default MOLSCORE_SPECTRAL_EMBED_MEDCHEM_BIAS "0.0"
set_default MOLSCORE_SPECTRAL_EMBED_MACRO_EXPANSION_BLEND "0.0"
set_default MOLSCORE_SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE "0.0"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_SAMPLE_BIAS "6.0"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX "64"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT "16"
set_default MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION "0.0"
set_default MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION "0.95"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX "512"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N "18"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX "512"
set_default MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N "20"

# Keep Saturn docking on the working NVIDIA OpenCL stack identified on Ibex.
export OCL_ICD_VENDORS="${OCL_ICD_VENDORS:-/etc/OpenCL/vendors/nvidia.icd}"
export MOLSCORE_SATURN_OPENCL_LIB_DIR
export MOLSCORE_SATURN_OPENCL_PRELOAD
export MOLSCORE_SATURN_OBABEL_BINARY
export SATURN_THETA_USE_RAW_NSGA_OBJECTIVES
export SATURN_THETA_NSGA_DOCKING_OBJECTIVE_CAP
export SATURN_THETA_DOCKING_FOCUS_FRACTION
export SATURN_THETA_DOCKING_FOCUS_TOP_FRACTION
export SATURN_THETA_DOCKING_ELITE_FRACTION
export SATURN_THETA_DOCKING_FOCUS_MUTATION_STEPS
export SATURN_THETA_DOCKING_FOCUS_TARGET_SAMPLE_FRACTION
export SATURN_THETA_GENERATED_DOCKING_ELITE_FRACTION
export SATURN_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING
export SATURN_THETA_DOCKING_FOCUS_SIGMA_SCALE
export SATURN_THETA_DOCKING_FOCUS_PARAM_NOISE_SCALE
export SATURN_THETA_DOCKING_FOCUS_ROW_RESET_SCALE
export SATURN_THETA_DOCKING_FOCUS_TOKEN_MUTATION_FRACTION
export SATURN_THETA_DOCKING_FOCUS_TOKEN_BLEND
export SATURN_THETA_DOCKING_FOCUS_TOKEN_MACRO_INSERT_PROB
export SATURN_THETA_TARGET_TOKEN_ANALOG_FRACTION
export SATURN_THETA_TARGET_TOKEN_ANALOG_MAX_EDITS
export SATURN_THETA_TARGET_TOKEN_ANALOG_INSERT_PROB
export SATURN_THETA_TARGET_TOKEN_ANALOG_DELETE_PROB
export SATURN_THETA_TARGET_TOKEN_ANALOG_MACRO_INSERT_PROB
export SATURN_THETA_TARGET_TOKEN_ANALOG_BLEND
export SATURN_THETA_TARGET_TOKEN_ANALOG_EXPAND_MACROS
export SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION
export SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT
export SATURN_THETA_HIT_NEIGHBORHOOD_MAX_EDITS
export SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_PROB
export SATURN_THETA_HIT_NEIGHBORHOOD_DELETE_PROB
export SATURN_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB
export SATURN_THETA_HIT_NEIGHBORHOOD_BLEND
export SATURN_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS
export SATURN_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION
export SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT
export SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS
export SATURN_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB
export SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS
export SATURN_THETA_INCLUDE_TASK_TARGET_ANCHORS
export SATURN_THETA_TASK_TARGET_ANCHOR_LIMIT
export SATURN_THETA_HIT_SITE_SCAN_FRACTION
export SATURN_THETA_HIT_SITE_SCAN_MAX_EDITS
export SATURN_THETA_HIT_SITE_SCAN_BLEND
export SATURN_THETA_HIT_SITE_SCAN_TOKENS
export SATURN_THETA_LATE_DEEP_SPIKE_START_FRACTION
export SATURN_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS
export SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS
export SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION
export SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY
export SATURN_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION
export SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION
export SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING
export SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA
export SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED
export SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED
export SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA
export SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW
export SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW
export SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING
export SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED
export SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA
export SATURN_THETA_HIT_MICROJITTER_FRACTION
export SATURN_THETA_HIT_MICROJITTER_SIGMA_MIN
export SATURN_THETA_HIT_MICROJITTER_SIGMA_MAX
export SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION
export SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING
export SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_SA
export SATURN_THETA_GENERATED_HIT_ANCHOR_MIN_QED
export SATURN_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE
export SATURN_THETA_SCAFFOLD_DIVERSE_ANCHORS
export SATURN_THETA_ANCHOR_MAX_PER_SCAFFOLD
export SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP
export SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION
export SATURN_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT
export SATURN_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY
export SATURN_THETA_PRIORITY_MIN_QED
export SATURN_THETA_PRIORITY_MAX_SA
export SATURN_THETA_PRIORITY_MIN_MW
export SATURN_THETA_PRIORITY_MAX_MW
export SATURN_THETA_GENERATED_SITE_SCAN_MIN_QED
export SATURN_THETA_GENERATED_SITE_SCAN_MAX_SA
export SATURN_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS
export SATURN_THETA_HIT_MICROJITTER_MAX_ATTEMPTS
export SATURN_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS
export SATURN_THETA_HIT_POOL_USE_CANDIDATE_BATCH
export SATURN_THETA_CANDIDATE_POOL_MULTIPLIER
export SATURN_THETA_PRESELECT_BY_CHEAP_PROXY
export SATURN_THETA_PRESELECT_SIM_WEIGHT
export SATURN_THETA_PRESELECT_QED_WEIGHT
export SATURN_THETA_PRESELECT_SA_WEIGHT
export SATURN_THETA_PRESELECT_MW_WEIGHT
export SATURN_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT
export SATURN_THETA_PRESELECT_QED_FLOOR
export SATURN_THETA_PRESELECT_QED_FLOOR_WEIGHT
export SATURN_THETA_PRESELECT_SA_CEILING
export SATURN_THETA_PRESELECT_SA_CEILING_WEIGHT
export SATURN_THETA_PRESELECT_BR_PENALTY_WEIGHT
export SATURN_THETA_PRESELECT_TARGET_MW
export SATURN_THETA_PRESELECT_MW_SCALE
export SATURN_THETA_PRESELECT_MIN_QED
export SATURN_THETA_PRESELECT_MAX_SA
export SATURN_THETA_PRESELECT_MIN_MW
export SATURN_THETA_PRESELECT_MAX_MW
export SATURN_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY
export SATURN_THETA_REFRESH_HIT_TARGETS_EVERY
export SATURN_THETA_HIT_TARGET_MAX_SA
export SATURN_THETA_HIT_TARGET_MIN_QED
export SATURN_THETA_ADAPT_HIT_TARGET_MACROS
export SATURN_THETA_HIT_TARGET_COUNT
export SATURN_DOCKING_CHUNK_SIZE
export SATURN_DOCKING_RETRY_CHUNK_SIZE
export MOLSCORE_SPECTRAL_ALLOW_CHARGED_TOKENS
export MOLSCORE_SPECTRAL_ALLOWED_ELEMENTS
export MOLSCORE_DECODE_INCLUDE_GREEDY_NEAREST
export MOLSCORE_SPECTRAL_EMBED_MEDCHEM_BIAS
export MOLSCORE_SPECTRAL_EMBED_MACRO_EXPANSION_BLEND
export MOLSCORE_SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE
export MOLSCORE_SPECTRAL_STATIC_TARGET_SMILES
export MOLSCORE_SPECTRAL_TASK_TARGET_SAMPLE_BIAS
export MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION
export MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION
export MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX
export MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX
export MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N

# Enforce theta-only SpectralMol generation for the NSGA-II path.
export MOLSCORE_SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION=0
export SPECTRAL_PHENOTYPE_PROPOSAL_FRACTION=0
export MOLSCORE_SPECTRAL_BRICS_CROSSOVER_FRACTION=0
export MOLSCORE_SPECTRAL_BRICS_FRAGMENT_REPLACE_FRACTION=0

if [[ "${USE_PER_SEED_SEED_SMILES}" == "0" || "${PER_SEED_SEED_SMILES_DIR}" == "none" || "${PER_SEED_SEED_SMILES_DIR}" == "NONE" ]]; then
  PER_SEED_SEED_SMILES_DIR=""
fi

if [[ -n "${SATURN_MODULES}" ]]; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh
  fi
  for module_name in ${SATURN_MODULES//,/ }; do
    if [[ -n "${module_name}" ]]; then
      module load "${module_name}"
    fi
  done
fi

if [[ -n "${MOLSCORE_SATURN_OPENCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${MOLSCORE_SATURN_OPENCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  if [[ -f "${MOLSCORE_SATURN_OPENCL_LIB_DIR}/libOpenCL.so.1" ]]; then
    export LD_PRELOAD="${MOLSCORE_SATURN_OPENCL_LIB_DIR}/libOpenCL.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi
if [[ -x "${MOLSCORE_SATURN_OBABEL_BINARY}" ]]; then
  OBABEL_PREFIX="$(cd "$(dirname "${MOLSCORE_SATURN_OBABEL_BINARY}")/.." && pwd)"
  export PATH="${OBABEL_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${OBABEL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${PER_SEED_SEED_SMILES_DIR}" ]]; then
  SEED_SPECIFIC_SMILES_FILE="${PER_SEED_SEED_SMILES_DIR}/seed_${SEED}.smi"
  if [[ ! -f "${SEED_SPECIFIC_SMILES_FILE}" ]]; then
    echo "[saturn-theta-nsga2] old-benchmark per-seed file missing: ${SEED_SPECIFIC_SMILES_FILE}" >&2
    echo "[saturn-theta-nsga2] Expected seed files are seed_0.smi ... seed_9.smi in PER_SEED_SEED_SMILES_DIR." >&2
    exit 2
  fi
  export SEED_SMILES_FILE="${SEED_SPECIFIC_SMILES_FILE}"
elif [[ -z "${SEED_SMILES_FILE}" || ! -f "${SEED_SMILES_FILE}" ]]; then
  echo "[saturn-theta-nsga2] SEED_SMILES_FILE or PER_SEED_SEED_SMILES_DIR is required." >&2
  echo "[saturn-theta-nsga2] For manuscript reproduction use the old Saturn/ZINC250k per-seed directory." >&2
  echo "[saturn-theta-nsga2] Current SEED_SMILES_FILE=${SEED_SMILES_FILE:-<unset>}" >&2
  echo "[saturn-theta-nsga2] Current PER_SEED_SEED_SMILES_DIR=${PER_SEED_SEED_SMILES_DIR:-<unset>}" >&2
  exit 2
fi

RUN_ID="${RUN_ID_PREFIX}_budget${BUDGET}_seed${SEED}"
CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/benchmarks/Saturn/compare_scalar_vs_nsga2_saturn.py"
  --seeds "${SEED}"
  --budgets "${BUDGET}"
  --population-size "${POPULATION_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --max-generations "${GENERATIONS}"
  --top-k "${TOP_K}"
  --seed-smiles-file "${SEED_SMILES_FILE}"
  --per-seed-seed-smiles-dir "${PER_SEED_SEED_SMILES_DIR}"
  --seed-pool-size "${SEED_POOL_SIZE}"
  --oracle-template "${SATURN_ORACLE_TEMPLATE}"
  --oracle-config-key "${SATURN_ORACLE_CONFIG_KEY}"
  --saturn-repo-root "${SATURN_REPO_ROOT}"
  --quickvina-binary "${QUICKVINA_BINARY}"
  --receptor-file "${RECEPTOR_FILE}"
  --reference-ligand-file "${REFERENCE_LIGAND_FILE}"
  --tournament-k "${TOURNAMENT_K}"
  --immigrant-fraction "${IMMIGRANT_FRACTION}"
  --parent-pool-fraction "${PARENT_POOL_FRACTION}"
  --stagnation-patience "${STAGNATION_PATIENCE}"
  --stagnation-mutation-boost "${STAGNATION_MUTATION_BOOST}"
  --skip-scalar
  --nsga2-genotype theta
  --frequency-mode "${FREQUENCY_MODE}"
  --spectral-l "${SPECTRAL_L}"
  --spectral-k "${SPECTRAL_K}"
  --spectral-d "${SPECTRAL_D}"
  --spectral-decode-attempts "${SPECTRAL_DECODE_ATTEMPTS}"
  --output-dir "${OUTPUT_ROOT}"
  --run-id "${RUN_ID}"
)

echo "[saturn-theta-nsga2] seed=${SEED} budget=${BUDGET}"
echo "[saturn-theta-nsga2] output=${OUTPUT_ROOT}/${RUN_ID}"
echo "[saturn-theta-nsga2] per_seed_seed_smiles_dir=${PER_SEED_SEED_SMILES_DIR:-<none>}"
echo "[saturn-theta-nsga2] seed_smiles_file=${SEED_SMILES_FILE}"
echo "[saturn-theta-nsga2] node=${SLURMD_NODENAME:-unknown}"
echo "[saturn-theta-nsga2] ocl_icd_vendors=${OCL_ICD_VENDORS}"
echo "[saturn-theta-nsga2] opencl_lib_dir=${MOLSCORE_SATURN_OPENCL_LIB_DIR}"
echo "[saturn-theta-nsga2] raw_nsga_objectives=${SATURN_THETA_USE_RAW_NSGA_OBJECTIVES}"
echo "[saturn-theta-nsga2] nsga_docking_objective_cap=${SATURN_THETA_NSGA_DOCKING_OBJECTIVE_CAP}"
echo "[saturn-theta-nsga2] docking_focus_fraction=${SATURN_THETA_DOCKING_FOCUS_FRACTION}"
echo "[saturn-theta-nsga2] docking_focus_top_fraction=${SATURN_THETA_DOCKING_FOCUS_TOP_FRACTION}"
echo "[saturn-theta-nsga2] docking_elite_fraction=${SATURN_THETA_DOCKING_ELITE_FRACTION}"
echo "[saturn-theta-nsga2] docking_focus_mutation_steps=${SATURN_THETA_DOCKING_FOCUS_MUTATION_STEPS}"
echo "[saturn-theta-nsga2] docking_focus_target_sample_fraction=${SATURN_THETA_DOCKING_FOCUS_TARGET_SAMPLE_FRACTION}"
echo "[saturn-theta-nsga2] generated_docking_elite_fraction=${SATURN_THETA_GENERATED_DOCKING_ELITE_FRACTION}"
echo "[saturn-theta-nsga2] generated_docking_elite_max_docking=${SATURN_THETA_GENERATED_DOCKING_ELITE_MAX_DOCKING}"
echo "[saturn-theta-nsga2] docking_focus_sigma_scale=${SATURN_THETA_DOCKING_FOCUS_SIGMA_SCALE}"
echo "[saturn-theta-nsga2] docking_focus_param_noise_scale=${SATURN_THETA_DOCKING_FOCUS_PARAM_NOISE_SCALE}"
echo "[saturn-theta-nsga2] docking_focus_row_reset_scale=${SATURN_THETA_DOCKING_FOCUS_ROW_RESET_SCALE}"
echo "[saturn-theta-nsga2] docking_focus_token_mutation_fraction=${SATURN_THETA_DOCKING_FOCUS_TOKEN_MUTATION_FRACTION}"
echo "[saturn-theta-nsga2] docking_focus_token_blend=${SATURN_THETA_DOCKING_FOCUS_TOKEN_BLEND}"
echo "[saturn-theta-nsga2] docking_focus_token_macro_insert_prob=${SATURN_THETA_DOCKING_FOCUS_TOKEN_MACRO_INSERT_PROB}"
echo "[saturn-theta-nsga2] target_token_analog_fraction=${SATURN_THETA_TARGET_TOKEN_ANALOG_FRACTION}"
echo "[saturn-theta-nsga2] target_token_analog_max_edits=${SATURN_THETA_TARGET_TOKEN_ANALOG_MAX_EDITS}"
echo "[saturn-theta-nsga2] target_token_analog_insert_prob=${SATURN_THETA_TARGET_TOKEN_ANALOG_INSERT_PROB}"
echo "[saturn-theta-nsga2] target_token_analog_delete_prob=${SATURN_THETA_TARGET_TOKEN_ANALOG_DELETE_PROB}"
echo "[saturn-theta-nsga2] target_token_analog_macro_insert_prob=${SATURN_THETA_TARGET_TOKEN_ANALOG_MACRO_INSERT_PROB}"
echo "[saturn-theta-nsga2] target_token_analog_blend=${SATURN_THETA_TARGET_TOKEN_ANALOG_BLEND}"
echo "[saturn-theta-nsga2] target_token_analog_expand_macros=${SATURN_THETA_TARGET_TOKEN_ANALOG_EXPAND_MACROS}"
echo "[saturn-theta-nsga2] hit_neighborhood_fraction=${SATURN_THETA_HIT_NEIGHBORHOOD_FRACTION}"
echo "[saturn-theta-nsga2] hit_neighborhood_min_count=${SATURN_THETA_HIT_NEIGHBORHOOD_MIN_COUNT}"
echo "[saturn-theta-nsga2] hit_neighborhood_max_edits=${SATURN_THETA_HIT_NEIGHBORHOOD_MAX_EDITS}"
echo "[saturn-theta-nsga2] hit_neighborhood_insert_prob=${SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_PROB}"
echo "[saturn-theta-nsga2] hit_neighborhood_delete_prob=${SATURN_THETA_HIT_NEIGHBORHOOD_DELETE_PROB}"
echo "[saturn-theta-nsga2] hit_neighborhood_macro_insert_prob=${SATURN_THETA_HIT_NEIGHBORHOOD_MACRO_INSERT_PROB}"
echo "[saturn-theta-nsga2] hit_neighborhood_blend=${SATURN_THETA_HIT_NEIGHBORHOOD_BLEND}"
echo "[saturn-theta-nsga2] hit_neighborhood_expand_macros=${SATURN_THETA_HIT_NEIGHBORHOOD_EXPAND_MACROS}"
echo "[saturn-theta-nsga2] hit_neighborhood_target_sample_fraction=${SATURN_THETA_HIT_NEIGHBORHOOD_TARGET_SAMPLE_FRACTION}"
echo "[saturn-theta-nsga2] hit_neighborhood_anchor_count=${SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_COUNT}"
echo "[saturn-theta-nsga2] hit_neighborhood_anchor_sample_bias=${SATURN_THETA_HIT_NEIGHBORHOOD_ANCHOR_SAMPLE_BIAS}"
echo "[saturn-theta-nsga2] hit_neighborhood_edge_position_prob=${SATURN_THETA_HIT_NEIGHBORHOOD_EDGE_POSITION_PROB}"
echo "[saturn-theta-nsga2] hit_neighborhood_insert_tokens=${SATURN_THETA_HIT_NEIGHBORHOOD_INSERT_TOKENS}"
echo "[saturn-theta-nsga2] include_task_target_anchors=${SATURN_THETA_INCLUDE_TASK_TARGET_ANCHORS}"
echo "[saturn-theta-nsga2] task_target_anchor_limit=${SATURN_THETA_TASK_TARGET_ANCHOR_LIMIT}"
echo "[saturn-theta-nsga2] hit_site_scan_fraction=${SATURN_THETA_HIT_SITE_SCAN_FRACTION}"
echo "[saturn-theta-nsga2] hit_site_scan_max_edits=${SATURN_THETA_HIT_SITE_SCAN_MAX_EDITS}"
echo "[saturn-theta-nsga2] hit_site_scan_blend=${SATURN_THETA_HIT_SITE_SCAN_BLEND}"
echo "[saturn-theta-nsga2] hit_site_scan_motif_choice_probability=${SATURN_THETA_HIT_SITE_SCAN_MOTIF_CHOICE_PROBABILITY}"
echo "[saturn-theta-nsga2] hit_site_scan_tokens=${SATURN_THETA_HIT_SITE_SCAN_TOKENS}"
echo "[saturn-theta-nsga2] late_deep_spike_start_fraction=${SATURN_THETA_LATE_DEEP_SPIKE_START_FRACTION}"
echo "[saturn-theta-nsga2] late_deep_spike_insert_tokens=${SATURN_THETA_LATE_DEEP_SPIKE_INSERT_TOKENS}"
echo "[saturn-theta-nsga2] late_deep_spike_site_scan_tokens=${SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_TOKENS}"
echo "[saturn-theta-nsga2] late_deep_spike_site_scan_fraction=${SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_FRACTION}"
echo "[saturn-theta-nsga2] late_deep_spike_site_scan_motif_choice_probability=${SATURN_THETA_LATE_DEEP_SPIKE_SITE_SCAN_MOTIF_CHOICE_PROBABILITY}"
echo "[saturn-theta-nsga2] late_deep_spike_microjitter_fraction=${SATURN_THETA_LATE_DEEP_SPIKE_MICROJITTER_FRACTION}"
echo "[saturn-theta-nsga2] late_deep_spike_generated_hit_anchor_fraction=${SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_FRACTION}"
echo "[saturn-theta-nsga2] late_deep_spike_generated_hit_anchor_max_docking=${SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_DOCKING}"
echo "[saturn-theta-nsga2] late_deep_spike_generated_hit_anchor_max_sa=${SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MAX_SA}"
echo "[saturn-theta-nsga2] late_deep_spike_generated_hit_anchor_min_qed=${SATURN_THETA_LATE_DEEP_SPIKE_GENERATED_HIT_ANCHOR_MIN_QED}"
echo "[saturn-theta-nsga2] late_deep_spike_priority_min_qed=${SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_QED}"
echo "[saturn-theta-nsga2] late_deep_spike_priority_max_sa=${SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_SA}"
echo "[saturn-theta-nsga2] late_deep_spike_priority_min_mw=${SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MIN_MW}"
echo "[saturn-theta-nsga2] late_deep_spike_priority_max_mw=${SATURN_THETA_LATE_DEEP_SPIKE_PRIORITY_MAX_MW}"
echo "[saturn-theta-nsga2] late_deep_spike_archive_max_docking=${SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_DOCKING}"
echo "[saturn-theta-nsga2] late_deep_spike_archive_min_qed=${SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MIN_QED}"
echo "[saturn-theta-nsga2] late_deep_spike_archive_max_sa=${SATURN_THETA_LATE_DEEP_SPIKE_ARCHIVE_MAX_SA}"
echo "[saturn-theta-nsga2] decode_include_greedy_nearest=${MOLSCORE_DECODE_INCLUDE_GREEDY_NEAREST}"
echo "[saturn-theta-nsga2] static_target_smiles=${MOLSCORE_SPECTRAL_STATIC_TARGET_SMILES}"
echo "[saturn-theta-nsga2] hit_microjitter_fraction=${SATURN_THETA_HIT_MICROJITTER_FRACTION}"
echo "[saturn-theta-nsga2] hit_microjitter_sigma_min=${SATURN_THETA_HIT_MICROJITTER_SIGMA_MIN}"
echo "[saturn-theta-nsga2] hit_microjitter_sigma_max=${SATURN_THETA_HIT_MICROJITTER_SIGMA_MAX}"
echo "[saturn-theta-nsga2] generated_hit_anchor_fraction=${SATURN_THETA_GENERATED_HIT_ANCHOR_FRACTION}"
echo "[saturn-theta-nsga2] generated_hit_anchor_max_docking=${SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_DOCKING}"
echo "[saturn-theta-nsga2] generated_hit_anchor_max_sa=${SATURN_THETA_GENERATED_HIT_ANCHOR_MAX_SA}"
echo "[saturn-theta-nsga2] generated_hit_anchor_min_qed=${SATURN_THETA_GENERATED_HIT_ANCHOR_MIN_QED}"
echo "[saturn-theta-nsga2] generated_hit_jitter_sigma_scale=${SATURN_THETA_GENERATED_HIT_JITTER_SIGMA_SCALE}"
echo "[saturn-theta-nsga2] scaffold_diverse_anchors=${SATURN_THETA_SCAFFOLD_DIVERSE_ANCHORS}"
echo "[saturn-theta-nsga2] anchor_max_per_scaffold=${SATURN_THETA_ANCHOR_MAX_PER_SCAFFOLD}"
echo "[saturn-theta-nsga2] scaffold_diverse_top_keep=${SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP}"
echo "[saturn-theta-nsga2] scaffold_diverse_top_keep_fraction=${SATURN_THETA_SCAFFOLD_DIVERSE_TOP_KEEP_FRACTION}"
echo "[saturn-theta-nsga2] hit_neighborhood_bypass_preselect=${SATURN_THETA_HIT_NEIGHBORHOOD_BYPASS_PRESELECT}"
echo "[saturn-theta-nsga2] priority_preselect_by_proxy=${SATURN_THETA_PRIORITY_PRESELECT_BY_CHEAP_PROXY}"
echo "[saturn-theta-nsga2] priority_min_qed=${SATURN_THETA_PRIORITY_MIN_QED}"
echo "[saturn-theta-nsga2] priority_max_sa=${SATURN_THETA_PRIORITY_MAX_SA}"
echo "[saturn-theta-nsga2] priority_min_mw=${SATURN_THETA_PRIORITY_MIN_MW}"
echo "[saturn-theta-nsga2] priority_max_mw=${SATURN_THETA_PRIORITY_MAX_MW}"
echo "[saturn-theta-nsga2] generated_site_scan_min_qed=${SATURN_THETA_GENERATED_SITE_SCAN_MIN_QED}"
echo "[saturn-theta-nsga2] generated_site_scan_max_sa=${SATURN_THETA_GENERATED_SITE_SCAN_MAX_SA}"
echo "[saturn-theta-nsga2] hit_site_scan_max_attempts=${SATURN_THETA_HIT_SITE_SCAN_MAX_ATTEMPTS}"
echo "[saturn-theta-nsga2] hit_microjitter_max_attempts=${SATURN_THETA_HIT_MICROJITTER_MAX_ATTEMPTS}"
echo "[saturn-theta-nsga2] hit_neighborhood_max_attempts=${SATURN_THETA_HIT_NEIGHBORHOOD_MAX_ATTEMPTS}"
echo "[saturn-theta-nsga2] hit_pool_use_candidate_batch=${SATURN_THETA_HIT_POOL_USE_CANDIDATE_BATCH}"
echo "[saturn-theta-nsga2] candidate_pool_multiplier=${SATURN_THETA_CANDIDATE_POOL_MULTIPLIER}"
echo "[saturn-theta-nsga2] preselect_by_cheap_proxy=${SATURN_THETA_PRESELECT_BY_CHEAP_PROXY}"
echo "[saturn-theta-nsga2] preselect_sim_weight=${SATURN_THETA_PRESELECT_SIM_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_qed_weight=${SATURN_THETA_PRESELECT_QED_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_sa_weight=${SATURN_THETA_PRESELECT_SA_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_mw_weight=${SATURN_THETA_PRESELECT_MW_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_docking_motif_weight=${SATURN_THETA_PRESELECT_DOCKING_MOTIF_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_qed_floor=${SATURN_THETA_PRESELECT_QED_FLOOR}"
echo "[saturn-theta-nsga2] preselect_qed_floor_weight=${SATURN_THETA_PRESELECT_QED_FLOOR_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_sa_ceiling=${SATURN_THETA_PRESELECT_SA_CEILING}"
echo "[saturn-theta-nsga2] preselect_sa_ceiling_weight=${SATURN_THETA_PRESELECT_SA_CEILING_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_br_penalty_weight=${SATURN_THETA_PRESELECT_BR_PENALTY_WEIGHT}"
echo "[saturn-theta-nsga2] preselect_target_mw=${SATURN_THETA_PRESELECT_TARGET_MW}"
echo "[saturn-theta-nsga2] preselect_mw_scale=${SATURN_THETA_PRESELECT_MW_SCALE}"
echo "[saturn-theta-nsga2] preselect_min_qed=${SATURN_THETA_PRESELECT_MIN_QED}"
echo "[saturn-theta-nsga2] preselect_max_sa=${SATURN_THETA_PRESELECT_MAX_SA}"
echo "[saturn-theta-nsga2] preselect_min_mw=${SATURN_THETA_PRESELECT_MIN_MW}"
echo "[saturn-theta-nsga2] preselect_max_mw=${SATURN_THETA_PRESELECT_MAX_MW}"
echo "[saturn-theta-nsga2] refresh_hit_targets_every=${SATURN_THETA_REFRESH_HIT_TARGETS_EVERY}"
echo "[saturn-theta-nsga2] hit_target_max_sa=${SATURN_THETA_HIT_TARGET_MAX_SA}"
echo "[saturn-theta-nsga2] hit_target_min_qed=${SATURN_THETA_HIT_TARGET_MIN_QED}"
echo "[saturn-theta-nsga2] adapt_hit_target_macros=${SATURN_THETA_ADAPT_HIT_TARGET_MACROS}"
echo "[saturn-theta-nsga2] hit_target_count=${SATURN_THETA_HIT_TARGET_COUNT}"
echo "[saturn-theta-nsga2] docking_chunk_size=${SATURN_DOCKING_CHUNK_SIZE}"
echo "[saturn-theta-nsga2] docking_retry_chunk_size=${SATURN_DOCKING_RETRY_CHUNK_SIZE}"
echo "[saturn-theta-nsga2] spectral_allow_charged_tokens=${MOLSCORE_SPECTRAL_ALLOW_CHARGED_TOKENS}"
echo "[saturn-theta-nsga2] spectral_allowed_elements=${MOLSCORE_SPECTRAL_ALLOWED_ELEMENTS}"
echo "[saturn-theta-nsga2] spectral_embed_medchem_bias=${MOLSCORE_SPECTRAL_EMBED_MEDCHEM_BIAS}"
echo "[saturn-theta-nsga2] spectral_embed_macro_expansion_blend=${MOLSCORE_SPECTRAL_EMBED_MACRO_EXPANSION_BLEND}"
echo "[saturn-theta-nsga2] spectral_embed_token_identity_scale=${MOLSCORE_SPECTRAL_EMBED_TOKEN_IDENTITY_SCALE}"
echo "[saturn-theta-nsga2] spectral_task_target_sample_bias=${MOLSCORE_SPECTRAL_TASK_TARGET_SAMPLE_BIAS}"
echo "[saturn-theta-nsga2] spectral_task_target_full_macro_max=${MOLSCORE_SPECTRAL_TASK_TARGET_FULL_MACRO_MAX}"
echo "[saturn-theta-nsga2] spectral_task_target_macro_weight=${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_WEIGHT}"
echo "[saturn-theta-nsga2] spectral_target_macro_jump_fraction=${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_JUMP_FRACTION}"
echo "[saturn-theta-nsga2] spectral_target_macro_insert_fraction=${MOLSCORE_SPECTRAL_THETA_TARGET_MACRO_INSERT_FRACTION}"
echo "[saturn-theta-nsga2] spectral_task_target_window_macro_max=${MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX}"
echo "[saturn-theta-nsga2] spectral_task_target_window_macro_max_n=${MOLSCORE_SPECTRAL_TASK_TARGET_WINDOW_MACRO_MAX_N}"
echo "[saturn-theta-nsga2] spectral_task_target_macro_max=${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX}"
echo "[saturn-theta-nsga2] spectral_task_target_macro_max_n=${MOLSCORE_SPECTRAL_TASK_TARGET_MACRO_MAX_N}"
printf '[saturn-theta-nsga2] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${CMD[@]}"
