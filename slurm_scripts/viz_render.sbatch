#!/bin/bash
# Render the SpectralMol showcase animations (viz/g*.py) on a CPU compute node.
# CPU only: matplotlib/RDKit/NumPy, no GPU needed. Submit from the repo root:
#   sbatch slurm_scripts/viz_render.sbatch                      # all targets
#   TARGETS="g3 g4" sbatch slurm_scripts/viz_render.sbatch      # a subset
#   TARGETS="g3" EXTRA="--seconds 4 --fps 12" sbatch ...        # quick preview (EXTRA goes to every target)
#SBATCH --job-name=viz-render
#SBATCH --partition=batch
#SBATCH --constraint=intel&cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=/ibex/user/${USER}/molevoDrugDiscovery
CONDA=/ibex/user/${USER}/miniconda3/bin/conda
ENV=/ibex/user/${USER}/molevoDrugDiscovery/env
TARGETS=${TARGETS:-"g1 g2 g3 g4"}
EXTRA=${EXTRA:-}

declare -A SCRIPT=(
  [g1]=viz/g1_mode_orchestra.py
  [g2]=viz/g2_theta_desk.py
  [g3]=viz/g3_nearest_star.py
  [g4]=viz/g4_crystallise.py
  [f10]=viz/f10_front_sweep.py
  [g10]=viz/g10_parallel_race.py
  [g7]=viz/g7_four_oceans.py
  [g8]=viz/g8_harmonic_peel.py
  [g9]=viz/g9_complexity_bloom.py
  [g12]=viz/g12_galaxy_dolly.py
  [g5]=viz/g5_frequency_brush.py
  [g0]=viz/g0_title_cards.py
)

mkdir -p "${REPO}/logs"
cd "${REPO}"

export LD_LIBRARY_PATH="${ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

for t in ${TARGETS}; do
  if [[ -z "${SCRIPT[$t]:-}" ]]; then
    echo "[viz] unknown target '${t}' (valid: ${!SCRIPT[*]})" >&2
    exit 2
  fi
  echo "[viz] $(date +%T) rendering ${t}: ${SCRIPT[$t]} ${EXTRA}"
  # shellcheck disable=SC2086
  "${CONDA}" run -p "${ENV}" --no-capture-output python "${SCRIPT[$t]}" ${EXTRA}
done

echo "[viz] done (VIZ_SCALE=${VIZ_SCALE:-1}). outputs in ${REPO}/viz/out*"
ls -la "${REPO}/viz/out"*/*.gif 2>/dev/null || true
