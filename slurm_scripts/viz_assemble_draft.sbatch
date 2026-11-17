#!/bin/bash
# Assemble the showcase clips (viz/out/*.mp4) into one draft video with crossfades.
# CPU only (ffmpeg/libx264). Submit from the repo root:
#   sbatch slurm_scripts/viz_assemble_draft.sbatch
#   ORDER="G1_mode_orchestra G4_crystallise" OUT=test sbatch slurm_scripts/viz_assemble_draft.sbatch
# Env: ORDER (clip basenames, storyboard order), XFADE (crossfade seconds, 0.6),
#      LOOPS (how many times to repeat in a second "loop test" file, 2), OUT (basename, draft_v1),
#      CRF (16), DIR (out | out_1080), TRIM (path to a trim plan json, or 'none'; default viz/trim_plan.json).
#SBATCH --job-name=viz-assemble
#SBATCH --partition=batch
#SBATCH --constraint=intel&cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO=/ibex/user/${USER}/molevoDrugDiscovery
ORDER=${ORDER:-"G0_title G1_mode_orchestra G2_theta_desk G3_nearest_star G4_crystallise G5_frequency_brush F10_front_sweep G8_harmonic_peel G7_four_oceans G9_complexity_bloom G10_same_destination G12_galaxy_dolly G13_outro"}
XFADE=${XFADE:-0.6}
OUT=${OUT:-draft_v1}
CRF=${CRF:-16}
LOOPS=${LOOPS:-2}

DIR=${DIR:-out}          # out (960x540) or out_1080
cd "${REPO}/viz/${DIR}"
module purge
module load ffmpeg/7.0.1/gnu-12.2.0
ffmpeg -version | head -1

inputs=()
durs=()
for n in ${ORDER}; do
  [[ -f "${n}.mp4" ]] || { echo "[assemble] missing ${n}.mp4" >&2; exit 2; }
  inputs+=(-i "${n}.mp4")
  durs+=("$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${n}.mp4")")
done
N=${#durs[@]}
echo "[assemble] ${N} clips, durations: ${durs[*]}, crossfade ${XFADE}s"

# per-clip trims/speed (viz/trim_plan.json; TRIM=none for whole clips) + cross-fades + dip at both ends
TRIM=${TRIM:-${REPO}/viz/trim_plan.json}
GEN=$(python3 "${REPO}/viz/assemble_filter.py" "${ORDER}" "${durs[*]}" "${XFADE}" "${TRIM}")
filter=$(sed -n 1p <<<"${GEN}")
L=$(sed -n 2p <<<"${GEN}")
echo "[assemble] total length ${L}s"

# -cpuflags 0: the SIMD xfade filter in ffmpeg 7.0.1 corrupts frames (green/magenta rows) on these nodes.
ffmpeg -hide_banner -y -cpuflags 0 "${inputs[@]}" -filter_complex "${filter}" -map "[out]" \
  -c:v libx264 -preset slow -crf "${CRF}" -pix_fmt yuv420p -movflags +faststart "${OUT}.mp4"

# loop test: the same file repeated, to eyeball the seam
ffmpeg -hide_banner -y -stream_loop $((LOOPS - 1)) -i "${OUT}.mp4" -c copy "${OUT}_loop${LOOPS}x.mp4"

# seam check: mean luma of the first and last frames (both should be ~ the dark background)
for f in 0 last; do
  if [[ "${f}" == "0" ]]; then
    sel="select=eq(n\,0)"
  else
    sel="select=eq(n\,$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "${OUT}.mp4")-1)"
  fi
  ffmpeg -hide_banner -v error -y -i "${OUT}.mp4" -vf "${sel},signalstats,metadata=print:key=lavfi.signalstats.YAVG" -vsync 0 -f null - 2>&1 | grep YAVG || true
done

# contact sheet: 12 evenly spaced frames (tiled with PIL; ffmpeg's tile filter output was unreliable)
FR=$(mktemp -d)
ffmpeg -hide_banner -v error -y -i "${OUT}.mp4" -vf "fps=12/${L},scale=320:180" "${FR}/f_%02d.png"
"${REPO}/env/bin/python" "${REPO}/viz/contact_sheet.py" "${FR}" "${OUT}_review.png"

ls -la "${OUT}"*.mp4 "${OUT}_review.png"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration -of default=nw=1 "${OUT}.mp4"
