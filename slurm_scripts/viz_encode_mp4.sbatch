#!/bin/bash
# Encode every PNG sequence in viz/out/<name>/<name>_%04d.png into viz/out/<name>.mp4 (H.264, yuv420p, PowerPoint-safe).
# CPU only. Uses the default CPU software stack: ffmpeg/7.0.1 there has libx264 (the rl9-gpustack ffmpeg 5.1.2 does NOT).
#   sbatch slurm_scripts/viz_encode_mp4.sbatch                  # all sequences
#   NAMES="G4_crystallise F10_front_sweep" sbatch ...           # a subset
#   FPS=24 CRF=16 sbatch ...
#   DIR=out_1080 sbatch ...                                   # the 1920x1080 renders (VIZ_SCALE=2)
#SBATCH --job-name=viz-encode
#SBATCH --partition=batch
#SBATCH --constraint=intel&cascadelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

REPO=/ibex/user/${USER}/molevoDrugDiscovery
FPS=${FPS:-24}
CRF=${CRF:-16}
NAMES=${NAMES:-}

mkdir -p "${REPO}/logs"
DIR=${DIR:-out}          # out (960x540) or out_1080
cd "${REPO}/viz/${DIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge >/dev/null 2>&1 || true
module load ffmpeg/7.0.1/gnu-12.2.0
ffmpeg -hide_banner -version | head -1
ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)   # capture first: grep -q + pipefail would false-fail via SIGPIPE
grep -q libx264 <<<"${ENCODERS}" || { echo "[encode] libx264 missing in this ffmpeg" >&2; exit 3; }

if [[ -z "${NAMES}" ]]; then
  NAMES=$(find . -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tr '\n' ' ')
fi

for n in ${NAMES}; do
  [[ -d "${n}" ]] || { echo "[encode] no sequence dir '${n}'" >&2; exit 2; }
  echo "[encode] $(date +%T) ${n} @ ${FPS} fps, crf ${CRF}"
  # -sws_flags: ffmpeg 7.0.1's default (SIMD) RGB->yuv420p conversion corrupts colour here
  # (alternating green/magenta rows); accurate_rnd+full_chroma_int is correct (must be an OUTPUT option, i.e. after -i).
  ffmpeg -hide_banner -loglevel error -y -framerate "${FPS}" -i "${n}/${n}_%04d.png" \
    -sws_flags accurate_rnd+full_chroma_int -c:v libx264 -preset slow -crf "${CRF}" -pix_fmt yuv420p -movflags +faststart "${n}.mp4"
  ffprobe -hide_banner -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration -of csv=p=0 "${n}.mp4"
done

echo "[encode] done"
ls -la "${REPO}"/viz/${DIR}/*.mp4
