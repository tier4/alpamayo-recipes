#!/usr/bin/env bash
# Visualise GT vs Base vs Fine-tuned trajectories for all val samples,
# then combine the images into a video with ffmpeg.
#
# GPU assignment:
#   GPUs 0 .. N/2-1  →  Base model (kept loaded, parallel shards)
#   GPUs N/2 .. N-1   →  FT model   (kept loaded, parallel shards)
#   Both groups run simultaneously — no model load/unload cycle.
#
# Predictions are saved as .npz files under <OUTPUT_DIR>/predictions/
# so visualisation and video can be re-run without re-inference.
#
# Usage:
#   cd recipes/alpamayo1_5_sft
#   bash visualize_wds_nav.sh BASE_CKPT FT_CKPT [OUTPUT_DIR]
#
# Environment variables:
#   NUM_GPUS   number of GPUs to use (default: 8, must be even)
#   FPS        video framerate (default: 2)

set -euo pipefail

unset WANDB_API_KEY || true

if [[ $# -lt 2 ]]; then
    echo "Usage: bash visualize_wds_nav.sh BASE_CKPT FT_CKPT [OUTPUT_DIR]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_CKPT="${1}"
FT_CKPT="${2}"
OUTPUT_DIR="${3:-${ALPAMAYO_OUTPUT_ROOT:-./outputs}/viz_all}"
NUM_GPUS="${NUM_GPUS:-8}"
FPS="${FPS:-2}"

echo "=========================================="
echo " Trajectory Visualisation + Video"
echo " Base checkpoint  : ${BASE_CKPT}"
echo " FT checkpoint    : ${FT_CKPT}"
echo " Output dir       : ${OUTPUT_DIR}"
echo " GPUs             : ${NUM_GPUS} (${NUM_GPUS}/2 base + ${NUM_GPUS}/2 ft)"
echo " FPS              : ${FPS}"
echo "=========================================="

.venv/bin/python visualize_wds_trajectories.py all \
    --base-ckpt "${BASE_CKPT}" \
    --ft-ckpt "${FT_CKPT}" \
    --all \
    --num-gpus "${NUM_GPUS}" \
    --fps "${FPS}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo " Done!"
echo " Predictions : ${OUTPUT_DIR}/predictions/"
echo " Frames      : ${OUTPUT_DIR}/frames/"
echo " Video       : ${OUTPUT_DIR}/trajectory_comparison.mp4"
echo "=========================================="
