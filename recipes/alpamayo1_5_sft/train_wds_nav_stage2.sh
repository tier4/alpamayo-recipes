#!/usr/bin/env bash
# Full-data Stage-2 trajectory-expert training on WDS navigation data.
# Freezes the Stage-1 VLM and trains the diffusion expert head.
# 4 cameras (FRONT_LEFT / FRONT_WIDE / FRONT_RIGHT / FRONT) × 4 temporal frames,
# matching the physical_ai_av dataset default setup.
#
# Usage:
#   cd recipes/alpamayo1_5_sft
#   bash train_wds_nav_stage2.sh STAGE1_CHECKPOINT_PATH [OUTPUT_DIR]
#
# STAGE1_CHECKPOINT_PATH is required: a Trainer output dir from the Stage-1
# WDS run (sft_stage1_wds_nav_4cam4frame), e.g.
#   $ALPAMAYO_OUTPUT_ROOT/stage1_wds_nav_4cam4frame_50ep/checkpoint-4700
# It must contain model.safetensors.index.json and its shards.
#
# The base (pretrained) checkpoint path is read from
# configs/sft_stage2_wds_nav_4cam4frame.yaml (model.pretrained_model_name_or_path).
#
# This script always runs the actual training inside a detached tmux session
# (named below) so it survives terminal/SSH disconnects. Re-running this
# script attaches to the existing session if one is already running.
#
# Environment variables:
#   NPROC_PER_NODE   number of GPUs per node (default: 8)
#   EXTRA_ARGS       any additional Hydra overrides appended to the command
#   TMUX_SESSION     tmux session name (default: alpamayo_wds_nav_train_stage2)
#   LOG_FILE         path to tee training output to (default: <OUTPUT_DIR>/train.log)

set -euo pipefail

unset WANDB_API_KEY || true

if [[ $# -lt 1 ]]; then
    echo "Usage: bash train_wds_nav_stage2.sh STAGE1_CHECKPOINT_PATH [OUTPUT_DIR]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STAGE1_CHECKPOINT_PATH="${1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${2:-${ALPAMAYO_OUTPUT_ROOT:-./outputs}/stage2_wds_nav_4cam4frame}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMUX_SESSION="${TMUX_SESSION:-alpamayo_wds_nav_train_stage2}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session '${TMUX_SESSION}' already exists; attaching."
    echo "  (Ctrl-b d to detach without stopping training)"
    exec tmux attach -t "${TMUX_SESSION}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo " Alpamayo 1.5 SFT — WDS Nav Stage-2 Full Training"
echo " GPUs per node      : ${NPROC_PER_NODE}"
echo " Stage-1 checkpoint : ${STAGE1_CHECKPOINT_PATH}"
echo " Output dir         : ${OUTPUT_DIR}"
echo " Config             : sft_stage2_wds_nav_4cam4frame"
echo " tmux session       : ${TMUX_SESSION}"
echo " Log file           : ${LOG_FILE}"
echo "=========================================="
echo "Launching in tmux. Attach with: tmux attach -t ${TMUX_SESSION}"
echo "Detach with Ctrl-b d. Training continues after detaching or closing this shell."

TRAIN_CMD=".venv/bin/torchrun \
    --nproc_per_node ${NPROC_PER_NODE} \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage2_wds_nav_4cam4frame \
    model.stage1_vlm_checkpoint_path=${STAGE1_CHECKPOINT_PATH} \
    paths.output_dir=${OUTPUT_DIR} \
    ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" "${TRAIN_CMD}"
tmux attach -t "${TMUX_SESSION}"
