#!/usr/bin/env bash
# Full-data Stage-1 SFT training on WDS navigation data.
# 4 cameras (FRONT_LEFT / FRONT_WIDE / FRONT_RIGHT / FRONT) × 4 temporal frames,
# matching the physical_ai_av dataset default setup.
#
# Usage:
#   cd recipes/alpamayo1_5_sft
#   bash train_wds_nav_stage1.sh [OUTPUT_DIR]
#
# The optional OUTPUT_DIR argument overrides paths.output_dir (default below).
# The model checkpoint path is read from configs/models/ar1_5_base.yaml; update
# that file or pass model.checkpoint_path=<path> as an extra argument.
#
# This script always runs the actual training inside a detached tmux session
# (named below) so it survives terminal/SSH disconnects. Re-running this
# script attaches to the existing session if one is already running.
#
# Environment variables:
#   NPROC_PER_NODE   number of GPUs per node (default: 8)
#   EXTRA_ARGS       any additional Hydra overrides appended to the command
#   TMUX_SESSION     tmux session name (default: alpamayo_wds_nav_train)
#   LOG_FILE         path to tee training output to (default: <OUTPUT_DIR>/train.log)

set -euo pipefail

unset WANDB_API_KEY || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${1:-${ALPAMAYO_OUTPUT_ROOT:-./outputs}/stage1_wds_nav_4cam4frame}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TMUX_SESSION="${TMUX_SESSION:-alpamayo_wds_nav_train}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session '${TMUX_SESSION}' already exists; attaching."
    echo "  (Ctrl-b d to detach without stopping training)"
    exec tmux attach -t "${TMUX_SESSION}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo " Alpamayo 1.5 SFT — WDS Nav Full Training"
echo " GPUs per node : ${NPROC_PER_NODE}"
echo " Output dir    : ${OUTPUT_DIR}"
echo " Config        : sft_stage1_wds_nav_4cam4frame"
echo " tmux session  : ${TMUX_SESSION}"
echo " Log file      : ${LOG_FILE}"
echo "=========================================="
echo "Launching in tmux. Attach with: tmux attach -t ${TMUX_SESSION}"
echo "Detach with Ctrl-b d. Training continues after detaching or closing this shell."

TRAIN_CMD=".venv/bin/torchrun \
    --nproc_per_node ${NPROC_PER_NODE} \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_wds_nav_4cam4frame \
    paths.output_dir=${OUTPUT_DIR} \
    ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}"

tmux new-session -d -s "${TMUX_SESSION}" -c "${SCRIPT_DIR}" "${TRAIN_CMD}"
tmux attach -t "${TMUX_SESSION}"
