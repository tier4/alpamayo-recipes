#!/usr/bin/env bash
# Full Stage-1 → Stage-2 training pipeline for WDS navigation data.
#
# Runs Stage-1 (VLM SFT) and Stage-2 (trajectory expert) sequentially,
# automatically passing the Stage-1 checkpoint to Stage-2.
#
# Usage:
#   cd recipes/alpamayo1_5_sft
#   bash train_wds_nav_pipeline.sh [OPTIONS]
#
# Options (environment variables):
#   STAGE1_CONFIG    Stage-1 Hydra config name (default: sft_stage1_wds_nav_4cam4frame_coc)
#   STAGE2_CONFIG    Stage-2 Hydra config name (default: sft_stage2_wds_nav_4cam4frame)
#   OUTPUT_ROOT      Root output directory (default: $ALPAMAYO_OUTPUT_ROOT or ./outputs)
#   RUN_NAME         Run name for output dirs (default: wds_nav_pipeline)
#   NPROC_PER_NODE   Number of GPUs (default: 8)
#   SKIP_STAGE1      Set to 1 to skip Stage-1 and use existing checkpoint
#   STAGE1_CKPT      Existing Stage-1 checkpoint (required if SKIP_STAGE1=1)

set -euo pipefail
unset WANDB_API_KEY || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --- Configuration ---
STAGE1_CONFIG="${STAGE1_CONFIG:-sft_stage1_wds_nav_4cam4frame_coc}"
STAGE2_CONFIG="${STAGE2_CONFIG:-sft_stage2_wds_nav_4cam4frame}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ALPAMAYO_OUTPUT_ROOT:-./outputs}}"
RUN_NAME="${RUN_NAME:-wds_nav_pipeline}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

STAGE1_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}_stage1"
STAGE2_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}_stage2"
LOG_DIR="${OUTPUT_ROOT}/${RUN_NAME}_logs"

mkdir -p "${STAGE1_OUTPUT}" "${STAGE2_OUTPUT}" "${LOG_DIR}"

echo "=========================================="
echo " Alpamayo 1.5 SFT — WDS Nav Pipeline"
echo " Stage-1 config : ${STAGE1_CONFIG}"
echo " Stage-2 config : ${STAGE2_CONFIG}"
echo " GPUs           : ${NPROC_PER_NODE}"
echo " Stage-1 output : ${STAGE1_OUTPUT}"
echo " Stage-2 output : ${STAGE2_OUTPUT}"
echo " Logs           : ${LOG_DIR}"
echo "=========================================="

# --- Stage 1: VLM SFT ---
if [[ "${SKIP_STAGE1}" == "1" ]]; then
    if [[ -z "${STAGE1_CKPT}" ]]; then
        echo "ERROR: SKIP_STAGE1=1 but STAGE1_CKPT is not set" >&2
        exit 1
    fi
    echo ""
    echo "[Stage-1] SKIPPED — using existing checkpoint: ${STAGE1_CKPT}"
else
    echo ""
    echo "[Stage-1] Starting VLM SFT training..."
    echo "  Config: ${STAGE1_CONFIG}"
    echo "  Output: ${STAGE1_OUTPUT}"
    echo ""

    .venv/bin/torchrun \
        --nproc_per_node "${NPROC_PER_NODE}" \
        -m alpamayo1_5_sft.train_hf \
        --config-path pkg://alpamayo1_5_sft/configs \
        --config-name "${STAGE1_CONFIG}" \
        paths.output_dir="${STAGE1_OUTPUT}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_DIR}/stage1_train.log"

    echo ""
    echo "[Stage-1] Training complete."

    # Find the latest checkpoint
    STAGE1_CKPT=$(ls -d "${STAGE1_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [[ -z "${STAGE1_CKPT}" ]]; then
        echo "ERROR: No Stage-1 checkpoint found in ${STAGE1_OUTPUT}" >&2
        exit 1
    fi
fi

echo "[Stage-1] Checkpoint: ${STAGE1_CKPT}"

# --- Stage 2: Trajectory Expert ---
echo ""
echo "[Stage-2] Starting trajectory expert training..."
echo "  Config: ${STAGE2_CONFIG}"
echo "  Stage-1 checkpoint: ${STAGE1_CKPT}"
echo "  Output: ${STAGE2_OUTPUT}"
echo ""

.venv/bin/torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name "${STAGE2_CONFIG}" \
    model.stage1_vlm_checkpoint_path="${STAGE1_CKPT}" \
    paths.output_dir="${STAGE2_OUTPUT}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_DIR}/stage2_train.log"

# Find the latest Stage-2 checkpoint
STAGE2_CKPT=$(ls -d "${STAGE2_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

echo ""
echo "=========================================="
echo " Pipeline Complete!"
echo " Stage-1 checkpoint : ${STAGE1_CKPT}"
echo " Stage-2 checkpoint : ${STAGE2_CKPT}"
echo " Logs               : ${LOG_DIR}/"
echo "=========================================="
