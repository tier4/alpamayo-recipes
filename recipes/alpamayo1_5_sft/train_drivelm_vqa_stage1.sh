#!/usr/bin/env bash
# Stage-1 VQA SFT on the local DriveLM-style VQA export.
#
# Usage:
#   cd recipes/alpamayo1_5_sft
#   CKPT_DIR_A1=/path/to/Alpamayo-1.5-10B-A1-format bash train_drivelm_vqa_stage1.sh
#
# Environment variables:
#   CKPT_DIR_A1        converted A1-format Alpamayo-1.5 checkpoint (required)
#   DRIVELM_VQA_ROOT  DriveLM VQA data root (default: /mnt/nvme/drivelm_vqa_t4)
#   NPROC_PER_NODE    number of GPUs per node (default: 8)
#   OUTPUT_DIR        output directory (default: output_stage1_drivelm_vqa)
#   EXTRA_ARGS        additional Hydra overrides appended to the command

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${CKPT_DIR_A1:-}" ]]; then
    echo "CKPT_DIR_A1 is required. It must point to the converted A1-format checkpoint." >&2
    exit 1
fi

export DRIVELM_VQA_ROOT="${DRIVELM_VQA_ROOT:-/mnt/nvme/drivelm_vqa_t4}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-output_stage1_drivelm_vqa}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
    if [[ -x "${SCRIPT_DIR}/.venv/bin/torchrun" ]]; then
        TORCHRUN_BIN="${SCRIPT_DIR}/.venv/bin/torchrun"
    elif [[ -x "${SCRIPT_DIR}/a1_5_sft/bin/torchrun" ]]; then
        TORCHRUN_BIN="${SCRIPT_DIR}/a1_5_sft/bin/torchrun"
    else
        TORCHRUN_BIN="torchrun"
    fi
fi

echo "=========================================="
echo " Alpamayo 1.5 SFT - DriveLM VQA Stage 1"
echo " GPUs per node    : ${NPROC_PER_NODE}"
echo " DriveLM VQA root : ${DRIVELM_VQA_ROOT}"
echo " Checkpoint       : ${CKPT_DIR_A1}"
echo " Output dir       : ${OUTPUT_DIR}"
echo " Config           : sft_stage1_drivelm_vqa"
echo " torchrun         : ${TORCHRUN_BIN}"
echo "=========================================="

"${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_drivelm_vqa \
    model.checkpoint_path="${CKPT_DIR_A1}" \
    data.train_dataset.data_root="${DRIVELM_VQA_ROOT}" \
    data.val_dataset.data_root="${DRIVELM_VQA_ROOT}" \
    paths.output_dir="${OUTPUT_DIR}" \
    ${EXTRA_ARGS}
