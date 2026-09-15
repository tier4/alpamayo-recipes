#!/bin/bash
# Stage 1 of Alpamayo 1.5 on T4 scenes, front camera only, navigation-conditioned.
#
# Eight GPUs per the cluster convention. `--time` is 24 h rather than the 7-day
# maximum: a shorter limit backfills into gaps a long one cannot reach, and the
# Trainer checkpoints, so a run that needs longer resumes with
# `trainer.resume_from_checkpoint`.
#
#SBATCH --job-name=a15-t4-nav-s1
#SBATCH --partition=advanced_e2e
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo1_5_sft

export ALPAMAYO_OUTPUT_ROOT="${ALPAMAYO_OUTPUT_ROOT:-/mnt/nvme/alpamayo_outputs}"
export HF_HOME=/mnt/storage_rdma/workspaces/takeuchi/.hf
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
unset WANDB_API_KEY

cd "$RECIPE"
exec ./.venv/bin/torchrun --standalone --nproc_per_node="${NGPU:-8}" \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_t4_nav_front \
  "$@"
