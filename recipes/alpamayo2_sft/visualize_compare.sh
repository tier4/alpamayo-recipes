#!/bin/bash
# Render GT vs released vs fine-tuned trajectories for a few windows.
#
# Node is a scheduling choice, not a data one: everything read here is on NFS.
# Pin it wherever the GPUs are actually free.
#
# All eight GPUs, per the repository convention. Rendering is split by frame:
# each rank holds both checkpoints -- inference needs no optimizer state, so the
# 32 B backbone and two 2.42 B experts fit on one card -- and draws its own subset.
#
#SBATCH --job-name=a2-viz
#SBATCH --partition=advanced_e2e
# Scheduled by partition: everything this job reads -- weights, T4 scenes, the venv
# and its interpreter -- is on NFS, so it runs wherever there is room. Outputs go to
# NFS too, which is what makes node01 fair game despite its small root disk.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
A=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft

export HF_HOME=/mnt/storage_rdma/workspaces/takeuchi/.hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

NGPU="${NGPU:-8}"
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2
exec ./.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" \
    "$RECIPE/visualize_compare.py" \
    --index "$A/index/prdjtval_tele.json" \
    --window-list "$A/index/prdjtval_windows.json" \
    --expert-weights "$A/weights/a2_sft_expert_v1/expert-final" \
    --decode gpu --device cpu \
    --out-dir "$A/viz/a2_sft_expert_v1" \
    "$@"
