#!/bin/bash
# Render the run-up to a keyframe as a 10 Hz comparison video.
#
# Scheduled by partition: everything this job reads -- weights, T4 scenes, the venv
# and its interpreter -- is on NFS, so it runs wherever there is room. Outputs go to
# NFS too, which is what makes node01 fair game despite its small root disk.
#SBATCH --job-name=a2-video
#SBATCH --partition=advanced_e2e
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
A=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft
TAG="${TAG:-clip}"

export HF_HOME=/mnt/storage_rdma/workspaces/takeuchi/.hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

NGPU="${NGPU:-8}"
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2
exec ./.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" \
    "$RECIPE/visualize_video.py" \
    --index "$A/index/prdjtval_tele.json" \
    --window-list "$A/index/prdjtval_windows.json" \
    --expert-weights "$A/weights/a2_sft_expert_v1/expert-final" \
    --decode gpu --device cpu \
    --work-dir "$A/viz/video_work/$TAG" \
    --out "$A/viz/clips_$TAG" \
    "$@"
