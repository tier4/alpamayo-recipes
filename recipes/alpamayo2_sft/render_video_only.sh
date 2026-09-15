#!/bin/bash
# Re-render a comparison video from saved predictions. No GPU: the inference phase
# already ran and wrote its output, which is the point of splitting the phases --
# a layout change costs a re-render, not a re-run.
#SBATCH --job-name=a2-video-render
#SBATCH --partition=advanced_e2e
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
A=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft
TAG="${TAG:-clip}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2
exec ./.venv/bin/python "$RECIPE/visualize_video.py" \
    --index "$A/index/prdjtval_tele.json" \
    --window-list "$A/index/prdjtval_windows.json" \
    --expert-weights "$A/weights/a2_sft_expert_v1/expert-final" \
    --decode gpu --device cpu --phase render \
    --work-dir "$A/viz/video_work/$TAG" \
    --out "$A/viz/clips_$TAG" \
    "$@"
