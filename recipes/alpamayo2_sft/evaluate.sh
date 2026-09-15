#!/bin/bash
# Score the fine-tuned expert against the released one on held-out prd_jt_val.
#
# All eight GPUs, per the repository convention. Scoring is data parallel: each
# rank keeps its own copy of the model -- inference needs no gradients or
# optimizer state, so 71.6 GB sits comfortably on one card -- and takes a strided
# slice of the windows. Both models are still scored in one process per rank, so
# the comparison remains on identical windows.
#
# Runs on `ubuntu` rather than node02. Everything it reads is on NFS -- the
# weights, the T4 scenes, the venv and its interpreter -- so unlike the cleanup
# jobs it is not tied to the node holding /mnt/nvme, and node02 is occupied.
#
# --mem-per-gpu, not --mem: the partition defaults to 257,500 M per GPU, which at
# eight GPUs is the whole node and can never be scheduled alongside anything else.
#
#SBATCH --job-name=a2-eval
#SBATCH --partition=advanced_e2e
# Scheduled by partition: everything this job reads -- weights, T4 scenes, the venv
# and its interpreter -- is on NFS, so it runs wherever there is room. Outputs go to
# NFS too, which is what makes node01 fair game despite its small root disk.
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-gpu=100G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
A=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft

export HF_HOME=/mnt/storage_rdma/workspaces/takeuchi/.hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

NGPU="${NGPU:-8}"
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2
exec ./.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" \
    "$RECIPE/evaluate.py" \
    --index "$A/index/prdjtval_tele.json" \
    --window-list "$A/index/prdjtval_windows.json" \
    --expert-weights "$A/weights/a2_sft_expert_v1/expert-final" \
    --decode gpu --device cpu \
    --out "$A/eval/a2_sft_expert_v1.json" \
    "$@"
