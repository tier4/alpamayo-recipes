#!/bin/bash
# SFT the Alpamayo2-Super action expert on curated T4 windows.
#
# Eight GPUs under ZeRO-3, and the sharding is not optional: 35.8 B in bf16 is
# 71.6 GB of weights, and the expert's gradients and AdamW state add ~24 GB, so a
# step does not fit on an 80 GB card even with the VLM frozen.
#
# It is ZeRO rather than device_map="auto" because the expert appends to the VLM's
# KV cache -- a cache spread over a layer map lands on the wrong device and
# torch.cat fails. ZeRO keeps the computation local, so the expert and the cache
# stay together and all eight GPUs do real work instead of seven holding weights.
#
# Stage 2, not 3. Only the 2.42 B expert trains; the 32 B VLM is frozen, so
# sharding its parameters buys nothing and costs an all-gather of every layer on
# every forward. Stage 2 leaves the weights replicated and shards only what is
# actually trained -- gradients and optimizer state -- with the optimizer offloaded
# to host memory to stay clear of the 80 GB ceiling.
#
# Checkpoints go to node-local NVMe so training does not hammer NFS; the finished
# weights are copied to NFS, because /mnt/nvme does not survive a move to another
# node -- which is how this project lost its previous outputs.
#
#SBATCH --job-name=a2-sft-expert
#SBATCH --partition=advanced_e2e
# Pinned to node02 on purpose: /mnt/nvme is node-local storage, and the data this
# job reads or writes lives on that node's copy. Removing the pin would silently
# operate on a different, empty directory.
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
# --mem-per-gpu, not --mem: the partition defaults to 257,500 M per GPU, which at
# eight GPUs is the node's entire 2,060 G. That request can never be satisfied
# while any other job holds memory, and the job sits on "Resources" forever.
#SBATCH --mem-per-gpu=100G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
A=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets
RECIPE=/mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft
RUN="${RUN_NAME:-a2_sft_expert}"

export HF_HOME=/mnt/storage_rdma/workspaces/takeuchi/.hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

NGPU="${NGPU:-8}"
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2
exec ./.venv/bin/torchrun --standalone --nproc_per_node="$NGPU" \
    "$RECIPE/train_expert.py" \
    --deepspeed "${ZERO_CONFIG:-$RECIPE/configs/zero2.json}" \
    --index "$A/index/prdjt_tele.json" \
    --window-list "$A/index/prdjt_windows.json" \
    --decode gpu \
    --output-dir "/mnt/nvme/alpamayo_outputs/$RUN" \
    --final-dir "/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets/weights/$RUN" \
    "$@"
