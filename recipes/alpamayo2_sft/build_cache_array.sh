#!/bin/bash
# Build the Alpamayo2 image cache for a whole T4 subtree, as a Slurm array.
#
# The cache lands on node02's local NVMe, so this job is pinned there: the
# cache is only reachable from the node that wrote it, and training is pinned
# to the same node for the same reason. /mnt/nvme itself is owned by `ubuntu`
# and not writable, so the root goes under alpamayo_outputs, which is 777.
#
# Usage:
#   sbatch --array=0-31 build_cache_array.sh <index.json> [extra build_cache.py args]
#
#SBATCH --job-name=a2-t4cache
#SBATCH --partition=advanced_e2e
# Pinned to node02 on purpose: /mnt/nvme is node-local storage, and the data this
# job reads or writes lives on that node's copy. Removing the pin would silently
# operate on a different, empty directory.
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=180G
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%A_%a.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%A_%a.err

set -euo pipefail

INDEX="${1:?usage: sbatch --array=0-N build_cache_array.sh <index.json> [args...]}"
shift || true

ALPAMAYO2=/mnt/storage_rdma/workspaces/takeuchi/alpamayo2
CACHE_DIR="${A2_CACHE_DIR:-/mnt/nvme/alpamayo_outputs/t4_cache}"

# uv lives only on node01 and the uv-managed python is node-local, so the venv
# is addressed by full path rather than activated through a tool that is not here.
PYTHON="$ALPAMAYO2/.venv/bin/python"

# Pillow and libav both spawn threads per decode; left alone they each take the
# whole node and fight the other 31 array tasks for it.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ALPAMAYO2"
exec "$PYTHON" /mnt/storage_rdma/workspaces/takeuchi/alpamayo-recipes/recipes/alpamayo2_sft/build_cache.py \
    --index "$INDEX" \
    --cache-dir "$CACHE_DIR" \
    "$@"
