#!/bin/bash
# Step 1 of the CoC autolabeler over T4 scenes: ego meta-actions from the pose track.
#
# CPU only and ~0.14 s/scene, so the whole prd_jt subtree is minutes of work. The
# QOS caps a user at 10 running jobs, so the parallelism lives inside the job as
# a process pool rather than in the array width.
#
#SBATCH --job-name=a2-metaaction
#SBATCH --partition=advanced_e2e
# Scheduled by partition, not pinned to a node: everything this job reads --
# weights, T4 scenes, the venv and its interpreter -- is on NFS, so it can run
# wherever the GPUs are free. node01 is excluded because it is the login node and
# carries a resident dashboard service.
#SBATCH --exclude=node01
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%A_%a.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%A_%a.err

set -euo pipefail
INDEX="${1:?usage: sbatch --array=0-N meta_action_array.sh <index.json> <save_dir>}"
SAVE_DIR="${2:?missing save_dir}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd /mnt/storage_rdma/workspaces/takeuchi/alpamayo2

exec ./.venv/bin/python -m meta_action.t4_labeling \
    --index "$INDEX" --save-dir "$SAVE_DIR" --workers "${SLURM_CPUS_PER_TASK:-8}"
