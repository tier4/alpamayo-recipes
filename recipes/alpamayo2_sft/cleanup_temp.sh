#!/bin/bash
# Remove /mnt/nvme/alpamayo_outputs/temp, recording what was there first.
#
# The listing is written to NFS before anything is unlinked: 458 GB is not
# recoverable, and "what did we delete" is a question worth being able to answer
# afterwards even when the answer is "intermediate files".
#SBATCH --job-name=a2-cleanup-temp
#SBATCH --partition=advanced_e2e
# Pinned to node02 on purpose: /mnt/nvme is node-local storage, and the data this
# job reads or writes lives on that node's copy. Removing the pin would silently
# operate on a different, empty directory.
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -uo pipefail
TARGET=/mnt/nvme/alpamayo_outputs/temp
MANIFEST=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets/deleted_temp_manifest.txt

if [ ! -d "$TARGET" ]; then echo "$TARGET is already gone"; exit 0; fi

{
  echo "# contents of $TARGET before deletion, $(date -Is)"
  du -sh "$TARGET"
  echo "## top level"
  du -sh "$TARGET"/* 2>/dev/null | sort -h
  echo "## file count"
  find "$TARGET" -type f 2>/dev/null | wc -l
} > "$MANIFEST" 2>&1
cat "$MANIFEST"

echo "--- deleting ---"
rm -rf "$TARGET"
echo "removed $TARGET"
df -h /mnt/nvme | tail -1
