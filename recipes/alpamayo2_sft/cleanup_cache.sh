#!/bin/bash
# Remove the image cache. It was built to pre-pay the decode cost -- three of the
# six rig slots are HEVC and the other three are 2880x1860 JPEG -- but the run
# decodes on the fly instead, and at 6.9 s/step a 2.3 s decode hides entirely in
# the workers. Nothing reads it any more.
#
# The listing is written to NFS first: 468 GB is not recoverable, and regenerating
# even the val third of it costs about 40 minutes.
#SBATCH --job-name=a2-cleanup-cache
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
TARGET=/mnt/nvme/alpamayo_outputs/t4_cache
MANIFEST=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets/deleted_cache_manifest.txt

if [ ! -d "$TARGET" ]; then echo "$TARGET is already gone"; exit 0; fi
{
  echo "# contents of $TARGET before deletion, $(date -Is)"
  du -sh "$TARGET"
  echo "## cached scenes: $(find "$TARGET" -maxdepth 1 -mindepth 1 -type d | wc -l)"
  find "$TARGET" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort
} > "$MANIFEST" 2>&1
head -5 "$MANIFEST"

echo "--- deleting ---"
rm -rf "$TARGET"
echo "removed $TARGET"
df -h /mnt/nvme | tail -1
