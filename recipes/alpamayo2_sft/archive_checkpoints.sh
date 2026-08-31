#!/bin/bash
# Move what is worth keeping off node-local NVMe, and drop what plainly is not.
#
# Deliberately conservative: it copies one mid-run checkpoint to NFS as a rollback
# point and removes only the smoke-test artefact this session created. The image
# cache and the older Alpamayo-1.5 outputs are left alone -- they are large enough
# that deleting them is a decision to take deliberately, not a side effect of
# tidying.
#SBATCH --job-name=a2-archive
#SBATCH --partition=advanced_e2e
# Pinned to node02 on purpose: /mnt/nvme is node-local storage, and the data this
# job reads or writes lives on that node's copy. Removing the pin would silently
# operate on a different, empty directory.
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -euo pipefail
SRC=/mnt/nvme/alpamayo_outputs/a2_sft_expert_v1
DST=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets/weights/a2_sft_expert_v1

# A mid-run checkpoint is the insurance against the evaluation showing that the
# run went past its best point; without one, "train less" means training again.
mkdir -p "$DST"
rsync -a --info=progress2 "$SRC/expert-step10000" "$DST/"
echo "archived expert-step10000 -> $DST"

# step20000 is the same state as expert-final, which is already on NFS.
if [ -d "$DST/expert-final" ] && [ -d "$SRC/expert-step20000" ]; then
  a=$(sha256sum "$DST/expert-final/model.safetensors" | cut -d" " -f1)
  b=$(sha256sum "$SRC/expert-step20000/model.safetensors" | cut -d" " -f1)
  echo "expert-final == expert-step20000 : $([ "$a" = "$b" ] && echo YES || echo NO)"
fi

rm -rf /mnt/nvme/alpamayo_outputs/a2_sft_smoke
echo "removed a2_sft_smoke (smoke-test artefact)"

echo "--- NFS weights now ---"; du -sh "$DST"/* 2>/dev/null
echo "--- nvme free ---"; df -h /mnt/nvme | tail -1
