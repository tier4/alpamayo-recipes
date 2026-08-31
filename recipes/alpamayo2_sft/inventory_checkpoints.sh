#!/bin/bash
# Inventory the run's node-local artefacts. /mnt/nvme is local to node02, so this
# has to run there; it only reads, so what to keep is still a decision to make
# afterwards rather than one baked in here.
#SBATCH --job-name=a2-inventory
#SBATCH --partition=advanced_e2e
# Pinned to node02 on purpose: /mnt/nvme is node-local storage, and the data this
# job reads or writes lives on that node's copy. Removing the pin would silently
# operate on a different, empty directory.
#SBATCH --nodelist=node02
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.log
#SBATCH --error=/mnt/storage_rdma/workspaces/takeuchi/logs/%x_%j.err

set -uo pipefail
OUT=/mnt/storage_rdma/workspaces/takeuchi/alpamayo_assets/nvme_inventory.txt
{
  echo "# node02 /mnt/nvme/alpamayo_outputs  $(date -Is)"
  df -h /mnt/nvme | tail -1
  echo
  echo "## this run's checkpoints"
  ls -1 /mnt/nvme/alpamayo_outputs/a2_sft_expert_v1/ 2>/dev/null
  du -sh /mnt/nvme/alpamayo_outputs/a2_sft_expert_v1/* 2>/dev/null | sort -k2
  echo
  echo "## everything under alpamayo_outputs"
  du -sh /mnt/nvme/alpamayo_outputs/* 2>/dev/null | sort -h
} > "$OUT" 2>&1
cat "$OUT"
