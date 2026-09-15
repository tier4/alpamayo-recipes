# CLAUDE.md

## Storage rules

`/mnt/nvme` is **node-local**, despite every node having a directory by that name.
Anything left there is invisible from every other node, and a run moved to a
different node finds an empty directory rather than an error. This has already
cost one migration's worth of outputs.

- **Final weights, evaluation results and anything else worth keeping go to
  `/mnt/storage_rdma/workspaces/takeuchi/`** (NFS, shared by every node).
- Intermediate checkpoints and scratch may stay on `/mnt/nvme/alpamayo_outputs/`
  while a run is in flight — writing them to NFS every few thousand steps is
  slower than it is worth. **Copy the final weights out to NFS when the run ends.**
- Never use `$HOME`, `./outputs`, or `/tmp`: those are node-local too, and the
  root disk is small enough that filling it takes training down with it.
- `/tmp` must be kept clear for torchrun's internal elastic-launch logging.
- Slurm `--output` / `--error` must name a path on NFS. The default lands in the
  compute node's `$HOME`, where the login node can never read it.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ALPAMAYO_CHECKPOINT_PATH` | Pretrained Alpamayo 1.5 checkpoint (A1 format) | (mandatory) |
| `ALPAMAYO_OUTPUT_ROOT` | Root for training/eval outputs | `/mnt/nvme/alpamayo_outputs` |
| `ALPAMAYO2_SUPER_MODEL_ID` | Alpamayo 2 checkpoint, or a local directory | `nvidia/Alpamayo2-Super` |
| `HF_HOME` | Hugging Face cache — point it at NFS so every node shares one copy | (mandatory for jobs) |

## Recipes

| Recipe | What it trains |
|---|---|
| `recipes/alpamayo1_5_sft` | Alpamayo 1.5 SFT: navigation, DriveLM VQA, LingoQA |
| `recipes/alpamayo1_x_rl` | Alpamayo 1.x RL on PhysicalAI-AV clips |
| `recipes/alpamayo2_sft` | Alpamayo 2 action expert on TIER IV T4 scenes |

## Training

- `dataloader_num_workers` should be ≤ 8 with `persistent_workers=true`. Beyond
  that the worker processes accumulate RAM over a long run until the OOM killer
  takes the job.
- Prefer `gradient_accumulation_steps` over a large
  `per_device_train_batch_size`: it lowers the per-step data-loading demand and
  keeps the GPUs busy.
- A multi-GPU job must pass `--mem-per-gpu`. The partition defaults to 257,500 M
  per GPU, so `--gres=gpu:8` silently asks for the whole node's memory and the
  job waits on `Resources` forever.
