# Alpamayo 1.5 SFT

This guide covers supervised fine-tuning (SFT) for Alpamayo 1.5 with navigation conditioning and LingoQA VQA. The SFT scripts live under [recipes/alpamayo1_5_sft](./).

Alpamayo 1.5 extends Alpamayo 1 with two new capabilities trained as Stage-1 (VLM) SFT tasks:

- **Navigation conditioning** — trajectory prediction guided by route instructions (e.g. "Turn left in 40m")
- **Visual Question Answering** — driving scene question answering (demonstrated with LingoQA)

Both tasks use the Stage-1 training entry point with different VLA processor configs and datasets. Navigation-conditioned trajectory fine-tuning can then continue into Stage 2 for the trajectory diffusion expert.

## Training Pipeline

Training uses a two-stage pipeline for convergence and stability:

1. **Stage 1:** Fine-tune the VLM (`base_model`) on supervised targets. For navigation, the target is the future trajectory conditioned on route instructions. For LingoQA VQA, the target is the answer text.
2. **Stage 2:** Freeze the Stage-1 VLM and train the action expert (trajectory diffusion) for continuous trajectories. This stage applies to trajectory prediction fine-tuning, such as the navigation-conditioned setup; VQA-only fine-tuning does not use Stage 2.

Alpamayo 1.5 uses Hydra, so you can extend or override configuration in a structured way.

**Weights & Biases:** To log runs to W&B, uncomment the `wandb` default, and set `report_to: wandb` under
`trainer` in [configs/sft_base.yaml](./configs/sft_base.yaml). Additionally, fill in `team` and `project` in
[configs/wandb/default.yaml](./configs/wandb/default.yaml), and have your W&B API key available when training starts.

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone and install

First, define your working directory (all subsequent commands reference `$YOUR_HOME`):

```bash
export YOUR_HOME="/path/to/your/workspace"
```

```bash
git clone https://github.com/NVlabs/alpamayo-recipes.git $YOUR_HOME/alpamayo-recipes
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
uv venv a1_5_sft
source a1_5_sft/bin/activate
uv sync --active
```

`alpamayo_r1` (inference models, action space, geometry) is automatically fetched from
[NVlabs/alpamayo](https://github.com/NVlabs/alpamayo.git) as part of `uv sync` — no separate
clone needed.


## Prepare dataset

### PAI dataset (for navigation)

You can download the full dataset, a subset by chunk, or individual components on demand.

Set your Hugging Face token first:

```bash
export HF_TOKEN=<your Hugging Face token>
```

Download the chunks referenced by the nav annotations:

```bash
cd $YOUR_HOME/alpamayo-recipes
python scripts/download_pai.py \
  --chunk-ids "214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868" \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir /path/to/pai_dataset
```

Navigation annotations are bundled [here](https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json). Each entry has `clip_id`, `t0_relative`, `nav_text`, and optionally `cot`. We'll use these 20 samples as an overfit smoke test for Stage-1 SFT.

### LingoQA dataset (for VQA)

Wayve distributes LingoQA via Google Drive; see the [official repo](https://github.com/wayveai/LingoQA) for the canonical links. Download the **Scenery** split (the loader here uses only Scenery):

- Scenery (Train): https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7
- Evaluation (Test): https://drive.google.com/drive/folders/1oA7W8-Ej_uJEuUxZIjPP5K8hQGGzYsPq

You can either pull the folders through the Drive web UI, or use a tool like [`gdown`](https://github.com/wkentaro/gdown):

```bash
uv pip install gdown
gdown --folder https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7 -O /path/to/LingoQA/
cd /path/to/LingoQA/ && unzip images.zip
```

The expected layout is:

```
LingoQA/
├── train.parquet      # 148k QA pairs, 3.5k video segments
└── images/train/      # front-camera JPEGs (5 frames per segment)
```

Set `data_root: /path/to/LingoQA/` in [sft_stage1_lingoqa](./configs/sft_stage1_lingoqa.yaml) config when post-training using LingoQA.

## Prepare Checkpoint

Download the pretrained Alpamayo 1.5 checkpoint from
[Hugging Face](https://huggingface.co/nvidia/Alpamayo-1.5-10B) into a local directory (Stage 1
loads weights from disk):

```bash
huggingface-cli download nvidia/Alpamayo-1.5-10B --local-dir <path/to/model>
```

Convert it to Alpamayo 1 format.

```bash

cd $YOUR_HOME/alpamayo-recipes
python scripts/convert_checkpoint.py to-a1 --input /path/to/Alpamayo-1.5-10B --output /path/to/Alpamayo-1.5-10B-A1-format
```

Then set `checkpoint_path` to `/path/to/Alpamayo-1.5-10B-A1-format` in [ar1_5_base.yaml](./configs/models/ar1_5_base.yaml):

```yaml
checkpoint_path: /path/to/Alpamayo-1.5-10B-A1-format
```

## VLA Processor Configs

Processor variants live under [vla_processor](./configs/vla_processor/):

| Variant   | Components                                      | Supervised on | Use case                          |
| --------- | ----------------------------------------------- | ------------- | --------------------------------- |
| `default` | image, traj_history, prompt, traj_future        | traj_future   | Alpamayo 1 trajectory SFT         |
| `nav`     | image, traj_history, route, prompt, traj_future | traj_future   | Navigation-conditioned trajectory |
| `vqa`     | image, question, answer                         | answer        | Visual question answering         |

Experiment configs select a variant via Hydra override, e.g.:

```yaml
defaults:
  - override /vla_processor@data.train_dataset.vla_preprocess_args: vqa
```

## Hyperparameters

Adjust settings such as `dataloader_num_workers` or the learning rate in the config as needed.

## Train with Navigation (overfitting on 20 PAI samples)

> Before starting, replace `local_dir` with the correct PAI directory and `annotations_path` with the JSON file from [Alpamayo1.5](https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json) in [configs/sft_stage1_nav.yaml](./configs/sft_stage1_nav.yaml).

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft/
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_nav
```

These 20 samples are not used to train the released Alpamayo 1.5 model, but the loss should also drop to near 0 after hundreds of steps.

![loss](./loss_A1-5_nav.png)

## Train Stage 2 Trajectory Expert (navigation-conditioned)


> **Before starting, fill in `local_dir` and `annotations_path`** in [configs/sft_stage2_nav.yaml](./configs/sft_stage2_nav.yaml) (both `train_dataset` and `val_dataset`)

Use the converted Alpamayo 1.5 checkpoint as the base model and point `model.stage1_vlm_checkpoint_path` to your Stage-1 navigation output checkpoint.

```bash
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  model.pretrained_model_name_or_path=/path/to/Alpamayo-1.5-10B-A1-format \
  model.stage1_vlm_checkpoint_path=/path/to/output_stage1_nav/checkpoint-xxxx
```

The Stage-1 checkpoint should be a Trainer output directory containing `model.safetensors.index.json` and its shards.

## Train with LingoQA

> Before starting, set `data_root` to the LingoQA directory (either edit [configs/sft_stage1_lingoqa.yaml](./configs/sft_stage1_lingoqa.yaml) or override on the CLI as shown below).

The Wayve Scenery split bundles only `train.parquet`; its `val.parquet` lives in the separate **Evaluation** Drive folder. For a quick smoke run, reuse `train.parquet` for the val side by overriding `data.val_dataset.parquet_name=train.parquet`. (For a real eval, download the Evaluation split into its own dir and point `data.val_dataset.data_root` at it with `parquet_name=val.parquet`.)

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_lingoqa \
  model.checkpoint_path=/path/to/Alpamayo-1.5-10B-A1-format \
  data.val_dataset.parquet_name=train.parquet
```

Note that `trainer.deepspeed` is passed as an absolute path because Hydra may change the working directory at runtime, and the shipped relative `configs/deepspeed/zero2.json` may not resolve from there. This also applies to the nav and Stage-2 launches if you hit `ValueError: Expected a string path to an existing deepspeed config`.

Because LingoQA was included in training for the released Alpamayo 1.5 model, the loss should remain low and stable.
![loss](./loss_A1-5_lingoqa.png)

## Train with WebDataset Navigation

This recipe variant trains navigation-conditioned trajectory prediction using locally stored WebDataset tar archives and pre-generated navigation labels, instead of the PAI dataset. No dataset download step is required — all data is on local storage.

### Data layout

| Path | Contents |
|------|----------|
| `$WDS_TAR_ROOT/{date}/scene-XXXXXX.tar` | Camera frames + trajectory (per scene) |
| `$WDS_TAR_ROOT/{date}/scene-XXXXXX.tar.idx` | JSON byte-offset index for random access |
| `$WDS_LABELS_ROOT/{date}/keyframes/segments_relative_timestamp_sampled.json` | Keyframe index (clip_id + event_start_frame per meta-action) |
| `$WDS_LABELS_ROOT/{date}/labels/{clip_id}/label_{frame}.yaml` | Labels (navigation_text, meta_action, effect_on_ego_behavior) |
| `$WDS_LABELS_ROOT/{date}/coc_labels/{clip_id}/cot_{ts}.yaml` | Chain-of-thought labels |

Ten dates are available (2025-11-19 through 2026-01-07), totalling ~3,449 labeled keyframes. Samples without a matching navigation label are silently skipped.

### Run training

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_wds_nav_4cam4frame \
  model.checkpoint_path=/path/to/Alpamayo-1.5-10B-A1-format
```

By default the config trains on all available dates and uses `2026-01-07` as the validation set.

### Override dates or cameras

```bash
# Single-date smoke test (fast iteration)
torchrun --nproc_per_node 1 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_wds_nav_4cam4frame \
  model.checkpoint_path=/path/to/checkpoint \
  "data.train_dataset.dates=[\"2025-11-19\"]" \
  "data.val_dataset.dates=[\"2025-11-19\"]" \
  trainer.max_steps=10

# Multi-camera (front-wide + front-left + front-right)
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_wds_nav_4cam4frame \
  model.checkpoint_path=/path/to/checkpoint \
  "data.train_dataset.cameras=[CAM_FRONT_WIDE,CAM_FRONT_LEFT,CAM_FRONT_RIGHT]" \
  "data.val_dataset.cameras=[CAM_FRONT_WIDE,CAM_FRONT_LEFT,CAM_FRONT_RIGHT]"
```

Available camera names: `CAM_FRONT_WIDE` (PAI index 1), `CAM_FRONT` (index 6), `CAM_FRONT_LEFT` (index 0), `CAM_FRONT_RIGHT` (index 2).

### Full-data 4-camera x 4-frame training

[configs/sft_stage1_wds_nav_4cam4frame.yaml](./configs/sft_stage1_wds_nav_4cam4frame.yaml) matches the
physical_ai_av dataset's default camera/frame setup (4 cameras x 4 temporal frames). It trains on all
10 dates (~2,991 samples after filtering out clips with incomplete camera coverage — see
[docs/wds_camera_coverage_investigation.md](./docs/wds_camera_coverage_investigation.md)) and holds out
`2026-01-07` for validation.

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
bash train_wds_nav_stage1.sh [OUTPUT_DIR]
```

The script launches training inside a detached `tmux` session (`alpamayo_wds_nav_train` by default) so it
survives terminal disconnects; re-running the script re-attaches to an already-running session. Set
`NPROC_PER_NODE`, `EXTRA_ARGS`, `TMUX_SESSION`, or `LOG_FILE` env vars to override defaults.

### Stage 2: trajectory expert on WDS navigation data

[configs/sft_stage2_wds_nav_4cam4frame.yaml](./configs/sft_stage2_wds_nav_4cam4frame.yaml) reuses the same
`WDSNavDataset` (4 cameras x 4 frames, all 10 dates) but swaps in the Stage-2 expert model
(`/models/ar1_5_expert`), freezing the Stage-1-tuned VLM and training the trajectory diffusion head, same
as [sft_stage2_nav.yaml](./configs/sft_stage2_nav.yaml) does for the PAI dataset.

Once the Stage-1 WDS run above has produced a checkpoint, launch Stage 2 with:

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
bash train_wds_nav_stage2.sh /path/to/stage1_output/checkpoint-XXXX [OUTPUT_DIR]
```

`STAGE1_CHECKPOINT_PATH` is required (a Trainer output dir containing `model.safetensors.index.json` and
its shards) and is passed through to `model.stage1_vlm_checkpoint_path`. The base pretrained checkpoint
path is read from `model.pretrained_model_name_or_path` in
[sft_stage2_wds_nav_4cam4frame.yaml](./configs/sft_stage2_wds_nav_4cam4frame.yaml).

## Evaluation

Evaluate the Stage-2 checkpoint against `val_dataset`:

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft
torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  evaluate.eval_ckpt=<path/to/output_stage2/checkpoint-xxxx>
```

Eval is intended for trajectory checkpoints (Stage 2 produced from a navigation Stage-1 run). VQA Stage-1 outputs do not have a trajectory target and aren't covered by the distance metrics above.
