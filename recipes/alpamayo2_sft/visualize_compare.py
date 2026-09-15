#!/usr/bin/env python
"""Draw ground truth, the released expert and the fine-tuned one on one figure.

`alpamayo2_super.t4.figure` renders a frame with one prediction, in one colour;
`viz_utils._plot_camera_panel` and `_plot_bev` are built the same way. Comparing
two checkpoints needs three trajectories distinguished by colour, so the panels
here are drawn directly -- but on top of the same projection helpers, so a
waypoint lands in the same pixel it would in the blog figures.

Layout, top to bottom: the front wide camera with all three trajectories
projected onto it, a bird's-eye view of the same three in ego coordinates at t0,
and the chain of thought each model produced.

Frames are split across all eight GPUs, per the repository convention. Each rank
holds both checkpoints -- inference needs no optimizer, so two 2.42 B experts and
one 32 B backbone still fit on a card -- and renders its own subset.

On the prompt. Training pinned both models to a CoT-free prompt, because T4 has
no ground-truth CoT and a prefix that inference cannot reproduce is worth less
than a consistent one. That leaves nothing to show in the CoT panel, so this
defaults to the released CoT prompt instead and labels what it did. The
fine-tuned expert is then reading a prefix it was not trained on, which is a real
caveat and the reason `--prompt` exists: `trajectory` reproduces training
conditions, `cot` shows the reasoning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_expert import build_inputs  # noqa: E402 - the CoT-free training prompt

from alpamayo2_super import helper, viz_utils  # noqa: E402
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.t4.dataset import T4SFTDataset  # noqa: E402

GT_COLOR = "#2ca02c"
BASE_COLOR = "#ff7f0e"
TUNED_COLOR = "#1f77b4"
HISTORY_COLOR = "#888888"
FRONT_WIDE_CAMERA_ID = 1
BEV_XLIM = (-10.0, 80.0)
BEV_YLIM = (-30.0, 30.0)


def infer(model: Alpamayo2Super, data: dict[str, Any], prompt: str, steps: int,
          samples: int, device: str) -> tuple[np.ndarray, str]:
    """Return `(pred_xyz [K, T, 3] in the t0 ego frame, chain of thought)`."""
    if prompt == "cot":
        messages = helper.create_messages(data, model.config)
        processor = helper.get_processor(model.tokenizer, model.config)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            add_vision_id=False, continue_final_message=False,
        )
        images = data["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized = dict(processor(text=text, images=images, videos=None, padding=False,
                                   return_tensors="pt", do_rescale=False))
        inputs = {"tokenized_data": tokenized,
                  "ego_history_xyz": data["ego_history_xyz"],
                  "ego_history_rot": data["ego_history_rot"]}
    else:
        inputs = build_inputs(data, model)

    inputs = helper.to_device(inputs, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, _, extra = model.sample_trajectories_from_data(
            data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=samples,
            diffusion_kwargs={"inference_step": steps}, return_extra=True,
        )
    cots = viz_utils._extract_cots(extra)
    return (pred_xyz.detach().float().cpu().numpy().reshape(-1, 64, 3),
            cots[0] if cots else "")


def _project(traj: np.ndarray, calibration: dict[str, Any], shape: tuple[int, int]):
    """Project one ego-frame trajectory to pixels, keeping only what is in front."""
    camera_xyz = viz_utils._camera_coordinates(traj, calibration)
    pixels = viz_utils._project_camera_points(camera_xyz, calibration)
    height, width = shape
    visible = (
        (camera_xyz[:, 2] > 0.5)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return pixels[visible]


def render(data: dict[str, Any], gt: np.ndarray, base: np.ndarray, tuned: np.ndarray,
           base_cot: str, tuned_cot: str, title: str, out_path: Path,
           prompt: str, tight: bool = True) -> None:
    """Write the camera / BEV / CoT figure.

    :param tight: trim the margins to the content. Good for a still; wrong for a
        video frame, because the trim depends on how long the chain of thought is,
        so consecutive frames come out different sizes and the encoder refuses the
        second one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    camera_ids = [int(v) for v in data["camera_indices"].tolist()]
    source = camera_ids.index(FRONT_WIDE_CAMERA_ID)
    image = viz_utils._tensor_to_image(data["image_frames"][source, -1])
    calibration = data.get("camera_calibrations", {}).get(FRONT_WIDE_CAMERA_ID)

    figure = plt.figure(figsize=(18, 12), dpi=110)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 0.55], width_ratios=[1.35, 1.0],
                               left=0.04, right=0.98, top=0.93, bottom=0.03,
                               wspace=0.12, hspace=0.18)

    # --- front camera ---
    ax = figure.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_autoscale_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("CAM_FRONT_WIDE", fontsize=11)
    if calibration is not None:
        for traj, colour, label in ((gt, GT_COLOR, "ground truth"),
                                    (base[0], BASE_COLOR, "released expert"),
                                    (tuned[0], TUNED_COLOR, "fine-tuned expert")):
            pixels = _project(traj, calibration, image.shape[:2])
            if len(pixels):
                ax.plot(pixels[:, 0], pixels[:, 1], color=colour, linewidth=2.6,
                        label=label, alpha=0.95)
                ax.scatter(pixels[:, 0], pixels[:, 1], color=colour, s=9, zorder=3)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    else:
        ax.text(0.5, 0.5, "no calibration for this camera", ha="center",
                transform=ax.transAxes)

    # --- BEV ---
    ax = figure.add_subplot(grid[0, 1])
    history = data["ego_history_xyz"].detach().float().cpu().numpy().reshape(-1, 3)
    ax.plot(history[:, 0], history[:, 1], color=HISTORY_COLOR, linewidth=2.0,
            linestyle="--", label="history")
    ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOR, linewidth=2.6, label="ground truth")
    # Every sampled trajectory is drawn, faintly after the first, so that spread
    # between samples is visible rather than averaged away.
    for k, traj in enumerate(base):
        ax.plot(traj[:, 0], traj[:, 1], color=BASE_COLOR, linewidth=2.2 if k == 0 else 1.0,
                alpha=1.0 if k == 0 else 0.35, label="released expert" if k == 0 else None)
    for k, traj in enumerate(tuned):
        ax.plot(traj[:, 0], traj[:, 1], color=TUNED_COLOR, linewidth=2.2 if k == 0 else 1.0,
                alpha=1.0 if k == 0 else 0.35, label="fine-tuned expert" if k == 0 else None)
    ax.plot(0, 0, marker="s", color="black", markersize=7)
    ax.set_xlim(*BEV_XLIM); ax.set_ylim(*BEV_YLIM)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")
    ax.set_title("BEV, ego frame at t0", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # --- chain of thought ---
    for column, (label, cot, colour) in enumerate(
        (("released expert", base_cot, BASE_COLOR), ("fine-tuned expert", tuned_cot, TUNED_COLOR))
    ):
        ax = figure.add_subplot(grid[1, column])
        ax.axis("off")
        ax.set_title(f"chain of thought — {label}", fontsize=11, color=colour, loc="left")
        body = "\n".join(textwrap.wrap(cot.strip(), 92)[:14]) if cot.strip() else (
            "(no chain of thought: the trajectory-only prompt asks for none)"
        )
        ax.text(0, 1, body, va="top", ha="left", fontsize=9.5, family="monospace",
                transform=ax.transAxes, wrap=True)

    def errors(pred: np.ndarray) -> tuple[float, float]:
        """min ADE and min FDE for one model's samples, as the evaluation defines them."""
        d = np.linalg.norm(pred[:, :, :2] - gt[None, :, :2], axis=-1)
        return float(d.mean(axis=-1).min()), float(d[:, -1].min())

    base_ade, base_fde = errors(base)
    tuned_ade, tuned_fde = errors(tuned)
    figure.suptitle(
        f"{title}   |   prompt: {prompt}\n"
        f"minADE  released {base_ade:.2f} m -> fine-tuned {tuned_ade:.2f} m"
        f"   ({tuned_ade - base_ade:+.2f})        "
        f"minFDE  released {base_fde:.2f} m -> fine-tuned {tuned_fde:.2f} m"
        f"   ({tuned_fde - base_fde:+.2f})",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, **({"bbox_inches": "tight"} if tight else {}))
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True)
    p.add_argument("--window-list", default=None)
    p.add_argument("--expert-weights", required=True)
    p.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    p.add_argument("--scene", default=None, help="substring of the scene dir to pick")
    p.add_argument("--behaviour", default=None, nargs="*",
                   help="meta-action classes to sample, e.g. strong_deceleration steer_left")
    p.add_argument("--rows", type=int, nargs="*", default=None, help="explicit dataset rows")
    p.add_argument("--n-frames", type=int, default=4, help="frames to render when picking")
    p.add_argument("--prompt", default="cot", choices=("cot", "trajectory"))
    p.add_argument("--num-traj-samples", type=int, default=3)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--decode", default="gpu", choices=("gpu", "cache"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cpu", help="decode device")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    dataset = T4SFTDataset(args.index, cache_dir=args.cache_dir,
                           window_list=args.window_list, decode=args.decode,
                           device=args.device)
    log(str(dataset))

    if args.rows:
        rows = list(args.rows)
    else:
        rows = []
        # Per behaviour, not pooled: asking for four frames of a mixed pool returns
        # four of whatever is most common, which is the opposite of the comparison
        # a behaviour breakdown is for.
        wanted = args.behaviour or [None]
        for behaviour in wanted:
            candidates = [
                i for i in range(len(dataset))
                if (args.scene is None or args.scene in dataset.scene_of(i))
                and (behaviour is None or (dataset.behaviours is not None
                                           and dataset.behaviours[i][0] == behaviour))
            ]
            if not candidates:
                log(f"no window matches scene={args.scene!r} behaviour={behaviour!r}")
                continue
            # Spread the picks across the match rather than taking the first few,
            # which would all be the same moment of the same scene.
            step = max(1, len(candidates) // args.n_frames)
            rows.extend(candidates[::step][: args.n_frames])
    if not rows:
        raise SystemExit("no windows selected")
    log(f"rendering {len(rows)} frames across {world_size} rank(s)")

    mine = rows[rank::world_size]
    if not mine:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return 0

    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                           device_map=device)
    model.eval()

    tuned_state = load_file(str(Path(args.expert_weights) / "model.safetensors"))
    base_state = {k: v.detach().clone() for k, v in model.expert.state_dict().items()}

    out_dir = Path(args.out_dir)
    for row in mine:
        data = dataset[row]
        gt = data["ego_future_xyz"].detach().float().cpu().numpy()[0, 0]

        model.expert.load_state_dict(base_state, strict=False)
        base_pred, base_cot = infer(model, data, args.prompt, args.diffusion_steps,
                                    args.num_traj_samples, device)
        model.expert.load_state_dict(tuned_state, strict=False)
        tuned_pred, tuned_cot = infer(model, data, args.prompt, args.diffusion_steps,
                                      args.num_traj_samples, device)

        scene = dataset.scene_of(row)
        label = f"{scene}  @frame {data['frame_index']}"
        if dataset.behaviours is not None:
            label += f"  [{dataset.behaviours[row][0]}]"
        name = scene.replace("/", "__") + f"__{data['frame_index']:05d}.png"
        render(data, gt, base_pred, tuned_pred, base_cot, tuned_cot, label,
               out_dir / name, args.prompt)
        print(f"wrote {out_dir / name}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    log(json.dumps({"rows": rows, "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
