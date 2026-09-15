#!/usr/bin/env python
"""Render the run-up to a keyframe as a 10 Hz comparison video.

`visualize_compare.py` answers "what do the two models do at this instant".
A keyframe is chosen because a behaviour *changes* there, and a still cannot show
the approach -- whether a model started braking late, or drifted wide before the
turn. This renders every 10 Hz frame from some seconds before the keyframe up to
it, so the divergence is visible as it develops.

Two phases, because they bottleneck differently and the split is what makes a
layout change cheap. Inference is GPU-bound at a few seconds a frame and is split
across all eight GPUs; rendering is matplotlib and a CPU decode per frame, which
is why it runs in a process pool. A whole 60 s clip is ~600 frames, and rendering
those one at a time takes longer than inferring them on eight GPUs did.
Re-running with `--phase render` redraws from the saved predictions without
touching a GPU at all.

Inference runs at every frame by default. The released renderer holds a
prediction between inferences and pose-compensates it, which is what a planner
slower than the camera looks like; here the question is what the two models
predict from the same input, so both are re-run each frame and nothing is held.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_compare import infer, render  # noqa: E402 - one figure definition

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.t4.dataset import T4SFTDataset  # noqa: E402


def frame_range(dataset: T4SFTDataset, row: int, lead_s: float, hz: float) -> list[int]:
    """Frames from `lead_s` seconds before the keyframe up to it, at `hz`.

    Clamped to the scene's usable t0 range rather than its frame count: a window
    needs 16 frames of history behind it and 65 ahead, so the first usable centre
    is not frame zero.
    """
    scene_idx, center = dataset._rows[row]
    entry_first, entry_last = dataset._bounds[scene_idx]
    stride = max(1, int(round(10.0 / hz)))
    first = max(entry_first, center - int(round(lead_s * 10)))
    return [f for f in range(first, min(center, entry_last) + 1, stride)]


def _content_bottom(image: np.ndarray, tolerance: int = 6) -> int:
    """Row index just past the last row that carries anything but page background.

    Background is read from the frame's own top-left pixel rather than assumed
    white, so this keeps working if the figure ever gets a different ground.
    """
    background = image[0, 0].astype(np.int16)
    ink = (np.abs(image.astype(np.int16) - background).max(axis=2) > tolerance)
    rows = np.flatnonzero(ink.any(axis=1))
    return int(rows[-1]) + 1 if rows.size else image.shape[0]


_RENDER_STATE: dict[str, Any] = {}


def _render_init(index: str, window_list: str | None, cache_dir: str | None,
                 decode: str, device: str) -> None:
    """Give each pool worker its own dataset handle."""
    _RENDER_STATE["dataset"] = T4SFTDataset(index, cache_dir=cache_dir,
                                            window_list=window_list, decode=decode,
                                            device=device)


def _render_one(job: tuple) -> str:
    """Draw one frame from its saved predictions."""
    npz_path, png_path, scene_idx, frame, center, scene_dir, label, prompt = job
    # Reuse a frame already drawn. Only the assembly below the figure -- the crop,
    # the encode -- changes often; redrawing 300 figures to change how they are
    # cropped costs ten minutes and produces the same pixels. Delete the PNGs to
    # force a redraw after changing the figure itself.
    if Path(png_path).exists():
        return png_path
    saved = np.load(npz_path, allow_pickle=False)
    data = _RENDER_STATE["dataset"]._load(scene_idx, frame)
    render(
        data, saved["gt"], saved["base_pred"], saved["tuned_pred"],
        str(saved["base_cot"]), str(saved["tuned_cot"]),
        f"{scene_dir}  @frame {frame}  (keyframe {center} [{label}], "
        f"t{(frame - center) / 10:+.1f} s)",
        Path(png_path), prompt, tight=False,
    )
    return png_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True)
    p.add_argument("--window-list", default=None)
    p.add_argument("--expert-weights", required=True)
    p.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    p.add_argument("--row", type=int, nargs="*", default=None,
                   help="dataset rows to render; one video each")
    p.add_argument("--behaviour", nargs="*", default=None,
                   help="pick one keyframe per named class; one video each")
    p.add_argument("--whole-clip", action="store_true",
                   help="render the scene's whole usable range instead of a run-up")
    p.add_argument("--lead-seconds", type=float, default=6.0)
    p.add_argument("--infer-hz", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--prompt", default="cot", choices=("cot", "trajectory"))
    p.add_argument("--num-traj-samples", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--decode", default="gpu", choices=("gpu", "cache"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--phase", default="all", choices=("all", "infer", "render"))
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out", required=True, help="output mp4, or a directory for several")
    p.add_argument("--render-workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = p.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    needs_gpu = args.phase in ("all", "infer")
    if world_size > 1 and needs_gpu and not dist.is_initialized():
        dist.init_process_group("nccl")
    if needs_gpu:
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if needs_gpu else "cpu"

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    dataset = T4SFTDataset(args.index, cache_dir=args.cache_dir,
                           window_list=args.window_list, decode=args.decode,
                           device=args.device)

    rows: list[int] = []
    if args.row:
        rows = list(args.row)
    elif args.behaviour and dataset.behaviours is not None:
        for behaviour in args.behaviour:
            matches = [i for i in range(len(dataset)) if dataset.behaviours[i][0] == behaviour]
            if not matches:
                log(f"no keyframe of behaviour {behaviour!r}")
                continue
            rows.append(matches[len(matches) // 2])
    if not rows:
        raise SystemExit("pass --row or --behaviour")

    # One clip per row, each with its own frame list and its own output. Distinct
    # scenes, because three videos of one scene is one video's worth of evidence.
    clips = []
    seen_scenes: set[int] = set()
    for row in rows:
        scene_idx, center = dataset._rows[row]
        if scene_idx in seen_scenes:
            continue
        seen_scenes.add(scene_idx)
        if args.whole_clip:
            first, last = dataset._bounds[scene_idx]
            stride = max(1, int(round(10.0 / args.infer_hz)))
            frames = list(range(first, last + 1, stride))
        else:
            frames = frame_range(dataset, row, args.lead_seconds, args.infer_hz)
        label = dataset.behaviours[row][0] if dataset.behaviours else "keyframe"
        clips.append({"row": row, "scene_idx": scene_idx, "center": center,
                      "scene_dir": dataset.scene_of(row), "frames": frames, "label": label})
        log(f"{dataset.scene_of(row)}\n  keyframe {center} [{label}], "
            f"{len(frames)} frames {frames[0]}..{frames[-1]} at {args.infer_hz} Hz")

    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    if args.phase in ("all", "infer"):
        model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                               device_map=device)
        model.eval()
        tuned_state = load_file(str(Path(args.expert_weights) / "model.safetensors"))
        base_state = {k: v.detach().clone() for k, v in model.expert.state_dict().items()}

        for clip in clips:
            work = work_root / clip["scene_dir"].replace("/", "__")
            work.mkdir(parents=True, exist_ok=True)
            mine = clip["frames"][rank::world_size]
            for n, frame in enumerate(mine, 1):
                out = work / f"frame_{frame:05d}.npz"
                if out.exists():
                    continue
                try:
                    data = dataset._load(clip["scene_idx"], frame)
                except Exception as error:  # noqa: BLE001 - a gap is not the run
                    print(f"  rank{rank} frame {frame} unreadable: {error}", flush=True)
                    continue
                model.expert.load_state_dict(base_state, strict=False)
                base_pred, base_cot = infer(model, data, args.prompt, args.diffusion_steps,
                                            args.num_traj_samples, device)
                model.expert.load_state_dict(tuned_state, strict=False)
                tuned_pred, tuned_cot = infer(model, data, args.prompt, args.diffusion_steps,
                                              args.num_traj_samples, device)
                np.savez_compressed(
                    out, base_pred=base_pred, tuned_pred=tuned_pred,
                    gt=data["ego_future_xyz"].detach().float().cpu().numpy()[0, 0],
                    base_cot=np.array(base_cot), tuned_cot=np.array(tuned_cot),
                )
                if rank == 0 and n % 10 == 0:
                    log(f"  [{clip['label']}] inferred {n}/{len(mine)}")
            if dist.is_initialized():
                dist.barrier()

    if rank != 0:
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    if args.phase == "infer":
        log(f"predictions in {work_root}")
        return 0

    import mediapy as media

    out_arg = Path(args.out)
    many = len(clips) > 1 or out_arg.suffix != ".mp4"
    written = []
    for clip in clips:
        work = work_root / clip["scene_dir"].replace("/", "__")
        # Rendering is a CPU decode plus a matplotlib figure per frame, about four
        # seconds each; a 600-frame clip is 40 minutes serial and two in a pool.
        jobs = [(str(work / f"frame_{f:05d}.npz"), str(work / f"frame_{f:05d}.png"),
                 clip["scene_idx"], f, clip["center"], clip["scene_dir"], clip["label"],
                 args.prompt) for f in clip["frames"]
                if (work / f"frame_{f:05d}.npz").exists()]
        log(f"rendering {len(jobs)} frames for {clip['label']} with {args.render_workers} workers")

        _RENDER_STATE["dataset"] = dataset
        with ProcessPoolExecutor(max_workers=max(1, args.render_workers),
                                 initializer=_render_init,
                                 initargs=(args.index, args.window_list, args.cache_dir,
                                           args.decode, args.device)) as pool:
            done = list(pool.map(_render_one, jobs))

        images = []
        for png in [j[1] for j in jobs]:
            if not Path(png).exists():
                continue
            image = media.read_image(png)
            # matplotlib writes RGBA; a video encoder wants three channels.
            if image.ndim == 3 and image.shape[-1] == 4:
                image = image[..., :3]
            images.append(image)
        if not images:
            log(f"no frames rendered for {clip['label']}")
            continue

        # The chain-of-thought panel is sized for the longest text the layout
        # allows, and the model writes a sentence or two, so most frames end in a
        # band of blank page. Crop it -- but to one height for the whole clip,
        # taken from the frame whose text runs longest, or the encoder rejects the
        # first frame that differs.
        bottom = max(_content_bottom(image) for image in images)
        limit = min(images[0].shape[0], bottom + 24)
        images = [image[: limit - limit % 2, : image.shape[1] - image.shape[1] % 2]
                  for image in images]

        stem = clip["scene_dir"].replace("/", "__") + f"__{clip['label']}"
        target = (out_arg / f"{stem}.mp4") if many else out_arg
        target.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(target), images, fps=args.fps)
        written.append({"scene": clip["scene_dir"], "keyframe": clip["center"],
                        "behaviour": clip["label"], "frames": len(images),
                        "out": str(target)})
        log(f"wrote {target}  ({len(images)} frames at {args.fps} fps)")

    print(json.dumps(written, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
