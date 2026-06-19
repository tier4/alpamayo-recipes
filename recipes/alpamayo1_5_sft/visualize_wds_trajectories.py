#!/usr/bin/env python3
"""Visualise GT vs base-model vs fine-tuned-model trajectories on WDS data.

Three-phase pipeline:
  1. Inference: parallel across GPUs, saves predictions as .npz
  2. Visualisation: CPU-only, reads .npz + dataset, generates PNGs
  3. Video: ffmpeg combines PNGs into MP4

Usage:
    cd recipes/alpamayo1_5_sft

    # Phase 1: inference (GPU-parallel)
    .venv/bin/python visualize_wds_trajectories.py infer \
        --base-ckpt /path/to/base --ft-ckpt /path/to/ft \
        --num-gpus 8 --output-dir /path/to/output

    # Phase 2: visualisation (CPU)
    .venv/bin/python visualize_wds_trajectories.py viz \
        --output-dir /path/to/output

    # Phase 3: video
    .venv/bin/python visualize_wds_trajectories.py video \
        --output-dir /path/to/output

    # All three phases at once:
    .venv/bin/python visualize_wds_trajectories.py all \
        --base-ckpt /path/to/base --ft-ckpt /path/to/ft \
        --num-gpus 8 --output-dir /path/to/output
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alpamayo.data.wds_nav import (
    WDS_CAM_CALIB_INDEX,
    _read_bytes,
    load_calibration,
)
from alpamayo.visualization.viz import (
    _plot_trajectory_with_fade,
    _set_tight_trajectory_limits,
    project_waypoints_pinhole,
    rotate_90cc,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    # --- shared args ---
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config-path", default="pkg://alpamayo1_5_sft/configs")
    shared.add_argument("--config-name", default="sft_stage2_wds_nav_4cam4frame")
    shared.add_argument("--output-dir", default=os.environ.get("ALPAMAYO_OUTPUT_ROOT", "./outputs") + "/viz_all")
    shared.add_argument("--sample-indices", type=int, nargs="+", default=None)
    shared.add_argument("--all", action="store_true")
    shared.add_argument("--num-samples", type=int, default=5)
    shared.add_argument("--camera", default="CAM_FRONT")
    shared.add_argument("--num-traj-samples", type=int, default=6)
    shared.add_argument("--tar-root", default=None, help="Override tar_root from config")
    shared.add_argument("--labels-root", default=None, help="Override labels_root from config")
    shared.add_argument("--dates", nargs="*", default=None,
                        help="Override dates (omit value for all dates, e.g. --dates or --dates 2025-06-12)")

    # --- infer ---
    p_infer = sub.add_parser("infer", parents=[shared])
    p_infer.add_argument("--base-ckpt", required=True)
    p_infer.add_argument("--ft-ckpt", required=True)
    p_infer.add_argument("--num-gpus", type=int, default=8)
    p_infer.add_argument("--_role", choices=["base", "ft"], default=None, help=argparse.SUPPRESS)
    p_infer.add_argument("--_gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    p_infer.add_argument("--_num-shards", type=int, default=None, help=argparse.SUPPRESS)

    # --- viz ---
    sub.add_parser("viz", parents=[shared])

    # --- video ---
    p_video = sub.add_parser("video", parents=[shared])
    p_video.add_argument("--fps", type=int, default=2)

    # --- all ---
    p_all = sub.add_parser("all", parents=[shared])
    p_all.add_argument("--base-ckpt", required=True)
    p_all.add_argument("--ft-ckpt", required=True)
    p_all.add_argument("--num-gpus", type=int, default=8)
    p_all.add_argument("--fps", type=int, default=2)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_hydra_config(config_path: str, config_name: str):
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize(config_path=config_path, version_base=None):
        return compose(config_name=config_name)


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def get_indices(args, ds_len: int) -> list[int]:
    if args.all:
        return list(range(ds_len))
    elif args.sample_indices is not None:
        return args.sample_indices
    else:
        rng = np.random.default_rng(42)
        return sorted(rng.choice(ds_len, size=min(args.num_samples, ds_len), replace=False))


def predictions_dir(output_dir: str) -> Path:
    return Path(output_dir) / "predictions"


def frames_dir(output_dir: str) -> Path:
    return Path(output_dir) / "frames"


# ---------------------------------------------------------------------------
# Phase 1: Inference
# ---------------------------------------------------------------------------
def load_model(cfg, ckpt_path: str, device):
    import hydra.utils as hyu
    from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
    model_cls = hyu.get_class(cfg.model._target_.rsplit(".", 1)[0])
    if issubclass(model_cls, TrainableReasoningVLA):
        cfg.model.checkpoint_path = ckpt_path
    elif issubclass(model_cls, TrainableAlpamayoR1):
        cfg.model.pretrained_model_name_or_path = ckpt_path
    model = hyu.instantiate(cfg.model, _convert_="partial")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, batch: dict, num_traj_samples: int = 6) -> tuple[np.ndarray, dict[str, str]]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data(
            data=batch,
            num_traj_samples=num_traj_samples,
            num_traj_sets=1,
            top_p=0.98,
            temperature=0.6,
            traj_only_generation=False,
            max_generation_length=1024,
            return_extra=True,
        )
    texts = {}
    if "cot" in extra and extra["cot"].size > 0:
        texts["cot"] = str(extra["cot"].flat[0])
    return pred_xyz.cpu().numpy(), texts


def run_infer_worker(args):
    """Single GPU worker: load one model, predict shard, save .npz."""
    import hydra.utils as hyu

    role = args._role       # "base" or "ft"
    gpu_id = args._gpu_id
    num_shards = args._num_shards

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    tag = f"[{role} GPU{gpu_id}]"

    ckpt = args.base_ckpt if role == "base" else args.ft_ckpt

    cfg = load_hydra_config(args.config_path, args.config_name)
    if args.tar_root:
        cfg.data.val_dataset.tar_root = args.tar_root
    if args.labels_root:
        cfg.data.val_dataset.labels_root = args.labels_root
    if args.dates is not None:
        cfg.data.val_dataset.dates = args.dates if args.dates else None
    # Generation starts at <|cot_start|>; model produces cot + trajectory.
    cfg.data.val_dataset.vla_preprocess_args.components_order = [
        "image", "traj_history", "route", "prompt", "cot",
    ]
    cfg.data.val_dataset.vla_preprocess_args.components_prompt = ["cot", "traj_future"]
    print(f"{tag} Loading model: {ckpt}")
    model = load_model(cfg.copy(), ckpt, device)

    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=model.config)
    collate_fn = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)

    all_indices = get_indices(args, len(ds))

    # Determine this shard's indices (within role group)
    # base uses GPU 0..N/2-1, ft uses GPU N/2..N-1
    gpus_per_role = num_shards
    shard_id = gpu_id if role == "base" else gpu_id - gpus_per_role
    my_indices = all_indices[shard_id::gpus_per_role]
    print(f"{tag} Processing {len(my_indices)}/{len(all_indices)} samples")

    pred_dir = predictions_dir(args.output_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    texts_all = {}
    for idx in my_indices:
        s = ds[idx]
        if s is None:
            continue
        batch = to_device(collate_fn([s]), device)
        pred, texts = predict(model, batch, args.num_traj_samples)
        results[str(idx)] = pred.squeeze(0).squeeze(0)  # (K, T, 3)
        texts_all[str(idx)] = texts

    out_file = pred_dir / f"{role}_gpu{gpu_id}.npz"
    np.savez_compressed(str(out_file), **results)
    # Save generated texts as JSON alongside
    import json as _json
    texts_file = pred_dir / f"{role}_gpu{gpu_id}_texts.json"
    with open(texts_file, "w") as f:
        _json.dump(texts_all, f, ensure_ascii=False)
    print(f"{tag} Saved {len(results)} predictions to {out_file}")


def cmd_infer(args):
    if args._role is not None:
        run_infer_worker(args)
        return

    num_gpus = args.num_gpus
    gpus_per_role = num_gpus // 2
    if gpus_per_role < 1:
        gpus_per_role = 1

    pred_dir = predictions_dir(args.output_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    procs = []

    for role in ["base", "ft"]:
        gpu_start = 0 if role == "base" else gpus_per_role
        for shard_id in range(gpus_per_role):
            gpu_id = gpu_start + shard_id
            cmd = [
                sys.executable, script, "infer",
                "--config-path", args.config_path,
                "--config-name", args.config_name,
                "--base-ckpt", args.base_ckpt,
                "--ft-ckpt", args.ft_ckpt,
                "--output-dir", args.output_dir,
                "--num-traj-samples", str(args.num_traj_samples),
                "--camera", args.camera,
                "--num-gpus", str(num_gpus),
                "--_role", role,
                "--_gpu-id", str(gpu_id),
                "--_num-shards", str(gpus_per_role),
            ]
            if args.all:
                cmd.append("--all")
            elif args.sample_indices is not None:
                cmd.extend(["--sample-indices"] + [str(i) for i in args.sample_indices])
            else:
                cmd.extend(["--num-samples", str(args.num_samples)])
            if args.tar_root:
                cmd.extend(["--tar-root", args.tar_root])
            if args.labels_root:
                cmd.extend(["--labels-root", args.labels_root])
            if args.dates is not None:
                cmd.append("--dates")
                cmd.extend(args.dates)
            procs.append((f"{role}/GPU{gpu_id}", subprocess.Popen(cmd)))

    print(f"Launched {len(procs)} workers ({gpus_per_role} base + {gpus_per_role} ft)")
    failed = False
    for name, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  {name} failed (exit {rc})")
            failed = True
    if failed:
        sys.exit(1)

    # Merge per-shard .npz and text JSONs into consolidated files
    for role in ["base", "ft"]:
        merged = {}
        for f in sorted(pred_dir.glob(f"{role}_gpu*.npz")):
            data = np.load(str(f))
            merged.update({k: data[k] for k in data.files})
            f.unlink()
        out = pred_dir / f"{role}.npz"
        np.savez_compressed(str(out), **merged)

        merged_texts = {}
        for f in sorted(pred_dir.glob(f"{role}_gpu*_texts.json")):
            with open(f) as fh:
                merged_texts.update(json.load(fh))
            f.unlink()
        texts_out = pred_dir / f"{role}_texts.json"
        with open(texts_out, "w") as fh:
            json.dump(merged_texts, fh, ensure_ascii=False)
        print(f"Merged {role}: {len(merged)} predictions + {len(merged_texts)} texts")


# ---------------------------------------------------------------------------
# Phase 2: Visualisation (CPU-only)
# ---------------------------------------------------------------------------
def load_front_camera_image(tar_path, name_idx, scene_name, frame_idx, cam_name):
    key = f"{scene_name}.cameras/{frame_idx:04d}_{cam_name}.jpg"
    if key not in name_idx:
        return None
    raw = _read_bytes(tar_path, name_idx[key])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def draw_trajectory_on_image(img, waypoints, cam_extr, cam_intr, color, thickness=2):
    h, w = img.shape[:2]
    wp_with_origin = np.concatenate([np.zeros((1, 3)), waypoints], axis=0)
    uv = project_waypoints_pinhole(wp_with_origin, cam_extr, cam_intr, img_hw=(h, w))
    if len(uv) == 0:
        return
    anchor = np.array([[uv[0, 0], h - 1]])
    uv = np.concatenate([anchor, uv], axis=0)
    pts = uv.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
    for p in uv[1:]:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, color, -1)


def compute_ade(pred_xyz, gt_xyz):
    diff = pred_xyz[:, :, :2] - gt_xyz[None, :, :2]
    return np.mean(np.linalg.norm(diff, axis=-1), axis=-1)


def visualize_sample(sample, pred_base, pred_ft, cam_intr, cam_extr,
                     front_img, sample_idx, output_path, camera_name,
                     base_texts=None, ft_texts=None):
    import textwrap

    gt_future = sample["ego_future_xyz"].squeeze(0).numpy()
    gt_history = sample["ego_history_xyz"].squeeze(0).numpy()

    ade_base = compute_ade(pred_base, gt_future)
    ade_ft = compute_ade(pred_ft, gt_future)
    best_base = np.argmin(ade_base)
    best_ft = np.argmin(ade_ft)

    has_camera = front_img is not None and cam_intr is not None

    if has_camera:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        ax_cam = fig.add_subplot(gs[0, 0])
        ax_bev = fig.add_subplot(gs[0, 1])
    else:
        fig, ax_bev = plt.subplots(figsize=(10, 10))
        ax_cam = None

    if has_camera and ax_cam is not None:
        overlay = front_img.copy()
        draw_trajectory_on_image(overlay, gt_future, cam_extr, cam_intr,
                                 color=(0, 200, 0), thickness=3)
        draw_trajectory_on_image(overlay, pred_base[best_base], cam_extr, cam_intr,
                                 color=(0, 0, 255), thickness=2)
        draw_trajectory_on_image(overlay, pred_ft[best_ft], cam_extr, cam_intr,
                                 color=(255, 0, 0), thickness=2)
        ax_cam.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax_cam.set_title(f"{camera_name} — GT(green) / Base(red) / FT(blue)")
        ax_cam.axis("off")

    plotted = []
    hist_xy = rotate_90cc(gt_history[:, :2].T)
    _plot_trajectory_with_fade(ax_bev, hist_xy, color="gray", label="GT history", fade_in=False)
    plotted.append(hist_xy)
    gt_xy = rotate_90cc(gt_future[:, :2].T)
    _plot_trajectory_with_fade(ax_bev, gt_xy, color="green", label="GT future", fade_in=True)
    plotted.append(gt_xy)

    for i in range(pred_base.shape[0]):
        xy = rotate_90cc(pred_base[i, :, :2].T)
        plotted.append(xy)
        if i == best_base:
            _plot_trajectory_with_fade(ax_bev, xy, color="red",
                                       label=f"Base best (ADE={ade_base[i]:.2f})")
        else:
            ax_bev.plot(xy[0], xy[1], "-", color="red", alpha=0.2, linewidth=0.8)

    for i in range(pred_ft.shape[0]):
        xy = rotate_90cc(pred_ft[i, :, :2].T)
        plotted.append(xy)
        if i == best_ft:
            _plot_trajectory_with_fade(ax_bev, xy, color="blue",
                                       label=f"FT best (ADE={ade_ft[i]:.2f})")
        else:
            ax_bev.plot(xy[0], xy[1], "-", color="blue", alpha=0.2, linewidth=0.8)

    _set_tight_trajectory_limits(ax_bev, plotted)
    ax_bev.set_title("BEV Trajectory")
    ax_bev.set_xlabel("← Left / Right →  (m)")
    ax_bev.set_ylabel("← Back / Forward →  (m)")
    ax_bev.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=8)

    nav_text = sample.get("nav_text", "")
    clip_id = sample.get("clip_id", "")
    gt_cot = sample.get("cot", "") or ""
    base_cot = (base_texts or {}).get("cot", "")
    ft_cot = (ft_texts or {}).get("cot", "")

    info_lines = [
        f"sample={sample_idx}  clip={clip_id}  nav=\"{nav_text}\"",
        f"Base minADE={ade_base[best_base]:.3f}  FT minADE={ade_ft[best_ft]:.3f}",
    ]
    if gt_cot:
        info_lines.append(f"GT  CoC: {textwrap.shorten(gt_cot, width=120)}")
    if base_cot:
        info_lines.append(f"Base CoC: {textwrap.shorten(base_cot, width=120)}")
    if ft_cot:
        info_lines.append(f"FT   CoC: {textwrap.shorten(ft_cot, width=120)}")

    info = "\n".join(info_lines)
    text_height = 0.04 + 0.02 * len(info_lines)
    fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=7, family="monospace")
    fig.tight_layout(rect=[0, text_height, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cmd_viz(args):
    import hydra.utils as hyu

    pred_dir = predictions_dir(args.output_dir)
    base_preds = dict(np.load(str(pred_dir / "base.npz")))
    ft_preds = dict(np.load(str(pred_dir / "ft.npz")))

    base_texts = {}
    ft_texts = {}
    base_texts_path = pred_dir / "base_texts.json"
    ft_texts_path = pred_dir / "ft_texts.json"
    if base_texts_path.exists():
        with open(base_texts_path) as f:
            base_texts = json.load(f)
    if ft_texts_path.exists():
        with open(ft_texts_path) as f:
            ft_texts = json.load(f)
    print(f"Loaded predictions: base={len(base_preds)}, ft={len(ft_preds)}, "
          f"texts: base={len(base_texts)}, ft={len(ft_texts)}")

    cfg = load_hydra_config(args.config_path, args.config_name)
    # Dataset without model_config (no vla_preprocess needed for viz)
    from alpamayo.data.wds_nav import WDSNavDataset
    ds_cfg = cfg.data.val_dataset
    if args.dates is not None:
        viz_dates = args.dates if args.dates else None
    elif ds_cfg.dates is not None:
        viz_dates = list(ds_cfg.dates)
    else:
        viz_dates = None
    ds = WDSNavDataset(
        tar_root=args.tar_root or ds_cfg.tar_root,
        labels_root=args.labels_root or ds_cfg.labels_root,
        dates=viz_dates,
        cameras=list(ds_cfg.cameras),
        num_image_frames=ds_cfg.num_image_frames,
    )

    out_dir = frames_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_calib_idx = WDS_CAM_CALIB_INDEX.get(args.camera)
    indices = sorted(int(k) for k in base_preds.keys() if k in ft_preds)
    print(f"Generating {len(indices)} visualisations...")

    for idx in indices:
        sample = ds[idx]
        if sample is None:
            continue

        front_img, cam_intr_mat, cam_extr_mat = None, None, None
        date, clip_id, t0_frame, _ = ds._samples[idx]
        key = (date, clip_id)
        if key in ds._tar_idx_cache and cam_calib_idx is not None:
            tar_path = f"{ds.tar_root}/{date}/{clip_id}.tar"
            name_idx = ds._tar_idx_cache[key]
            scene_name = ds._scene_name_cache[key]
            try:
                intr, extr = load_calibration(tar_path, name_idx, scene_name)
                cam_intr_mat = intr[cam_calib_idx]
                cam_extr_mat = extr[cam_calib_idx]
                front_img = load_front_camera_image(
                    tar_path, name_idx, scene_name, t0_frame, args.camera
                )
            except Exception as e:
                print(f"  WARNING: camera data for sample {idx}: {e}")

        visualize_sample(
            sample=sample,
            pred_base=base_preds[str(idx)],
            pred_ft=ft_preds[str(idx)],
            cam_intr=cam_intr_mat,
            cam_extr=cam_extr_mat,
            front_img=front_img,
            sample_idx=idx,
            output_path=str(out_dir / f"sample_{idx:04d}.png"),
            camera_name=args.camera,
            base_texts=base_texts.get(str(idx)),
            ft_texts=ft_texts.get(str(idx)),
        )
    print(f"Done — {len(indices)} frames saved to {out_dir}")


# ---------------------------------------------------------------------------
# Phase 3: Video
# ---------------------------------------------------------------------------
def cmd_video(args):
    fdir = frames_dir(args.output_dir)
    num = len(list(fdir.glob("sample_*.png")))
    if num == 0:
        print("No frames found, run 'viz' first.")
        sys.exit(1)

    video_path = Path(args.output_dir) / "trajectory_comparison.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-pattern_type", "glob", "-i", str(fdir / "sample_*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video saved: {video_path} ({num} frames, {args.fps} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "viz":
        cmd_viz(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "all":
        args.command = "infer"
        args._role = None
        args._gpu_id = None
        args._num_shards = None
        cmd_infer(args)
        cmd_viz(args)
        fps = getattr(args, "fps", 2)
        args.fps = fps
        cmd_video(args)
    else:
        print("Usage: visualize_wds_trajectories.py {infer,viz,video,all} ...")
        sys.exit(1)


if __name__ == "__main__":
    main()
