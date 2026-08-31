#!/usr/bin/env python
"""Score the fine-tuned action expert against the released one on held-out T4 val.

Both are measured in the same process and on the same windows: the base model is
loaded once, scored, then the fine-tuned expert weights are swapped in and the
identical window list is scored again. Loading the 35.8 B backbone twice would
double a 40-minute job for no additional information, and scoring different
windows would make the difference unreadable.

The prompt is the CoT-free one training used. At inference the released model
writes a chain of thought and the expert conditions on the cache containing it;
T4 has no ground-truth CoT, so training pinned both sides to the same CoT-free
prefix, and evaluation has to honour that or it measures a mismatch instead of a
model.

Scoring is data-parallel over all eight GPUs: each rank holds its own copy of the
model -- inference needs no gradients or optimizer, so 71.6 GB fits on one card --
and takes a disjoint slice of the windows. The per-window metrics are gathered at
the end, so the aggregate is over exactly the same set a single rank would have
scored, just eight times sooner.

Metrics follow `viz_utils._compute_metrics`: ADE and FDE over the 64 waypoints of
the 6.4 s horizon, xy only, in the ego frame at t0, minimised over the sampled
trajectories. Per-behaviour breakdown comes from the curated window list, because
an aggregate hides whether the gain is in the turns or only in the straights.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_expert import build_inputs  # noqa: E402 - same prompt as training

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.t4.dataset import T4SFTDataset  # noqa: E402


def metrics(pred_xyz: torch.Tensor, gt_future_xyz: torch.Tensor) -> tuple[float, float]:
    """min ADE and min FDE in metres over the sampled trajectories."""
    gt = gt_future_xyz.detach().float().cpu().numpy()[0, 0, :, :2]
    pred = pred_xyz.detach().float().cpu().numpy().reshape(-1, gt.shape[0], 3)[:, :, :2]
    d = np.linalg.norm(pred - gt[None], axis=-1)
    return float(d.mean(axis=-1).min()), float(d[:, -1].min())


def score(
    model: Alpamayo2Super,
    dataset: T4SFTDataset,
    rows: list[int],
    samples: int,
    steps: int,
    label: str,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the model over this rank's slice of `rows`, then gather every rank's.

    The slice is strided rather than contiguous so that a difficulty that
    correlates with position -- a hard vehicle-day, a stretch of one scene --
    lands on every rank instead of one, keeping the ranks in step.
    """
    mine = rows[rank::world_size]
    ade: list[float] = []
    fde: list[float] = []
    by_behaviour: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    failures = 0
    started = time.perf_counter()

    for n, row in enumerate(mine, 1):
        try:
            data = dataset[row]
            inputs = build_inputs(data, model)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            inputs["tokenized_data"] = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in inputs["tokenized_data"].items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _, _, _ = model.sample_trajectories_from_data(
                    data=inputs, top_p=0.98, temperature=0.6,
                    num_traj_samples=samples,
                    diffusion_kwargs={"inference_step": steps}, return_extra=True,
                )
            a, f = metrics(pred_xyz, data["ego_future_xyz"])
            ade.append(a)
            fde.append(f)
            if dataset.behaviours is not None:
                by_behaviour[dataset.behaviours[row][0]].append((a, f))
        except Exception as error:  # noqa: BLE001 - one bad window is not the run
            failures += 1
            print(f"  [{label}] window {row} failed: {type(error).__name__}: {error}", flush=True)

        if rank == 0 and n % 10 == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{label}] rank0 {n}/{len(mine)}  minADE {np.mean(ade):.3f}  "
                  f"{elapsed/n:.1f} s/window", flush=True)

    local = {"ade": ade, "fde": fde, "failures": failures,
             "by_behaviour": {k: list(v) for k, v in by_behaviour.items()}}
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]

    ade = [x for part in gathered for x in part["ade"]]
    fde = [x for part in gathered for x in part["fde"]]
    failures = sum(part["failures"] for part in gathered)
    merged: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for part in gathered:
        for k, v in part["by_behaviour"].items():
            merged[k].extend(v)

    return {
        "n": len(ade),
        "failures": failures,
        "minADE_m": float(np.mean(ade)) if ade else None,
        "minFDE_m": float(np.mean(fde)) if fde else None,
        "by_behaviour": {
            k: {"n": len(v),
                "minADE_m": float(np.mean([x[0] for x in v])),
                "minFDE_m": float(np.mean([x[1] for x in v]))}
            for k, v in sorted(merged.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", required=True)
    p.add_argument("--window-list", default=None)
    p.add_argument("--expert-weights", required=True, help="fine-tuned expert safetensors dir")
    p.add_argument("--model-id", default="nvidia/Alpamayo2-Super")
    p.add_argument("--decode", default="gpu", choices=("gpu", "cache"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cpu", help="decode device")
    p.add_argument("--n-windows", type=int, default=400)
    p.add_argument("--center-stride", type=int, default=50)
    p.add_argument("--num-traj-samples", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)

    dataset = T4SFTDataset(
        args.index, cache_dir=args.cache_dir, window_list=args.window_list,
        decode=args.decode, device=args.device, center_stride=args.center_stride,
    )
    log(str(dataset))

    rng = np.random.default_rng(args.seed)
    # Every rank draws the same rows from the same seed, then takes its own stride
    # of them -- no broadcast needed, and the set scored is identical whatever the
    # world size, so results are comparable across runs on different node counts.
    rows = sorted(rng.choice(len(dataset), size=min(args.n_windows, len(dataset)),
                             replace=False).tolist())
    log(f"scoring {len(rows)} windows across {world_size} rank(s)")

    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16,
                                           device_map=device)
    model.eval()

    # Keep the released weights so the comparison is against what shipped, not
    # against whatever the fine-tuned module leaves behind.
    baseline_state = {k: v.detach().clone() for k, v in model.expert.state_dict().items()}
    results: dict[str, Any] = {"config": vars(args), "n_windows": len(rows)}

    log("\n=== baseline (released expert) ===")
    results["baseline"] = score(model, dataset, rows, args.num_traj_samples,
                                args.diffusion_steps, "base", rank, world_size, device)

    log("\n=== fine-tuned expert ===")
    tuned = load_file(str(Path(args.expert_weights) / "model.safetensors"))
    missing, unexpected = model.expert.load_state_dict(tuned, strict=False)
    # Reported rather than asserted: the released checkpoint carries buffers the
    # save may not repeat, and silently loading nothing would look like "no gain".
    log(f"loaded expert weights: {len(missing)} missing, {len(unexpected)} unexpected")
    if len(tuned) == 0:
        raise SystemExit("fine-tuned state dict is empty")
    results["finetuned"] = score(model, dataset, rows, args.num_traj_samples,
                                 args.diffusion_steps, "tuned", rank, world_size, device)

    base, tune = results["baseline"], results["finetuned"]
    if base["minADE_m"] and tune["minADE_m"]:
        results["delta"] = {
            "minADE_m": tune["minADE_m"] - base["minADE_m"],
            "minFDE_m": tune["minFDE_m"] - base["minFDE_m"],
            "minADE_pct": 100 * (tune["minADE_m"] - base["minADE_m"]) / base["minADE_m"],
            "minFDE_pct": 100 * (tune["minFDE_m"] - base["minFDE_m"]) / base["minFDE_m"],
        }

    if not is_main:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 62)
    print(f"{'':22s} {'minADE (m)':>12s} {'minFDE (m)':>12s}")
    print(f"{'baseline':22s} {base['minADE_m']:12.3f} {base['minFDE_m']:12.3f}")
    print(f"{'fine-tuned':22s} {tune['minADE_m']:12.3f} {tune['minFDE_m']:12.3f}")
    if "delta" in results:
        d = results["delta"]
        print(f"{'delta':22s} {d['minADE_m']:+12.3f} {d['minFDE_m']:+12.3f}")
        print(f"{'':22s} {d['minADE_pct']:+11.1f}% {d['minFDE_pct']:+11.1f}%")
    print("=" * 62)
    print(f"\nwritten to {out}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
