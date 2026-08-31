#!/usr/bin/env python
"""Plot the training curve from a run's log.

The log is the record -- there is no metrics file -- so this parses the step
lines `train_expert.py` prints and draws what they carry: loss, learning rate,
step time and peak memory.

The loss is already an average over `log_every * grad_accum` samples, but a
diffusion loss is noisy even so: each sample draws its own timestep, and the
noise level that timestep implies dominates the difference between a good
prediction and a poor one. So the raw series is drawn faintly with a rolling
median over it, and the trend is read from the median.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

STEP_LINE = re.compile(
    r"^step\s+(?P<step>\d+)/(?P<total>\d+)\s+loss\s+(?P<loss>[\d.]+)\s+"
    r"lr\s+(?P<lr>[\d.e+-]+)\s+(?P<sec>[\d.]+) s/step\s+"
    r"mem\s+(?P<mem>[\d.]+) GB/rank\s+samples/s\s+(?P<sps>[\d.]+)"
)


def parse(path: Path) -> dict[str, np.ndarray]:
    rows = [m.groupdict() for line in path.read_text(errors="ignore").splitlines()
            if (m := STEP_LINE.match(line))]
    if not rows:
        raise SystemExit(f"no step lines found in {path}")
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("step", "loss", "lr", "sec", "mem", "sps")}


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median over a centred window, shrinking at the edges rather than padding."""
    half = window // 2
    return np.array([
        np.median(values[max(0, i - half):min(len(values), i + half + 1)])
        for i in range(len(values))
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smooth", type=int, default=21, help="rolling median window, in points")
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = parse(Path(args.log))
    smooth = rolling_median(d["loss"], args.smooth)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120)
    figure.suptitle(args.title or f"training curve — {Path(args.log).name}", fontsize=13)

    ax = axes[0][0]
    ax.plot(d["step"], d["loss"], color="#1f77b4", alpha=0.25, linewidth=0.9, label="logged")
    ax.plot(d["step"], smooth, color="#1f77b4", linewidth=2.0,
            label=f"rolling median ({args.smooth})")
    ax.set_xlabel("step"); ax.set_ylabel("expert flow-matching loss")
    ax.set_title("loss"); ax.grid(alpha=0.25); ax.legend(fontsize=9)

    # Log scale makes the late, small improvements legible; on a linear axis the
    # first thousand steps flatten everything after them.
    ax = axes[0][1]
    ax.plot(d["step"], smooth, color="#1f77b4", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
    ax.set_title("loss, log scale (smoothed)"); ax.grid(alpha=0.25, which="both")

    ax = axes[1][0]
    ax.plot(d["step"], d["lr"], color="#d62728", linewidth=1.8)
    ax.set_xlabel("step"); ax.set_ylabel("learning rate")
    ax.set_title("learning rate (warmup then cosine)"); ax.grid(alpha=0.25)

    ax = axes[1][1]
    ax.plot(d["step"], d["sec"], color="#2ca02c", alpha=0.4, linewidth=0.9)
    ax.plot(d["step"], rolling_median(d["sec"], args.smooth), color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("step"); ax.set_ylabel("s/step", color="#2ca02c")
    ax.set_title("step time and peak memory"); ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(d["step"], d["mem"], color="#9467bd", linewidth=1.6)
    twin.set_ylabel("GB/rank", color="#9467bd")
    twin.set_ylim(0, 84)
    twin.axhline(80, color="#9467bd", linestyle=":", linewidth=1.0)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)

    first, last = smooth[0], smooth[-1]
    print(f"{len(d['step']):d} logged points over {int(d['step'][-1])} steps")
    print(f"loss (smoothed)  {first:.4f} -> {last:.4f}   ({100*(last-first)/first:+.1f}%)")
    print(f"last 10%% mean    {d['loss'][int(len(d['loss'])*0.9):].mean():.4f}")
    print(f"step time median {np.median(d['sec']):.2f} s   peak mem {d['mem'].max():.1f} GB/rank")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
