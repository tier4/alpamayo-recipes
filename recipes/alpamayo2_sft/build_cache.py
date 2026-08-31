#!/usr/bin/env python
"""Build the image cache for one contiguous slice of a T4 scene index.

`prepare_t4_cache` handles one scene and is idempotent, so the only thing a
fleet-wide build needs is a way to split the index across array tasks and keep
going when one scene fails. That is this.

A slice, not a stride: consecutive index entries are consecutive scenes of one
vehicle-day, so a slice keeps one task reading one region of the tree rather
than seeking across all of it.

Each task then fans its slice across a process pool, because the cluster caps a
user at 10 running jobs. Concurrency has to live inside the job or the node sits
idle: one scene takes ~209 s and is effectively single-threaded (six cores build
it no faster than two), so throughput here is a count of concurrent scenes, not
cores per scene.

One scene failing does not fail the task. A scene whose channels are short, or
whose mp4 will not open, is a fact about the export; the run records it and
moves on, and the summary at the end is what decides whether to care. An
exit code is reserved for "nothing at all worked", which is a different thing
and usually means the cache root is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from alpamayo2_super.t4.camera import build_rig
from alpamayo2_super.t4.frames import build_cache
from alpamayo2_super.t4.scene import T4Scene


def slice_bounds(n_items: int, task_id: int, n_tasks: int) -> tuple[int, int]:
    """Split ``n_items`` into ``n_tasks`` near-equal contiguous slices.

    The first ``n_items % n_tasks`` tasks take one extra item, so no task is
    ever more than one scene behind another and none is handed an empty slice
    while another carries two extras.
    """
    if not 0 <= task_id < n_tasks:
        raise ValueError(f"task_id {task_id} outside [0, {n_tasks})")
    base, extra = divmod(n_items, n_tasks)
    start = task_id * base + min(task_id, extra)
    return start, start + base + (1 if task_id < extra else 0)


def build_one(job: tuple[str, str, str, str, str, bool]) -> tuple[str, str | None]:
    """Build one scene's cache in a pool worker.

    Takes plain strings rather than the rig object because a pool argument has to
    pickle, and returns the failure as a value rather than raising: one unbuildable
    scene is a fact about the export, and it must not take the other 4,912 with it.
    """
    scene_dir, cache_dir, front_tele, rear, rel_dir, force = job
    try:
        rig = build_rig(front_tele=front_tele, rear=rear)
        build_cache(T4Scene(scene_dir), rig, cache_dir, force=force, progress=False)
        return rel_dir, None
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return rel_dir, f"{type(error).__name__}: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, help="scene index JSON from t4.index")
    parser.add_argument("--cache-dir", required=True, help="cache root")
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="this task's position (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)),
        help="total tasks (default: $SLURM_ARRAY_TASK_COUNT)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap scenes, for a trial run")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help="scenes to build concurrently (default: $SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    args = parser.parse_args(argv)

    index = json.loads(Path(args.index).read_text())
    root = Path(index["root"])
    scenes = index["scenes"]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    start, end = slice_bounds(len(scenes), args.task_id, args.n_tasks)
    mine = scenes[start:end]

    print(
        f"task {args.task_id}/{args.n_tasks}: scenes [{start}, {end}) of {len(scenes)} "
        f"-> {args.cache_dir}",
        flush=True,
    )

    built = failed = 0
    failures: list[tuple[str, str]] = []
    t_start = time.perf_counter()

    front_tele, rear = index["rig"]["front_tele"], index["rig"]["rear"]
    jobs = [
        (
            str(root / entry["scene_dir"]),
            str(args.cache_dir),
            front_tele,
            rear,
            entry["scene_dir"],
            args.force,
        )
        for entry in mine
    ]
    workers = max(1, min(args.workers, len(jobs)))
    print(f"  {len(jobs)} scenes across {workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rel_dir, reason = future.result()
            if reason is None:
                built += 1
            else:
                failed += 1
                failures.append((rel_dir, reason))
                print(f"FAIL {rel_dir}: {reason}", flush=True)

            if done % 10 == 0 or done == len(jobs):
                elapsed = time.perf_counter() - t_start
                rate = elapsed / done
                print(
                    f"  {done}/{len(jobs)} scenes, {rate:.1f} s/scene wall, "
                    f"eta {rate * (len(jobs) - done) / 60:.0f} min",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    print(f"task {args.task_id}: built {built}, failed {failed}, {elapsed / 60:.1f} min")
    for scene_dir, reason in failures:
        print(f"  failed: {scene_dir}: {reason}")

    # Only a task that accomplished nothing is an error. A task with some
    # failures did its job; the caller reads the counts.
    return 1 if built == 0 and mine else 0


if __name__ == "__main__":
    sys.exit(main())
