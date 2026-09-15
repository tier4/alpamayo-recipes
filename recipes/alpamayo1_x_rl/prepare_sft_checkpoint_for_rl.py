#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare an Alpamayo 1.5 SFT checkpoint for Cosmos-RL startup.

SFT checkpoints are saved with ``model_type="alpamayo_r1"`` and architecture
``TrainableAlpamayoR1``. The RL wrapper registers
``model_type="alpamayo_reasoning_vla"`` with Transformers, so Cosmos-RL must
point at a lightly converted checkpoint directory.

This script creates a sibling checkpoint directory, symlinks large model files,
copies small metadata files, and rewrites ``config.json``. It never modifies the
source SFT checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TOKENIZER_PROCESSOR_FILES: tuple[str, ...] = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _is_large_state_file(path: Path) -> bool:
    name = path.name
    return (
        name.endswith(".safetensors")
        or name.endswith(".bin")
        or name.endswith(".pt")
        or name.endswith(".pth")
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if _is_large_state_file(src):
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def prepare_checkpoint(
    src_dir: Path,
    dst_dir: Path,
    vlm_name_or_path: str,
    tokenizer_source: Path | None,
) -> None:
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory not found: {src_dir}")
    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"SFT checkpoint is missing config.json: {config_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        if src.name == "config.json":
            continue
        _link_or_copy(src, dst_dir / src.name)

    if tokenizer_source is not None:
        tokenizer_source = tokenizer_source.resolve()
        if not tokenizer_source.is_dir():
            raise FileNotFoundError(f"tokenizer source not found: {tokenizer_source}")
        for name in TOKENIZER_PROCESSOR_FILES:
            src = tokenizer_source / name
            if src.exists() and src.is_file():
                shutil.copy2(src, dst_dir / name)

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["model_type"] = "alpamayo_reasoning_vla"
    config["architectures"] = ["ReasoningVLA"]
    # These fields are required by the RL tokenizer/processor initialization.
    config["vlm_name_or_path"] = vlm_name_or_path
    config.setdefault("include_camera_ids", True)
    config.setdefault("include_frame_nums", True)

    with (dst_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vlm-name-or-path",
        default="nvidia/Cosmos-Reason2-8B",
        help=(
            "Local processor/tokenizer directory or HF model id used for "
            "config.vlm_name_or_path. Prefer a local path for offline RL runs."
        ),
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the expanded Alpamayo tokenizer and "
            "processor files to copy into the RL-ready checkpoint."
        ),
    )
    args = parser.parse_args()

    prepare_checkpoint(
        args.sft_checkpoint,
        args.output_dir,
        args.vlm_name_or_path,
        args.tokenizer_source,
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
