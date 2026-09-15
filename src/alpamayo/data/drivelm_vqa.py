# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DriveLM-style VQA dataset for Alpamayo Stage-1 SFT.

The expected input is the local DriveLM VQA export with frame-level JSON files:

    data_root/
    ├── perception_vqa/**/frame_*.json
    └── driving_context_vqa/**/frame_*.json

Each frame JSON contains a ``QA`` mapping with category lists. This dataset
flattens those frame records into one sample per QA pair and returns the same
minimal VQA contract as :class:`alpamayo.data.lingoqa.LingoQADataset`.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

import alpamayo.common.constants as constants
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)

MAX_CHAR_LENGTH = 2048
DEFAULT_SOURCES = ("perception_vqa", "driving_context_vqa")
DEFAULT_CATEGORIES = ("perception", "prediction", "planning")
FRONT_CAMERA_INDEX = constants.CAMERA_NAMES_TO_INDICES[constants.FRONT_WIDE_CAMERA_NAME]


def _as_list(value: Any, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class DriveLMVQADataset(torch.utils.data.Dataset):
    """DriveLM VQA dataset with deterministic scene split and epoch curation."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        sampling_mode: str = "balanced_epoch",
        epoch_size: int | None = 50_000,
        perception_image_mode: str = "labeled_first",
        index_cache_path: str | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        n_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.sources = _as_list(sources, DEFAULT_SOURCES)
        self.categories = _as_list(categories, DEFAULT_CATEGORIES)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.sampling_mode = sampling_mode
        self.epoch_size = None if epoch_size is None else int(epoch_size)
        self.perception_image_mode = perception_image_mode
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None
        self.n_frames = n_frames

        if self.split not in {"train", "val", "all"}:
            raise ValueError(f"split must be one of train, val, all; got {self.split!r}")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1); got {self.val_ratio}")
        if self.sampling_mode not in {"balanced_epoch", "fixed"}:
            raise ValueError(
                f"sampling_mode must be one of balanced_epoch, fixed; got {self.sampling_mode!r}"
            )
        if self.perception_image_mode not in {"labeled_first", "raw", "labeled"}:
            raise ValueError(
                "perception_image_mode must be one of labeled_first, raw, labeled; "
                f"got {self.perception_image_mode!r}"
            )

        self._records = self._load_or_build_index()
        self._records = self._apply_scene_split(self._records)
        if not self._records:
            raise ValueError(
                f"No DriveLM VQA records found for split={self.split!r}, "
                f"sources={self.sources}, categories={self.categories} under {self.data_root}"
            )

        self._all_indices = list(range(len(self._records)))
        self._curated_indices = self._all_indices
        if self.sampling_mode == "balanced_epoch" and self.split == "train":
            self.set_epoch(0)

        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.vla_preprocess_func: Callable[..., Any] | None = None
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

        logger.info(
            "[DriveLMVQADataset] split=%s records=%d active=%d sources=%s categories=%s "
            "sampling_mode=%s epoch_size=%s",
            self.split,
            len(self._records),
            len(self._curated_indices),
            self.sources,
            self.categories,
            self.sampling_mode,
            self.epoch_size,
        )

    def __len__(self) -> int:
        return len(self._curated_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._records[self._curated_indices[idx]]

        question = str(record["question"])
        answer = str(record["answer"])
        if len(question) > MAX_CHAR_LENGTH or len(answer) > MAX_CHAR_LENGTH:
            return self.__getitem__(random.randint(0, len(self) - 1))

        image_path = Path(record["image_path"])
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        frame = torch.from_numpy(arr).permute(2, 0, 1)
        frames = frame.unsqueeze(0)  # (N_frames=1, 3, H, W)
        if self.n_frames is not None:
            frames = frames[-self.n_frames :]

        image_frames = frames.unsqueeze(0)  # (1_camera, N_frames, 3, H, W)
        n = image_frames.shape[1]
        sample_data: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": torch.tensor([FRONT_CAMERA_INDEX], dtype=torch.int64),
            "relative_timestamps": torch.arange(n, dtype=torch.float32).unsqueeze(0),
            "question": question,
            "answer": answer,
            "source": record["source"],
            "category": record["category"],
            "scene_name": record["scene_name"],
            "frame_json_path": record["frame_json_path"],
        }

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data

    def set_epoch(self, epoch: int) -> None:
        """Refresh the train subset for a new epoch."""
        if self.sampling_mode != "balanced_epoch" or self.split != "train":
            return

        target_size = self.epoch_size or len(self._all_indices)
        if target_size <= 0:
            raise ValueError(f"epoch_size must be positive; got {target_size}")

        buckets: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, record in enumerate(self._records):
            buckets[(record["source"], record["category"])][record["scene_name"]].append(idx)

        rng = random.Random(self.seed + int(epoch))
        bucket_keys = sorted(k for k, v in buckets.items() if v)
        if not bucket_keys:
            self._curated_indices = []
            return

        base = target_size // len(bucket_keys)
        remainder = target_size % len(bucket_keys)
        curated: list[int] = []
        for bucket_pos, key in enumerate(bucket_keys):
            take = base + (1 if bucket_pos < remainder else 0)
            curated.extend(self._sample_scene_balanced(buckets[key], take, rng))

        rng.shuffle(curated)
        self._curated_indices = curated

    def _sample_scene_balanced(
        self,
        scene_buckets: dict[str, list[int]],
        take: int,
        rng: random.Random,
    ) -> list[int]:
        """Sample one source/category bucket while spreading picks over scenes."""
        if take <= 0:
            return []

        scene_names = sorted(scene_buckets)
        rng.shuffle(scene_names)
        base = take // len(scene_names)
        remainder = take % len(scene_names)
        sampled: list[int] = []
        for scene_pos, scene_name in enumerate(scene_names):
            scene_take = base + (1 if scene_pos < remainder else 0)
            bucket = scene_buckets[scene_name]
            if scene_take <= len(bucket):
                sampled.extend(rng.sample(bucket, scene_take))
            else:
                sampled.extend(bucket)
                sampled.extend(rng.choice(bucket) for _ in range(scene_take - len(bucket)))
        return sampled

    def _load_or_build_index(self) -> list[dict[str, str]]:
        if self.index_cache_path is not None and self.index_cache_path.exists():
            return self._read_index_cache(self.index_cache_path)

        records = self._build_index()
        if self.index_cache_path is not None:
            self._write_index_cache(self.index_cache_path, records)
        return records

    def _read_index_cache(self, path: Path) -> list[dict[str, str]]:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return [
            record
            for record in records
            if record.get("source") in self.sources and record.get("category") in self.categories
        ]

    def _write_index_cache(self, path: Path, records: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_index(self) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for source in self.sources:
            source_root = self.data_root / source
            if not source_root.is_dir():
                logger.warning("[DriveLMVQADataset] source directory not found: %s", source_root)
                continue
            for json_path in sorted(source_root.rglob("frame_*.json")):
                records.extend(self._records_from_frame_json(source, json_path))
        return records

    def _records_from_frame_json(self, source: str, json_path: Path) -> list[dict[str, str]]:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                frame = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[DriveLMVQADataset] failed to load %s: %s", json_path, exc)
            return []

        image_path = self._select_image_path(source, frame)
        if not image_path:
            return []

        qa_by_category = frame.get("QA", {})
        if not isinstance(qa_by_category, dict):
            return []

        out: list[dict[str, str]] = []
        for category in self.categories:
            qa_list = qa_by_category.get(category, [])
            if not isinstance(qa_list, list):
                continue
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("Q")
                answer = qa.get("A")
                if not question or not answer:
                    continue
                out.append(
                    {
                        "source": source,
                        "category": str(qa.get("category") or category),
                        "scene_name": str(frame.get("scene_name") or "unknown"),
                        "frame_json_path": str(json_path),
                        "image_path": image_path,
                        "question": str(question),
                        "answer": str(answer),
                    }
                )
        return out

    def _select_image_path(self, source: str, frame: dict[str, Any]) -> str:
        raw_path = str(frame.get("image_path") or "")
        labeled_path = str(frame.get("labeled_image_path") or "")
        if source != "perception_vqa":
            return raw_path
        if self.perception_image_mode == "raw":
            return raw_path
        if self.perception_image_mode == "labeled":
            return labeled_path
        return labeled_path or raw_path

    def _apply_scene_split(self, records: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.split == "all":
            return records

        scenes = sorted({record["scene_name"] for record in records})
        rng = random.Random(self.seed)
        rng.shuffle(scenes)
        n_val = int(len(scenes) * self.val_ratio)
        if self.val_ratio > 0.0 and n_val == 0 and len(scenes) > 1:
            n_val = 1
        val_scenes = set(scenes[:n_val])

        if self.split == "val":
            return [record for record in records if record["scene_name"] in val_scenes]
        return [record for record in records if record["scene_name"] not in val_scenes]
