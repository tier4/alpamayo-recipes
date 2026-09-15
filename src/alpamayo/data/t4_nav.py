# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TIER IV T4 scenes with navigation text, for Alpamayo 1.5 SFT.

``WDSNavDataset`` reads the same quantities out of WebDataset tars. Those are
gone; T4 carries all of them, and in the same shapes, so this is that dataset
with its storage layer replaced and its geometry copied rather than reinvented:

==============================  =========================================
trajectory ``[T, 4]`` x,y,cos,sin  ``derived/scalars.npz`` -- identical
camera frames                      ``data/<CHANNEL>/%05d.jpg`` -- five digits
navigation text                    ``nav_labels/<clip>/nav_<us>.yaml``
keyframes                          ``segments_relative_timestamp_sampled.json``
==============================  =========================================

The default camera is ``CAM_FRONT_WIDE`` alone. On the prd_jt rig it is a JPEG
directory, so a sample decodes four images and no video: 0.21 s against 1.53 s
for the four-camera set, which is what lets eight dataloader workers stay ahead
of the GPUs without a pre-built image cache. The wide front view is PAI camera 1
(``camera_front_wide_120fov``), the same index ``WDSNavDataset`` gave it.

``derived/`` is required. It is the byte-verified rendering of the same scenes
the tars held, present on 4,613 of prd_jt's 4,955 scene directories; a scene
without it is skipped at index time rather than failing mid-epoch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

#: T4 channel -> PAI camera index, matching ``wds_nav.WDS_CAM_TO_PAI_INDEX``.
#: The WIDE variants are the ones stored as JPEG on the prd_jt taxi rig; the
#: narrow channels of the same name are HEVC and are deliberately not offered.
T4_CAM_TO_PAI_INDEX: dict[str, int] = {
    "CAM_FRONT_WIDE": 1,        # camera_front_wide_120fov
    "CAM_FRONT_LEFT_WIDE": 0,   # camera_cross_left_120fov
    "CAM_FRONT_RIGHT_WIDE": 2,  # camera_cross_right_120fov
    "CAM_BACK_LEFT_WIDE": 3,    # camera_rear_left_70fov
    "CAM_BACK_RIGHT_WIDE": 5,   # camera_rear_right_70fov
}

DEFAULT_CAMERAS: list[str] = ["CAM_FRONT_WIDE"]

_US_PER_FRAME: int = 100_000  # 10 Hz


def _scene_dir_from_clip_id(clip_id: str) -> str:
    """Invert ``meta_action.t4_labeling.clip_id_for``.

    The labelling pipeline flattens a scene's last four path components with
    ``__``; no subtree, vehicle id, date or scene name contains a double
    underscore, so the split is unambiguous.
    """
    parts = clip_id.split("__")
    if len(parts) != 4:
        raise ValueError(f"clip_id {clip_id!r} does not have four __-separated parts")
    return "/".join(parts)


class T4NavDataset(Dataset):
    """T4 windows with navigation text, in the sample format Alpamayo 1.5 expects.

    The sample dict is ``WDSNavDataset``'s: ``image_frames``, ``camera_indices``,
    ``ego_history_xyz``, ``ego_history_rot``, ``ego_future_xyz``,
    ``ego_future_rot``, ``relative_timestamps``, ``absolute_timestamps``,
    ``t0_us``, ``clip_id``, ``nav_text``, ``meta_action``, ``cot``.

    ``image_frames`` is ``(N_cams, num_image_frames, 3, H, W)`` uint8, and the
    ``ego_*`` tensors keep their leading singleton so that after collation they
    are ``[B, n_traj=1, T, ...]``, which is what
    ``sft_base_model.tokenize_future_trajectory`` asserts.
    """

    def __init__(
        self,
        root: str,
        keyframes: str,
        nav_labels_root: str,
        coc_labels_root: str | None = None,
        scenes: list[str] | None = None,
        cameras: list[str] = DEFAULT_CAMERAS,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        num_image_frames: int = 4,
        image_size: tuple[int, int] | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        :param root: T4 dataset root, the parent of ``prd_jt`` and friends.
        :param keyframes: ``segments_relative_timestamp_sampled.json``.
        :param nav_labels_root: directory of ``<clip_id>/nav_<us>.yaml``.
        :param coc_labels_root: optional directory of ``<clip_id>/cot_<us>.yaml``.
            A window without one gets ``cot=None``, which is what the ``nav``
            processor expects; only ``nav_coc`` needs them.
        :param scenes: restrict to these scene dirs, e.g. a train/val split.
        :param cameras: T4 channels to load, in the order the register expects.
        :param image_size: optional ``(width, height)`` to resize to. ``None``
            keeps the source 2880x1860, which the processor then rescales.
        """
        self.root = Path(root)
        self.nav_labels_root = Path(nav_labels_root)
        self.coc_labels_root = Path(coc_labels_root) if coc_labels_root else None
        self.cameras = list(cameras)
        self.num_history_steps = int(num_history_steps)
        self.num_future_steps = int(num_future_steps)
        self.num_image_frames = int(num_image_frames)
        self.image_size = tuple(image_size) if image_size else None

        unknown = [c for c in self.cameras if c not in T4_CAM_TO_PAI_INDEX]
        if unknown:
            raise ValueError(
                f"cameras {unknown} have no PAI index; known: {sorted(T4_CAM_TO_PAI_INDEX)}"
            )
        self.camera_indices = torch.tensor(
            [T4_CAM_TO_PAI_INDEX[c] for c in self.cameras], dtype=torch.int64
        )

        allowed = set(scenes) if scenes else None
        self._samples: list[tuple[str, str, int, str]] = []   # clip_id, scene_dir, frame, nav yaml
        self._trajectories: dict[str, np.ndarray] = {}
        self._build_index(Path(keyframes), allowed)
        if not self._samples:
            raise ValueError(
                f"no usable windows: check that {nav_labels_root} holds labels for {keyframes}"
            )
        logger.info(
            "[T4NavDataset] %d windows over %d scenes, cameras %s",
            len(self._samples),
            len({s[1] for s in self._samples}),
            self.cameras,
        )

        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.vla_preprocess_func = None
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _build_index(self, keyframes: Path, allowed: set[str] | None) -> None:
        """Keep the windows that have a navigation label and all their images.

        Every check that can be made from metadata is made here rather than in
        ``__getitem__``: a window that fails mid-epoch takes the whole run down,
        and a missing camera frame is a property of the export, not of the step
        that happened to reach it.
        """
        payload = json.loads(keyframes.read_text())
        seen: set[tuple[str, int]] = set()
        skipped = {"scene": 0, "nav": 0, "frames": 0, "window": 0}

        for records in payload.values():
            for record in records:
                clip_id = record["clip_id"]
                frame = int(record["event_start_frame"])
                # A frame is tagged once per taxonomy, so the same window arrives
                # up to three times; it is one training sample.
                if (clip_id, frame) in seen:
                    continue
                seen.add((clip_id, frame))

                try:
                    scene_dir = _scene_dir_from_clip_id(clip_id)
                except ValueError:
                    skipped["scene"] += 1
                    continue
                if allowed is not None and scene_dir not in allowed:
                    skipped["scene"] += 1
                    continue

                nav_yaml = self.nav_labels_root / clip_id / f"nav_{frame * _US_PER_FRAME}.yaml"
                if not nav_yaml.is_file():
                    skipped["nav"] += 1
                    continue

                trajectory = self._trajectory(scene_dir)
                if trajectory is None:
                    skipped["scene"] += 1
                    continue
                if frame >= len(trajectory):
                    skipped["window"] += 1
                    continue

                scene = self.root / scene_dir
                if not all(
                    (scene / "data" / camera / f"{index:05d}.jpg").is_file()
                    for camera in self.cameras
                    for index in self._image_frame_indices(frame)
                ):
                    skipped["frames"] += 1
                    continue

                self._samples.append((clip_id, scene_dir, frame, str(nav_yaml)))

        if any(skipped.values()):
            logger.info("[T4NavDataset] skipped %s", skipped)

    def _trajectory(self, scene_dir: str) -> np.ndarray | None:
        """The scene's ``[T, 4]`` pose track, cached; None when it has none.

        `scalars.npz` is 42 KB and every window of a scene shares it, so it is
        read once per scene rather than once per window.
        """
        cached = self._trajectories.get(scene_dir)
        if cached is not None:
            return cached
        path = self.root / scene_dir / "derived" / "scalars.npz"
        try:
            with np.load(path) as z:
                trajectory = np.asarray(z["trajectory"], dtype=np.float64)
        except (OSError, KeyError, ValueError):
            return None
        self._trajectories[scene_dir] = trajectory
        return trajectory

    def _image_frame_indices(self, t0_frame: int) -> list[int]:
        """Frame indices for the image window ending at ``t0_frame``.

        Left-padded by repeating the oldest available frame, exactly as
        ``WDSNavDataset._image_frame_indices`` does, so a window near the start
        of a scene has the same shape as any other.
        """
        start = max(0, t0_frame - self.num_image_frames + 1)
        indices = list(range(start, t0_frame + 1))
        if len(indices) < self.num_image_frames:
            indices = [indices[0]] * (self.num_image_frames - len(indices)) + indices
        return indices

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_id, scene_dir, t0_frame, nav_yaml = self._samples[index]
        scene = self.root / scene_dir

        label = yaml.safe_load(Path(nav_yaml).read_text()) or {}
        nav_text = label.get("navigation_text", "")
        meta_action = label.get("meta_action", "")

        cot = None
        if self.coc_labels_root is not None:
            cot_path = (
                self.coc_labels_root / clip_id / f"cot_{t0_frame * _US_PER_FRAME}.yaml"
            )
            if cot_path.is_file():
                payload = yaml.safe_load(cot_path.read_text()) or {}
                cot = (
                    payload.get("final_content", {})
                    .get("ego_behavior_schema", {})
                    .get("effect_on_ego_behavior")
                )

        trajectory = self._trajectory(scene_dir)
        n_frames = len(trajectory)

        # Frame index lists, padded at the edges the way WDSNavDataset pads them.
        history_start = max(0, t0_frame - self.num_history_steps + 1)
        history_idx = list(range(history_start, t0_frame + 1))
        if len(history_idx) < self.num_history_steps:
            history_idx = [history_idx[0]] * (
                self.num_history_steps - len(history_idx)
            ) + history_idx

        future_end = min(n_frames, t0_frame + self.num_future_steps + 1)
        future_idx = list(range(t0_frame + 1, future_end))
        if not future_idx:
            future_idx = [min(t0_frame, n_frames - 1)]
        if len(future_idx) < self.num_future_steps:
            future_idx = future_idx + [future_idx[-1]] * (
                self.num_future_steps - len(future_idx)
            )

        # Everything is expressed in the t0 ego frame. R_inv is R_t0 transposed,
        # rotation matrices being orthogonal.
        x0, y0 = trajectory[t0_frame, 0], trajectory[t0_frame, 1]
        cos0, sin0 = trajectory[t0_frame, 2], trajectory[t0_frame, 3]
        r_inv = np.array(
            [[cos0, sin0, 0.0], [-sin0, cos0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        def to_local_xyz(indices: list[int]) -> np.ndarray:
            points = trajectory[np.asarray(indices), :2]
            delta = points - np.array([x0, y0])
            return (r_inv @ np.column_stack([delta, np.zeros(len(indices))]).T).T

        def to_rot(indices: list[int]) -> np.ndarray:
            c = trajectory[np.asarray(indices), 2]
            s = trajectory[np.asarray(indices), 3]
            zero, one = np.zeros(len(c)), np.ones(len(c))
            steps = np.stack(
                [
                    np.stack([c, -s, zero], -1),
                    np.stack([s, c, zero], -1),
                    np.stack([zero, zero, one], -1),
                ],
                axis=-2,
            )
            return np.einsum("ij,njk->nik", r_inv, steps)

        # The leading singleton is deliberate: after collation these become
        # [B, n_traj=1, T, ...], which tokenize_future_trajectory asserts.
        ego_history_xyz = torch.from_numpy(to_local_xyz(history_idx)).float().unsqueeze(0)
        ego_history_rot = torch.from_numpy(to_rot(history_idx)).float().unsqueeze(0)
        ego_future_xyz = torch.from_numpy(to_local_xyz(future_idx)).float().unsqueeze(0)
        ego_future_rot = torch.from_numpy(to_rot(future_idx)).float().unsqueeze(0)

        image_idx = self._image_frame_indices(t0_frame)
        per_camera: list[torch.Tensor] = []
        stamps: list[torch.Tensor] = []
        for camera in self.cameras:
            frames = []
            for i in image_idx:
                with Image.open(scene / "data" / camera / f"{i:05d}.jpg") as handle:
                    if self.image_size is not None:
                        # draft() decodes at a reduced size in the DCT domain,
                        # which is most of the saving; the resize lands it
                        # exactly. It must be called before the pixels are read.
                        handle.draft("RGB", self.image_size)
                        image = handle.convert("RGB").resize(self.image_size)
                    else:
                        image = handle.convert("RGB")
                    # np.array, not np.asarray: asarray hands back PIL's own
                    # read-only buffer and torch then warns that writing through
                    # the tensor is undefined.
                    array = np.array(image, dtype=np.uint8)
                frames.append(torch.from_numpy(array).permute(2, 0, 1))
            per_camera.append(torch.stack(frames))
            stamps.append(torch.tensor([i * _US_PER_FRAME for i in image_idx], dtype=torch.int64))

        image_frames = torch.stack(per_camera)                      # (N_cams, N_frames, 3, H, W)
        absolute_timestamps = torch.stack(stamps)                   # (N_cams, N_frames)
        relative_timestamps = (
            (absolute_timestamps - absolute_timestamps.min()).float() * 1e-6
        )

        sample: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": self.camera_indices,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "relative_timestamps": relative_timestamps,
            "absolute_timestamps": absolute_timestamps,
            "t0_us": t0_frame * _US_PER_FRAME,
            "clip_id": clip_id,
            "nav_text": nav_text,
            "meta_action": meta_action,
            "cot": cot,
        }
        if self.vla_preprocess_func is not None:
            sample["tokenized_data"] = self.vla_preprocess_func(data=sample)
        return sample
