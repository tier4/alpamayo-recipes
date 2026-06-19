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

"""WebDataset-based dataset with navigation labels for Alpamayo 1.5 SFT training.

Data layout
-----------
Tar archives:  {tar_root}/{date}/scene-XXXXXX.tar
Tar indices:   {tar_root}/{date}/scene-XXXXXX.tar.idx  (JSON, offset = data start)
Labels:        {labels_root}/{date}/labels/{clip_id}/label_{frame:06d}.yaml
CoC labels:    {labels_root}/{date}/coc_labels/{clip_id}/cot_{ts}.yaml
Keyframes:     {labels_root}/{date}/keyframes/segments_relative_timestamp_sampled.json

Each tar contains one scene (~60 s at 10 Hz):
  {scene_name}.trajectory.npy.zst   — shape [T, 4]: (x, y, cos_h, sin_h) global
  {scene_name}.cameras/NNNN_CAM_*.jpg
  {scene_name}.cam_intrinsics.npy.zst — shape [11, 3, 3] pinhole K matrices
  {scene_name}.cam_extrinsics.npy.zst — shape [11, 4, 4] camera-to-vehicle SE(3)
"""

import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zstandard
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Maps WDS camera names to PAI CAMERA_NAMES_TO_INDICES values (constants.py)
WDS_CAM_TO_PAI_INDEX: dict[str, int] = {
    "CAM_FRONT": 6,        # camera_front_tele_30fov
    "CAM_FRONT_WIDE": 1,   # camera_front_wide_120fov
    "CAM_FRONT_LEFT": 0,   # camera_cross_left_120fov
    "CAM_FRONT_RIGHT": 2,  # camera_cross_right_120fov
}

DEFAULT_CAMERAS: list[str] = ["CAM_FRONT_WIDE"]

# Calibration arrays in the tar are ordered alphabetically by camera name.
WDS_CAM_CALIB_INDEX: dict[str, int] = {
    "CAM_BACK_LEFT": 0, "CAM_BACK_LEFT_WIDE": 1,
    "CAM_BACK_RIGHT": 2, "CAM_BACK_RIGHT_WIDE": 3,
    "CAM_FRONT": 4, "CAM_FRONT_LEFT": 5, "CAM_FRONT_LEFT_WIDE": 6,
    "CAM_FRONT_RIGHT": 7, "CAM_FRONT_RIGHT_WIDE": 8,
    "CAM_FRONT_WIDE": 9, "CAM_TRAFFIC_LIGHT_FAR": 10,
}

_US_PER_FRAME: int = 100_000  # 10 Hz → 100 ms per frame in microseconds
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _read_bytes(tar_path: str, entry: dict) -> bytes:
    """Read raw bytes for a tar member using a tar.idx entry.

    The tar.idx ``offset`` field is the byte offset to the member *data*
    (not the 512-byte header), so we seek directly to it.
    """
    with open(tar_path, "rb") as f:
        f.seek(entry["offset"])
        return f.read(entry["size"])


def _read_npy_zst(tar_path: str, entry: dict) -> np.ndarray:
    """Read a zstd-compressed numpy array from a tar file.

    Uses ``stream_reader`` because the zstd frames in this dataset do not
    embed content size, which causes ``ZstdDecompressor.decompress()`` to fail.
    """
    raw = _read_bytes(tar_path, entry)
    dctx = zstandard.ZstdDecompressor()
    data = dctx.stream_reader(io.BytesIO(raw)).read()
    return np.load(io.BytesIO(data))


def load_calibration(
    tar_path: str, name_idx: dict, scene_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics and extrinsics from a WDS tar.

    Returns:
        intrinsics: (11, 3, 3) pinhole K matrices, indexed by WDS_CAM_CALIB_INDEX.
        extrinsics: (11, 4, 4) SE(3) camera-to-vehicle transforms, same ordering.
    """
    intr = _read_npy_zst(tar_path, name_idx[f"{scene_name}.cam_intrinsics.npy.zst"])
    extr = _read_npy_zst(tar_path, name_idx[f"{scene_name}.cam_extrinsics.npy.zst"])
    return intr, extr


class WDSNavDataset(Dataset):
    """Dataset loading from WebDataset tar archives with navigation text labels.

    Drop-in replacement for ``PAIDatasetWithNav`` in the SFT training pipeline.
    Reads camera frames and trajectory data directly from ``.tar`` files using
    ``.tar.idx`` index files for O(1) random access (no full archive scans).

    The sample dict matches the format returned by ``PAIDatasetWithNav``:
    ``image_frames``, ``camera_indices``, ``ego_history_xyz``, ``ego_history_rot``,
    ``ego_future_xyz``, ``ego_future_rot``, ``relative_timestamps``,
    ``absolute_timestamps``, ``t0_us``, ``clip_id``, ``nav_text``.

    ``image_frames`` shape: ``(N_cams, num_image_frames, 3, H, W)``.
    ``relative_timestamps`` shape: ``(N_cams, num_image_frames)`` in seconds,
    0.0 at the oldest loaded frame (matching the alpamayo_r1 convention).
    """

    def __init__(
        self,
        tar_root: str,
        labels_root: str,
        dates: list[str] | None = None,
        cameras: list[str] = DEFAULT_CAMERAS,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        num_image_frames: int = 1,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize dataset and build sample index.

        Args:
            tar_root: Root directory containing ``{date}/scene-XXXXXX.tar`` files.
            labels_root: Root directory containing ``{date}/nav_labels/`` and
                ``{date}/keyframes/`` subdirectories.
            dates: List of date strings (``"YYYY-MM-DD"``) to include. If None,
                all date directories found under ``labels_root`` are used.
            cameras: WDS camera names to load (must be keys of WDS_CAM_TO_PAI_INDEX).
            num_history_steps: Number of past trajectory steps at 10 Hz (default 16 = 1.6 s).
            num_future_steps: Number of future trajectory steps at 10 Hz (default 64 = 6.4 s).
            num_image_frames: Number of consecutive camera frames to load per camera,
                ending at t0_frame (default 1 = t0 only). At 10 Hz, each extra frame
                adds 100 ms of temporal context.
            model_config: Optional model config dict (or OmegaConf); forwarded to
                ``vla_preprocess_args`` instantiation, same pattern as PAIDataset.
            vla_preprocess_args: Hydra config dict to instantiate a VLA preprocessor.
                When set, each sample will include a ``tokenized_data`` key.
        """
        self.tar_root = tar_root
        self.labels_root = labels_root
        self.cameras = list(cameras)
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.num_image_frames = num_image_frames

        for cam in self.cameras:
            if cam not in WDS_CAM_TO_PAI_INDEX:
                raise ValueError(
                    f"Unknown camera '{cam}'. Valid cameras: {list(WDS_CAM_TO_PAI_INDEX)}"
                )

        # Resolve dates
        if dates is None:
            dates = sorted(
                d for d in Path(labels_root).iterdir()
                if d.is_dir() and _DATE_RE.match(d.name)
            )
            dates = [str(d.name) for d in dates]  # type: ignore[assignment]
        self._dates = list(dates)

        # Build flat sample list and cache tar indices
        self._samples: list[tuple[str, str, int, str]] = []  # (date, clip_id, t0_frame, nav_path)
        self._tar_idx_cache: dict[tuple[str, str], dict[str, dict]] = {}
        self._scene_name_cache: dict[tuple[str, str], str] = {}

        self._build_index()

        logger.info(
            "[WDSNavDataset] %d samples from %d dates; cameras=%s num_image_frames=%d",
            len(self._samples),
            len(self._dates),
            self.cameras,
            self.num_image_frames,
        )

        # VLA preprocessor (same pattern as PAIDataset)
        self.vla_preprocess_func = None
        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    # ------------------------------------------------------------------
    # Index building (runs once at __init__)
    # ------------------------------------------------------------------

    def _image_frame_indices(self, t0_frame: int) -> list[int]:
        """Frame indices for the camera-image window ending at t0_frame."""
        img_start = max(0, t0_frame - self.num_image_frames + 1)
        img_frame_idx = list(range(img_start, t0_frame + 1))
        if len(img_frame_idx) < self.num_image_frames:
            img_frame_idx = (
                [img_frame_idx[0]] * (self.num_image_frames - len(img_frame_idx)) + img_frame_idx
            )
        return img_frame_idx

    def _build_index(self) -> None:
        for date in self._dates:
            kf_path = Path(self.labels_root) / date / "keyframes" / "segments_relative_timestamp_sampled.json"
            if not kf_path.exists():
                logger.warning("[WDSNavDataset] Keyframes file not found: %s; skipping date", kf_path)
                continue

            with open(kf_path) as f:
                kf_data: dict[str, list[dict]] = json.load(f)

            # Flatten all meta_action buckets
            all_entries: list[dict] = []
            for entries in kf_data.values():
                all_entries.extend(entries)

            for entry in all_entries:
                clip_id: str = entry["clip_id"]
                t0_frame: int = entry["event_start_frame"]

                tar_path = Path(self.tar_root) / date / f"{clip_id}.tar"
                if not tar_path.exists():
                    logger.warning(
                        "[WDSNavDataset] tar not found: %s; skipping sample", tar_path
                    )
                    continue

                label_path = (
                    Path(self.labels_root)
                    / date
                    / "labels"
                    / clip_id
                    / f"label_{t0_frame:06d}.yaml"
                )
                if not label_path.exists():
                    continue  # no label for this keyframe — silently skip

                # Load and cache the tar index for this clip (if not already done)
                key = (date, clip_id)
                if key not in self._tar_idx_cache:
                    idx_path = Path(self.tar_root) / date / f"{clip_id}.tar.idx"
                    try:
                        with open(idx_path) as f:
                            idx_list: list[dict] = json.load(f)
                        self._tar_idx_cache[key] = {e["name"]: e for e in idx_list}
                        # Scene name = prefix before first dot in first entry's name
                        self._scene_name_cache[key] = idx_list[0]["name"].split(".")[0]
                    except Exception as exc:
                        logger.warning(
                            "[WDSNavDataset] Failed to load tar index %s: %s", idx_path, exc
                        )
                        continue  # cannot verify camera frames; skip sample

                # Verify all requested cameras have every frame in the image
                # window present in the tar; some clips drop frames for a single
                # camera, which would otherwise surface as a None sample at
                # __getitem__ time and crash collation.
                name_idx = self._tar_idx_cache[key]
                scene_name = self._scene_name_cache[key]
                missing = False
                for cam_name in self.cameras:
                    for fi in self._image_frame_indices(t0_frame):
                        img_key = f"{scene_name}.cameras/{fi:04d}_{cam_name}.jpg"
                        if img_key not in name_idx:
                            missing = True
                            break
                    if missing:
                        break
                if missing:
                    continue  # incomplete camera coverage for this keyframe; skip

                self._samples.append((date, clip_id, t0_frame, str(label_path)))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        date, clip_id, t0_frame, label_yaml_path = self._samples[idx]

        # 1. Labels (nav_text, meta_action, effect_on_ego_behavior)
        with open(label_yaml_path) as f:
            label = yaml.safe_load(f)
        nav_text: str = label.get("navigation_text", "")
        meta_action: str = label.get("meta_action", "")

        # CoC (optional — not all keyframes have a corresponding CoC label)
        coc_text: str | None = None
        ts = t0_frame * _US_PER_FRAME
        coc_path = Path(self.labels_root) / date / "coc_labels" / clip_id / f"cot_{ts}.yaml"
        if coc_path.exists():
            with open(coc_path) as f:
                coc_data = yaml.safe_load(f)
            coc_text = (
                coc_data.get("final_content", {})
                .get("ego_behavior_schema", {})
                .get("effect_on_ego_behavior")
            )

        # 2. Cached tar index
        key = (date, clip_id)
        name_idx = self._tar_idx_cache[key]
        scene_name = self._scene_name_cache[key]
        tar_path = str(Path(self.tar_root) / date / f"{clip_id}.tar")

        # 3. Trajectory: [T, 4] = (x_global, y_global, cos_h, sin_h) at 10 Hz
        traj_key = f"{scene_name}.trajectory.npy.zst"
        try:
            traj = _read_npy_zst(tar_path, name_idx[traj_key])
        except Exception as exc:
            logger.warning("[WDSNavDataset] Failed to read trajectory for %s/%s: %s", date, clip_id, exc)
            return None
        T = len(traj)

        # 4. Frame index lists with edge padding
        hist_start = max(0, t0_frame - self.num_history_steps + 1)
        hist_idx = list(range(hist_start, t0_frame + 1))
        if len(hist_idx) < self.num_history_steps:
            hist_idx = [hist_idx[0]] * (self.num_history_steps - len(hist_idx)) + hist_idx

        fut_end = min(T, t0_frame + self.num_future_steps + 1)
        fut_idx = list(range(t0_frame + 1, fut_end))
        if not fut_idx:
            fut_idx = [min(t0_frame, T - 1)]
        if len(fut_idx) < self.num_future_steps:
            fut_idx = fut_idx + [fut_idx[-1]] * (self.num_future_steps - len(fut_idx))

        # 5. Local frame transformation (t0 pose as reference)
        x0, y0 = traj[t0_frame, 0], traj[t0_frame, 1]
        cos0, sin0 = traj[t0_frame, 2], traj[t0_frame, 3]
        # R_t0_inv = R_t0^T  (rotation matrices are orthogonal)
        R_inv = np.array(
            [[cos0, sin0, 0.0], [-sin0, cos0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        def to_local_xyz(indices: list[int]) -> np.ndarray:
            pts = traj[np.array(indices), :2]                       # (N, 2)
            dxy = pts - np.array([x0, y0])                          # (N, 2)
            xyz = np.column_stack([dxy, np.zeros(len(indices))])    # (N, 3)
            return (R_inv @ xyz.T).T                                 # (N, 3)

        def to_rot_mat(indices: list[int]) -> np.ndarray:
            c = traj[np.array(indices), 2]   # cos_h  (N,)
            s = traj[np.array(indices), 3]   # sin_h  (N,)
            z = np.zeros(len(c))
            o = np.ones(len(c))
            # R_step[i] = [[c,-s,0],[s,c,0],[0,0,1]]
            R_steps = np.stack(
                [
                    np.stack([c, -s, z], -1),
                    np.stack([s, c, z], -1),
                    np.stack([z, z, o], -1),
                ],
                axis=-2,
            )  # (N, 3, 3)
            return np.einsum("ij,njk->nik", R_inv, R_steps)         # (N, 3, 3)

        # 6. Ego tensors — shape after unsqueeze(0) matches PAIDataset after squeeze(0):
        #    ego_history_xyz: (1, H, 3), ego_history_rot: (1, H, 3, 3), etc.
        ego_history_xyz = torch.from_numpy(to_local_xyz(hist_idx)).float().unsqueeze(0)
        ego_history_rot = torch.from_numpy(to_rot_mat(hist_idx)).float().unsqueeze(0)
        ego_future_xyz = torch.from_numpy(to_local_xyz(fut_idx)).float().unsqueeze(0)
        ego_future_rot = torch.from_numpy(to_rot_mat(fut_idx)).float().unsqueeze(0)

        # 7. Camera frames: num_image_frames consecutive frames ending at t0_frame
        #    img_frame_idx[0] is the oldest frame, img_frame_idx[-1] == t0_frame
        img_frame_idx = self._image_frame_indices(t0_frame)

        cam_frames_list: list[torch.Tensor] = []  # each: (num_image_frames, 3, H, W)
        cam_idx_list: list[int] = []
        cam_ts_list: list[torch.Tensor] = []       # each: (num_image_frames,) in μs

        for cam_name in self.cameras:
            per_cam_frames: list[torch.Tensor] = []
            per_cam_ts: list[int] = []
            for fi in img_frame_idx:
                img_key = f"{scene_name}.cameras/{fi:04d}_{cam_name}.jpg"
                if img_key not in name_idx:
                    logger.warning(
                        "[WDSNavDataset] Missing camera %s at frame %d in %s/%s; returning None",
                        cam_name, fi, date, clip_id,
                    )
                    return None
                try:
                    raw = _read_bytes(tar_path, name_idx[img_key])
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    frame = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (3, H, W)
                except Exception as exc:
                    logger.warning(
                        "[WDSNavDataset] Failed to decode image %s in %s/%s: %s",
                        img_key, date, clip_id, exc,
                    )
                    return None
                per_cam_frames.append(frame)
                per_cam_ts.append(fi * _US_PER_FRAME)

            cam_frames_list.append(torch.stack(per_cam_frames, dim=0))   # (N_frames, 3, H, W)
            cam_ts_list.append(torch.tensor(per_cam_ts, dtype=torch.int64))
            cam_idx_list.append(WDS_CAM_TO_PAI_INDEX[cam_name])

        image_frames = torch.stack(cam_frames_list, dim=0)       # (N_cams, N_frames, 3, H, W)
        camera_indices = torch.tensor(cam_idx_list, dtype=torch.int64)
        absolute_timestamps = torch.stack(cam_ts_list, dim=0)    # (N_cams, N_frames)
        # relative_timestamps: seconds from the earliest loaded frame (≥ 0, matches alpamayo_r1)
        ts_min = absolute_timestamps.min()
        relative_timestamps = (absolute_timestamps - ts_min).float() * 1e-6  # (N_cams, N_frames)
        t0_us = t0_frame * _US_PER_FRAME

        # 8. Assemble sample dict
        sample_data: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "relative_timestamps": relative_timestamps,
            "absolute_timestamps": absolute_timestamps,
            "t0_us": t0_us,
            "clip_id": clip_id,
            "nav_text": nav_text,
            "meta_action": meta_action,
            "cot": coc_text,
        }

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
