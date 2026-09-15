from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _dataset_cls():
    pytest.importorskip("torch")
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    pytest.importorskip("alpamayo_r1")
    from alpamayo.data.drivelm_vqa import DriveLMVQADataset

    return DriveLMVQADataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _add_perception_scene(root: Path, scene: str, frame: int = 1) -> None:
    base = root / "perception_vqa" / scene / "CAM_FRONT_WIDE"
    raw = base / f"raw_{frame:04d}.jpg"
    labeled = base / f"labeled_{frame:04d}.jpg"
    _write_image(raw, (10, 20, 30))
    _write_image(labeled, (200, 10, 10))
    _write_json(
        base / f"frame_{frame:04d}.json",
        {
            "scene_name": scene,
            "camera": "CAM_FRONT_WIDE",
            "frame_idx": frame,
            "image_path": str(raw),
            "labeled_image_path": str(labeled),
            "QA": {
                "perception": [
                    {"Q": "What objects are visible?", "A": "A car is visible."},
                    {"Q": "Where is the car?", "A": "The car is ahead."},
                ],
                "prediction": [{"Q": "What will it do?", "A": "It will continue."}],
                "planning": [{"Q": "What should ego do?", "A": "Maintain speed."}],
            },
        },
    )


def _add_driving_scene(root: Path, date: str, scene: str, frame: int = 30) -> None:
    base = root / "driving_context_vqa" / date / scene
    image = base / f"cam_{frame:04d}.jpg"
    _write_image(image, (20, 200, 20))
    _write_json(
        base / f"frame_{frame:04d}.json",
        {
            "scene_name": scene,
            "camera": "CAM_FRONT_WIDE",
            "frame_idx": frame,
            "image_path": str(image),
            "QA": {
                "perception": [{"Q": "What is the traffic light?", "A": "It is red."}],
                "prediction": [{"Q": "What happens next?", "A": "The ego should stop."}],
                "planning": [{"Q": "What action is needed?", "A": "Brake smoothly."}],
            },
        },
    )


def test_drivelm_vqa_dataset_flattens_sources_and_prefers_labeled_images(tmp_path: Path) -> None:
    DriveLMVQADataset = _dataset_cls()
    _add_perception_scene(tmp_path, "perception_scene")
    _add_driving_scene(tmp_path, "2025-12-24", "driving_scene")

    ds = DriveLMVQADataset(
        data_root=str(tmp_path),
        split="all",
        sampling_mode="fixed",
        categories=["perception", "prediction", "planning"],
    )

    assert len(ds) == 7
    perception_records = [record for record in ds._records if record["source"] == "perception_vqa"]
    assert perception_records
    assert Path(perception_records[0]["image_path"]).name.startswith("labeled_")

    item = ds[0]
    assert item["image_frames"].shape[:3] == (1, 1, 3)
    assert item["question"]
    assert item["answer"]


def test_drivelm_vqa_dataset_scene_split_has_no_overlap(tmp_path: Path) -> None:
    DriveLMVQADataset = _dataset_cls()
    for idx in range(12):
        _add_perception_scene(tmp_path, f"scene_{idx:02d}")

    train = DriveLMVQADataset(
        data_root=str(tmp_path),
        split="train",
        sampling_mode="fixed",
        val_ratio=0.25,
        seed=7,
    )
    val = DriveLMVQADataset(
        data_root=str(tmp_path),
        split="val",
        sampling_mode="fixed",
        val_ratio=0.25,
        seed=7,
    )

    train_scenes = {record["scene_name"] for record in train._records}
    val_scenes = {record["scene_name"] for record in val._records}
    assert train_scenes
    assert val_scenes
    assert train_scenes.isdisjoint(val_scenes)


def test_drivelm_vqa_dataset_balanced_epoch_is_deterministic_and_refreshes(
    tmp_path: Path,
) -> None:
    DriveLMVQADataset = _dataset_cls()
    for idx in range(8):
        _add_perception_scene(tmp_path, f"perception_{idx:02d}")
        _add_driving_scene(tmp_path, "2025-12-24", f"driving_{idx:02d}", frame=idx)

    ds = DriveLMVQADataset(
        data_root=str(tmp_path),
        split="train",
        sampling_mode="balanced_epoch",
        epoch_size=24,
        val_ratio=0.1,
        seed=11,
    )

    ds.set_epoch(0)
    epoch0 = list(ds._curated_indices)
    ds.set_epoch(1)
    epoch1 = list(ds._curated_indices)
    ds.set_epoch(0)
    epoch0_again = list(ds._curated_indices)

    assert len(ds) == 24
    assert epoch0 == epoch0_again
    assert epoch0 != epoch1
