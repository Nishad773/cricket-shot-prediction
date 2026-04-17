"""Dataset assembly helpers for cricket shot classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_engineering import aggregate_window_features, build_feature_frame

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


def _iter_labeled_videos(input_dir: str | Path) -> list[tuple[str, Path]]:
    """Return ``(label, video_path)`` pairs from label-based subdirectories."""

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    labeled_videos: list[tuple[str, Path]] = []
    for label_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
        for video_path in sorted(label_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                labeled_videos.append((label_dir.name, video_path))

    return labeled_videos


def process_video_to_features(
    video_path: str | Path,
    target_fps: float = 10.0,
    frame_size: tuple[int, int] = (224, 224),
) -> dict[str, float]:
    """Extract window-aggregated features for a single video."""

    from src.pose_estimation import estimate_pose
    from src.video_processing import extract_frames

    frames = extract_frames(video_path, target_fps=target_fps, frame_size=frame_size)
    keypoints_sequence = estimate_pose(frames)
    return aggregate_window_features(keypoints_sequence)


def build_dataset(samples: list[dict[str, Any]], output_path: str | Path) -> pd.DataFrame:
    """Build and persist a tabular dataset from extracted samples."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_feature_frame(samples)
    dataset.to_csv(output_path, index=False)
    return dataset


def create_dataset_from_video_folders(
    input_dir: str | Path,
    output_path: str | Path,
    target_fps: float = 10.0,
    frame_size: tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Create a dataset CSV from labeled video folders."""

    rows: list[dict[str, Any]] = []

    for label, video_path in _iter_labeled_videos(input_dir):
        feature_row = process_video_to_features(
            video_path,
            target_fps=target_fps,
            frame_size=frame_size,
        )
        feature_row["label"] = label
        rows.append(feature_row)

    dataset = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset
