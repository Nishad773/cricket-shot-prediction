"""Video loading and frame extraction utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_FPS = 10.0
DEFAULT_SIZE = (224, 224)


def extract_frames(
    video_path: str | Path,
    target_fps: float = DEFAULT_FPS,
    frame_size: tuple[int, int] = DEFAULT_SIZE,
) -> list[np.ndarray]:
    """Load a video, sample frames at a fixed FPS, resize, and return them."""

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = target_fps

    frame_interval = max(int(round(source_fps / target_fps)), 1)
    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_interval == 0:
                resized_frame = cv2.resize(frame, frame_size)
                frames.append(resized_frame)

            frame_index += 1
    finally:
        capture.release()

    return frames
