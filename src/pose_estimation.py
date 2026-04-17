"""Pose estimation helpers built on top of MediaPipe."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
NUM_KEYPOINTS = 33
KEYPOINT_DIMENSIONS = 3
DEFAULT_CONFIDENCE = 0.5


def _empty_keypoints() -> np.ndarray:
    """Return a zero-filled keypoint array for undetected frames."""

    return np.zeros((NUM_KEYPOINTS, KEYPOINT_DIMENSIONS), dtype=np.float32)


def _extract_frame_keypoints(
    results: mp.solutions.pose.PoseLandmarker | object,
    min_visibility: float,
) -> np.ndarray:
    """Convert MediaPipe pose results into a fixed-size keypoint array."""

    if not getattr(results, "pose_landmarks", None):
        return _empty_keypoints()

    keypoints = _empty_keypoints()
    confident_landmarks = 0

    for index, landmark in enumerate(results.pose_landmarks.landmark[:NUM_KEYPOINTS]):
        if landmark.visibility < min_visibility:
            continue

        keypoints[index] = (landmark.x, landmark.y, landmark.z)
        confident_landmarks += 1

    if confident_landmarks == 0:
        return _empty_keypoints()

    return keypoints


def estimate_pose_on_frame(
    frame: np.ndarray,
    pose_model: mp.solutions.pose.Pose,
    min_visibility: float = DEFAULT_CONFIDENCE,
) -> tuple[np.ndarray, object]:
    """Process a single frame and return keypoints plus raw MediaPipe results."""

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_frame)
    keypoints = _extract_frame_keypoints(results, min_visibility=min_visibility)
    return keypoints, results


def draw_pose_overlay(frame: np.ndarray, results: object) -> np.ndarray:
    """Draw the MediaPipe pose skeleton on a frame."""

    overlay_frame = frame.copy()
    if getattr(results, "pose_landmarks", None):
        mp_drawing.draw_landmarks(
            overlay_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
    return overlay_frame


def estimate_pose(
    frames: list[np.ndarray],
    min_detection_confidence: float = DEFAULT_CONFIDENCE,
    min_tracking_confidence: float = DEFAULT_CONFIDENCE,
    min_visibility: float = DEFAULT_CONFIDENCE,
) -> list[np.ndarray]:
    """Process frames and return a keypoint array for each frame.

    Each returned array has shape ``(33, 3)`` and stores ``x, y, z`` values.
    Frames with missing or low-confidence detections return zero-filled arrays.
    """

    if not frames:
        return []

    keypoints_per_frame: list[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        for frame in frames:
            keypoints, _ = estimate_pose_on_frame(
                frame,
                pose_model=pose,
                min_visibility=min_visibility,
            )
            keypoints_per_frame.append(keypoints)

    return keypoints_per_frame
