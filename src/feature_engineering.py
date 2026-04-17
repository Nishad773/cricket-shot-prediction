"""Feature engineering utilities for cricket shot classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _is_valid_point(point: np.ndarray) -> bool:
    """Treat all-zero keypoints as missing detections."""

    return bool(np.any(point))


def _calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """Return the angle ABC in degrees."""

    if not (_is_valid_point(point_a) and _is_valid_point(point_b) and _is_valid_point(point_c)):
        return float("nan")

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    magnitude = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    if magnitude == 0:
        return float("nan")

    cosine = np.clip(np.dot(vector_ba, vector_bc) / magnitude, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Return the Euclidean distance between two keypoints."""

    if not (_is_valid_point(point_a) and _is_valid_point(point_b)):
        return float("nan")
    return float(np.linalg.norm(point_a - point_b))


def _nanmean_pair(left_value: float, right_value: float) -> float:
    """Average left/right measurements while tolerating missing values."""

    values = np.asarray([left_value, right_value], dtype=float)
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _wrist_center(frame_keypoints: np.ndarray) -> np.ndarray | None:
    """Return the midpoint of detected wrists for a frame."""

    wrists = []
    for wrist_index in (LEFT_WRIST, RIGHT_WRIST):
        wrist = frame_keypoints[wrist_index]
        if _is_valid_point(wrist):
            wrists.append(wrist)

    if not wrists:
        return None
    return np.mean(np.vstack(wrists), axis=0)


def _compute_joint_angles(frame_keypoints: np.ndarray) -> dict[str, float]:
    """Extract joint-angle features for a single frame."""

    left_elbow = _calculate_angle(
        frame_keypoints[LEFT_SHOULDER],
        frame_keypoints[LEFT_ELBOW],
        frame_keypoints[LEFT_WRIST],
    )
    right_elbow = _calculate_angle(
        frame_keypoints[RIGHT_SHOULDER],
        frame_keypoints[RIGHT_ELBOW],
        frame_keypoints[RIGHT_WRIST],
    )

    left_shoulder = _calculate_angle(
        frame_keypoints[LEFT_ELBOW],
        frame_keypoints[LEFT_SHOULDER],
        frame_keypoints[LEFT_HIP],
    )
    right_shoulder = _calculate_angle(
        frame_keypoints[RIGHT_ELBOW],
        frame_keypoints[RIGHT_SHOULDER],
        frame_keypoints[RIGHT_HIP],
    )

    left_knee = _calculate_angle(
        frame_keypoints[LEFT_HIP],
        frame_keypoints[LEFT_KNEE],
        frame_keypoints[LEFT_ANKLE],
    )
    right_knee = _calculate_angle(
        frame_keypoints[RIGHT_HIP],
        frame_keypoints[RIGHT_KNEE],
        frame_keypoints[RIGHT_ANKLE],
    )

    return {
        "elbow_angle": _nanmean_pair(left_elbow, right_elbow),
        "shoulder_angle": _nanmean_pair(left_shoulder, right_shoulder),
        "knee_angle": _nanmean_pair(left_knee, right_knee),
    }


def _compute_distances(frame_keypoints: np.ndarray) -> dict[str, float]:
    """Extract distance-based features for a single frame."""

    left_wrist_to_hip = _calculate_distance(frame_keypoints[LEFT_WRIST], frame_keypoints[LEFT_HIP])
    right_wrist_to_hip = _calculate_distance(frame_keypoints[RIGHT_WRIST], frame_keypoints[RIGHT_HIP])
    left_wrist_to_shoulder = _calculate_distance(
        frame_keypoints[LEFT_WRIST],
        frame_keypoints[LEFT_SHOULDER],
    )
    right_wrist_to_shoulder = _calculate_distance(
        frame_keypoints[RIGHT_WRIST],
        frame_keypoints[RIGHT_SHOULDER],
    )

    return {
        "wrist_to_hip_distance": _nanmean_pair(left_wrist_to_hip, right_wrist_to_hip),
        "wrist_to_shoulder_distance": _nanmean_pair(left_wrist_to_shoulder, right_wrist_to_shoulder),
    }


def extract_frame_features(frame_keypoints: np.ndarray) -> dict[str, float]:
    """Extract angle and distance features for one frame."""

    features = {}
    features.update(_compute_joint_angles(frame_keypoints))
    features.update(_compute_distances(frame_keypoints))
    return features


def compute_wrist_velocity(keypoints_sequence: list[np.ndarray]) -> np.ndarray:
    """Compute wrist velocity magnitudes across consecutive frames."""

    if len(keypoints_sequence) < 2:
        return np.array([], dtype=float)

    centers = [_wrist_center(frame_keypoints) for frame_keypoints in keypoints_sequence]
    velocities = []

    for previous_center, current_center in zip(centers, centers[1:]):
        if previous_center is None or current_center is None:
            velocities.append(float("nan"))
            continue
        velocities.append(float(np.linalg.norm(current_center - previous_center)))

    return np.asarray(velocities, dtype=float)


def _summarize_series(name: str, values: np.ndarray) -> dict[str, float]:
    """Summarize a time series into stable video-level statistics."""

    if values.size == 0 or np.isnan(values).all():
        return {
            f"{name}_mean": 0.0,
            f"{name}_std": 0.0,
            f"{name}_min": 0.0,
            f"{name}_max": 0.0,
        }

    return {
        f"{name}_mean": float(np.nanmean(values)),
        f"{name}_std": float(np.nanstd(values)),
        f"{name}_min": float(np.nanmin(values)),
        f"{name}_max": float(np.nanmax(values)),
    }


def extract_video_features(keypoints_sequence: list[np.ndarray]) -> dict[str, float]:
    """Convert frame-wise keypoints into a single feature dictionary per video."""

    if not keypoints_sequence:
        empty_feature_names = [
            "elbow_angle",
            "shoulder_angle",
            "knee_angle",
            "wrist_to_hip_distance",
            "wrist_to_shoulder_distance",
            "wrist_velocity",
        ]
        features: dict[str, float] = {}
        for name in empty_feature_names:
            features.update(_summarize_series(name, np.array([], dtype=float)))
        return features

    per_frame_features = [extract_frame_features(frame_keypoints) for frame_keypoints in keypoints_sequence]
    features = {}

    for feature_name in per_frame_features[0]:
        series = np.asarray([frame_features[feature_name] for frame_features in per_frame_features], dtype=float)
        features.update(_summarize_series(feature_name, series))

    wrist_velocity = compute_wrist_velocity(keypoints_sequence)
    features.update(_summarize_series("wrist_velocity", wrist_velocity))
    return features


def aggregate_video_features_mean(keypoints_sequence: list[np.ndarray]) -> dict[str, float]:
    """Aggregate frame-wise features into a mean feature vector per video."""

    feature_names = [
        "elbow_angle",
        "shoulder_angle",
        "knee_angle",
        "wrist_to_hip_distance",
        "wrist_to_shoulder_distance",
    ]

    if not keypoints_sequence:
        return {name: 0.0 for name in [*feature_names, "wrist_velocity"]}

    per_frame_features = [extract_frame_features(frame_keypoints) for frame_keypoints in keypoints_sequence]
    aggregated_features: dict[str, float] = {}

    for feature_name in feature_names:
        values = np.asarray([frame_features[feature_name] for frame_features in per_frame_features], dtype=float)
        aggregated_features[feature_name] = (
            0.0 if values.size == 0 or np.isnan(values).all() else float(np.nanmean(values))
        )

    wrist_velocity = compute_wrist_velocity(keypoints_sequence)
    aggregated_features["wrist_velocity"] = (
        0.0 if wrist_velocity.size == 0 or np.isnan(wrist_velocity).all() else float(np.nanmean(wrist_velocity))
    )
    return aggregated_features


def aggregate_window_features(keypoints_sequence: list[np.ndarray]) -> dict[str, float]:
    """Aggregate a sliding window into mean and variance features."""

    feature_names = [
        "elbow_angle",
        "shoulder_angle",
        "knee_angle",
        "wrist_to_hip_distance",
        "wrist_to_shoulder_distance",
    ]

    if not keypoints_sequence:
        empty = {f"{name}_mean": 0.0 for name in feature_names}
        empty.update({f"{name}_var": 0.0 for name in feature_names})
        empty["wrist_velocity_mean"] = 0.0
        empty["wrist_velocity_var"] = 0.0
        return empty

    per_frame_features = [extract_frame_features(frame_keypoints) for frame_keypoints in keypoints_sequence]
    aggregated_features: dict[str, float] = {}

    for feature_name in feature_names:
        values = np.asarray([frame_features[feature_name] for frame_features in per_frame_features], dtype=float)
        if values.size == 0 or np.isnan(values).all():
            aggregated_features[f"{feature_name}_mean"] = 0.0
            aggregated_features[f"{feature_name}_var"] = 0.0
        else:
            aggregated_features[f"{feature_name}_mean"] = float(np.nanmean(values))
            aggregated_features[f"{feature_name}_var"] = float(np.nanvar(values))

    wrist_velocity = compute_wrist_velocity(keypoints_sequence)
    if wrist_velocity.size == 0 or np.isnan(wrist_velocity).all():
        aggregated_features["wrist_velocity_mean"] = 0.0
        aggregated_features["wrist_velocity_var"] = 0.0
    else:
        aggregated_features["wrist_velocity_mean"] = float(np.nanmean(wrist_velocity))
        aggregated_features["wrist_velocity_var"] = float(np.nanvar(wrist_velocity))

    return aggregated_features


def video_features_to_vector(keypoints_sequence: list[np.ndarray]) -> np.ndarray:
    """Return a deterministic numeric feature vector for a video."""

    feature_map = extract_video_features(keypoints_sequence)
    ordered_feature_names = sorted(feature_map)
    return np.asarray([feature_map[name] for name in ordered_feature_names], dtype=float)


def build_feature_frame(samples: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert labeled video samples into a tabular training DataFrame."""

    rows = []
    for sample in samples:
        keypoints_sequence = sample.get("keypoints_sequence", [])
        feature_map = aggregate_window_features(keypoints_sequence)
        if "label" in sample:
            feature_map["label"] = sample["label"]
        rows.append(feature_map)

    return pd.DataFrame(rows)
