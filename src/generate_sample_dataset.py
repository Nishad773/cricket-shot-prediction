"""Generate a small synthetic cricket-shot dataset for demo and testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DATASET_PATH = Path("data") / "sample_dataset.csv"

FEATURE_TEMPLATES: dict[str, dict[str, float]] = {
    "cover_drive": {
        "elbow_angle_mean": 152.0,
        "elbow_angle_var": 55.0,
        "shoulder_angle_mean": 122.0,
        "shoulder_angle_var": 40.0,
        "knee_angle_mean": 165.0,
        "knee_angle_var": 18.0,
        "wrist_to_hip_distance_mean": 0.62,
        "wrist_to_hip_distance_var": 0.010,
        "wrist_to_shoulder_distance_mean": 0.38,
        "wrist_to_shoulder_distance_var": 0.008,
        "wrist_velocity_mean": 0.085,
        "wrist_velocity_var": 0.0015,
    },
    "pull_shot": {
        "elbow_angle_mean": 108.0,
        "elbow_angle_var": 95.0,
        "shoulder_angle_mean": 88.0,
        "shoulder_angle_var": 62.0,
        "knee_angle_mean": 146.0,
        "knee_angle_var": 34.0,
        "wrist_to_hip_distance_mean": 0.74,
        "wrist_to_hip_distance_var": 0.020,
        "wrist_to_shoulder_distance_mean": 0.58,
        "wrist_to_shoulder_distance_var": 0.012,
        "wrist_velocity_mean": 0.148,
        "wrist_velocity_var": 0.0045,
    },
    "sweep": {
        "elbow_angle_mean": 126.0,
        "elbow_angle_var": 72.0,
        "shoulder_angle_mean": 102.0,
        "shoulder_angle_var": 48.0,
        "knee_angle_mean": 118.0,
        "knee_angle_var": 52.0,
        "wrist_to_hip_distance_mean": 0.44,
        "wrist_to_hip_distance_var": 0.013,
        "wrist_to_shoulder_distance_mean": 0.50,
        "wrist_to_shoulder_distance_var": 0.010,
        "wrist_velocity_mean": 0.131,
        "wrist_velocity_var": 0.0031,
    },
}


def generate_sample_dataset(
    output_path: str | Path = OUTPUT_DATASET_PATH,
    samples_per_class: int = 60,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic training dataset with webcam-compatible features."""

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | str]] = []

    for label, template in FEATURE_TEMPLATES.items():
        for _ in range(samples_per_class):
            row: dict[str, float | str] = {"label": label}
            for feature_name, center in template.items():
                if feature_name.endswith("_var"):
                    scale = max(center * 0.35, 0.002)
                    value = max(float(rng.normal(center, scale)), 0.0)
                elif "distance" in feature_name or "velocity" in feature_name:
                    scale = max(center * 0.12, 0.005)
                    value = max(float(rng.normal(center, scale)), 0.0)
                else:
                    scale = max(center * 0.08, 2.0)
                    value = max(float(rng.normal(center, scale)), 0.0)
                row[feature_name] = value
            rows.append(row)

    dataset = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


if __name__ == "__main__":
    generated = generate_sample_dataset()
    print(f"Saved synthetic dataset to {OUTPUT_DATASET_PATH} with shape {generated.shape}.")
