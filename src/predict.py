"""Prediction utilities for trained cricket shot classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.dataset_builder import process_video_to_features


DEFAULT_MODEL_PATH = Path("models") / "model.pkl"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a serialized classification model."""

    return joblib.load(model_path)


def _build_feature_frame(feature_map: dict[str, float], model: Any) -> pd.DataFrame:
    """Create a single-row feature frame aligned to the trained model."""

    features = pd.DataFrame([feature_map])
    model_feature_names = list(getattr(model, "feature_names_in_", features.columns))
    return features.reindex(columns=model_feature_names, fill_value=0.0)


def predict_shot(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    target_fps: float = 10.0,
    frame_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Predict the cricket shot label and confidence score for a video."""

    model = load_model(model_path)
    feature_map = process_video_to_features(
        video_path,
        target_fps=target_fps,
        frame_size=frame_size,
    )
    feature_frame = _build_feature_frame(feature_map, model)

    predicted_label = model.predict(feature_frame)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
        confidence = float(probabilities.max())

    return {
        "label": predicted_label,
        "confidence": confidence,
    }
