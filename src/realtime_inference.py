"""Real-time webcam-based cricket shot prediction."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from src.feature_engineering import aggregate_window_features
from src.pose_estimation import DEFAULT_CONFIDENCE, draw_pose_overlay, estimate_pose_on_frame
from src.predict import DEFAULT_MODEL_PATH, _build_feature_frame, load_model


WINDOW_SIZE = 24
FRAME_SIZE = (224, 224)
FRAME_SKIP = 2


def _predict_from_window(window_keypoints: list[Any], model: Any) -> tuple[str | None, float | None]:
    """Predict a shot label and confidence from a sliding keypoint window."""

    if len(window_keypoints) < 2:
        return None, None

    feature_map = aggregate_window_features(window_keypoints)
    feature_frame = _build_feature_frame(feature_map, model)
    label = model.predict(feature_frame)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
        confidence = float(probabilities.max())

    return str(label), confidence


def run_realtime_prediction(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    window_size: int = WINDOW_SIZE,
    frame_size: tuple[int, int] = FRAME_SIZE,
    frame_skip: int = FRAME_SKIP,
) -> None:
    """Run real-time cricket shot prediction on a webcam stream."""

    model = load_model(model_path)
    keypoint_window: deque = deque(maxlen=window_size)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    frame_index = 0
    predicted_label = "Waiting..."
    confidence_text = "--"

    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=DEFAULT_CONFIDENCE,
            min_tracking_confidence=DEFAULT_CONFIDENCE,
        ) as pose:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                display_frame = frame.copy()

                if frame_index % frame_skip == 0:
                    resized_frame = cv2.resize(frame, frame_size)
                    keypoints, results = estimate_pose_on_frame(
                        resized_frame,
                        pose_model=pose,
                        min_visibility=DEFAULT_CONFIDENCE,
                    )
                    keypoint_window.append(keypoints)

                    overlay_source = cv2.resize(frame, frame_size)
                    overlay_frame = draw_pose_overlay(overlay_source, results)
                    display_frame = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

                    label, confidence = _predict_from_window(list(keypoint_window), model)
                    if label is not None:
                        predicted_label = label
                        confidence_text = "--" if confidence is None else f"{confidence:.2%}"

                cv2.putText(
                    display_frame,
                    f"Shot: {predicted_label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Confidence: {confidence_text}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Press 'q' to quit",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Cricket Shot Prediction", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_prediction()
