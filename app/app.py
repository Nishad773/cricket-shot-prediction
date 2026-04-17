"""Streamlit app entry point for cricket shot classification."""

from __future__ import annotations

from collections import deque
import tempfile
from pathlib import Path
from typing import Any
import threading

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.feature_engineering import aggregate_window_features
from src.pose_estimation import DEFAULT_CONFIDENCE, draw_pose_overlay, estimate_pose_on_frame
from src.predict import DEFAULT_MODEL_PATH, predict_shot
from src.predict import _build_feature_frame, load_model
from src.video_processing import extract_frames


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
WEBCAM_WINDOW_SIZE = 24
WEBCAM_FRAME_SIZE = (224, 224)
WEBCAM_FRAME_SKIP = 2


class LiveShotVideoProcessor(VideoProcessorBase):
    """Run pose-based shot prediction on streaming webcam frames."""

    def __init__(self) -> None:
        self.model = load_model(DEFAULT_MODEL_PATH)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=DEFAULT_CONFIDENCE,
            min_tracking_confidence=DEFAULT_CONFIDENCE,
        )
        self.keypoint_window: deque[Any] = deque(maxlen=WEBCAM_WINDOW_SIZE)
        self.frame_index = 0
        self.latest_label = "Waiting..."
        self.latest_confidence: float | None = None
        self.state_lock = threading.Lock()

    def _predict_from_window(self) -> tuple[str | None, float | None]:
        """Predict a shot label and confidence from buffered keypoints."""

        if len(self.keypoint_window) < 2:
            return None, None

        feature_map = aggregate_window_features(list(self.keypoint_window))
        feature_frame = _build_feature_frame(feature_map, self.model)
        label = self.model.predict(feature_frame)[0]

        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_frame)[0]
            confidence = float(probabilities.max())

        return str(label), confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a webcam frame and return an annotated frame."""

        image = frame.to_ndarray(format="bgr24")
        display_frame = image.copy()

        if self.frame_index % WEBCAM_FRAME_SKIP == 0:
            resized_frame = cv2.resize(image, WEBCAM_FRAME_SIZE)
            keypoints, results = estimate_pose_on_frame(
                resized_frame,
                pose_model=self.pose,
                min_visibility=DEFAULT_CONFIDENCE,
            )
            self.keypoint_window.append(keypoints)

            overlay_frame = draw_pose_overlay(resized_frame, results)
            display_frame = cv2.resize(overlay_frame, (image.shape[1], image.shape[0]))

            label, confidence = self._predict_from_window()
            if label is not None:
                with self.state_lock:
                    self.latest_label = label
                    self.latest_confidence = confidence

        with self.state_lock:
            label_text = self.latest_label
            confidence_text = (
                "--"
                if self.latest_confidence is None
                else f"{self.latest_confidence:.2%}"
            )

        cv2.putText(
            display_frame,
            f"Shot: {label_text}",
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

        self.frame_index += 1
        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


def _save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Persist an uploaded video to a temporary file."""

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def _render_pose_overlay(video_path: str | Path) -> object | None:
    """Return a sampled frame with a pose skeleton overlay."""

    frames = extract_frames(video_path, target_fps=1.0, frame_size=(480, 270))
    if not frames:
        return None

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            if not results.pose_landmarks:
                continue

            overlay_frame = rgb_frame.copy()
            mp_drawing.draw_landmarks(
                overlay_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
            return overlay_frame

    return None


def main() -> None:
    """Render the Streamlit interface."""

    st.set_page_config(page_title="Cricket Shot Classifier", layout="centered")
    st.title("Cricket Shot Classifier")
    st.write("Upload a cricket shot video or use your webcam for live prediction.")

    model_path = DEFAULT_MODEL_PATH

    if not model_path.exists():
        st.warning(f"Model file not found at {model_path}. Train the model before running predictions.")

    upload_tab, webcam_tab = st.tabs(["Upload Video", "Webcam"])

    with upload_tab:
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
        )
        show_overlay = st.checkbox("Display skeleton overlay", value=False)

        if uploaded_video is not None:
            st.video(uploaded_video)

        if st.button("Predict Shot", type="primary", use_container_width=True):
            if uploaded_video is None:
                st.info("Upload a video to start prediction.")
                return

            if not model_path.exists():
                st.error("Prediction is unavailable because the trained model file is missing.")
                return

            temp_video_path = _save_uploaded_video(uploaded_video)

            try:
                with st.spinner("Running prediction..."):
                    result = predict_shot(temp_video_path, model_path=model_path)

                st.subheader("Prediction")
                st.metric("Shot Label", str(result["label"]))

                confidence = result.get("confidence")
                if confidence is None:
                    st.write("Confidence score unavailable for this model.")
                else:
                    st.metric("Confidence", f"{confidence:.2%}")

                if show_overlay:
                    overlay_image = _render_pose_overlay(temp_video_path)
                    if overlay_image is None:
                        st.info("No confident pose was detected for the overlay preview.")
                    else:
                        st.image(overlay_image, caption="Pose Skeleton Overlay", use_container_width=True)
            except Exception as error:
                st.error(f"Prediction failed: {error}")
            finally:
                temp_video_path.unlink(missing_ok=True)

    with webcam_tab:
        st.write("Start the webcam stream to see live pose overlays and shot predictions.")

        if not model_path.exists():
            st.info("Webcam prediction will be available once `models/model.pkl` exists.")
        else:
            webrtc_streamer(
                key="cricket-shot-webcam",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=LiveShotVideoProcessor,
                async_processing=True,
            )
            st.caption("Prediction and confidence are drawn directly on the live video feed.")


if __name__ == "__main__":
    main()
