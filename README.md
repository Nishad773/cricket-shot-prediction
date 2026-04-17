# Cricket Shot Classification System

This project classifies cricket shots from uploaded videos using a simple pipeline:

- extract video frames with OpenCV
- estimate body pose with MediaPipe Pose
- compute pose-based features
- run a trained `RandomForestClassifier`
- serve predictions through a Streamlit app

## Project Structure

```text
cricket/
├── app.py
├── README.md
├── requirements.txt
├── app/
│   └── app.py
├── data/
├── models/
└── src/
    ├── dataset_builder.py
    ├── feature_engineering.py
    ├── pose_estimation.py
    ├── predict.py
    ├── train_model.py
    └── video_processing.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

Create a dataset CSV first, then train the classifier:

```python
from src.dataset_builder import create_dataset_from_video_folders
from src.train_model import train_classifier

create_dataset_from_video_folders("data", "data/dataset.csv")
train_classifier("data/dataset.csv", "models/model.pkl")
```

Expected labeled data layout:

```text
data/
├── cover_drive/
│   ├── video1.mp4
│   └── video2.mp4
├── pull/
│   └── video3.mp4
└── cut/
    └── video4.mp4
```

## Run the App Locally

Use the root entry point:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

The app now includes:

- uploaded video prediction
- optional pose overlay preview
- live webcam prediction through `streamlit-webrtc`

When using the webcam tab, allow browser camera access when prompted.

## Run Real-Time Webcam Prediction

Use the webcam inference script:

```bash
python -m src.realtime_inference
```

This opens an OpenCV window that:

- captures frames from `cv2.VideoCapture(0)`
- runs MediaPipe Pose on each sampled frame
- keeps a sliding window of recent keypoints
- predicts the shot label from the trained model
- overlays the pose skeleton and prediction on screen

Press `q` to quit the window.

## Deploy on Streamlit Cloud

1. Push this project to a GitHub repository.
2. Make sure `requirements.txt` and `app.py` are in the repository root.
3. Train your model and include `models/model.pkl`, or arrange for that file to be available at runtime.
4. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
5. Click **Create app**.
6. Select your GitHub repository and branch.
7. Set the main file path to `app.py`.
8. Deploy the app.

## Notes for Streamlit Cloud

- If deployment fails during install, confirm all packages in `requirements.txt` are supported by the selected Python runtime.
- The app expects the trained model at `models/model.pkl`.
- Large model files or private training assets may be better stored outside the repo and loaded during startup.
