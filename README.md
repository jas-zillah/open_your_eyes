# OpenCV Face + Pose Detection App

This application opens the local webcam on startup and uses face detection + PoseNet pose estimation to detect humans in the scene.

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required models:
   ```bash
   python download_models.py
   ```
4. Enroll known faces and train the recognizer:
   ```bash
   python train_faces.py --name "Alice" --samples 20 --train
   ```
   Repeat for each person you want to recognize.
5. Run the app:
   ```bash
   python main.py
   ```

## Controls

- Press `q` to exit the webcam viewer.

## Face Recognition

- The app uses an LBPH face recognizer to label known faces.
- Known identities are stored in `known_faces/<name>/`.
- If no trained recognizer exists, the app still shows face detection and pose estimation.

## Recognition Log

- Recognized people are logged to `logs/recognition_log.csv`.
- Each row includes a timestamp, name, and confidence score.
- The app avoids logging the same person too frequently by default.

## Notes

- The app uses OpenCV DNN face detection and a TensorFlow Lite PoseNet model.
- If your webcam is on a different device index, edit `main.py` and change `camera_index`.
