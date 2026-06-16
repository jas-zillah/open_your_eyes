OYS — OpenCV + PoseNet Face Calibration and Recognition

Overview

OYS is a lightweight Python app that uses your webcam to detect faces and body pose (PoseNet), perform a pose-driven face calibration flow, and provide a simple face recognition pipeline using OpenCV's LBPH recognizer.

Key features

- Live webcam face detection and pose estimation (PoseNet via TensorFlow Lite)
- Interactive calibration: capture straight/left/right/tilt face images driven by pose cues
- Local face enrollment and LBPH training
- Simple GUI control panel (toggle insight, calibrate, save, stop)
- CSV logging of recognition events (local)

Requirements

- Python 3.8+ (3.10+ recommended)
- OpenCV with contrib modules: `pip install opencv-contrib-python`
- TensorFlow (for TFLite Interpreter): `pip install tensorflow` (or `tensorflow-cpu`)
- NumPy
- Tkinter (usually included with standard Python installs on Windows/Mac)

Install

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows PowerShell
pip install -r requirements.txt
```

2. Download models (if not present):

```bash
python download_models.py
```

Run

```bash
python main.py
```

Usage

- Click `Insight ON` to view face boxes and pose keypoints.
- Click `Calibrate Face` to run the pose-driven calibration flow. Follow the on-screen prompts shown on the right of the video window.
- After calibration completes, enter a name and click `Save Calibration` to store images and train the LBPH recognizer.
- Press `q` in the video window or `Stop App` in the control panel to exit.

Calibration notes

- The app listens for a short tone when each stage starts; turn your head when you hear it and hold the target pose briefly.
- Calibration captures multiple images per stage (straight, left, right, tilt) and trains a local LBPH model.
- If pose detection fails or is noisy, ensure good lighting, center your face, and try again.

Privacy and Git

This project stores face images and recognition logs locally under the project folder by default:

- `known_faces/` — saved calibrated/enrolled images (per-name folders)
- `logs/recognition_log.csv` — timestamped recognition events
- `models/face_recognizer.yml`, `models/face_labels.npy` — trained model and label map containing names

Before committing this repository to a public GitHub repo, the project already includes a `.gitignore` to exclude these sensitive directories/files. If you have already committed personal images or logs, do not push them — follow the earlier steps to untrack and purge them from history (BFG or `git filter-repo`).

Commands to untrack sensitive files locally (won't delete your local copies):

```bash
git rm --cached -r logs/ known_faces/ models/face_recognizer.yml models/face_labels.npy
git add .gitignore
git commit -m "Add .gitignore and remove tracked sensitive files (logs, faces, labels)"
```

If you previously pushed sensitive files, remove them from remote history with a history-rewrite tool (BFG or `git filter-repo`) and coordinate with any collaborators.

Troubleshooting

- No tone / audio cue on Windows: ensure speakers are not muted. The app falls back to `winsound.MessageBeep` or a console bell if `Beep` fails.
- Pose not recognized as expected: try relaxing your distance to the camera, ensure even lighting, and enable `Insight` to see pose keypoints.
- Face recognizer not available: install `opencv-contrib-python` to get the `cv2.face` module.

Contributing

Contributions welcome. Open an issue or PR with a short description of the change and rationale. Avoid committing personal face images or logs.

License

This repository does not include an explicit license file. Add one if you intend to publish with usage terms.

