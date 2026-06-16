import csv
import datetime
import os
import platform
import sys
import time
import tkinter as tk

import cv2
import numpy as np

if platform.system() == "Windows":
    import winsound
else:
    winsound = None

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is required to run this application. Install it with: pip install tensorflow")
    sys.exit(1)

from face_recognizer import FaceRecognizer, crop_face

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_PROTO = os.path.join(MODEL_DIR, "face_detector.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "face_detector.caffemodel")
POSENET_MODEL = os.path.join(MODEL_DIR, "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KEYPOINT_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar",
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist",
    "rightWrist", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]


class FacePoseDetector:
    def __init__(self):
        self.face_net = self._load_face_detector()
        self.pose_interpreter, self.input_details, self.output_details = self._load_posenet()

    def _load_face_detector(self):
        if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
            raise FileNotFoundError(
                "Face detection model files not found. Run download_models.py to download the models."
            )
        net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        return net

    def _load_posenet(self):
        if not os.path.exists(POSENET_MODEL):
            raise FileNotFoundError(
                "PoseNet model file not found. Run download_models.py to download the models."
            )
        interpreter = tf.lite.Interpreter(model_path=POSENET_MODEL)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    def detect_faces(self, frame, confidence_threshold=0.6):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            boxes.append((x1, y1, x2, y2, confidence))
        return boxes

    def detect_pose(self, frame, threshold=0.25):
        input_shape = self.input_details[0]["shape"]
        input_height, input_width = input_shape[1], input_shape[2]
        image = cv2.resize(frame, (input_width, input_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(image_rgb.astype(np.float32), axis=0)
        input_data = (input_data - 127.5) / 127.5
        self.pose_interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.pose_interpreter.invoke()
        heatmaps = self.pose_interpreter.get_tensor(self.output_details[0]["index"])
        offsets = self.pose_interpreter.get_tensor(self.output_details[1]["index"])
        keypoints = self._decode_pose(heatmaps, offsets, frame.shape[:2], threshold)
        return keypoints

    @staticmethod
    def _decode_pose(heatmaps, offsets, image_size, threshold):
        heatmaps = heatmaps[0]
        offsets = offsets[0]
        height, width = heatmaps.shape[:2]
        output_stride = 32
        keypoints = []
        for kp in range(heatmaps.shape[2]):
            flat = heatmaps[:, :, kp]
            max_pos = np.unravel_index(np.argmax(flat), flat.shape)
            score = float(flat[max_pos])
            if score < threshold:
                keypoints.append((None, None, score))
                continue
            y, x = max_pos
            offset_y = offsets[y, x, kp]
            offset_x = offsets[y, x, kp + heatmaps.shape[2]]
            pos_y = y * output_stride + offset_y
            pos_x = x * output_stride + offset_x
            scale_x = image_size[1] / (width * output_stride)
            scale_y = image_size[0] / (height * output_stride)
            keypoints.append((int(pos_x * scale_x), int(pos_y * scale_y), score))
        return keypoints

    def draw_faces(self, frame, boxes, identities=None):
        identities = identities or [None] * len(boxes)
        for (x1, y1, x2, y2, confidence), identity in zip(boxes, identities):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 120, 255), 2)
            if identity is not None and identity[0] != "Unknown" and identity[1] is not None:
                label = f"{identity[0]} ({identity[1]:.1f})"
            elif identity is not None and identity[0] == "Unknown":
                label = "Unknown"
            else:
                label = f"Face {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 120, 255), 2)

    def draw_pose(self, frame, keypoints):
        for idx, (x, y, score) in enumerate(keypoints):
            if x is None or y is None:
                continue
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for p1, p2 in POSE_CONNECTIONS:
            if keypoints[p1][0] is None or keypoints[p2][0] is None:
                continue
            cv2.line(frame, (keypoints[p1][0], keypoints[p1][1]), (keypoints[p2][0], keypoints[p2][1]), (0, 220, 255), 2)


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "recognition_log.csv")


def ensure_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["timestamp", "name", "confidence"])


def log_recognitions(recognized_identities, last_log_times, min_interval=5.0):
    now = datetime.datetime.now()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for name, confidence in recognized_identities:
            if name == "Unknown" or confidence is None:
                continue
            last_time = last_log_times.get(name)
            if last_time is not None and (now - last_time).total_seconds() < min_interval:
                continue
            writer.writerow([now.isoformat(timespec="seconds"), name, f"{confidence:.1f}"])
            last_log_times[name] = now


def beep_signal():
    if winsound:
        try:
            winsound.Beep(1200, 200)
            return
        except Exception:
            try:
                winsound.MessageBeep(winsound.MB_OK)
                return
            except Exception:
                pass
    print("\a", end="", flush=True)


def create_control_panel(detector, cap, recognizer):
    root = tk.Tk()
    root.title("OYS Insight Control")
    root.geometry("280x260")
    root.resizable(False, False)

    insight_state = tk.BooleanVar(value=False)
    calibration_state = {
        "active": False,
        "stage": -1,
        "stage_start": None,
        "captured_faces": [],
        "best_face": None,
        "completed": False,
        "ready_to_save": False,
        "instructions": [
            ("Look straight at the camera", 4),
            ("Turn your head left", 4),
            ("Turn your head right", 4),
            ("Tilt your chin up or down", 4),
        ],
    }

    def toggle_insight():
        insight_state.set(not insight_state.get())
        state_text = "Insight ON" if insight_state.get() else "Insight OFF"
        toggle_button.config(text=state_text)

    def set_status(text):
        status_label.config(text=text)
        root.update_idletasks()

    def start_calibration():
        if calibration_state["active"]:
            return
        calibration_state.update({
            "active": True,
            "stage": 0,
            "stage_start": None,
            "captured_faces": [],
            "best_face": None,
            "completed": False,
            "ready_to_save": False,
        })
        if not name_label.winfo_ismapped():
            name_label.pack(pady=(10, 0))
            name_entry.pack(pady=(0, 10))
        if save_button.winfo_ismapped():
            save_button.pack_forget()
        save_button.config(state="disabled")
        set_status("Calibration started. Follow prompts on screen.")
        calibrate_button.config(state="disabled")

    def get_pose_direction(keypoints, frame_width, frame_height):
        def kp(name):
            try:
                idx = KEYPOINT_NAMES.index(name)
            except ValueError:
                return None
            x, y, score = keypoints[idx]
            return (x, y, score) if x is not None else None

        nose = kp("nose")
        left_eye = kp("leftEye")
        right_eye = kp("rightEye")
        left_ear = kp("leftEar")
        right_ear = kp("rightEar")
        left_shoulder = kp("leftShoulder")
        right_shoulder = kp("rightShoulder")

        if not nose:
            return None
        nose_offset = nose[0] - frame_width / 2
        center_threshold = frame_width * 0.12

        if left_ear and left_ear[2] > 0.35 and (not right_ear or right_ear[2] < 0.25):
            return "right"
        if right_ear and right_ear[2] > 0.35 and (not left_ear or left_ear[2] < 0.25):
            return "left"

        if left_eye and right_eye:
            if abs(nose_offset) < center_threshold:
                if left_shoulder and right_shoulder:
                    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    if nose[1] < shoulder_center_y - 20:
                        return "tilt_up"
                    if nose[1] > shoulder_center_y + 20:
                        return "tilt_down"
                return "straight"
            return "left" if nose_offset < 0 else "right"

        if left_ear and right_ear:
            if abs(nose_offset) < center_threshold:
                return "straight"
            return "left" if nose_offset < 0 else "right"

        if left_shoulder and right_shoulder:
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
            if nose[1] < shoulder_center_y - 20:
                return "tilt_up"
            if nose[1] > shoulder_center_y + 20:
                return "tilt_down"

        return None

    def update_calibration(frame, face_boxes, pose_keypoints):
        if not calibration_state["active"] or calibration_state["completed"]:
            return

        stage = calibration_state["stage"]
        if stage < 0 or stage >= len(calibration_state["instructions"]):
            calibration_state["active"] = False
            calibration_state["completed"] = True
            set_status("Calibration complete. Enter name and click Save.")
            calibration_state["ready_to_save"] = any(f is not None for f in calibration_state["captured_faces"])
            if calibration_state["ready_to_save"]:
                if not save_button.winfo_ismapped():
                    save_button.pack(pady=5)
                save_button.config(state="normal")
            calibrate_button.config(state="normal")
            return

        if calibration_state["stage_start"] is None:
            calibration_state["stage_start"] = time.time()
            calibration_state["best_face"] = None
            calibration_state["stable_count"] = 0
            beep_signal()

        instruction, _ = calibration_state["instructions"][stage]
        target = ["straight", "left", "right", "tilt"][stage]
        direction = None
        if pose_keypoints:
            direction = get_pose_direction(pose_keypoints, frame.shape[1], frame.shape[0])

        if face_boxes:
            face_gray = crop_face(frame, face_boxes[0])
            if face_gray is not None:
                calibration_state["best_face"] = face_gray
                cv2.rectangle(frame, (face_boxes[0][0], face_boxes[0][1]), (face_boxes[0][2], face_boxes[0][3]), (0, 255, 0), 2)

        match = False
        if direction == target:
            match = True
        if target == "tilt" and direction in ("tilt_up", "tilt_down"):
            match = True

        if match:
            calibration_state["stable_count"] += 1
        else:
            calibration_state["stable_count"] = 0

        if direction:
            direction_text = direction.replace("_", " ").title()
        else:
            direction_text = "Looking for a good angle..."

        text_block = [
            instruction,
            "Turn your head when the tone sounds",
            f"Target: {target.replace('_', ' ').title()}",
            f"Pose: {direction_text}",
        ]

        box_width = 320
        box_height = 110
        x1 = 10
        y1 = 10
        x2 = x1 + box_width
        y2 = y1 + box_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        for line_index, line_text in enumerate(text_block):
            y_text = y1 + 25 + (line_index * 22)
            cv2.putText(frame, line_text, (x1 + 8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if calibration_state["stable_count"] >= 5 and calibration_state["best_face"] is not None:
            calibration_state["captured_faces"].append(calibration_state["best_face"])
            calibration_state["stage"] += 1
            calibration_state["stage_start"] = None
            if calibration_state["stage"] >= len(calibration_state["instructions"]):
                calibration_state["active"] = False
                calibration_state["completed"] = True
                calibration_state["ready_to_save"] = any(f is not None for f in calibration_state["captured_faces"])
                if calibration_state["ready_to_save"]:
                    if not save_button.winfo_ismapped():
                        save_button.pack(pady=5)
                    save_button.config(state="normal")
                    set_status("Calibration complete. Enter name and click Save.")
                else:
                    set_status("Calibration complete but no faces captured. Retry.")
                calibrate_button.config(state="normal")
            else:
                beep_signal()
                set_status(f"Next stage: {calibration_state['instructions'][calibration_state['stage']][0]}")

    def save_calibration():
        name = name_entry.get().strip()
        if not name:
            set_status("Enter a face name before saving.")
            return
        valid_faces = [face for face in calibration_state["captured_faces"] if face is not None]
        if len(valid_faces) == 0:
            set_status("No valid calibrated faces available. Calibrate again.")
            return

        try:
            saved_count = recognizer.save_calibrated_faces(name, valid_faces, replace=True)
            recognizer.train()
            set_status(f"Saved {saved_count} images and trained {name}.")
            calibration_state.update({
                "active": False,
                "stage": -1,
                "stage_start": None,
                "captured_faces": [],
                "best_face": None,
                "completed": False,
                "ready_to_save": False,
            })
            save_button.config(state="disabled")
            name_entry.delete(0, tk.END)
            name_label.pack_forget()
            name_entry.pack_forget()
        except Exception as error:
            set_status(f"Save failed: {error}")

    toggle_button = tk.Button(root, text="Insight OFF", width=24, command=toggle_insight)
    toggle_button.pack(pady=(15, 5))

    calibrate_button = tk.Button(root, text="Calibrate Face", width=24, command=start_calibration)
    calibrate_button.pack(pady=5)

    name_label = tk.Label(root, text="Face name:")
    name_entry = tk.Entry(root, width=32)

    save_button = tk.Button(root, text="Save Calibration", width=24, command=save_calibration, state="disabled")

    stop_button = tk.Button(root, text="Stop App", width=24, command=root.destroy)
    stop_button.pack(pady=(5, 10))

    status_label = tk.Label(root, text="Ready", anchor="w", justify="left")
    status_label.pack(fill="x", padx=10)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    return root, insight_state, calibration_state, save_button, name_label, name_entry, update_calibration


def main():
    detector = FacePoseDetector()
    recognizer = FaceRecognizer()
    ensure_log_file()
    last_log_times = {}
    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Unable to open webcam index {camera_index}. Trying fallback camera.")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Unable to open any webcam. Exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    control_root, insight_state, calibration_state, save_button, name_label, name_entry, update_calibration = create_control_panel(detector, cap, recognizer)

    if recognizer.is_loaded:
        print("Face recognizer loaded. Known identities will be displayed.")
    else:
        print("No trained face recognizer found. Run train_faces.py to enroll and train.")

    while True:
        if not control_root.winfo_exists():
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        face_boxes = detector.detect_faces(frame)
        pose_keypoints = detector.detect_pose(frame)

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        frame_width = frame.shape[1]

        # Adjust face boxes for flipped frame
        face_boxes = [(frame_width - x2, y1, frame_width - x1, y2, conf)
                      for x1, y1, x2, y2, conf in face_boxes]

        # Adjust keypoints for flipped frame
        pose_keypoints = [(frame_width - x if x is not None else None, y, score)
                          for x, y, score in pose_keypoints]

        if calibration_state["active"]:
            update_calibration(frame, face_boxes, pose_keypoints)

        face_identities = []
        for box in face_boxes:
            face_gray = crop_face(frame, box)
            identity = recognizer.predict(face_gray)
            face_identities.append(identity)

        log_recognitions(face_identities, last_log_times)

        if insight_state.get():
            detector.draw_faces(frame, face_boxes, face_identities)
            detector.draw_pose(frame, pose_keypoints)

        face_count = len(face_boxes)
        pose_points = sum(1 for kp in pose_keypoints if kp[0] is not None)
        cv2.putText(frame, f"Faces: {face_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Pose keypoints: {pose_points}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("OpenCV Face + Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        try:
            control_root.update()
        except tk.TclError:
            break

    cap.release()
    cv2.destroyAllWindows()
    if control_root.winfo_exists():
        control_root.destroy()


if __name__ == "__main__":
    main()
