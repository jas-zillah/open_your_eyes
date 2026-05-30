import csv
import datetime
import os
import sys
import cv2
import numpy as np

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

    if recognizer.is_loaded:
        print("Face recognizer loaded. Known identities will be displayed.")
    else:
        print("No trained face recognizer found. Run train_faces.py to enroll and train.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        face_boxes = detector.detect_faces(frame)
        pose_keypoints = detector.detect_pose(frame)

        face_identities = []
        for box in face_boxes:
            face_gray = crop_face(frame, box)
            identity = recognizer.predict(face_gray)
            face_identities.append(identity)

        log_recognitions(face_identities, last_log_times)
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
