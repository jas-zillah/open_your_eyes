import os
import glob
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "face_labels.npy")

FACE_PROTO = os.path.join(MODEL_DIR, "face_detector.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "face_detector.caffemodel")


def ensure_directories():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_face_detector():
    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
        raise FileNotFoundError(
            "Face detector model files not found. Run download_models.py to download the required models."
        )
    return cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


def detect_faces(frame, confidence_threshold=0.6):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net = load_face_detector()
    net.setInput(blob)
    detections = net.forward()
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


def crop_face(frame, box):
    x1, y1, x2, y2, _ = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 - x1 < 50 or y2 - y1 < 50:
        return None
    face = frame[y1:y2, x1:x2]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (200, 200))
    return face_gray


class FaceRecognizer:
    def __init__(self, threshold=80.0):
        ensure_directories()
        self.threshold = threshold
        self.recognizer = None
        self.labels = {}
        self._loaded = False
        self.load()

    def load(self):
        if not os.path.exists(RECOGNIZER_PATH) or not os.path.exists(LABELS_PATH):
            return
        if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "OpenCV face module not available. Install opencv-contrib-python."
            )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(RECOGNIZER_PATH)
        self.labels = np.load(LABELS_PATH, allow_pickle=True).item()
        self._loaded = True

    @property
    def is_loaded(self):
        return self._loaded

    def predict(self, face_gray):
        if not self._loaded or face_gray is None:
            return "Unknown", None
        label_id, confidence = self.recognizer.predict(face_gray)
        if confidence > self.threshold:
            return "Unknown", confidence
        name = self.labels.get(label_id, "Unknown")
        return name, confidence

    def train(self):
        if not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "OpenCV face module not available. Install opencv-contrib-python."
            )

        image_paths = []
        labels = []
        label_map = {}
        next_label = 0

        for person_dir in sorted(os.listdir(KNOWN_FACES_DIR)):
            person_path = os.path.join(KNOWN_FACES_DIR, person_dir)
            if not os.path.isdir(person_path):
                continue
            label_map[next_label] = person_dir
            for image_file in glob.glob(os.path.join(person_path, "*.png")):
                image_paths.append(image_file)
                labels.append(next_label)
            next_label += 1

        if len(image_paths) == 0:
            raise RuntimeError("No known face images found. Capture faces before training.")

        faces = []
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            faces.append(image)

        if len(faces) == 0:
            raise RuntimeError("No valid face images available for training.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.write(RECOGNIZER_PATH)
        np.save(LABELS_PATH, label_map)
        self.recognizer = recognizer
        self.labels = label_map
        self._loaded = True


def capture_face_samples(name, sample_count=20, skip_frames=5):
    ensure_directories()
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam to enroll faces.")

    existing = len(os.listdir(person_dir))
    sample_index = existing + 1
    detection_net = load_face_detector()
    saved = 0
    frame_count = 0

    print(f"Capturing {sample_count} samples for '{name}'. Press q to stop early.")
    while saved < sample_count:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame)
        if boxes:
            frame_count += 1
            if frame_count % skip_frames == 0:
                face_gray = crop_face(frame, boxes[0])
                if face_gray is not None:
                    filename = os.path.join(person_dir, f"img_{sample_index:03d}.png")
                    cv2.imwrite(filename, face_gray)
                    saved += 1
                    sample_index += 1
                    print(f"Saved sample {saved}/{sample_count}")
                    cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (0, 255, 0), 2)

        cv2.putText(frame, f"Samples: {saved}/{sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll Face", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} face samples for {name}.")
