import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://raw.githubusercontent.com/parthsawant36/Real-Time-Face-Detection-Using-DNN/main/res10_300x300_ssd_iter_140000_fp16.caffemodel"
POSENET_TFLITE_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

FILES = {
    "face_detector.prototxt": FACE_PROTO_URL,
    "face_detector.caffemodel": FACE_MODEL_URL,
    "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite": POSENET_TFLITE_URL,
}


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def download_file(url, path):
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    print("Done")


def main():
    ensure_directory(MODEL_DIR)
    for filename, url in FILES.items():
        target_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(target_path):
            print(f"Already exists: {filename}")
            continue
        try:
            download_file(url, target_path)
        except Exception as exc:
            print(f"Failed to download {filename}: {exc}")
            print("Please download the file manually and place it in the 'models' folder.")
            return
    print("All model files are present.")


if __name__ == "__main__":
    main()
