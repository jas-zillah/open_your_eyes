import argparse
import sys
from face_recognizer import FaceRecognizer, capture_face_samples


def main():
    parser = argparse.ArgumentParser(description="Enroll face samples and train the face recognizer.")
    parser.add_argument("--name", help="Person name to enroll samples for.")
    parser.add_argument("--samples", type=int, default=20, help="Number of face samples to capture.")
    parser.add_argument("--train", action="store_true", help="Train the face recognizer after capture.")
    args = parser.parse_args()

    if args.name:
        try:
            capture_face_samples(args.name, sample_count=args.samples)
        except Exception as error:
            print(f"Error capturing samples: {error}")
            sys.exit(1)

    if args.train:
        recognizer = FaceRecognizer()
        try:
            recognizer.train()
            print("Face recognizer trained successfully.")
        except Exception as error:
            print(f"Error training recognizer: {error}")
            sys.exit(1)

    if not args.name and not args.train:
        parser.print_help()


if __name__ == "__main__":
    main()
