"""Run arecanut detection on a live camera feed.

This is intentionally separate from the Flask upload application. It reuses
the same ONNX model and inference helpers, so camera detections behave exactly
like uploaded-image detections.

Install the camera-only dependencies with:

    pip install -r requirements-camera.txt

Then start the default camera with:

    python realtime_camera.py
"""

import argparse
import sys

import cv2
import numpy as np
from PIL import Image

from api.index import detect, draw, summarise

WINDOW_NAME = "Arecanut live detection"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect healthy and YLD arecanut trees from a live camera."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera height (default: 720).",
    )
    return parser.parse_args()


def add_status(frame, stats):
    """Show the same summary metrics as the upload application."""
    text = (
        f"YLD: {stats['yld_percentage']:.1f}%  |  "
        f"Healthy: {stats['detected_healthy']}  |  "
        f"YLD trees: {stats['detected_yld']}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (16, 44, 53), -1)
    cv2.putText(
        frame,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run(camera_index, width, height):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "Check that it is connected and not being used by another app."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    except cv2.error as error:
        camera.release()
        raise RuntimeError(
            "OpenCV was installed without Windows GUI support, so the live "
            "preview cannot be displayed. Run:\n"
            "  pip uninstall opencv-python-headless opencv-contrib-python-headless\n"
            "  pip install -r requirements-camera.txt"
        ) from error

    print("Live detection started. Press Q or Esc in the camera window to stop.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera stopped returning frames.")

            # OpenCV supplies BGR frames; the model and PIL use RGB.
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections = detect(image)
            rendered = draw(image, detections)
            output = cv2.cvtColor(
                np.asarray(rendered), cv2.COLOR_RGB2BGR
            )
            add_status(output, summarise(detections))

            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Some headless OpenCV builds also fail during GUI cleanup.
            pass


def main():
    args = parse_args()
    try:
        run(args.camera, args.width, args.height)
    except (RuntimeError, KeyboardInterrupt) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
