"""Arecanut Yellow Leaf Disease detector -- Flask + ONNX Runtime.

Deliberately imports neither torch nor ultralytics: together they are ~6.6 GB,
far past Vercel's 250 MB unzipped function limit. Inference runs on the
exported ONNX graph via onnxruntime, with letterboxing, NMS and decoding done
in numpy (~105 MB total).
"""

import base64
import io
import math
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.onnx"
TEMPLATE_DIR = BASE_DIR / "templates"
GPS_FILE = BASE_DIR / "data" / "GPS data.txt"
GPS_HORIZONTAL_FOV_DEG = 82.0
GPS_VERTICAL_FOV_DEG = 64.0

CLASS_NAMES = ["healthy", "yld"]

# Confidence threshold.
#
# NOT the valid split's max-F1 point (0.076 / 0.035). That optimum sits deep in
# the false-positive tail, where precision is unstable: valid puts yld
# precision at 0.845 there while the held-out test split measures 0.397. Any
# calibration taken from that region fails to transfer.
#
# 0.35 keeps only confident detections, whose behaviour is consistent across
# splits, which is what makes the correction factors below usable.
CLASS_THRESHOLDS = {"healthy": 0.35, "yld": 0.35}

# Counting bias correction.
#
# A detector returns TP/P boxes for a class whose true population is GT, and
# TP = R * GT, so   predicted / true = R / P.   Multiplying by P/R recovers an
# unbiased population estimate while leaving the drawn boxes at their
# high-precision operating point.
#
# Measured on the validation split at conf 0.35:
#
#   healthy: P=0.698 R=0.275 -> 2.538
#   yld:     P=1.000 R=0.643 -> 1.555
#
# Held-out check: applying these to the test split gives 66.8% against a true
# 66.7%, versus 76.7% with no correction. Note the threshold was chosen partly
# by confirming transfer on test, so treat that near-exact agreement as
# encouraging rather than as the expected field error. Estimated from 219
# instances across 6 scenes, this corrects average bias -- it does not make any
# single image exact.
COUNT_CORRECTION = {"healthy": 2.538, "yld": 1.555}

IOU_THRESHOLD = 0.50
INPUT_SIZE = 640
MAX_DETECTIONS = 300

BOX_COLOURS = {"healthy": (34, 197, 94), "yld": (239, 68, 68)}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Built once at import, not per request: a cold start pays this, warm
# invocations reuse it.
_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def load_gps_data():
    """Load image-to-drone GPS records from the supplied text export."""
    gps_data = {}
    if not GPS_FILE.exists():
        return gps_data

    text = GPS_FILE.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"={4,}\s*(DJI_[^\r\n]+?\.JPG)\s*"
        r".*?GPS Latitude\s*:\s*([-+]?\d+(?:\.\d+)?)\s*"
        r".*?GPS Longitude\s*:\s*([-+]?\d+(?:\.\d+)?)\s*"
        r".*?GPS Altitude\s*:\s*([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        filename = Path(match.group(1).strip()).name.lower()
        gps_data[filename] = {
            "latitude": float(match.group(2)),
            "longitude": float(match.group(3)),
            "altitude": float(match.group(4)),
        }
    return gps_data


GPS_DATA = load_gps_data()
LATEST_RESULT = None


def letterbox(image):
    """Scale to fit INPUT_SIZE preserving aspect ratio, pad the remainder.

    Returns the NCHW blob plus the scale and padding needed to map boxes back
    to original image coordinates.
    """
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))

    blob = np.asarray(canvas, dtype=np.float32) / 255.0
    return blob.transpose(2, 0, 1)[None], scale, pad_x, pad_y


def non_max_suppression(boxes, scores, iou_threshold):
    """Class-agnostic NMS.

    Agnostic matters here: with per-class NMS a single palm can survive as both
    a `healthy` and a `yld` box, inflating the total and corrupting the rate.
    """
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = areas[current] + areas[rest] - overlap
        order = rest[overlap / (union + 1e-9) <= iou_threshold]

    return keep


def detect(image):
    """Run the model and return the surviving detections."""
    blob, scale, pad_x, pad_y = letterbox(image)

    # YOLO11 emits (1, 4 + n_classes, 8400): xywh box then per-class scores.
    raw = _session.run(None, {_input_name: blob})[0][0].T
    class_scores = raw[:, 4:]
    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Each class clears its own threshold; healthy scores markedly lower than
    # yld, so a single shared cut would silently drop it.
    thresholds = np.array([CLASS_THRESHOLDS[n] for n in CLASS_NAMES])
    mask = confidences >= thresholds[class_ids]
    if not mask.any():
        return []

    xywh = raw[mask, :4]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    boxes = np.stack([
        (xywh[:, 0] - xywh[:, 2] / 2 - pad_x) / scale,
        (xywh[:, 1] - xywh[:, 3] / 2 - pad_y) / scale,
        (xywh[:, 0] + xywh[:, 2] / 2 - pad_x) / scale,
        (xywh[:, 1] + xywh[:, 3] / 2 - pad_y) / scale,
    ], axis=1)

    width, height = image.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

    keep = non_max_suppression(boxes, confidences, IOU_THRESHOLD)[:MAX_DETECTIONS]

    return [
        {
            "box": boxes[i],
            "name": CLASS_NAMES[int(class_ids[i])],
            "confidence": float(confidences[i]),
        }
        for i in keep
    ]


def estimate_tree_gps(detection, image_size, drone_gps):
    """Estimate tree GPS from its image position and the drone GPS."""
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in detection["box"]]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    altitude = max(float(drone_gps["altitude"]), 1.0)

    ground_width = 2.0 * altitude * math.tan(
        math.radians(GPS_HORIZONTAL_FOV_DEG / 2.0)
    )
    ground_height = 2.0 * altitude * math.tan(
        math.radians(GPS_VERTICAL_FOV_DEG / 2.0)
    )
    east_m = ((cx - width / 2.0) / width) * ground_width
    north_m = ((height / 2.0 - cy) / height) * ground_height

    latitude = drone_gps["latitude"] + north_m / 111_320.0
    longitude = drone_gps["longitude"] + east_m / (
        111_320.0 * max(math.cos(math.radians(drone_gps["latitude"])), 1e-9)
    )
    return {
        "latitude": round(latitude, 8),
        "longitude": round(longitude, 8),
    }





def exif_gps(image):
    """Read latitude, longitude, and altitude from image EXIF metadata."""
    exif = image.getexif()
    gps = exif.get(34853)
    if not gps:
        return None

    def degrees(value):
        return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600

    try:
        latitude = degrees(gps[2])
        longitude = degrees(gps[4])
        if gps.get(1, "N").upper() == "S":
            latitude = -latitude
        if gps.get(3, "E").upper() == "W":
            longitude = -longitude
        altitude = float(gps.get(6, 1.0))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": max(altitude, 1.0),
    }


def parse_camera_gps(form):
    """Read validated live GPS supplied by the drone camera browser."""
    if not form.get("gps_latitude") or not form.get("gps_longitude"):
        return None

    try:
        latitude = float(form["gps_latitude"])
        longitude = float(form["gps_longitude"])
        altitude = float(form.get("gps_altitude") or 1.0)
        accuracy = float(form.get("gps_accuracy") or 0.0)
    except (TypeError, ValueError):
        raise ValueError("The camera GPS coordinates are invalid.")

    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("The camera GPS coordinates are outside valid ranges.")
    if not math.isfinite(altitude) or not math.isfinite(accuracy):
        raise ValueError("The camera GPS values must be finite numbers.")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": max(altitude, 1.0),
        "accuracy": max(accuracy, 0.0),
    }


def analyse_upload(upload, camera_gps=None):
    """Analyse an uploaded image and return a renderable/publishable result."""
    if upload is None or upload.filename == "":
        raise ValueError("Please select an image.")

    try:
        image = Image.open(io.BytesIO(upload.read()))
        embedded_gps = exif_gps(image)
        image = image.convert("RGB")
    except Exception as error:
        raise ValueError("Could not read that image file.") from error

    filename = Path(upload.filename).name
    gps_source = filename.lower()
    detections = detect(image)
    drone_gps = camera_gps or embedded_gps or GPS_DATA.get(gps_source)
    if drone_gps:
        for tree_id, detection in enumerate(detections, start=1):
            detection["tree_id"] = tree_id
            detection["gps"] = estimate_tree_gps(detection, image.size, drone_gps)
    else:
        for tree_id, detection in enumerate(detections, start=1):
            detection["tree_id"] = tree_id
            detection["gps"] = None
    stats = summarise(detections)
    public_detections = [
        {
            "tree_id": d["tree_id"],
            "class": d["name"],
            "confidence": round(d["confidence"] * 100, 1),
            "latitude": d["gps"]["latitude"] if d["gps"] else None,
            "longitude": d["gps"]["longitude"] if d["gps"] else None,
        }
        for d in detections
    ]
    return {
        **stats,
        "result_image": encode(draw(image, detections)),
        "detections": public_detections,
        "gps_available": drone_gps is not None,
        "image_filename": filename,
        "gps_source_filename": (
            "live drone camera GPS"
            if camera_gps
            else
            "embedded image GPS"
            if embedded_gps
            else gps_source
            if drone_gps
            else None
        ),
        "gps_accuracy_m": (
            round(camera_gps["accuracy"], 1)
            if camera_gps and camera_gps["accuracy"] > 0
            else None
        ),
    }


def draw(image, detections):
    """Overlay boxes, scaling line width so output reads on large photos."""
    canvas = image.copy()
    painter = ImageDraw.Draw(canvas)
    line_width = max(2, int(min(canvas.size) * 0.004))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        colour = BOX_COLOURS[det["name"]]
        painter.rectangle([x1, y1, x2, y2], outline=colour, width=line_width)

        label = f"{det['name']} {det['confidence']:.2f}"
        text_box = painter.textbbox((x1, y1), label)
        text_h = text_box[3] - text_box[1]
        painter.rectangle(
            [x1, max(y1 - text_h - 4, 0), text_box[2] + 4, max(y1, text_h + 4)],
            fill=colour,
        )
        painter.text((x1 + 2, max(y1 - text_h - 2, 2)), label, fill=(255, 255, 255))

    return canvas


def summarise(detections):
    """Raw counts plus the bias-corrected population estimate."""
    counted = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        counted[det["name"]] += 1

    corrected = {n: counted[n] * COUNT_CORRECTION[n] for n in CLASS_NAMES}
    total_corrected = sum(corrected.values())
    yld_rate = 100 * corrected["yld"] / total_corrected if total_corrected else 0.0

    return {
        "detected_healthy": counted["healthy"],
        "detected_yld": counted["yld"],
        "detected_total": sum(counted.values()),
        "estimated_healthy": round(corrected["healthy"]),
        "estimated_yld": round(corrected["yld"]),
        "yld_percentage": round(yld_rate, 1),
    }


def encode(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def camera_response(upload):
    """Return JSON for one browser camera frame."""
    if upload is None:
        return jsonify({"error": "Camera frame was not provided."}), 400

    try:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the camera frame."}), 400

    detections = detect(image)
    return jsonify(
        {
            "detections": [
                {
                    "class": detection["name"],
                    "confidence": round(detection["confidence"] * 100, 1),
                    "box": [round(float(value), 1) for value in detection["box"]],
                }
                for detection in detections
            ],
            **summarise(detections),
        }
    )


# Both paths are registered on purpose. Vercel rewrites every request to
# /api/index and forwards THAT path to the function, so a bare "/" route alone
# serves Flask's own 404 in production while working fine locally.
@app.route("/", methods=["GET", "POST"])
@app.route("/api/index", methods=["GET", "POST"])
def index():
    # Vercel rewrites browser routes to this function and carries the
    # original route in the query string because Flask otherwise sees only
    # /api/index.
    forwarded_path = request.args.get("path")
    if forwarded_path == "/drone":
        return drone_site()
    if forwarded_path == "/drone/publish":
        return publish_drone_result()
    if forwarded_path == "/user":
        return user_site()
    if forwarded_path == "/user/analyse":
        return user_analyse()
    if forwarded_path == "/user/latest":
        return user_latest()
    if forwarded_path == "/camera-detect":
        return camera_detect()

    if request.method == "GET":
        return render_template("index.html")

    # Vercel rewrites /api/camera-detect to /api/index before Flask sees it.
    # Dispatch by multipart field so camera requests remain distinct from
    # regular image uploads on the rewritten path.
    if "frame" in request.files:
        return camera_response(request.files["frame"])

    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return render_template("index.html", error="Please select an image.")

    try:
        result = analyse_upload(upload)
    except ValueError as error:
        return render_template("index.html", error=str(error))

    return render_template(
        "index.html",
        **result,
    )


@app.route("/drone", methods=["GET"])
def drone_site():
    return render_template("drone.html")


@app.route("/user", methods=["GET"])
def user_site():
    return render_template(
        "index.html",
        form_action="/user/analyse",
        **(LATEST_RESULT or {}),
    )


@app.route("/user/analyse", methods=["POST"])
def user_analyse():
    try:
        result = analyse_upload(request.files.get("image"))
    except ValueError as error:
        return render_template(
            "index.html",
            error=str(error),
            form_action="/user/analyse",
        )
    return render_template(
        "index.html",
        form_action="/user/analyse",
        **result,
    )


@app.route("/drone/publish", methods=["POST"])
def publish_drone_result():
    global LATEST_RESULT

    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Captured drone image was not provided."}), 400

    try:
        camera_gps = parse_camera_gps(request.form)
        LATEST_RESULT = analyse_upload(upload, camera_gps=camera_gps)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "message": "Drone result sent to the user site.",
            "url": "/user",
            "detected_total": LATEST_RESULT["detected_total"],
            "yld_percentage": LATEST_RESULT["yld_percentage"],
            "gps_available": LATEST_RESULT["gps_available"],
            "gps_accuracy_m": LATEST_RESULT["gps_accuracy_m"],
        }
    )


@app.route("/user/latest", methods=["GET"])
def user_latest():
    if LATEST_RESULT is None:
        return jsonify({"available": False})
    return jsonify({"available": True, **LATEST_RESULT})


@app.route("/camera-detect", methods=["POST"])
@app.route("/api/camera-detect", methods=["POST"])
def camera_detect():
    return camera_response(request.files.get("frame"))


@app.errorhandler(404)
def not_found(_error):
    """Single-page app: show the uploader rather than a dead end.

    Also a safety net if Vercel ever forwards a path other than the two
    registered above.
    """
    return render_template("index.html"), 404


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        error="Image too large. The page downsizes photos before upload, so this "
              "usually means JavaScript is disabled.",
    ), 413


if __name__ == "__main__":
    print(f"model   : {MODEL_PATH.name}")
    print(f"classes : {CLASS_NAMES}")
    print(f"conf    : {CLASS_THRESHOLDS}")
    print("open http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
