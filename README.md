# Arecanut Yellow Leaf Disease Detection

Detects yellow leaf disease (YLD) in arecanut palms from drone imagery and
estimates what fraction of a plot is affected.

YOLO11s object detector, two classes (`healthy`, `yld`), served as a Flask app
that runs ONNX Runtime on CPU — no PyTorch at inference time, so it fits inside
a Vercel serverless function.

---

## Results

Held-out test split, 40 images:

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| healthy | 0.634 | 0.475 | 0.474 | 0.176 |
| yld | 0.739 | 0.850 | 0.811 | 0.376 |
| **all** | **0.687** | **0.663** | **0.642** | **0.276** |

The headline number the app reports is the **YLD rate**. On the test split:

| | YLD rate |
|---|---|
| ground truth | 66.7% |
| raw detection counts | 76.7% |
| **what the app reports** (bias-corrected) | **66.8%** |

### Why the counts are corrected

The detector finds diseased palms far more reliably than healthy ones, so
simply counting boxes overstates disease. For a class with true population
`GT`, a detector returns `TP/P` boxes where `TP = R·GT`, hence:

```
predicted_count / true_count = R / P
```

Multiplying each class's count by `P/R` recovers an unbiased population
estimate while leaving the drawn boxes at their high-precision operating point.
Factors live in `api/index.py` (`COUNT_CORRECTION`).

This is why the UI shows two different numbers: the palms actually **marked**
(fewer, high confidence) and the **estimated** composition (bias-corrected).
That is intentional, not a bug.

### Choosing the threshold

Confidence is **0.35**, not the validation split's max-F1 optimum of
0.035/0.076. That optimum sits in the false-positive tail where precision does
not transfer between splits — validation measures yld precision at 0.845 there,
while the held-out test split measures 0.397. At 0.35 only confident detections
survive, and their behaviour is consistent enough that the correction factors
carry over.

---

## Project layout

```
api/index.py                       Flask app + ONNX inference
templates/index.html               UI (client-side downscaling before upload)
best.onnx                          deployed model — committed on purpose
scripts/augment_dataset.py         offline augmentation, polygon-safe
scripts/build_colab_notebook.py    generates the training notebook
notebooks/train_yld_colab.ipynb    Colab training pipeline
requirements.txt                   runtime deps only (no torch)
requirements-train.txt             training/dataset tooling
```

Datasets, `runs/` and `*.pt` checkpoints are gitignored — they live in Google
Drive and are regenerable.

---

## Run locally

```bash
pip install -r requirements.txt
python api/index.py
# http://127.0.0.1:5000
```

### Two-site local workflow

The same Flask process provides two interfaces:

- `http://127.0.0.1:5000/drone` — open the camera once and continuously publish
  YOLO/YLD results with live camera GPS. There is no manual capture step.
- `http://127.0.0.1:5000/user` — view the continuously updated drone result,
  including the annotated frame and each detected tree's latitude/longitude,
  or upload an image and run the same analysis.

The latest drone result is held in process memory and the user site polls it
once per second, so this workflow is intended for local use with one running
Flask process. Restarting the server clears the latest published result.

On the user site, each detection with GPS coordinates has a navigation arrow.
Selecting it requests the user's current browser GPS position, shows the
distance to that tree, and opens a Leaflet map with the user and tree markers.
The map requests a road route from the public OSRM service and falls back to a
straight-line path if routing is unavailable. Browser GPS requires HTTPS (or
localhost) and location permission; map tiles and routing also require network
access. While the popup is open, the browser watches the user's position,
updates the user marker and remaining distance continuously, and refreshes the
route after movement. Tracking stops when the popup is closed.

The drone site provides the same navigation controls for its live detections.
The latest tree list is updated as frames are published, and its navigation
popup watches the drone-site operator's GPS position, updates the remaining
distance, and refreshes the route while the operator moves.

GPS resolution is automatic for uploaded original drone images: filenames are
matched against `data/GPS data.txt`, and embedded EXIF GPS metadata is preferred
when present. The drone site continuously watches a high-accuracy browser GPS
fix and sends latitude, longitude, altitude (when supplied), and accuracy with
each analysed frame. The server uses that camera position and each detection's
image position to estimate the tree coordinates. Browser GPS
requires HTTPS (or localhost) and location permission. Because a camera
position alone cannot measure the tree's ground position exactly, the estimate
also assumes the configured camera field of view and a nadir-facing camera;
use a drone feed that supplies reliable altitude for the best result.

---

## Rebuilding the dataset

Labels are **YOLO polygon format** (`class x1 y1 x2 y2 …`), not `class cx cy w h`.
Ultralytics converts polygons to boxes automatically for detection training.

`scripts/augment_dataset.py` reads only the original (non-`_aug_`) images and
writes a fresh augmented dataset. It never emits an empty label file: if a
transform crops away every annotation it re-rolls, and per-polygon it drops
anything that loses more than 65% of its bounding-box area.

```bash
# unbalanced: reproduces the source class ratio (1 : 3.4)
python scripts/augment_dataset.py --dst dataset_aug

# balanced: weights augmentation toward healthy-rich images (1 : 2.1)
python scripts/augment_dataset.py --dst dataset_bal --balance
```

Balancing applies to **train only** — rebalancing valid/test would make the
metrics describe a plot that does not exist. It measurably reduced the class
bias:

| | healthy mAP50 | yld mAP50 | gap |
|---|---|---|---|
| unbalanced | 0.275 | 0.815 | 2.96× |
| balanced | 0.474 | 0.811 | 1.71× |

> **Note on `A.Affine`:** the pipeline uses `border_mode=cv2.BORDER_REPLICATE`.
> Reflect-style border modes make albumentations mirror keypoints into the
> reflected tiles (3 points in → 27 out), which silently scrambles every
> polygon. The script asserts the keypoint count is unchanged.

---

## Retraining

Open `notebooks/train_yld_colab.ipynb` in Colab (GPU runtime), upload
`dataset_bal` to `MyDrive/YLD/`, and run the cells in order. Checkpoints are
written straight to Drive, so a disconnected session resumes from `last.pt`
with nothing lost.

Cell 7 calibrates thresholds from the PR curve. It reads `f1_curve` / `p_curve`
/ `r_curve` from a single `val()` pass rather than sweeping `model.val(conf=…)`
— that sweep does not work, because Ultralytics reports P and R at the curve's
own max-F1 point regardless of the `conf` passed, so the results look flat and
then "peak" wherever the curve first gets clipped.

Cell 9 runs the exported ONNX through numpy + onnxruntime only. If it prints
detections, the Vercel deployment will work.

---

## Deploying to Vercel

`best.onnx` must be committed. `requirements.txt` deliberately excludes torch
and ultralytics:

| runtime stack | | rejected stack | |
|---|---|---|---|
| onnxruntime | 45 MB | **torch** | **6,625 MB** |
| numpy | 33 MB | opencv | 156 MB |
| Pillow | 16 MB | ultralytics | 8 MB |
| flask + model | 11 MB | | |
| **≈105 MB** ✅ | under the 250 MB limit | **≈6.8 GB** ❌ | |

Vercel caps request bodies at ~4.5 MB while drone stills routinely exceed 8 MB,
so `index.html` downscales images in the browser (long edge 1600 px) before
upload. The model sees 640 px regardless, so this costs no accuracy.

NMS is **class-agnostic**. With per-class NMS one palm can survive as both a
`healthy` and a `yld` box, inflating the total and corrupting the rate.

> **Routing gotcha:** `vercel.json` rewrites `/(.*)` to `/api/index`, and Vercel
> forwards the *rewritten* path to the function. A Flask app declaring only
> `@app.route("/")` therefore serves its own 404 in production while working
> perfectly locally. `api/index.py` registers both `/` and `/api/index`, plus a
> 404 handler that falls back to the uploader. If you see Flask's "The requested
> URL was not found on the server", this is the cause — not a bundling problem.

---

## Limitations

Read these before trusting a number from this model.

- **25 unique training scenes.** The 300 training images are augmentations of
  75 Roboflow variants of ~25 real photographs. Augmentation multiplies pixels,
  not information — this is the binding constraint on accuracy.
- **Validation and test are 6 and 4 real scenes.** Metrics have wide error
  bars, and thresholds calibrated on one split transfer imperfectly to the
  other (see above). Treat all figures as directional.
- **Correction factors are estimated from 219 instances.** They correct average
  bias across many palms; they do not make any single image exact.
- **`healthy` still lags `yld`** (1.71× mAP50 gap). Balancing extracted what the
  existing 114 healthy annotations can give — closing the rest needs more
  healthy palms annotated in more plots.
- Training images include a few AI-generated ones and Roboflow cutout patches
  (black rectangles); validation and test are pure DJI imagery.

The single highest-value improvement is **annotating more real plots**,
particularly healthy palms.
#   A r e c a n u t - L e a f - D i s e a s e - D e t e c t i o n  
 