"""Generates notebooks/train_yld_colab.ipynb from the cell list below.

Keeping the notebook in a generator makes it reviewable in git diffs; run this
script after editing a cell.
"""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
(MD, """# Arecanut YLD — YOLO11 training (Colab)

Trains on the augmented dataset stored in Google Drive, checkpoints to Drive so a
disconnect costs nothing, and exports an ONNX model for Vercel deployment.

**Runtime → Change runtime type → GPU (T4)** before running anything.

Cell order matters on a first run. If Colab drops the session, just re-run
cells 1–4 then cell 5 — it auto-resumes from the last checkpoint."""),

(CODE, """# 1. GPU check
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install "ultralytics==8.4.136" onnx onnxruntime onnxslim

import torch, ultralytics
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)"""),

(CODE, '''# 2. Set paths, then mount Drive
from pathlib import Path

# ---- EDIT THESE TWO IF YOUR DRIVE LAYOUT DIFFERS ----------------------------
DRIVE_ROOT   = Path("/content/drive/MyDrive/YLD")      # your project folder
DATASET_NAME = "dataset_aug"                           # folder OR dataset_aug.zip
# -----------------------------------------------------------------------------

RUNS_DIR = DRIVE_ROOT / "runs"          # checkpoints live here (survive restarts)
RUN_NAME = "yld_v2"
LOCAL_DS = Path("/content/dataset_aug") # training reads from local disk, not Drive

# The constants above are assigned BEFORE mounting on purpose. If drive.mount()
# fails or its auth popup is dismissed, those names still exist, so the next
# cell reports a useful error instead of a bare NameError.
from google.colab import drive
drive.mount("/content/drive")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
print("drive root :", DRIVE_ROOT, "|", "exists" if DRIVE_ROOT.exists() else "MISSING")
print("runs dir   :", RUNS_DIR)'''),

(CODE, '''# 3. Copy the dataset to local disk
#
# Drive is a FUSE mount -- reading 400 small images from it every epoch is
# painfully slow. Copying to /content first cuts epoch time by roughly 5-10x.
import shutil, time

missing = [n for n in ("DRIVE_ROOT", "DATASET_NAME", "LOCAL_DS") if n not in globals()]
if missing:
    raise NameError(
        f"{', '.join(missing)} not defined -- run cell 2 first.\\n"
        "If you restarted the runtime (Colab prompts for this after cell 1's "
        "pip install), every variable was cleared: re-run cells 1-2, "
        "completing the Drive authorisation popup."
    )
if not Path("/content/drive/MyDrive").exists():
    raise RuntimeError("Drive is not mounted -- re-run cell 2 and approve the popup.")

zip_src    = DRIVE_ROOT / f"{DATASET_NAME}.zip"
folder_src = DRIVE_ROOT / DATASET_NAME

if LOCAL_DS.exists():
    shutil.rmtree(LOCAL_DS)

t0 = time.time()
if zip_src.exists():
    print("unzipping", zip_src)
    shutil.unpack_archive(str(zip_src), "/content")
elif folder_src.exists():
    print("copying", folder_src, "(a few minutes for ~400 files over Drive)")
    shutil.copytree(folder_src, LOCAL_DS)
else:
    raise FileNotFoundError(
        f"Found neither {zip_src} nor {folder_src}.\\n"
        f"Upload your dataset_aug folder (or a zip of it) to {DRIVE_ROOT}."
    )

# Both a zip and a Drive folder upload can land one level deeper than expected
# (e.g. /content/dataset_aug/dataset_aug/...), so locate the real root by
# looking for the split layout rather than trusting the path.
def find_root(start):
    if (start / "train" / "images").is_dir():
        return start
    for cand in sorted(start.glob("**/train/images")):
        return cand.parent.parent
    return None

resolved = find_root(LOCAL_DS) if LOCAL_DS.exists() else find_root(Path("/content"))
if resolved is None:
    raise FileNotFoundError(
        f"No train/images directory found under {LOCAL_DS}. "
        "The upload may be incomplete -- check the folder in Drive."
    )
if resolved != LOCAL_DS:
    print("resolved dataset root ->", resolved)
LOCAL_DS = resolved

# Drive's web uploader drops files silently on flaky connections, so confirm
# each split is actually present before burning GPU time on a partial dataset.
for split in ("train", "valid", "test"):
    n_img = len(list((LOCAL_DS / split / "images").glob("*")))
    n_lbl = len(list((LOCAL_DS / split / "labels").glob("*.txt")))
    print(f"  {split:5s} {n_img:4d} images / {n_lbl:4d} labels"
          + ("   <-- MISMATCH" if n_img != n_lbl else ""))

print(f"ready in {time.time() - t0:.1f}s")'''),

(CODE, '''# 4. Verify the dataset and write an absolute-path data.yaml
#
# Ultralytics resolves relative `train:` paths against its own settings dir,
# which is a classic source of "dataset not found" in Colab. An absolute
# `path:` key removes the ambiguity.
import collections, yaml

counts = {}
for split in ("train", "valid", "test"):
    img_dir = LOCAL_DS / split / "images"
    lbl_dir = LOCAL_DS / split / "labels"
    if not img_dir.exists():
        print(f"{split:5s} MISSING")
        continue

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    per_class, empty, orphan = collections.Counter(), 0, 0
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            orphan += 1
            continue
        rows = [r for r in lbl.read_text().splitlines() if r.strip()]
        if not rows:
            empty += 1
        for r in rows:
            per_class[int(float(r.split()[0]))] += 1

    counts[split] = per_class
    flag = "  <-- FIX THIS" if (empty or orphan) else ""
    print(f"{split:5s} {len(images):4d} images | empty={empty} unlabelled={orphan} "
          f"| healthy={per_class[0]} yld={per_class[1]}{flag}")

h, y = counts["train"][0], counts["train"][1]
print(f"\\ntrain class ratio healthy:yld = 1:{y / max(h, 1):.1f}")

DATA_YAML = LOCAL_DS / "data_colab.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path":  str(LOCAL_DS),
    "train": "train/images",
    "val":   "valid/images",
    "test":  "test/images",
    "nc":    2,
    "names": ["healthy", "yld"],
}, sort_keys=False))
print("\\nwrote", DATA_YAML)'''),

(CODE, '''# 5. Train (auto-resumes after a disconnect)
#
# Ultralytics already writes weights/best.pt (highest fitness so far) and
# weights/last.pt every epoch. save_period=10 additionally keeps periodic
# epochN.pt snapshots. All of it lands on Drive via project=RUNS_DIR.
from ultralytics import YOLO

last_ckpt = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
resuming  = last_ckpt.exists()

if resuming:
    print("resuming from", last_ckpt)
    model = YOLO(str(last_ckpt))
    results = model.train(resume=True)
else:
    print("starting fresh")
    # yolo11s over yolo11n: with only 25 unique scenes the bottleneck is data,
    # but the small model still converges better than nano at the same cost.
    model = YOLO("yolo11s.pt")
    results = model.train(
        data=str(DATA_YAML),
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,

        epochs=150,
        patience=40,          # early stop if val fitness stalls
        batch=16,
        imgsz=640,
        seed=0,
        device=0,
        workers=2,

        save=True,
        save_period=10,       # extra epochN.pt snapshots on Drive
        plots=True,
        val=True,

        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=5,

        # Offline augmentation is already baked into the dataset, so keep the
        # online pass light -- stacking both over-distorts these images.
        hsv_h=0.010, hsv_s=0.35, hsv_v=0.25,
        degrees=0.0, translate=0.05, scale=0.25, shear=0.0,
        fliplr=0.5, flipud=0.5,
        mosaic=0.5, close_mosaic=15,
        mixup=0.0, erasing=0.0,
    )

BEST = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
print("\\nbest weights ->", BEST)'''),

(CODE, '''# 6. Evaluate best.pt on the held-out test split
from ultralytics import YOLO

best = YOLO(str(BEST))
metrics = best.val(data=str(DATA_YAML), split="test", imgsz=640, plots=True)

print(f"\\nmAP50    {metrics.box.map50:.3f}")
print(f"mAP50-95 {metrics.box.map:.3f}")
print(f"precision {metrics.box.mp:.3f}   recall {metrics.box.mr:.3f}")
for i, name in enumerate(["healthy", "yld"]):
    p, r, ap50, ap = metrics.box.class_result(i)
    print(f"  {name:8s} P={p:.3f} R={r:.3f} mAP50={ap50:.3f}")'''),

(CODE, '''# 7. Calibrate PER-CLASS thresholds from the valid-split PR curve
#
# Do NOT calibrate by sweeping model.val(conf=...). Ultralytics derives P and R
# from ap_per_class(), which reports them at the CURVE'S OWN max-F1 point --
# so they stay pinned no matter what conf you pass. conf only truncates the
# prediction list, making the sweep look flat and then "peak" at whatever value
# first clips the curve. It measures the clipping point, not the optimum.
#
# The full curve is already computed in a single val() pass. Read it directly.
import numpy as np
from ultralytics import YOLO

model = YOLO(str(BEST))
res = model.val(data=str(DATA_YAML), split="val", imgsz=640,
                plots=True, verbose=False)

box = res.box
px = np.asarray(box.px)         # confidence axis: 1000 points from 0 to 1
f1 = np.asarray(box.f1_curve)   # (n_classes, 1000)
pc = np.asarray(box.p_curve)
rc = np.asarray(box.r_curve)

NAMES = ["healthy", "yld"]
CONF_BY_CLASS = {}

print(f"{'class':10s} {'conf':>7} {'F1':>7} {'P':>7} {'R':>7}")
for row, cls_idx in enumerate(box.ap_class_index):
    k = int(f1[row].argmax())
    name = NAMES[int(cls_idx)]
    CONF_BY_CLASS[name] = round(float(px[k]), 3)
    print(f"{name:10s} {px[k]:7.3f} {f1[row][k]:7.3f} {pc[row][k]:7.3f} {rc[row][k]:7.3f}")

k = int(f1.mean(0).argmax())
print(f"\\nsingle global threshold would be {px[k]:.3f} (mean F1 {f1.mean(0)[k]:.3f})")

# --- count-balanced thresholds -------------------------------------------
#
# Max-F1 is the right objective for per-box quality, but this app reports a
# PERCENTAGE, and percentages care about counts. Since
#
#     predicted_count = TP / P = (R * GT) / P     =>   predicted/true = R/P
#
# a class is counted without bias exactly when P == R. Picking each class's
# threshold at that crossover makes the YLD rate unbiased, even though it
# sacrifices a little F1.
COUNT_CONF = {}
print(f"\\n{'class':10s} {'objective':16s} {'conf':>7} {'P':>7} {'R':>7} {'R/P':>6}")
for row, cls_idx in enumerate(box.ap_class_index):
    name = NAMES[int(cls_idx)]
    k_f1 = int(f1[row].argmax())
    # Ignore the degenerate tail where recall collapses to ~0 and P==R trivially.
    usable = rc[row] > 0.02
    k_bal = int(np.where(usable, np.abs(pc[row] - rc[row]), np.inf).argmin())
    COUNT_CONF[name] = round(float(px[k_bal]), 3)
    for label, k2 in (("max-F1", k_f1), ("count-balanced", k_bal)):
        p, r = pc[row][k2], rc[row][k2]
        print(f"{name:10s} {label:16s} {px[k2]:7.3f} {p:7.3f} {r:7.3f} "
              f"{(r / p if p else 0):6.2f}")

print("\\nmax-F1 thresholds        :", CONF_BY_CLASS)
print("count-balanced thresholds:", COUNT_CONF)

# Use the count-balanced set in the Flask app: the headline number is a rate.
CONF_BY_CLASS = COUNT_CONF
BEST_CONF = min(CONF_BY_CLASS.values())
print(f"\\n>>> infer at conf={BEST_CONF:.3f}, then filter per class")'''),

(CODE, '''# 8. Export to ONNX for Vercel
#
# Vercel's Python functions cap at 250 MB unzipped. torch + ultralytics is well
# over that on its own, so the deployed app must NOT import them. ONNX +
# onnxruntime + numpy + Pillow fits comfortably.
from ultralytics import YOLO
import shutil

model = YOLO(str(BEST))
onnx_path = model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False,   # fixed 1x3x640x640 -- smaller and faster to load cold
    half=False,      # Vercel is CPU-only; fp16 would be slower there
)

deploy_dir = DRIVE_ROOT / "deploy"
deploy_dir.mkdir(parents=True, exist_ok=True)
final_onnx = deploy_dir / "best_model.onnx"
shutil.copy(str(onnx_path), final_onnx)
shutil.copy(str(BEST), deploy_dir / "best_model.pt")   # keep the torch one too

print("ONNX  ->", final_onnx, f"({final_onnx.stat().st_size / 1e6:.1f} MB)")
print("conf threshold from cell 7:", f"{BEST_CONF:.2f}")'''),

(CODE, '''# 9. Prove the export works WITHOUT torch or ultralytics
#
# This is the actual Vercel compatibility test: the exact numpy + onnxruntime
# code path the deployed function will use. If this cell prints detections,
# the deployment will work.
import numpy as np, onnxruntime as ort, glob
from PIL import Image

IOU = 0.50
NAMES = ["healthy", "yld"]
THR = np.array([CONF_BY_CLASS[n] for n in NAMES])   # per-class cut from cell 7


def letterbox(img, size=640):
    """Resize preserving aspect ratio, pad to square. Returns (chw, scale, pads)."""
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], s, (px, py)


def nms(boxes, scores, thr):
    """Class-agnostic NMS: one palm must not be counted as healthy AND yld."""
    idx, keep = scores.argsort()[::-1], []
    while idx.size:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        idx = rest[inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9) <= thr]
    return keep


sess = ort.InferenceSession(str(final_onnx), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
print("input", sess.get_inputs()[0].shape, "-> output", sess.get_outputs()[0].shape)

for path in sorted(glob.glob(str(LOCAL_DS / "test/images/*.jpg")))[:3]:
    img = Image.open(path).convert("RGB")
    blob, scale, (px, py) = letterbox(img)

    # YOLO11 ONNX output is (1, 4 + nc, 8400): xywh then per-class scores.
    out = sess.run(None, {inp: blob})[0][0].T
    cls_scores = out[:, 4:]
    conf, cls = cls_scores.max(1), cls_scores.argmax(1)
    # Each class must clear its OWN threshold, not a shared one -- healthy
    # scores markedly lower than yld, so a single cut silently drops it.
    m = conf >= THR[cls]
    if not m.any():
        print(f"{Path(path).name[:30]:32s} no detections")
        continue

    xywh, conf, cls = out[m, :4], conf[m], cls[m]
    boxes = np.stack([
        (xywh[:, 0] - xywh[:, 2] / 2 - px) / scale,
        (xywh[:, 1] - xywh[:, 3] / 2 - py) / scale,
        (xywh[:, 0] + xywh[:, 2] / 2 - px) / scale,
        (xywh[:, 1] + xywh[:, 3] / 2 - py) / scale,
    ], 1)
    keep = nms(boxes, conf, IOU)
    n_h = int((cls[keep] == 0).sum())
    n_y = int((cls[keep] == 1).sum())
    rate = 100 * n_y / max(n_h + n_y, 1)
    print(f"{Path(path).name[:30]:32s} healthy={n_h} yld={n_y}  YLD rate={rate:.1f}%")'''),

(MD, """## Deploying to Vercel

Download `MyDrive/YLD/deploy/best_model.onnx` and commit it (~40 MB, under the
100 MB git limit).

`requirements.txt` — **no torch, no ultralytics, no opencv**:

```
flask
onnxruntime
numpy
pillow
```

`vercel.json` — note the pattern must be `dataset/**`, not `train/**`, or the
whole dataset gets uploaded:

```json
{
  "functions": {
    "api/index.py": {
      "includeFiles": "{templates/**,best_model.onnx}",
      "excludeFiles": "{dataset/**,dataset_aug/**,scripts/**,runs/**,*.pt}"
    }
  }
}
```

Three things to carry into `api/index.py`:

1. Use the single `CONF_THRESHOLD` from cell 7 — drop the split
   `HEALTHY_THRESHOLD` / `YLD_THRESHOLD` pair.
2. Keep NMS **class-agnostic** (cell 9 does this). Otherwise one palm can be
   emitted as both `healthy` and `yld`, inflating the total and corrupting the
   YLD percentage.
3. Load the ONNX session once at module scope, not per request.

If the function still exceeds 250 MB, `onnxruntime` is the only heavy
dependency left — swap it for `onnxruntime-slim`, or move inference to a
Hugging Face Space and have Vercel call it."""),
]


def main():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out = Path(__file__).resolve().parent.parent / "notebooks" / "train_yld_colab.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
