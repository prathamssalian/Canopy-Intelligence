"""
Offline augmentation for the arecanut YLD dataset.

Reads ONLY the original (non-augmented) images + labels from `dataset/`,
generates augmented copies, and writes a fresh dataset to `dataset_aug/`.

Labels are YOLO polygon format:  class x1 y1 x2 y2 ... xn yn   (normalised)
Polygons are transformed with the image, clipped to the frame, and any
annotation that gets cropped away is dropped.  If an augmentation would
leave an image with NO annotations it is retried with a new random
transform -- so unlike the previous Roboflow pass, no empty label files
are produced.

Usage
-----
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --train 300 --valid 60 --test 40
    python scripts/augment_dataset.py --dst dataset_aug --seed 0
"""

import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# An annotation is kept only if, after transforming + clipping, it retains at
# least this fraction of its bounding-box area and is still reasonably large.
MIN_VISIBLE_AREA = 0.35
MIN_SIDE_PX = 10

# How many times to re-roll the random transform before giving up on a sample.
MAX_ATTEMPTS = 12


# ---------------------------------------------------------------------------
# Label IO
# ---------------------------------------------------------------------------

def read_polygons(path):
    """-> list of (class_id, ndarray[N, 2]) with normalised xy coords."""
    polys = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue                      # need class + at least 3 xy pairs
        cls = int(float(parts[0]))
        xy = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        polys.append((cls, xy.reshape(-1, 2)))
    return polys


def write_polygons(path, polys):
    lines = []
    for cls, xy in polys:
        coords = " ".join(f"{v:.6f}" for v in xy.reshape(-1))
        lines.append(f"{cls} {coords}")
    path.write_text("\n".join(lines) + "\n")


def bbox_of(xy):
    return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    """Geometry + photometry suited to nadir drone imagery.

    Nadir shots have no canonical 'up', so flips and 90-degree rotations are
    label-preserving.  Affine is kept mild (scale 0.85-1.2, +/-10% shift) so
    crowns near the frame edge are not sliced off.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.75),
            A.Affine(
                # Kept mild: RandomRotate90 above already supplies the large
                # orientation changes for free, so Affine only needs to add
                # small jitter -- which keeps the smeared border region thin.
                scale=(0.90, 1.15),
                translate_percent=(-0.07, 0.07),
                rotate=(-12, 12),
                shear=(-5, 5),
                # BORDER_REPLICATE, not a reflect mode: albumentations mirrors
                # keypoints into the reflected tiles (3 points in -> 27 out),
                # which would scramble the polygon-to-annotation mapping.
                border_mode=cv2.BORDER_REPLICATE,
                p=0.90,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.22, contrast_limit=0.22, p=0.80
            ),
            # Hue is deliberately tight: YLD is diagnosed by yellowing, so a
            # large hue shift would destroy the very signal that separates
            # the two classes.
            A.HueSaturationValue(
                hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=14, p=0.60
            ),
            A.RandomGamma(gamma_limit=(85, 118), p=0.35),
            A.CLAHE(clip_limit=2.0, p=0.15),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    # The 2.x default (std_range up to 0.44) is far too strong
                    # and buries the crown texture entirely.
                    A.GaussNoise(std_range=(0.03, 0.12)),
                ],
                p=0.30,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def augment_once(pipeline, image, polys):
    """Apply one random transform.

    Returns (image, polys) or None if the transform destroyed every
    annotation.
    """
    h, w = image.shape[:2]

    keypoints, spans = [], []
    for _, xy in polys:
        start = len(keypoints)
        keypoints.extend((float(x) * w, float(y) * h) for x, y in xy)
        spans.append((start, len(keypoints)))

    out = pipeline(image=image, keypoints=keypoints)
    aug_img = out["image"]
    aug_kps = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2)
    ah, aw = aug_img.shape[:2]

    # Some transforms (reflect-style border modes) silently duplicate
    # keypoints, which would misalign every polygon.  Fail loudly instead.
    if len(aug_kps) != len(keypoints):
        raise RuntimeError(
            f"keypoint count changed: {len(keypoints)} -> {len(aug_kps)}; "
            "a transform in the pipeline is not keypoint-safe"
        )

    kept = []
    for (cls, _), (start, end) in zip(polys, spans):
        pts = aug_kps[start:end]
        x1, y1, x2, y2 = bbox_of(pts)
        raw_area = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)

        clipped = pts.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, aw - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, ah - 1)
        cx1, cy1, cx2, cy2 = bbox_of(clipped)
        cw, ch = cx2 - cx1, cy2 - cy1

        if cw < MIN_SIDE_PX or ch < MIN_SIDE_PX:
            continue
        if (cw * ch) / raw_area < MIN_VISIBLE_AREA:
            continue

        norm = clipped / np.asarray([aw, ah], dtype=np.float32)
        kept.append((cls, np.clip(norm, 0.0, 1.0)))

    return (aug_img, kept) if kept else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def originals(split_dir):
    """Original (non-`_aug_`) image/label pairs that actually carry labels."""
    pairs = []
    for img in sorted((split_dir / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXTS or "_aug_" in img.name:
            continue
        lbl = split_dir / "labels" / f"{img.stem}.txt"
        if lbl.exists() and read_polygons(lbl):
            pairs.append((img, lbl))
    return pairs


def allocate(pairs, needed, balance_floor):
    """How many augmentations each original gets.

    Uniform when `balance_floor` is None.  Otherwise images are weighted by
    their share of minority-class (healthy) boxes, which pulls the 1:3.4
    healthy:yld ratio toward parity.  The floor guarantees every original
    still gets augmented, so scene diversity is not sacrificed for balance.
    """
    if not pairs:
        return []

    if balance_floor is None:
        counts = [needed // len(pairs)] * len(pairs)
        for i in range(needed % len(pairs)):
            counts[i] += 1
        return counts

    weights = []
    for _, lbl in pairs:
        classes = [cls for cls, _ in read_polygons(lbl)]
        healthy = sum(1 for c in classes if c == 0)
        share = healthy / len(classes) if classes else 0.0
        weights.append(share + balance_floor)

    total = sum(weights)
    counts = [max(1, int(round(needed * w / total))) for w in weights]

    # Trim/pad to land exactly on the requested total.
    while sum(counts) > needed:
        i = max(range(len(counts)), key=lambda j: counts[j])
        if counts[i] == 1:
            break
        counts[i] -= 1
    while sum(counts) < needed:
        i = max(range(len(counts)), key=lambda j: weights[j] / counts[j])
        counts[i] += 1
    return counts


def process_split(src_split, dst_split, target, pipeline, quality, balance_floor=None):
    pairs = originals(src_split)
    if not pairs:
        print(f"  no labelled originals in {src_split} -- skipped")
        return 0

    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)

    # Originals are copied through verbatim and count toward the target.
    for img, lbl in pairs:
        shutil.copy2(img, dst_split / "images" / img.name)
        shutil.copy2(lbl, dst_split / "labels" / lbl.name)

    written = len(pairs)
    needed = max(target - written, 0)
    if needed == 0:
        print(f"  {written} originals already meet the target of {target}")
        return written

    per_image = allocate(pairs, needed, balance_floor)

    failures = 0
    for (img_path, lbl_path), n in zip(pairs, per_image):
        if n == 0:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ! unreadable: {img_path.name}")
            continue
        polys = read_polygons(lbl_path)

        for k in range(n):
            result = None
            for _ in range(MAX_ATTEMPTS):
                result = augment_once(pipeline, image, polys)
                if result is not None:
                    break
            if result is None:
                failures += 1
                continue

            aug_img, aug_polys = result
            stem = f"{img_path.stem}_aug{k:03d}"
            cv2.imwrite(
                str(dst_split / "images" / f"{stem}.jpg"),
                aug_img,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            write_polygons(dst_split / "labels" / f"{stem}.txt", aug_polys)
            written += 1

    if failures:
        print(f"  ! {failures} sample(s) dropped: no annotation survived")
    print(f"  {len(pairs)} originals + {written - len(pairs)} augmented = {written}")
    return written


def summarise(split_dir):
    counts = {}
    empty = 0
    labels = list((split_dir / "labels").glob("*.txt"))
    for lbl in labels:
        polys = read_polygons(lbl)
        if not polys:
            empty += 1
        for cls, _ in polys:
            counts[cls] = counts.get(cls, 0) + 1
    return len(labels), empty, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_aug")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--valid", type=int, default=60)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--quality", type=int, default=95, help="output JPEG quality")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--balance",
        nargs="?",
        type=float,
        const=0.15,
        default=None,
        metavar="FLOOR",
        help="weight TRAIN augmentation toward healthy-rich images to reduce "
             "class imbalance; optional floor keeps every scene represented "
             "(default 0.15, lower = more aggressive balancing)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)

    pipeline = build_pipeline()
    targets = {"train": args.train, "valid": args.valid, "test": args.test}

    for split, target in targets.items():
        if not (src / split).exists():
            continue
        # Balancing is a training-set intervention only.  Rebalancing valid or
        # test would make the metrics describe a field that does not exist.
        floor = args.balance if split == "train" else None
        note = f" (balanced, floor={floor})" if floor is not None else ""
        print(f"[{split}] target {target}{note}")
        process_split(src / split, dst / split, target, pipeline, args.quality, floor)

    (dst / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names:\n"
        "- healthy\n"
        "- yld\n"
    )

    print("\nsummary")
    for split in targets:
        if not (dst / split).exists():
            continue
        n, empty, counts = summarise(dst / split)
        print(
            f"  {split:5s} {n:4d} images  empty={empty}  "
            f"healthy={counts.get(0, 0)}  yld={counts.get(1, 0)}"
        )
    print(f"\nwrote {dst.resolve()}")


if __name__ == "__main__":
    main()
