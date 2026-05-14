"""
visualize_bbox.py
=================
Visualise bounding boxes (ground truth and/or predictions) on RWDS-CZ tiles.

Features:
  1. Visualise ground-truth labels from a YOLO label directory
  2. Run inference with a trained model and overlay predictions
  3. Create ID vs OOD comparison grids (paper-style)

Usage:
    # Visualise GT boxes in CZ_A test set
    python visualize_bbox.py \
        --img_dir data/CZ_A/images/test \
        --lbl_dir data/CZ_A/labels/test \
        --out_dir results/predictions/gt_CZA \
        --n_samples 20

    # GT + predictions for a trained model
    python visualize_bbox.py \
        --img_dir data/CZ_B/images/test \
        --lbl_dir data/CZ_B/labels/test \
        --weights results/runs/baseline_CZA/weights/best.pt \
        --out_dir results/predictions/baseline_CZA_on_CZB \
        --n_samples 20

    # Cross-domain comparison grid (ID + OOD side-by-side)
    python visualize_bbox.py \
        --weights results/runs/baseline_CZA/weights/best.pt \
        --mode grid \
        --data_dir data \
        --id_zone CZ_A --ood_zones CZ_B CZ_C \
        --out_dir results/predictions
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    "Building", "Small Car", "Truck", "Bus",
    "Cargo Truck", "Shipping Container", "Vehicle Lot", "Shed",
]

# Distinct colours per class (BGR for OpenCV)
PALETTE = [
    (255, 56,  56 ),  # Building       – red
    (255, 157, 151),  # Small Car      – pink
    (255, 112, 31 ),  # Truck          – orange
    (255, 178, 29 ),  # Bus            – yellow
    (207, 210, 49 ),  # Cargo Truck    – lime
    (72,  249, 10 ),  # Ship. Container– green
    (146, 204, 23 ),  # Vehicle Lot    – light green
    (61,  219, 134),  # Shed           – teal
]


def load_yolo_labels(lbl_path: Path, img_w: int, img_h: int) -> list:
    """Return list of (class_idx, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(img_bgr: np.ndarray,
               boxes: list,
               color_override=None,
               thickness: int = 2,
               label_prefix: str = "") -> np.ndarray:
    """Draw bounding boxes with class labels on image (in-place)."""
    img = img_bgr.copy()
    for box in boxes:
        cls, x1, y1, x2, y2 = box[:5]
        conf = box[5] if len(box) > 5 else None
        color = color_override if color_override else PALETTE[cls % len(PALETTE)]
        name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        label = f"{label_prefix}{name}"
        if conf is not None:
            label += f" {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(img,
                      (x1, y1 - text_size[1] - 4),
                      (x1 + text_size[0], y1),
                      color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def run_inference(model, img_path: str, conf: float = 0.25) -> list:
    """Return list of (cls, x1, y1, x2, y2, conf)."""
    results = model.predict(img_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf_ = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((cls, x1, y1, x2, y2, conf_))
    return boxes


def visualize_single(img_path: Path,
                     lbl_path: Path | None,
                     model=None,
                     conf: float = 0.25,
                     show_gt: bool = True) -> np.ndarray:
    """Return BGR image with GT (green) and/or predictions (red) drawn."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    h, w = img_bgr.shape[:2]

    canvas = img_bgr.copy()

    if show_gt and lbl_path is not None:
        gt_boxes = load_yolo_labels(lbl_path, w, h)
        canvas = draw_boxes(canvas, gt_boxes,
                            color_override=(0, 200, 0),
                            label_prefix="GT:")

    if model is not None:
        pred_boxes = run_inference(model, str(img_path), conf=conf)
        canvas = draw_boxes(canvas, pred_boxes,
                            color_override=(0, 0, 220),
                            label_prefix="P:")

    return canvas


def make_comparison_grid(images_bgr: list,
                         titles: list,
                         cols: int = 3) -> np.ndarray:
    """Stack images into a grid with titles."""
    assert len(images_bgr) == len(titles)
    rows = (len(images_bgr) + cols - 1) // cols
    cell_h = max(img.shape[0] for img in images_bgr)
    cell_w = max(img.shape[1] for img in images_bgr)
    title_h = 28

    grid = np.zeros(
        ((cell_h + title_h) * rows, cell_w * cols, 3), dtype=np.uint8)
    grid[:] = 30  # dark background

    for idx, (img, title) in enumerate(zip(images_bgr, titles)):
        r, c = divmod(idx, cols)
        y0 = r * (cell_h + title_h)
        x0 = c * cell_w
        # paste image
        grid[y0 + title_h: y0 + title_h + img.shape[0],
             x0: x0 + img.shape[1]] = img
        # paste title bar
        cv2.rectangle(grid, (x0, y0), (x0 + cell_w, y0 + title_h),
                      (60, 60, 60), -1)
        cv2.putText(grid, title, (x0 + 6, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return grid


# ──────────────────────────── modes ─────────────────────────────────────────

def mode_single(args):
    """Visualise GT and/or predictions for N random images in img_dir."""
    img_dir = Path(args.img_dir)
    lbl_dir = Path(args.lbl_dir) if args.lbl_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.weights:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    img_files = sorted(list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.png")))
    random.shuffle(img_files)
    img_files = img_files[:args.n_samples]

    for img_path in img_files:
        lbl_path = None
        if lbl_dir:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
        vis = visualize_single(img_path, lbl_path, model=model,
                               conf=args.conf)
        cv2.imwrite(str(out_dir / img_path.name), vis)
        print(f"  Saved {out_dir / img_path.name}")

    print(f"\n✓ {len(img_files)} images saved to {out_dir}")


def mode_grid(args):
    """
    Create a cross-domain comparison grid:
    rows = test zones, cols = models trained on different domains
    """
    from ultralytics import YOLO
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    id_zone   = args.id_zone
    ood_zones = args.ood_zones or [z for z in ["CZ_A", "CZ_B", "CZ_C"]
                                   if z != id_zone]
    all_zones = [id_zone] + ood_zones

    # Pick a few test images per zone
    n = args.n_samples
    grid_cells = []
    titles     = []

    for zone in all_zones:
        img_dir = data_dir / zone / "images" / "test"
        lbl_dir = data_dir / zone / "labels" / "test"
        imgs = sorted(list(img_dir.glob("*.jpg")) +
                      list(img_dir.glob("*.png")))
        random.shuffle(imgs)
        imgs = imgs[:n]

        for img_path in imgs:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            vis = visualize_single(img_path, lbl_path, model=model,
                                   conf=args.conf)
            # Resize to fixed size for grid
            vis = cv2.resize(vis, (512, 512))
            grid_cells.append(vis)
            tag = "(ID)" if zone == id_zone else "(OOD)"
            titles.append(f"{zone}{tag}  {img_path.stem[:12]}")

    grid = make_comparison_grid(grid_cells, titles, cols=n)
    out_path = out_dir / f"grid_{id_zone}_vs_{'_'.join(ood_zones)}.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"\n✓ Grid saved → {out_path}")


# ──────────────────────────── legend ────────────────────────────────────────

def save_legend(out_path: Path):
    """Save a colour legend for the class palette."""
    h = 30
    w = 300
    img = np.ones((h * len(CLASS_NAMES), w, 3), dtype=np.uint8) * 240
    for i, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        y0 = i * h
        cv2.rectangle(img, (5, y0 + 4), (30, y0 + h - 4), color, -1)
        cv2.putText(img, name, (38, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), img)
    print(f"Legend saved → {out_path}")


# ──────────────────────────── main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "grid"],
                        default="single")
    # single mode
    parser.add_argument("--img_dir",  default=None)
    parser.add_argument("--lbl_dir",  default=None)
    # grid mode
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--id_zone",  default="CZ_A")
    parser.add_argument("--ood_zones", nargs="+", default=None)
    # shared
    parser.add_argument("--weights",  default=None)
    parser.add_argument("--out_dir",  default="results/predictions")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--conf",      type=float, default=0.25)
    args = parser.parse_args()

    # Save legend
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    save_legend(Path(args.out_dir) / "class_legend.jpg")

    if args.mode == "grid":
        mode_grid(args)
    else:
        mode_single(args)


if __name__ == "__main__":
    main()
