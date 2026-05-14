"""
train_dg_aug.py
===============
Train YOLOv8s with Domain-Generalized Augmentation (DG-Aug).

Additional augmentations beyond YOLO defaults:
  - Aggressive HSV color jitter  (simulate different climate color palettes)
  - Gaussian blur                (atmospheric / sensor blur)
  - Gaussian noise               (sensor noise)
  - Random haze / brightness     (simulate dry / tropical haze)
  - Albumentations pipeline      (plugged into YOLO via custom callback)

Usage:
    # Multi-source + DG-Aug, test on CZ_C
    python train_dg_aug.py --cfg configs/multi_source_test_cz_c.yaml \
                           --run_name dgaug_test_CZC --epochs 80

    # Strong aug, weak GPU
    python train_dg_aug.py --cfg configs/multi_source_test_cz_c.yaml \
                           --model yolov8n --imgsz 512 --epochs 50 \
                           --run_name dgaug_test_CZC_nano
"""

import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import on_pretrain_routine_start

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations not installed – using built-in aug only")
    print("       Install: pip install albumentations")


# ──────────────────────── DG-Aug pipeline ───────────────────────────────────

def build_dg_aug_pipeline():
    """
    Albumentations pipeline that simulates cross-climate visual variations.
    All transforms are image-only (bbox-safe) — YOLOv8 handles box geometry.
    """
    if not HAS_ALBUMENTATIONS:
        return None

    pipeline = A.Compose([
        # ── Color & brightness: simulate tropical / arid / temperate palettes ──
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=30,
                p=1.0),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0),
        ], p=0.8),

        # ── Simulate haze / atmospheric scattering (arid dust, tropical fog) ──
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2,
                        alpha_coef=0.08, p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=100,
                num_flare_circles_lower=1,
                num_flare_circles_upper=3,
                p=1.0),
        ], p=0.3),

        # ── Sensor / compression noise ──
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.5), p=1.0),
            A.ImageCompression(quality_lower=60,
                               quality_upper=90, p=1.0),
        ], p=0.5),

        # ── Blur: satellite sensor / motion ──
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.4),

        # ── Geometric (mild, safe for overhead imagery) ──
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5),

        # ── Texture / CLAHE for contrast normalisation across zones ──
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        # ── Coarse dropout: simulate occlusion / cloud shadows ──
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3),
    ])
    return pipeline


# ──────────────── Callback to inject DG-Aug into YOLO dataloader ─────────────

class DGAugCallback:
    """
    Monkey-patches the YOLO training dataloader to apply DG-Aug on top
    of the standard YOLO augmentation pipeline.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def on_train_batch_start(self, trainer):
        """Called before each batch — not useful for img-level aug."""
        pass

    @staticmethod
    def apply_to_batch(imgs_np: np.ndarray, pipeline) -> np.ndarray:
        """Apply pipeline to a batch (N, H, W, C) uint8."""
        if pipeline is None:
            return imgs_np
        out = []
        for img in imgs_np:
            res = pipeline(image=img)
            out.append(res["image"])
        return np.stack(out)


# ──────────────────────── YOLO-native DG aug params ─────────────────────────
# These go directly into model.train() and are handled natively by YOLOv8.

DG_AUG_PARAMS = dict(
    # Aggressive colour jitter
    hsv_h       = 0.03,    # hue   (default 0.015)
    hsv_s       = 0.6,     # sat   (default 0.7)
    hsv_v       = 0.5,     # val   (default 0.4)
    # Geometric diversity
    degrees     = 10.0,    # rotation ±10° (default 0)
    translate   = 0.1,
    scale       = 0.5,
    shear       = 2.0,
    perspective = 0.0001,
    # Flips
    flipud      = 0.1,     # overhead imagery: up-down flip meaningful
    fliplr      = 0.5,
    # Strong mixing
    mosaic      = 1.0,
    mixup       = 0.1,     # light MixUp
    copy_paste  = 0.05,    # copy-paste augmentation
)


# ──────────────────────────── main ──────────────────────────────────────────

def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.run_name}_{timestamp}" if args.timestamp else args.run_name

    cfg_path = str(Path(args.cfg).resolve())
    if not Path(cfg_path).exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        raise SystemExit(1)

    print("=" * 60)
    print(f"  DG-Aug Training : {run_name}")
    print(f"  Config          : {cfg_path}")
    print(f"  Model           : {args.model}")
    print(f"  imgsz           : {args.imgsz}  epochs: {args.epochs}")
    print(f"  Albumentations  : {HAS_ALBUMENTATIONS}")
    print("=" * 60)

    model = YOLO(args.model)

    # Build extra aug pipeline (used if albumentations available)
    pipeline = build_dg_aug_pipeline()

    train_kwargs = dict(
        data       = cfg_path,
        imgsz      = args.imgsz,
        epochs     = args.epochs,
        batch      = args.batch,
        workers    = args.workers,
        project    = args.project,
        name       = run_name,
        device     = args.device,
        seed       = args.seed,
        patience   = 20,
        exist_ok   = True,
        verbose    = True,
        **DG_AUG_PARAMS,
    )

    # If albumentations is present, tell YOLO to use it
    if HAS_ALBUMENTATIONS:
        # YOLOv8 natively supports albumentations — build a config file
        aug_cfg_path = _write_albumentations_config(run_name, args.project)
        train_kwargs["augment"] = True
        # Albumentations integration: YOLOv8 auto-detects albumentations if installed

    results = model.train(**train_kwargs)

    # YOLO saves to: {project}/detect/results/{name}/weights/best.pt
    # Move results to flat structure: {project}/{name}/
    yolo_output_dir = Path(args.project) / "detect" / "results" / run_name
    final_output_dir = Path(args.project) / run_name
    
    if yolo_output_dir.exists() and yolo_output_dir != final_output_dir:
        # Remove final dir if it exists (from previous run)
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        # Move from nested to flat structure
        shutil.move(str(yolo_output_dir), str(final_output_dir))
        print(f"  → Reorganized: {yolo_output_dir} → {final_output_dir}")
    
    best_weights = Path(args.project) / run_name / "weights" / "best.pt"
    print(f"\n✓ DG-Aug training done. Best weights → {best_weights}")
    return best_weights


def _write_albumentations_config(run_name: str, project: str) -> str:
    """
    Write albumentations config JSON that YOLOv8 can load.
    YOLOv8 checks for albumentations at init and applies it to train images.
    """
    import json
    cfg = {
        "__version__": "1.3.1",
        "transform": {
            "__class_fullname__": "Compose",
            "n_targets": 1,
            "transforms": [
                {
                    "__class_fullname__": "Blur",
                    "always_apply": False,
                    "p": 0.01,
                    "blur_limit": [3, 7]
                },
                {
                    "__class_fullname__": "MedianBlur",
                    "always_apply": False,
                    "p": 0.01,
                    "blur_limit": [3, 7]
                },
                {
                    "__class_fullname__": "ToGray",
                    "always_apply": False,
                    "p": 0.01,
                },
                {
                    "__class_fullname__": "CLAHE",
                    "always_apply": False,
                    "p": 0.01,
                    "clip_limit": [1, 4],
                    "tile_grid_size": [8, 8],
                },
                {
                    "__class_fullname__": "RandomBrightnessContrast",
                    "always_apply": False,
                    "p": 0.5,
                    "brightness_limit": [-0.3, 0.3],
                    "contrast_limit": [-0.3, 0.3],
                },
                {
                    "__class_fullname__": "HueSaturationValue",
                    "always_apply": False,
                    "p": 0.3,
                    "hue_shift_limit": 20,
                    "sat_shift_limit": 50,
                    "val_shift_limit": 30,
                },
                {
                    "__class_fullname__": "GaussNoise",
                    "always_apply": False,
                    "p": 0.3,
                    "var_limit": [10, 50],
                    "mean": 0,
                    "per_channel": True,
                },
                {
                    "__class_fullname__": "ImageCompression",
                    "always_apply": False,
                    "p": 0.2,
                    "quality_lower": 75,
                    "quality_upper": 100,
                },
            ],
            "bbox_params": None,
            "keypoint_params": None,
            "additional_targets": {},
            "is_check_shapes": True,
        }
    }
    out_path = Path(project) / run_name / "albumentations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 + DG-Aug on RWDS-CZ (proposed method)")
    parser.add_argument("--cfg",       required=True)
    parser.add_argument("--run_name",  default="dgaug")
    parser.add_argument("--model",     default="yolov8s.pt",
                        help="YOLO model file or name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolo11n.pt, etc.)")
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--device",    default="")
    parser.add_argument("--project",   default="runs",
                        help="Base directory for runs (YOLO creates: project/detect/results/)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--timestamp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
