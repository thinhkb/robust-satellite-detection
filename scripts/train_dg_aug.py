"""
Train YOLO with domain-generalized augmentation (DG-Aug).

Photometric, sensor, and atmospheric transforms are injected into the actual
Ultralytics training dataloader. Geometry remains in YOLO's native pipeline so
bounding boxes are always transformed consistently.
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

try:
    import albumentations as A
except ImportError as exc:
    raise RuntimeError(
        "DG-Aug requires albumentations. Install: pip install -r requirements.txt"
    ) from exc


DG_AUG_PARAMS = {
    "hsv_h": 0.03,
    "hsv_s": 0.60,
    "hsv_v": 0.50,
    "degrees": 10.0,
    "translate": 0.10,
    "scale": 0.50,
    "shear": 2.0,
    "perspective": 0.0001,
    "flipud": 0.10,
    "fliplr": 0.50,
    "mosaic": 1.0,
    "mixup": 0.10,
    "copy_paste": 0.05,
}


def _image_compression():
    """Build ImageCompression across Albumentations 1.x and 2.x."""
    try:
        return A.ImageCompression(quality_range=(60, 90), p=1.0)
    except (TypeError, ValueError):
        return A.ImageCompression(quality_lower=60, quality_upper=90, p=1.0)


def _gauss_noise():
    """Build GaussNoise across Albumentations 1.x and 2.x."""
    try:
        return A.GaussNoise(std_range=(0.04, 0.12), p=1.0)
    except (TypeError, ValueError):
        return A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)


def build_dg_aug_transforms() -> list:
    """Return bbox-safe pixel transforms consumed by Ultralytics."""
    return [
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.30, 0.30),
                    contrast_limit=(-0.30, 0.30),
                    p=1.0,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=50,
                    val_shift_limit=30,
                    p=1.0,
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0,
                ),
            ],
            p=0.65,
        ),
        A.OneOf(
            [
                _gauss_noise(),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0,
                ),
                _image_compression(),
            ],
            p=0.35,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ],
            p=0.25,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.20),
        A.CoarseDropout(p=0.20),
    ]


class DGAugTrainer(DetectionTrainer):
    """Attach custom Albumentations before the training dataset is built."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            from ultralytics.data import augment as yolo_augment

            source = inspect.getsource(yolo_augment.v8_transforms)
            supports_custom = (
                "hyp.augmentations" in source
                or 'getattr(hyp, "augmentations"' in source
            )
            if not supports_custom:
                raise RuntimeError(
                    "This Ultralytics version cannot inject custom transforms. "
                    "Install the version pinned in requirements.txt."
                )

            transforms = build_dg_aug_transforms()
            self.args.augmentations = transforms
            LOGGER.info(
                "DG-Aug dataloader transforms: "
                + ", ".join(type(transform).__name__ for transform in transforms)
            )
        return super().build_dataset(img_path, mode=mode, batch=batch)


def find_yolo_output(project: str | Path, run_name: str) -> Path | None:
    project = Path(project)
    for candidate in [
        project / run_name,
        project / "detect" / run_name,
        project / "detect" / "runs" / run_name,
        project / "detect" / "results" / run_name,
    ]:
        if candidate.exists():
            return candidate
    return None


def train(args) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}" if args.timestamp else args.run_name
    cfg_path = Path(args.cfg).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(cfg_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "device": args.device,
        "seed": args.seed,
        "patience": args.patience,
        "exist_ok": True,
        "verbose": True,
        **DG_AUG_PARAMS,
    }
    model.train(trainer=DGAugTrainer, **train_kwargs)

    final_dir = Path(args.project) / run_name
    output_dir = find_yolo_output(args.project, run_name)
    if output_dir and output_dir.resolve() != final_dir.resolve():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(output_dir), str(final_dir))

    final_dir.mkdir(parents=True, exist_ok=True)
    with open(final_dir / "dg_aug_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "native_yolo": DG_AUG_PARAMS,
                "albumentations": [
                    repr(transform) for transform in build_dg_aug_transforms()
                ],
            },
            handle,
            indent=2,
        )

    best_weights = final_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"Best weights not found: {best_weights}")
    print(f"DG-Aug training done. Best weights: {best_weights}")
    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--run_name", default="dgaug")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--timestamp", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
