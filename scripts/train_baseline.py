"""
train_baseline.py
=================
Train YOLOv8s (or YOLOv8n) on RWDS-CZ under single-source or multi-source
setup WITHOUT domain-generalization augmentation.

Usage examples:
    # Single-source, train on CZ_A
    python train_baseline.py --cfg configs/single_source_cz_a.yaml \
                             --run_name baseline_CZA --epochs 80

    # Multi-source, test on CZ_C
    python train_baseline.py --cfg configs/multi_source_test_cz_c.yaml \
                             --run_name multisrc_test_CZC --epochs 80

    # Fast (weak GPU)
    python train_baseline.py --cfg configs/single_source_cz_a.yaml \
                             --model yolov8n --imgsz 512 --epochs 50 \
                             --run_name baseline_CZA_nano
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


# ────────────────────── default hyper-params ────────────────────────────────
DEFAULTS = dict(
    model      = "yolov8s.pt",   # pretrained on COCO
    imgsz      = 640,
    epochs     = 80,
    batch      = 16,
    workers    = 4,
    patience   = 20,             # early stopping
    device     = "",             # "" → auto (CUDA if available)
    project    = "runs",         # YOLO will create: runs/detect/results/{name}/
    exist_ok   = True,
    verbose    = True,
    seed       = 42,
    # Standard augmentations (no DG-Aug extras)
    hsv_h      = 0.015,
    hsv_s      = 0.7,
    hsv_v      = 0.4,
    degrees    = 0.0,
    translate  = 0.1,
    scale      = 0.5,
    shear      = 0.0,
    perspective= 0.0,
    flipud     = 0.0,
    fliplr     = 0.5,
    mosaic     = 1.0,
    mixup      = 0.0,
    copy_paste = 0.0,
)


def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.run_name}_{timestamp}" if args.timestamp else args.run_name

    # Resolve config path to absolute so YOLO finds it regardless of cwd
    cfg_path = str(Path(args.cfg).resolve())
    if not Path(cfg_path).exists():
        print(f"[ERROR] Config file not found: {cfg_path}")
        print("  → Run scripts/split_domain.py first, then scripts/check_data.py")
        raise SystemExit(1)

    print("=" * 60)
    print(f"  Training: {run_name}")
    print(f"  Config  : {cfg_path}")
    print(f"  Model   : {args.model}")
    print(f"  imgsz   : {args.imgsz}  epochs: {args.epochs}  batch: {args.batch}")
    print("=" * 60)

    model = YOLO(args.model)

    # Merge defaults with CLI overrides
    train_kwargs = {**DEFAULTS}
    train_kwargs.update(dict(
        data    = cfg_path,
        imgsz   = args.imgsz,
        epochs  = args.epochs,
        batch   = args.batch,
        workers = args.workers,
        project = args.project,
        name    = run_name,
        device  = args.device,
        seed    = args.seed,
        patience= args.patience,
    ))

    # Baseline augmentation (YOLO defaults, no DG-Aug)
    results = model.train(**train_kwargs)

    # YOLO saves to: {project}/detect/{name}/weights/best.pt  (or nested deeper)
    # Move results to flat structure: {project}/{name}/
    # Try multiple possible YOLO output locations
    yolo_output_dir = None
    for candidate in [
        Path(args.project) / "detect" / run_name,
        Path(args.project) / "detect" / "runs" / run_name,
        Path(args.project) / "detect" / "results" / run_name,
    ]:
        if candidate.exists():
            yolo_output_dir = candidate
            break
    final_output_dir = Path(args.project) / run_name
    
    if yolo_output_dir is not None and yolo_output_dir != final_output_dir:
        # Remove final dir if it exists (from previous run)
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        # Move from nested to flat structure
        try:
            shutil.move(str(yolo_output_dir), str(final_output_dir))
            print(f"  → Reorganized: {yolo_output_dir} → {final_output_dir}")
        except Exception as e:
            print(f"[WARN] Failed to reorganize results: {e}")
            print(f"       Results may be in: {yolo_output_dir}")
    elif not final_output_dir.exists() and yolo_output_dir is None:
        print(f"[WARN] Training output not found. Expected: {final_output_dir}")
    
    best_weights = Path(args.project) / run_name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n✓ Training done. Best weights → {best_weights}")
    else:
        print(f"\n[ERROR] Best weights not found at {best_weights}")
    return best_weights


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 baseline on RWDS-CZ")
    parser.add_argument("--cfg",       required=True,
                        help="Dataset YAML (from split_domain.py)")
    parser.add_argument("--run_name",  default="baseline",
                        help="Experiment name")
    parser.add_argument("--model",     default="yolov8s.pt",
                        help="YOLO model file or name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolo11n.pt, etc.)")
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--epochs",    type=int, default=80)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--device",    default="",
                        help="'' = auto, '0' = GPU 0, 'cpu' = CPU")
    parser.add_argument("--project",   default="runs",
                        help="Base directory for runs (YOLO creates: project/detect/results/)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--patience",  type=int, default=20)
    parser.add_argument("--timestamp", action="store_true",
                        help="Append timestamp to run_name")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
