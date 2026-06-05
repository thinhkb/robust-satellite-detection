"""
train_acs_yolo.py
=================
Train ACS-YOLO: Adaptive Correspondence Scoring for robust satellite object
detection under climate-zone distribution shift.

This script adapts the core idea from Adaptive Correspondence Scoring (AdaCS):
training residuals should not be trusted uniformly when domain-specific nuisance
factors create unstable supervision. For object detection, we estimate an
image-level correspondence/reliability score from prediction consistency under
style-only domain perturbations, then reweight YOLO detection loss during the
final training stage.

Pipeline:
  1. Warm up a YOLO detector for a small number of epochs.
  2. Score every training image by comparing predictions on the original image
     and on a climate-style augmented copy, optionally checked against GT boxes.
  3. Continue training from the warm-up weights with an ACS-weighted YOLO loss.

Usage:
    python scripts/train_acs_yolo.py \
        --cfg configs/multi_source_test_cz_c.yaml \
        --run_name acs_test_CZC \
        --model yolov8s.pt \
        --epochs 80 \
        --warmup_epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# YOLO-native augmentation used in the final ACS stage. It intentionally mirrors
# the existing DG-Aug recipe so ACS is evaluated as "DG-Aug + adaptive scoring".
ACS_AUG_PARAMS = dict(
    hsv_h=0.03,
    hsv_s=0.60,
    hsv_v=0.50,
    degrees=10.0,
    translate=0.10,
    scale=0.50,
    shear=2.0,
    perspective=0.0001,
    flipud=0.10,
    fliplr=0.50,
    mosaic=1.0,
    mixup=0.10,
    copy_paste=0.05,
)

BASELINE_AUG_PARAMS = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)


ACS_CONTEXT: dict[str, Any] = {}


def norm_path(path: str | Path) -> str:
    """Normalize paths for stable score lookup across Windows/Unix separators."""
    return str(Path(path).resolve()).replace("\\", "/").lower()


def find_label_for_image(img_path: str | Path) -> Path:
    """Convert a YOLO image path to its matching label path."""
    img_path = Path(img_path)
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name


def load_data_yaml(cfg_path: str | Path) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def collect_images(paths: Any) -> list[Path]:
    """Collect image files from YOLO yaml train entries."""
    images: list[Path] = []
    for item in as_list(paths):
        p = Path(item)
        if p.is_dir():
            for ext in IMG_EXTS:
                images.extend(p.rglob(f"*{ext}"))
        elif p.is_file() and p.suffix.lower() == ".txt":
            with open(p, encoding="utf-8") as f:
                images.extend(Path(line.strip()) for line in f if line.strip())
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted({Path(x).resolve() for x in images})


def read_yolo_labels(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read YOLO labels and return cls, xyxy in pixel units."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    h, w = shape
    rows = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            rows.append([cls, x1, y1, x2, y2])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def result_to_numpy(result: Any) -> np.ndarray:
    """Convert an Ultralytics result to cls, conf, xyxy rows."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(np.float32)
    return np.column_stack([cls, conf, xyxy]).astype(np.float32)


def gt_alignment_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean class-aware GT alignment score in [0, 1]."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.5
    if len(pred) == 0:
        return 0.0

    vals = []
    for gt_row in gt:
        same_cls = pred[:, 0] == gt_row[0]
        if not np.any(same_cls):
            vals.append(0.0)
            continue
        candidates = pred[same_cls]
        ious = box_iou_matrix(gt_row[None, 1:5], candidates[:, 2:6])[0]
        best = int(np.argmax(ious))
        vals.append(float(ious[best] * candidates[best, 1]))
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def prediction_consistency_score(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric class-aware prediction consistency in [0, 1]."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0

    def one_way(src: np.ndarray, dst: np.ndarray) -> float:
        vals = []
        for row in src:
            same_cls = dst[:, 0] == row[0]
            if not np.any(same_cls):
                vals.append(0.0)
                continue
            candidates = dst[same_cls]
            ious = box_iou_matrix(row[None, 2:6], candidates[:, 2:6])[0]
            best = int(np.argmax(ious))
            vals.append(float(ious[best] * min(row[1], candidates[best, 1])))
        return float(np.mean(vals)) if vals else 0.0

    return float(np.clip(0.5 * (one_way(a, b) + one_way(b, a)), 0.0, 1.0))


def apply_domain_style_aug(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Style-only perturbation that simulates climate/sensor shift without moving boxes."""
    out = img_bgr.copy()

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8.0, 8.0)) % 180.0
    hsv[..., 1] *= rng.uniform(0.70, 1.35)
    hsv[..., 2] *= rng.uniform(0.65, 1.35)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-25.0, 25.0)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < 0.35:
        noise = rng.normal(0, rng.uniform(4.0, 14.0), size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.25:
        haze = np.full_like(out, rng.uniform(175, 230), dtype=np.uint8)
        out = cv2.addWeighted(out, rng.uniform(0.75, 0.90), haze, rng.uniform(0.10, 0.25), 0)

    if rng.random() < 0.25:
        h, w = out.shape[:2]
        overlay = out.copy()
        for _ in range(int(rng.integers(1, 5))):
            rw = int(rng.uniform(0.04, 0.16) * w)
            rh = int(rng.uniform(0.04, 0.16) * h)
            x1 = int(rng.integers(0, max(1, w - rw)))
            y1 = int(rng.integers(0, max(1, h - rh)))
            cv2.rectangle(overlay, (x1, y1), (x1 + rw, y1 + rh), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)

    return out


def score_one_pair(gt: np.ndarray, pred_orig: np.ndarray, pred_aug: np.ndarray, score_floor: float) -> dict[str, float]:
    """Combine supervised residual and augmentation consistency into an ACS score."""
    gt_orig = gt_alignment_score(gt, pred_orig)
    gt_aug = gt_alignment_score(gt, pred_aug)
    pred_cons = prediction_consistency_score(pred_orig, pred_aug)

    raw = 0.25 * gt_orig + 0.45 * gt_aug + 0.30 * pred_cons
    score = score_floor + (1.0 - score_floor) * float(np.clip(raw, 0.0, 1.0))
    return {
        "score": float(np.clip(score, score_floor, 1.0)),
        "raw": float(np.clip(raw, 0.0, 1.0)),
        "gt_orig": float(gt_orig),
        "gt_aug": float(gt_aug),
        "pred_consistency": float(pred_cons),
    }


def compute_acs_scores(
    weights: str | Path,
    cfg_path: str | Path,
    out_json: str | Path,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    score_floor: float,
    aug_repeats: int,
    seed: int,
) -> Path:
    """Score training images using warm-up model consistency under domain perturbations."""
    data = load_data_yaml(cfg_path)
    images = collect_images(data.get("train"))
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not images:
        raise RuntimeError(f"No training images found in {cfg_path}")

    rng = np.random.default_rng(seed)
    model = YOLO(str(weights))
    score_items: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="acs_aug_") as tmp:
        tmp_dir = Path(tmp)
        for start in tqdm(range(0, len(images), batch), desc="ACS scoring"):
            batch_paths = images[start : start + batch]
            sources: list[str] = []
            meta: list[tuple[Path, Path]] = []

            for img_path in batch_paths:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    LOGGER.warning(f"ACS scoring skipped unreadable image: {img_path}")
                    continue

                for repeat in range(max(1, aug_repeats)):
                    aug = apply_domain_style_aug(img, rng)
                    aug_path = tmp_dir / f"{start}_{repeat}_{img_path.stem}.jpg"
                    cv2.imwrite(str(aug_path), aug)
                    sources.extend([str(img_path), str(aug_path)])
                    meta.append((img_path, aug_path))

            if not sources:
                continue

            results = model.predict(
                source=sources,
                imgsz=imgsz,
                batch=max(1, min(batch * 2, len(sources))),
                device=device,
                conf=conf,
                iou=iou,
                verbose=False,
            )

            per_image: dict[str, list[dict[str, float]]] = {}
            for idx, (img_path, _aug_path) in enumerate(meta):
                pred_orig = result_to_numpy(results[2 * idx])
                pred_aug = result_to_numpy(results[2 * idx + 1])
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gt = read_yolo_labels(find_label_for_image(img_path), img.shape[:2])
                item = score_one_pair(gt, pred_orig, pred_aug, score_floor=score_floor)
                per_image.setdefault(norm_path(img_path), []).append(item)

            for key, items in per_image.items():
                avg = {
                    field: float(np.mean([it[field] for it in items]))
                    for field in ["score", "raw", "gt_orig", "gt_aug", "pred_consistency"]
                }
                score_items[key] = {
                    **avg,
                    "image": str(Path(key)),
                    "n_repeats": len(items),
                }

    scores = np.array([v["score"] for v in score_items.values()], dtype=np.float32)
    summary = {
        "n_images": int(len(score_items)),
        "mean": float(scores.mean()) if len(scores) else 1.0,
        "std": float(scores.std()) if len(scores) else 0.0,
        "min": float(scores.min()) if len(scores) else 1.0,
        "p10": float(np.percentile(scores, 10)) if len(scores) else 1.0,
        "p50": float(np.percentile(scores, 50)) if len(scores) else 1.0,
        "p90": float(np.percentile(scores, 90)) if len(scores) else 1.0,
        "max": float(scores.max()) if len(scores) else 1.0,
    }
    payload = {
        "method": "ACS-YOLO",
        "description": "Image-level adaptive correspondence scores from prediction consistency under style shift.",
        "cfg": str(Path(cfg_path).resolve()),
        "weights": str(Path(weights).resolve()),
        "score_floor": float(score_floor),
        "aug_repeats": int(max(1, aug_repeats)),
        "summary": summary,
        "scores": score_items,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_json.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "score", "raw", "gt_orig", "gt_aug", "pred_consistency", "n_repeats"],
        )
        writer.writeheader()
        for key, item in sorted(score_items.items()):
            writer.writerow({"image": key, **item})

    LOGGER.info(f"ACS scores saved to {out_json}")
    LOGGER.info(f"ACS score summary: {summary}")
    return out_json


def load_score_table(score_json: str | Path) -> tuple[dict[str, float], float]:
    """Load score JSON and return score map + dataset mean."""
    with open(score_json, encoding="utf-8") as f:
        payload = json.load(f)

    raw_scores = payload.get("scores", {})
    scores = {}
    for key, value in raw_scores.items():
        if isinstance(value, dict):
            scores[norm_path(key)] = float(value.get("score", 1.0))
        else:
            scores[norm_path(key)] = float(value)

    mean_score = float(payload.get("summary", {}).get("mean", 0.0))
    if mean_score <= 0 and scores:
        mean_score = float(np.mean(list(scores.values())))
    return scores, max(mean_score, 1e-6)


class ACSV8DetectionLoss(v8DetectionLoss):
    """YOLOv8 detection loss with ACS image-level reliability weighting."""

    def __init__(
        self,
        model: torch.nn.Module,
        score_by_path: dict[str, float],
        score_mean: float,
        default_score: float = 1.0,
        min_weight: float = 0.25,
        max_weight: float = 1.50,
        normalize_scores: bool = True,
    ):
        super().__init__(model)
        self.score_by_path = score_by_path
        self.score_mean = max(float(score_mean), 1e-6)
        self.default_score = float(default_score)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_scores = bool(normalize_scores)

    def batch_image_weights(self, batch: dict[str, Any], batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        paths = batch.get("im_file", [])
        if isinstance(paths, (str, Path)):
            paths = [paths]
        weights = []
        for path in list(paths)[:batch_size]:
            score = self.score_by_path.get(norm_path(path))
            if score is None:
                score = self.score_mean if self.normalize_scores else self.default_score
            weights.append(float(score))
        if len(weights) < batch_size:
            fill = self.score_mean if self.normalize_scores else self.default_score
            weights.extend([fill] * (batch_size - len(weights)))

        out = torch.tensor(weights, device=self.device, dtype=dtype)
        if self.normalize_scores:
            out = out / self.score_mean
        return out.clamp_(self.min_weight, self.max_weight)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Copy of Ultralytics loss with ACS weights injected into cls/box/dfl terms."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        image_weights = self.batch_image_weights(batch, batch_size, dtype)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        weighted_target_scores = target_scores * image_weights.view(batch_size, 1, 1)
        target_scores_sum = weighted_target_scores.sum().clamp(min=1.0)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        bce_loss = bce_loss * image_weights.view(batch_size, 1, 1)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                weighted_target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class ACSDetectionTrainer(DetectionTrainer):
    """Detection trainer that swaps in ACSV8DetectionLoss after model setup."""

    def set_model_attributes(self):
        super().set_model_attributes()
        scores = ACS_CONTEXT.get("scores", {})
        self.model.criterion = ACSV8DetectionLoss(
            self.model,
            score_by_path=scores,
            score_mean=ACS_CONTEXT.get("score_mean", 1.0),
            default_score=ACS_CONTEXT.get("default_score", 1.0),
            min_weight=ACS_CONTEXT.get("min_weight", 0.25),
            max_weight=ACS_CONTEXT.get("max_weight", 1.50),
            normalize_scores=ACS_CONTEXT.get("normalize_scores", True),
        )
        LOGGER.info(
            "ACS-YOLO loss enabled: "
            f"{len(scores)} scored images, mean={ACS_CONTEXT.get('score_mean', 1.0):.4f}, "
            f"weight clamp=[{ACS_CONTEXT.get('min_weight', 0.25)}, {ACS_CONTEXT.get('max_weight', 1.5)}]"
        )

    def set_class_weights(self):
        super().set_class_weights()
        criterion = getattr(self.model, "criterion", None)
        class_weights = getattr(self.model, "class_weights", None)
        if criterion is not None and class_weights is not None:
            criterion.class_weights = class_weights.to(criterion.device).view(1, 1, -1)


def yolo_train_kwargs(args: argparse.Namespace, cfg_path: str, run_name: str, epochs: int, aug: str) -> dict[str, Any]:
    aug_params = ACS_AUG_PARAMS if aug == "dg_aug" else BASELINE_AUG_PARAMS
    return dict(
        data=cfg_path,
        imgsz=args.imgsz,
        epochs=epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=run_name,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
        **aug_params,
    )


def flatten_yolo_run(project: str | Path, run_name: str) -> Path:
    """Move Ultralytics nested output to project/run_name, matching the repo scripts."""
    project = Path(project)
    final = project / run_name
    candidates = [
        final,
        project / "detect" / run_name,
        project / "detect" / "runs" / run_name,
        project / "detect" / "results" / run_name,
    ]

    for cand in candidates:
        if (cand / "weights" / "best.pt").exists():
            if cand.resolve() != final.resolve():
                if final.exists():
                    shutil.rmtree(final)
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cand), str(final))
                LOGGER.info(f"Reorganized: {cand} -> {final}")
            return final

    for cand in candidates:
        if cand.exists():
            if cand.resolve() != final.resolve():
                if final.exists():
                    shutil.rmtree(final)
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(cand), str(final))
                LOGGER.info(f"Reorganized: {cand} -> {final}")
            return final
    return final


def best_weights_for(project: str | Path, run_name: str) -> Path:
    run_dir = flatten_yolo_run(project, run_name)
    return run_dir / "weights" / "best.pt"


def train(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}" if args.timestamp else args.run_name
    cfg_path = str(Path(args.cfg).resolve())
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if args.warmup_epochs >= args.epochs:
        adjusted = max(args.epochs - 1, 0)
        LOGGER.warning(f"warmup_epochs={args.warmup_epochs} >= epochs={args.epochs}; using warmup_epochs={adjusted}")
        args.warmup_epochs = adjusted
    final_epochs = max(args.epochs - args.warmup_epochs, 1)

    final_weights = best_weights_for(args.project, run_name)
    if final_weights.exists() and not args.force_retrain:
        LOGGER.info(f"[SKIP] ACS-YOLO weights already exist: {final_weights}")
        return final_weights

    print("=" * 70)
    print(f"  ACS-YOLO Training : {run_name}")
    print(f"  Config            : {cfg_path}")
    print(f"  Base model        : {args.model}")
    print(f"  Epoch budget      : warm-up {args.warmup_epochs} + ACS {final_epochs} = {args.epochs}")
    print(f"  imgsz/batch       : {args.imgsz}/{args.batch}")
    print("=" * 70)

    warmup_weights: Path | None = None
    warmup_name = f"{run_name}_warmup"
    if args.warmup_epochs > 0:
        warmup_weights = best_weights_for(args.project, warmup_name)
        if warmup_weights.exists() and not args.force_retrain:
            LOGGER.info(f"[SKIP] Warm-up weights already exist: {warmup_weights}")
        else:
            warmup_model = YOLO(args.model)
            warmup_model.train(
                **yolo_train_kwargs(
                    args,
                    cfg_path=cfg_path,
                    run_name=warmup_name,
                    epochs=args.warmup_epochs,
                    aug=args.warmup_aug,
                )
            )
            warmup_weights = best_weights_for(args.project, warmup_name)
            if not warmup_weights.exists():
                raise RuntimeError(f"Warm-up weights not found: {warmup_weights}")

    scoring_weights = Path(args.score_weights).resolve() if args.score_weights else warmup_weights
    if scoring_weights is None:
        scoring_weights = Path(args.model).resolve() if Path(args.model).exists() else Path(args.model)

    score_json = (
        Path(args.score_json)
        if args.score_json
        else Path(args.project) / "acs_scores" / f"{run_name}_acs_scores.json"
    )
    if score_json.exists() and not args.force_rescore:
        LOGGER.info(f"[SKIP] Reusing ACS scores: {score_json}")
    else:
        compute_acs_scores(
            weights=scoring_weights,
            cfg_path=cfg_path,
            out_json=score_json,
            imgsz=args.imgsz,
            batch=args.score_batch,
            device=args.device,
            conf=args.score_conf,
            iou=args.score_iou,
            score_floor=args.score_floor,
            aug_repeats=args.aug_repeats,
            seed=args.seed,
        )

    scores, score_mean = load_score_table(score_json)
    ACS_CONTEXT.clear()
    ACS_CONTEXT.update(
        scores=scores,
        score_mean=score_mean,
        default_score=args.default_score,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        normalize_scores=not args.no_normalize_scores,
    )

    start_model = str(warmup_weights) if args.train_from_warmup and warmup_weights and warmup_weights.exists() else args.model
    final_model = YOLO(start_model)
    final_model.train(
        trainer=ACSDetectionTrainer,
        **yolo_train_kwargs(
            args,
            cfg_path=cfg_path,
            run_name=run_name,
            epochs=final_epochs,
            aug=args.final_aug,
        ),
    )

    final_weights = best_weights_for(args.project, run_name)
    if not final_weights.exists():
        raise RuntimeError(f"ACS-YOLO best weights not found: {final_weights}")
    print(f"\nDone. ACS-YOLO best weights -> {final_weights}")
    print(f"ACS scores -> {score_json}")
    return final_weights


def main():
    parser = argparse.ArgumentParser(description="Train ACS-YOLO for robust satellite detection")
    parser.add_argument("--cfg", required=True, help="YOLO dataset YAML")
    parser.add_argument("--run_name", default="acs_yolo")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80, help="Total epoch budget: warm-up + ACS final stage")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--timestamp", action="store_true")
    parser.add_argument("--force_retrain", action="store_true")

    parser.add_argument("--warmup_aug", choices=["baseline", "dg_aug"], default="dg_aug")
    parser.add_argument("--final_aug", choices=["baseline", "dg_aug"], default="dg_aug")
    parser.add_argument("--train_from_warmup", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--score_json", default=None, help="Optional existing/output ACS score JSON")
    parser.add_argument("--score_weights", default=None, help="Optional detector weights used only for scoring")
    parser.add_argument("--force_rescore", action="store_true")
    parser.add_argument("--score_batch", type=int, default=8)
    parser.add_argument("--score_conf", type=float, default=0.05)
    parser.add_argument("--score_iou", type=float, default=0.60)
    parser.add_argument("--score_floor", type=float, default=0.25)
    parser.add_argument("--aug_repeats", type=int, default=1)

    parser.add_argument("--default_score", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.25)
    parser.add_argument("--max_weight", type=float, default=1.50)
    parser.add_argument("--no_normalize_scores", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
