"""
evaluate_domains.py
===================
Cross-domain evaluation for RWDS-CZ experiments.

Given a trained YOLO weights file, evaluates it on every climate zone's
test set and computes:
  - mAP50, mAP50-95, Precision, Recall  (per zone)
  - Performance Drop (PD)  =  100 × (mAP_ID − mAP_OOD) / mAP_ID
  - Harmonic Mean (H)      =  2 × mAP_ID × mAP_OOD / (mAP_ID + mAP_OOD)

Usage:
    # Evaluate a single run across all 3 zones
    python evaluate_domains.py \
        --weights results/runs/baseline_CZA/weights/best.pt \
        --data_dir data \
        --id_zone CZ_A \
        --run_name baseline_CZA

    # Evaluate a multi-source run (source = CZ_A + CZ_B, OOD = CZ_C)
    python evaluate_domains.py \
        --weights results/runs/multisrc_test_CZC/weights/best.pt \
        --data_dir data \
        --id_zones CZ_A CZ_B \
        --ood_zones CZ_C \
        --run_name multisrc_test_CZC
"""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO

ZONES = ["CZ_A", "CZ_B", "CZ_C"]


def dataset_manifest_hashes(data_dir: Path) -> dict[str, str | None]:
    """Hash split manifests so report rows cannot silently mix datasets."""
    hashes = {}
    for zone in ZONES:
        manifest = data_dir / zone / "split_manifest.csv"
        hashes[zone] = (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.exists()
            else None
        )
    return hashes


# ─────────────────────── helpers ────────────────────────────────────────────

def evaluate_on_zone(model: YOLO,
                     data_dir: Path,
                     zone: str,
                     imgsz: int = 640,
                     batch: int = 16,
                     device: str = "",
                     conf: float = 0.001,
                     iou: float = 0.6) -> dict:
    """
    Run model.val() on <zone>/images/test and return metric dict.
    """
    test_img_dir = data_dir / zone / "images" / "test"
    if not test_img_dir.exists():
        print(f"  [SKIP] {test_img_dir} not found")
        return {}

    # Build a minimal YAML for this zone's test set
    # Use cross-platform temp directory (works on Windows, Linux, macOS)
    tmp_yaml = Path(tempfile.gettempdir()) / f"eval_{zone}.yaml"
    zone_data = {
        "train": str(data_dir / zone / "images" / "train"),
        "val":   str(data_dir / zone / "images" / "test"),  # use test as val
        "test":  str(data_dir / zone / "images" / "test"),
        "nc":    8,
        "names": [
            "Building", "Small Car", "Truck", "Bus",
            "Cargo Truck", "Shipping Container", "Vehicle Lot", "Shed",
        ],
    }
    # Try to load nc/names from classes.yaml
    classes_yaml = data_dir / ".." / "configs" / "classes.yaml"
    if classes_yaml.exists():
        with open(classes_yaml) as f:
            cls_data = yaml.safe_load(f)
            zone_data["nc"]    = cls_data["nc"]
            zone_data["names"] = cls_data["names"]

    with open(tmp_yaml, "w") as f:
        yaml.dump(zone_data, f)

    try:
        metrics = model.val(
            data   = str(tmp_yaml),
            imgsz  = imgsz,
            batch  = batch,
            device = device,
            conf   = conf,
            iou    = iou,
            verbose= False,
            split  = "val",   # evaluates on the 'val' key above (= test set)
        )
        names = metrics.names
        class_names = list(names.values()) if isinstance(names, dict) else list(names)
        return {
            "mAP50":     float(metrics.box.map50),
            "mAP50_95":  float(metrics.box.map),
            "Precision": float(metrics.box.mp),
            "Recall":    float(metrics.box.mr),
            "per_class_mAP50_95": {
                str(name): float(value)
                for name, value in zip(
                    class_names,
                    metrics.box.maps,
                )
            },
            "zone":      zone,
        }
    except Exception as e:
        print(f"  [ERROR] Evaluation on {zone} failed: {e}")
        return {}


def compute_pd(map_id: float, map_ood: float) -> float:
    """Performance Drop (%)."""
    if map_id == 0:
        return 0.0
    return 100.0 * (map_id - map_ood) / map_id


def compute_harmonic(map_id: float, map_ood: float) -> float:
    """Harmonic Mean of ID and OOD mAP."""
    if map_id + map_ood == 0:
        return 0.0
    return 2.0 * map_id * map_ood / (map_id + map_ood)


def avg_maps(results: list[dict], metric: str = "mAP50_95") -> float:
    vals = [r[metric] for r in results if metric in r]
    return float(np.mean(vals)) if vals else 0.0


# ────────────────────────────── main ────────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir).resolve()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Weights  : {args.weights}")
    print(f"  Run name : {args.run_name}")
    print(f"  ID zones : {args.id_zones}")
    print(f"  OOD zones: {args.ood_zones}")
    print("=" * 60)

    # Determine ID/OOD zones
    if args.id_zones:
        id_zones  = args.id_zones
        ood_zones = args.ood_zones if args.ood_zones else \
                    [z for z in ZONES if z not in id_zones]
    else:
        # Default: all zones evaluated; first is ID (single-source mode)
        id_zones  = [args.id_zone] if args.id_zone else [ZONES[0]]
        ood_zones = [z for z in ZONES if z not in id_zones]

    model = YOLO(args.weights)

    # Evaluate on every zone
    zone_results = {}
    for zone in ZONES:
        print(f"\n→ Evaluating on {zone} ...")
        res = evaluate_on_zone(
            model, data_dir, zone,
            imgsz  = args.imgsz,
            batch  = args.batch,
            device = args.device,
        )
        if res:
            zone_results[zone] = res
            print(f"  mAP50={res['mAP50']:.4f}  mAP50:95={res['mAP50_95']:.4f}"
                  f"  P={res['Precision']:.4f}  R={res['Recall']:.4f}")

    if not zone_results:
        print("No results collected. Exiting.")
        return

    # Compute aggregate ID / OOD
    id_results  = [zone_results[z] for z in id_zones  if z in zone_results]
    ood_results = [zone_results[z] for z in ood_zones if z in zone_results]

    map_id_50    = avg_maps(id_results,  "mAP50")
    map_ood_50   = avg_maps(ood_results, "mAP50")
    map_id_5095  = avg_maps(id_results,  "mAP50_95")
    map_ood_5095 = avg_maps(ood_results, "mAP50_95")

    pd_50    = compute_pd(map_id_50,   map_ood_50)
    pd_5095  = compute_pd(map_id_5095, map_ood_5095)
    h_50     = compute_harmonic(map_id_50,   map_ood_50)
    h_5095   = compute_harmonic(map_id_5095, map_ood_5095)

    # ─── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Summary — {args.run_name}")
    print("=" * 60)
    header = f"{'Zone':<10} {'mAP50':>8} {'mAP50:95':>10} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * 50)
    for zone in ZONES:
        if zone not in zone_results:
            print(f"{zone:<10} {'N/A':>8}")
            continue
        r = zone_results[zone]
        tag = "(ID)" if zone in id_zones else "(OOD)"
        print(f"{zone+tag:<14} {r['mAP50']:>8.4f} {r['mAP50_95']:>10.4f}"
              f" {r['Precision']:>10.4f} {r['Recall']:>8.4f}")
    print("-" * 50)
    print(f"{'ID avg':<14} {map_id_50:>8.4f} {map_id_5095:>10.4f}")
    print(f"{'OOD avg':<14} {map_ood_50:>8.4f} {map_ood_5095:>10.4f}")
    print(f"{'PD (%)':<14} {pd_50:>8.2f} {pd_5095:>10.2f}")
    print(f"{'H-score':<14} {h_50:>8.4f} {h_5095:>10.4f}")
    print("=" * 60)

    # ─── Save JSON result ──────────────────────────────────────────────────
    summary = {
        "run_name"    : args.run_name,
        "base_run"    : args.base_run or args.run_name,
        "method"      : args.method,
        "seed"        : args.seed,
        "dataset_manifests": dataset_manifest_hashes(data_dir),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "weights"     : args.weights,
        "id_zones"    : id_zones,
        "ood_zones"   : ood_zones,
        "per_zone"    : zone_results,
        "aggregate"   : {
            "mAP50_ID"   : map_id_50,
            "mAP50_OOD"  : map_ood_50,
            "mAP5095_ID" : map_id_5095,
            "mAP5095_OOD": map_ood_5095,
            "PD_50"      : pd_50,
            "PD_5095"    : pd_5095,
            "H_50"       : h_50,
            "H_5095"     : h_5095,
        }
    }
    json_path = out_dir / f"{args.run_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved → {json_path}")

    # ─── Append to master CSV (for easy comparison) ────────────────────────
    csv_path = out_dir / "all_results.csv"
    row = {
        "run_name"   : args.run_name,
        "base_run"   : args.base_run or args.run_name,
        "method"     : args.method,
        "seed"       : args.seed,
        "id_zones"   : "+".join(id_zones),
        "ood_zones"  : "+".join(ood_zones),
        "mAP50_ID"   : round(map_id_50,    4),
        "mAP50_OOD"  : round(map_ood_50,   4),
        "mAP5095_ID" : round(map_id_5095,  4),
        "mAP5095_OOD": round(map_ood_5095, 4),
        "PD_50"      : round(pd_50,        2),
        "PD_5095"    : round(pd_5095,      2),
        "H_50"       : round(h_50,         4),
        "H_5095"     : round(h_5095,       4),
    }
    # per-zone mAP50 columns
    for zone in ZONES:
        key = f"mAP50_{zone}"
        row[key] = round(zone_results.get(zone, {}).get("mAP50", 0), 4)

    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["run_name"],
            keep="last",
        )
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"✓ Appended to master CSV → {csv_path}")

    return summary


def main_cli():
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation for RWDS-CZ")
    parser.add_argument("--weights",   required=True,
                        help="Path to best.pt weights")
    parser.add_argument("--data_dir",  default="data",
                        help="Root data dir with CZ_A / CZ_B / CZ_C")
    parser.add_argument("--run_name",  default="experiment")
    parser.add_argument("--base_run", default=None)
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_zone",   default=None,
                        help="Single ID zone (single-source mode)")
    parser.add_argument("--id_zones",  nargs="+", default=None,
                        help="ID zones for multi-source mode")
    parser.add_argument("--ood_zones", nargs="+", default=None,
                        help="OOD zones to evaluate on (default: all non-ID)")
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--batch",     type=int, default=16)
    parser.add_argument("--device",    default="")
    parser.add_argument("--out_dir",   default="results/tables")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()
