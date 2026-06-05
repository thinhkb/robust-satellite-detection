"""
check_data.py
=============
Diagnostic script — run this BEFORE training to verify everything is in order.
Checks data directories, YAML configs, and reports what needs to be done.

Usage:
    python scripts/check_data.py
    python scripts/check_data.py --data_dir data --cfg_dir configs
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ZONES  = ["CZ_A", "CZ_B", "CZ_C"]
SPLITS = ["train", "val", "test"]


def check_split_manifest(zone_dir: Path) -> bool:
    """Verify that every original source image belongs to exactly one split."""
    manifest = zone_dir / "split_manifest.csv"
    if not manifest.exists():
        print("    X  split_manifest.csv missing (legacy tile-level split)")
        return False

    source_splits = defaultdict(set)
    tile_rows = 0
    with open(manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_splits[row["source_image"]].add(row["split"])
            tile_rows += 1

    leaked = {
        source: splits
        for source, splits in source_splits.items()
        if len(splits) != 1
    }
    split_sources = {
        split: sum(split in splits for splits in source_splits.values())
        for split in SPLITS
    }
    if leaked:
        print(f"    X  spatial leakage in {len(leaked)} source images")
        return False
    if any(count == 0 for count in split_sources.values()):
        print(f"    X  empty source-image split: {split_sources}")
        return False

    print(
        "    OK split manifest: "
        f"{len(source_splits)} source images, {tile_rows} tiles, no leakage"
    )
    return True


def check_path(p: Path, label: str) -> bool:
    exists = p.exists()
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {label}")
    print(f"          {p}")
    return exists


def count_files(directory: Path, exts=(".jpg", ".png", ".txt")) -> dict:
    counts = {}
    for ext in exts:
        counts[ext] = len(list(directory.glob(f"*{ext}"))) if directory.exists() else 0
    return counts


def main(args):
    data_dir = Path(args.data_dir).resolve()
    cfg_dir  = Path(args.cfg_dir).resolve()
    all_ok   = True

    print("=" * 65)
    print("  RWDS-CZ Data Diagnostic")
    print("=" * 65)

    # ── 1. Check data root ────────────────────────────────────────────────
    print(f"\n[1] Data root directory")
    if not check_path(data_dir, "data_dir"):
        print("\n  ⚠ data_dir does not exist!")
        print("  → Run scripts/convert_to_yolo.py first.")
        raise SystemExit(1)

    # ── 2. Check per-zone directories ─────────────────────────────────────
    print(f"\n[2] Zone directories")
    zone_status = {}
    for zone in ZONES:
        zone_dir = data_dir / zone
        print(f"\n  {zone}:")
        has_all = check_path(zone_dir / "images" / "all", "images/all")
        has_lbl = check_path(zone_dir / "labels" / "all", "labels/all")

        if has_all and has_lbl:
            img_cnt = count_files(zone_dir / "images" / "all", (".jpg", ".png"))
            lbl_cnt = count_files(zone_dir / "labels" / "all", (".txt",))
            total_img = sum(img_cnt.values())
            total_lbl = lbl_cnt[".txt"]
            print(f"          images: {total_img}   labels: {total_lbl}")
            if total_img == 0:
                print(f"    ⚠  images/all is EMPTY → convert_to_yolo.py may have failed")
                all_ok = False
            if total_lbl == 0:
                print(f"    ⚠  labels/all is EMPTY → convert_to_yolo.py may have failed")
                all_ok = False
        else:
            all_ok = False

        # Check splits
        split_ok = True
        for split in SPLITS:
            img_dir = zone_dir / "images" / split
            lbl_dir = zone_dir / "labels" / split
            img_exists = img_dir.exists()
            lbl_exists = lbl_dir.exists()
            if img_exists:
                n_img = count_files(img_dir, (".jpg", ".png"))
                n = sum(n_img.values())
            else:
                n = 0
            status = "✓" if (img_exists and n > 0) else "✗"
            print(f"    {status}  images/{split:<6}  {n} images", end="")
            if not img_exists or n == 0:
                split_ok = False
                print("  ← MISSING", end="")
            print()

        if split_ok:
            split_ok = check_split_manifest(zone_dir)
        zone_status[zone] = split_ok
        if not split_ok:
            all_ok = False
            print(
                "    -> Rebuild grouped splits: "
                f"python scripts/split_domain.py --data_dir {args.data_dir} --force"
            )

    # ── 3. Check YAML configs ─────────────────────────────────────────────
    print(f"\n[3] YAML config files  ({cfg_dir})")
    expected_yamls = (
        [f"single_source_{z.lower()}.yaml" for z in ZONES] +
        [f"multi_source_test_{z.lower()}.yaml" for z in ZONES]
    )
    yamls_ok = True
    for fname in expected_yamls:
        yaml_path = cfg_dir / fname
        exists = yaml_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status}  {fname}")
        if not exists:
            yamls_ok = False
            all_ok = False

    if not yamls_ok:
        print(f"\n  → Run: python scripts/split_domain.py "
              f"--data_dir {args.data_dir} --cfg_dir {args.cfg_dir}")

    # ── 4. Validate YAML paths ─────────────────────────────────────────────
    print(f"\n[4] Validating paths inside YAML files")
    yaml_path_ok = True
    for fname in expected_yamls:
        yaml_path = cfg_dir / fname
        if not yaml_path.exists():
            continue
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        broken = []
        for key in ("train", "val", "test"):
            entries = data.get(key, [])
            if isinstance(entries, str):
                entries = [entries]
            for entry in entries:
                if not Path(entry).exists():
                    broken.append(f"{key}: {entry}")
        if broken:
            print(f"  ✗  {fname}  — broken paths:")
            for b in broken:
                print(f"       {b}")
            yaml_path_ok = False
            all_ok = False
        else:
            print(f"  ✓  {fname}")

    if not yaml_path_ok:
        print(f"\n  → Re-run: python scripts/split_domain.py "
              f"--data_dir {args.data_dir} --cfg_dir {args.cfg_dir} --force")

    # ── 5. Summary & recommended next steps ───────────────────────────────
    print(f"\n{'='*65}")
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — ready to train!")
        print()
        print("  Quick start (weak GPU):")
        print("    python scripts/train_baseline.py \\")
        print(f"        --cfg {args.cfg_dir}/single_source_cz_a.yaml \\")
        print("        --run_name baseline_CZA --model yolov8n.pt \\")
        print("        --imgsz 512 --epochs 50 --batch 8")
        print()
        print("  Run all experiments:")
        print("    python scripts/run_experiments.py \\")
        print("        --model yolov8n.pt --epochs 50 --imgsz 512 --batch 8")
    else:
        print("  ✗ ISSUES FOUND — follow the → instructions above")
        print()
        if not all(z in zone_status for z in ZONES):
            print("  Step 1: python scripts/convert_to_yolo.py \\")
            print(f"              --xview_img_dir /path/to/xview/train_images \\")
            print(f"              --geojson /path/to/xView_train.geojson \\")
            print(f"              --koppen_raster /path/to/Beck_KG_V1_present_0p0083.tif \\")
            print(f"              --out_dir {args.data_dir}")
        if not all(zone_status.get(z, False) for z in ZONES):
            print()
            print("  Step 2: python scripts/split_domain.py \\")
            print(f"              --data_dir {args.data_dir} --cfg_dir {args.cfg_dir}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check RWDS-CZ data before training")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--cfg_dir",  default="configs")
    args = parser.parse_args()
    main(args)
