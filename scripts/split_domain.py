"""
split_domain.py
===============
Splits each climate zone's tiles into train/val/test and writes YOLO YAML configs.

After convert_to_yolo.py has written tiles into:
    data/CZ_A/images/all/   data/CZ_A/labels/all/
    data/CZ_B/images/all/   data/CZ_B/labels/all/
    data/CZ_C/images/all/   data/CZ_C/labels/all/

This script:
  1. Splits images 60% train / 20% val / 20% test (random)
  2. Copies image+label pairs into correct subfolders
  3. Writes absolute-path YAML configs for all experiment setups

Usage:
    python scripts/split_domain.py --data_dir data --cfg_dir configs
    python scripts/split_domain.py --data_dir data --cfg_dir configs --force
"""

import argparse
import csv
import re
import shutil
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ZONES       = ["CZ_A", "CZ_B", "CZ_C"]
SPLITS      = ["train", "val", "test"]
SPLIT_RATIO = (0.60, 0.20, 0.20)

CLASS_NAMES = [
    "Building", "Small Car", "Truck", "Bus",
    "Cargo Truck", "Shipping Container", "Vehicle Lot", "Shed",
]


# ──────────────────────────── helpers ────────────────────────────────────────

def ensure_dirs(zone_dir: Path):
    for split in SPLITS:
        (zone_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (zone_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_image(img_all_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_all_dir / (stem + ext)
        if p.exists():
            return p
    return None


def read_label_classes(lbl_path: Path) -> set:
    classes = set()
    try:
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    except Exception:
        pass
    return classes


def count_instances(lbl_files: list) -> dict:
    counts = defaultdict(int)
    for p in lbl_files:
        for cls in read_label_classes(Path(p)):
            counts[cls] += 1
    return dict(counts)


TILE_STEM_RE = re.compile(r"^(?P<source>.+)_(?P<x>-?\d+)_(?P<y>-?\d+)$")


def source_image_id(path: str | Path) -> str:
    """Return the original image id from a <source>_<tile_x>_<tile_y> filename."""
    stem = Path(path).stem
    match = TILE_STEM_RE.match(stem)
    if not match:
        raise ValueError(
            f"Cannot derive source image from tile '{Path(path).name}'. "
            "Expected <source>_<tile_x>_<tile_y>."
        )
    return match.group("source")


def grouped_split(files: list, seed: int = 42):
    """Split source-image groups while balancing tile and class counts."""
    groups = defaultdict(list)
    for path in files:
        groups[source_image_id(path)].append(path)

    if len(groups) < 3:
        raise ValueError(
            f"Need at least 3 source images for train/val/test, found {len(groups)}."
        )

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total_tiles = len(files)
    targets = {
        split: total_tiles * ratio
        for split, ratio in zip(SPLITS, SPLIT_RATIO)
    }
    group_classes = {
        source: sum((_label_histogram(path) for path in paths), Counter())
        for source, paths in group_items
    }
    all_classes = sum(group_classes.values(), Counter())
    class_targets = {
        split: {
            class_id: count * ratio
            for class_id, count in all_classes.items()
        }
        for split, ratio in zip(SPLITS, SPLIT_RATIO)
    }
    assigned = {split: [] for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    class_counts = {split: Counter() for split in SPLITS}

    for split, item in zip(SPLITS, group_items[:3]):
        assigned[split].append(item)
        counts[split] += len(item[1])
        class_counts[split].update(group_classes[item[0]])

    for item in group_items[3:]:
        source, paths = item

        def assignment_cost(candidate: str) -> float:
            cost = 0.0
            for split_name in SPLITS:
                tile_count = counts[split_name]
                histogram = class_counts[split_name].copy()
                if split_name == candidate:
                    tile_count += len(paths)
                    histogram.update(group_classes[source])
                cost += ((tile_count - targets[split_name]) / max(targets[split_name], 1)) ** 2
                for class_id, target in class_targets[split_name].items():
                    cost += 0.35 * (
                        (histogram[class_id] - target) / max(target, 1)
                    ) ** 2
            return cost

        split = min(SPLITS, key=assignment_cost)
        assigned[split].append(item)
        counts[split] += len(item[1])
        class_counts[split].update(group_classes[source])

    return tuple(
        [path for _source, paths in assigned[split] for path in paths]
        for split in SPLITS
    )


def _label_histogram(path: str | Path) -> Counter:
    """Count object instances in one YOLO label file."""
    histogram = Counter()
    label_path = Path(path)
    if not label_path.exists():
        return histogram
    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                histogram[int(fields[0])] += 1
    return histogram


def write_split_manifest(zone_dir: Path, split_files: dict[str, list[Path]]) -> Path:
    """Write tile-to-source assignments and reject source-image leakage."""
    source_splits = defaultdict(set)
    rows = []
    for split, files in split_files.items():
        for path in files:
            source = source_image_id(path)
            source_splits[source].add(split)
            rows.append((split, source, Path(path).name))

    leaked = {
        source: sorted(splits)
        for source, splits in source_splits.items()
        if len(splits) > 1
    }
    if leaked:
        preview = ", ".join(
            f"{source}:{'/'.join(splits)}"
            for source, splits in list(leaked.items())[:5]
        )
        raise RuntimeError(f"Source-image leakage detected: {preview}")

    manifest = zone_dir / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "source_image", "tile"])
        writer.writerows(sorted(rows))

    stats = {
        split: {
            "source_images": len({source_image_id(path) for path in files}),
            "tiles": len(files),
            "class_instances": dict(
                sorted(
                    sum((_label_histogram(path) for path in files), Counter()).items()
                )
            ),
        }
        for split, files in split_files.items()
    }
    with open(zone_dir / "split_stats.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(stats, handle, sort_keys=False)
    return manifest


def copy_files(zone_dir: Path, lbl_files: list, split: str) -> int:
    img_all = zone_dir / "images" / "all"
    img_out = zone_dir / "images" / split
    lbl_out = zone_dir / "labels" / split
    copied  = 0
    for lbl in lbl_files:
        lbl = Path(lbl)
        img = find_image(img_all, lbl.stem)
        if img is None:
            print(f"    [WARN] No image for {lbl.name} – skipped")
            continue
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)
        copied += 1
    return copied


def to_posix(p) -> str:
    """Absolute path with forward slashes (works on Windows + Linux)."""
    return str(Path(p).resolve()).replace("\\", "/")


def write_yaml(yaml_path: Path, train_dirs, val_dirs, test_dirs, class_names):
    data = {
        "train": [to_posix(d) for d in train_dirs],
        "val":   [to_posix(d) for d in val_dirs],
        "test":  [to_posix(d) for d in test_dirs],
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
    print(f"    Wrote → {yaml_path.name}")


# ─────────────────────────────── main ────────────────────────────────────────

def main(args):
    data_dir = Path(args.data_dir).resolve()
    cfg_dir  = Path(args.cfg_dir).resolve()

    print("=" * 60)
    print("  RWDS-CZ Domain Split")
    print(f"  data_dir : {data_dir}")
    print(f"  cfg_dir  : {cfg_dir}")
    print("=" * 60)

    if not data_dir.exists():
        print(f"\n[ERROR] data_dir not found: {data_dir}")
        print("  → Run scripts/convert_to_yolo.py first.")
        raise SystemExit(1)

    # Load / create class names
    cfg_dir.mkdir(parents=True, exist_ok=True)
    classes_yaml = cfg_dir / "classes.yaml"
    if classes_yaml.exists():
        with open(classes_yaml, encoding="utf-8") as f:
            class_names = yaml.safe_load(f).get("names", CLASS_NAMES)
    else:
        class_names = CLASS_NAMES
        with open(classes_yaml, "w", encoding="utf-8") as f:
            yaml.dump({"nc": len(class_names), "names": class_names}, f)

    zone_img_dirs = {}   # zone → {"train": str, "val": str, "test": str}

    for zone in ZONES:
        zone_dir = data_dir / zone
        all_lbl  = zone_dir / "labels" / "all"
        all_img  = zone_dir / "images" / "all"

        print(f"\n[{zone}]  {zone_dir}")

        # ── Check source exists ───────────────────────────────────────
        if not all_lbl.exists():
            print(f"  [SKIP] labels/all not found – run convert_to_yolo.py first")
            continue
        lbl_files = sorted(all_lbl.glob("*.txt"))
        if not lbl_files:
            print(f"  [SKIP] No .txt files found in {all_lbl}")
            continue

        print(f"  Tiles available : {len(lbl_files)}")

        # ── Class statistics ──────────────────────────────────────────
        cls_counts   = count_instances(lbl_files)
        valid_classes = {
            cls for cls, cnt in cls_counts.items()
            if cnt >= args.min_samples
        }
        if not valid_classes:
            print(f"  [WARN] No class ≥ {args.min_samples} samples – keeping all")
            valid_classes = set(cls_counts.keys())

        print(f"  Classes (threshold {args.min_samples}):")
        for cls_idx in sorted(cls_counts):
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"cls{cls_idx}"
            cnt  = cls_counts[cls_idx]
            ok   = "✓" if cls_idx in valid_classes else "✗"
            print(f"    {ok} [{cls_idx}] {name:<22} {cnt:>6} instances")

        # Filter tiles
        filtered = [
            lbl for lbl in lbl_files
            if read_label_classes(lbl) & valid_classes
        ]
        print(f"  Tiles after filter : {len(filtered)}")

        if not filtered:
            print(f"  [SKIP] 0 tiles after filtering")
            continue

        # ── Create subdirectories ─────────────────────────────────────
        ensure_dirs(zone_dir)

        # ── Check if already split ────────────────────────────────────
        n_existing = sum(
            len(list((zone_dir / "images" / s).glob("*.jpg")) +
                list((zone_dir / "images" / s).glob("*.png")))
            for s in SPLITS
        )
        if n_existing > 0 and not args.force:
            print(f"  Already split ({n_existing} total images). Use --force to redo.")
        else:
            if args.force:
                for split in SPLITS:
                    for f in (zone_dir / "images" / split).glob("*"):
                        f.unlink()
                    for f in (zone_dir / "labels" / split).glob("*"):
                        f.unlink()

            train_f, val_f, test_f = grouped_split(filtered, seed=args.seed)
            split_files = {
                "train": train_f,
                "val": val_f,
                "test": test_f,
            }
            manifest = write_split_manifest(zone_dir, split_files)
            print(f"    manifest: {manifest}")
            for split, files in split_files.items():
                n = copy_files(zone_dir, files, split)
                print(f"    {split:<8}: {n:>5} tiles copied")

        # Final counts
        print(f"  Final split sizes:")
        for split in SPLITS:
            imgs = list((zone_dir / "images" / split).glob("*.jpg")) + \
                   list((zone_dir / "images" / split).glob("*.png"))
            print(f"    {split:<8}: {len(imgs):>5} images")

        zone_img_dirs[zone] = {
            s: str(zone_dir / "images" / s)
            for s in SPLITS
        }

    if not zone_img_dirs:
        print("\n[ERROR] No zones processed. Check your data directory.")
        raise SystemExit(1)

    # ── Write YAML configs ────────────────────────────────────────────────
    print(f"\n[Writing YAML configs → {cfg_dir}]")

    # Single-source (one zone as both train and test)
    for zone in ZONES:
        if zone not in zone_img_dirs:
            continue
        write_yaml(
            cfg_dir / f"single_source_{zone.lower()}.yaml",
            train_dirs  = [zone_img_dirs[zone]["train"]],
            val_dirs    = [zone_img_dirs[zone]["val"]],
            test_dirs   = [zone_img_dirs[zone]["test"]],
            class_names = class_names,
        )

    # Multi-source (two zones train, one zone test)
    for ood_zone in ZONES:
        src_zones = [z for z in ZONES
                     if z != ood_zone and z in zone_img_dirs]
        if len(src_zones) < 1 or ood_zone not in zone_img_dirs:
            continue
        write_yaml(
            cfg_dir / f"multi_source_test_{ood_zone.lower()}.yaml",
            train_dirs  = [zone_img_dirs[z]["train"] for z in src_zones],
            val_dirs    = [zone_img_dirs[z]["val"]   for z in src_zones],
            test_dirs   = [zone_img_dirs[ood_zone]["test"]],
            class_names = class_names,
        )

    print(f"\n{'='*60}")
    print("  ✓ Split complete. YAML configs written:")
    for f in sorted(cfg_dir.glob("*.yaml")):
        print(f"    {f.name}")
    print()
    print("  Next steps:")
    print("    python scripts/train_baseline.py \\")
    print("        --cfg configs/single_source_cz_a.yaml \\")
    print("        --run_name baseline_CZA")
    print()
    print("  Or run all experiments at once:")
    print("    python scripts/run_experiments.py --model yolov8n.pt --epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split RWDS-CZ tiles into train/val/test and write YAML configs")
    parser.add_argument("--data_dir",    default="data",
                        help="Root data dir containing CZ_A, CZ_B, CZ_C")
    parser.add_argument("--cfg_dir",     default="configs",
                        help="Directory to write YAML configs into")
    parser.add_argument("--min_samples", type=int, default=30,
                        help="Min instances per class; classes below are excluded")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--force",       action="store_true",
                        help="Delete and redo splits even if they exist")
    args = parser.parse_args()
    main(args)
