"""
convert_to_yolo.py
==================
Parse xView dataset → map images to Köppen climate zones → crop 512×512 tiles
→ write YOLO-format labels.

xView format (what xviewdataset.org actually provides):
    train_images/            ← folder of GeoTIFF (.tif) files
    xView_train.geojson      ← single annotation file

GeoJSON feature structure:
    {
      "geometry": { "type": "Point", "coordinates": [lon, lat] },
      "properties": {
        "image_id":        "1.tif",
        "bounds_imcoords": "x1,y1,x2,y2",   ← pixel bbox (comma-separated)
        "type_id":         17,               ← xView class ID
        "feature_id":      "...",
        "det_source":      "..."
      }
    }

Köppen raster (Beck et al. 2018, 1 km resolution):
    Beck_KG_V1_present_0p0083.tif
    Download: https://figshare.com/articles/dataset/6396959

Usage:
    python scripts/convert_to_yolo.py \
        --xview_img_dir  data_raw/train_images \
        --geojson        data_raw/xView_train.geojson \
        --koppen_raster  data_raw/Beck_KG_V1_present_0p0083.tif \
        --out_dir        data

    # Inspect only (no files written) — run this first!
    python scripts/convert_to_yolo.py \
        --xview_img_dir  data_raw/train_images \
        --geojson        data_raw/xView_train.geojson \
        --koppen_raster  data_raw/Beck_KG_V1_present_0p0083.tif \
        --out_dir        data \
        --inspect
"""

import argparse
import sys
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_cli_path(value: str) -> Path:
    """Resolve CLI paths from cwd first, then from the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (PROJECT_ROOT / path).resolve()

# ──────────────────── check critical dependencies ───────────────────────────

def check_dependencies():
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    if missing:
        print("[ERROR] Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        if "rasterio" in missing:
            print()
            print("  Note for Windows — if 'pip install rasterio' fails, try:")
            print("  conda install -c conda-forge rasterio")
            print("  OR download wheel from: https://github.com/cgohlke/geospatial-wheels/releases")
        sys.exit(1)

check_dependencies()

import numpy as np
import rasterio
from rasterio.transform import rowcol
from PIL import Image


# ──────────────────── xView class map ───────────────────────────────────────
# xView uses integer type_id. We map to our 8 target class names.

# First: map each xView type_id to a readable name
XVIEW_ID_TO_NAME = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck w/Box",
    26: "Truck Tractor",
    27: "Trailer",
    28: "Truck w/Flatbed",
    29: "Truck w/Liquid",
    32: "Crane Truck",
    33: "Railway Vehicle",
    34: "Passenger Car",
    35: "Cargo Car",
    36: "Flat Car",
    37: "Tank Car",
    38: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower Crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front Loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping Container Lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}

# Second: map xView names → our 8 RWDS-CZ-mini classes
# (multiple xView classes can map to one of our classes)
NAME_TO_TARGET = {
    # Building
    "Building":         "Building",
    "Shed":             "Shed",
    "Hut/Tent":         "Building",
    "Aircraft Hangar":  "Building",
    "Facility":         "Building",
    # Small Car
    "Small Car":        "Small Car",
    "Passenger Vehicle":"Small Car",
    # Truck
    "Truck":            "Truck",
    "Pickup Truck":     "Truck",
    "Utility Truck":    "Truck",
    "Truck w/Box":      "Truck",
    "Truck w/Flatbed":  "Truck",
    "Truck w/Liquid":   "Truck",
    "Truck Tractor":    "Truck",
    # Bus
    "Bus":              "Bus",
    # Cargo Truck
    "Cargo Truck":      "Cargo Truck",
    "Dump Truck":       "Cargo Truck",
    "Haul Truck":       "Cargo Truck",
    # Shipping Container
    "Shipping Container":     "Shipping Container",
    "Shipping Container Lot": "Shipping Container",
    # Vehicle Lot
    "Vehicle Lot":      "Vehicle Lot",
    # Shed
    "Shed":             "Shed",
    "Storage Tank":     "Shed",
}

TARGET_CLASSES = [
    "Building",
    "Small Car",
    "Truck",
    "Bus",
    "Cargo Truck",
    "Shipping Container",
    "Vehicle Lot",
    "Shed",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TARGET_CLASSES)}

def xview_id_to_target_idx(type_id: int):
    """Return (class_idx, class_name) or (None, None) if not in our target set."""
    xview_name = XVIEW_ID_TO_NAME.get(type_id)
    if xview_name is None:
        return None, None
    target_name = NAME_TO_TARGET.get(xview_name)
    if target_name is None:
        return None, None
    return CLASS_TO_IDX[target_name], target_name


# ──────────────────── Köppen zone mapping ───────────────────────────────────
# Beck et al. 2018 integer codes → macro-zone (A / B / C only)
KOPPEN_TO_ZONE = {}
for c in range(1,  4):  KOPPEN_TO_ZONE[c] = "CZ_A"   # Tropical
for c in range(4,  10): KOPPEN_TO_ZONE[c] = "CZ_B"   # Arid/Dry
for c in range(10, 18): KOPPEN_TO_ZONE[c] = "CZ_C"   # Temperate
# Zones D (18–28) and E (29–30) are excluded


def get_zone_for_lonlat(lon: float, lat: float, koppen_src) -> str | None:
    """Look up Köppen zone for a (lon, lat) point. Returns 'CZ_A/B/C' or None."""
    try:
        row, col = rowcol(koppen_src.transform, lon, lat)
        h, w = koppen_src.height, koppen_src.width
        if not (0 <= row < h and 0 <= col < w):
            return None
        code = int(koppen_src.read(1)[row, col])
        return KOPPEN_TO_ZONE.get(code, None)
    except Exception:
        return None


# ──────────────────── GeoJSON parser ────────────────────────────────────────

def parse_geojson(geojson_path: str) -> dict:
    """
    Parse xView GeoJSON into a dict:
        { image_id: [ {class_idx, class_name, lon, lat, bbox_px} ] }

    bbox_px = (x1, y1, x2, y2) pixel coordinates from bounds_imcoords.
    """
    print(f"  Parsing {geojson_path} ...")
    with open(geojson_path, encoding="utf-8") as f:
        data = json.load(f)

    by_image   = defaultdict(list)
    total      = 0
    kept       = 0
    no_bounds  = 0
    no_class   = 0

    for i, feat in enumerate(tqdm(data.get("features", []), desc="  Features", unit="feat")):
        total += 1
        props = feat.get("properties", {})

        # ── image id ──────────────────────────────────────────────────
        image_id = str(props.get("image_id", "")).strip()
        if not image_id:
            continue

        # ── type_id → our class ───────────────────────────────────────
        type_id = props.get("type_id", -1)
        try:
            type_id = int(float(type_id))
        except (ValueError, TypeError):
            type_id = -1

        cls_idx, cls_name = xview_id_to_target_idx(type_id)
        if cls_idx is None:
            no_class += 1
            continue

        # ── pixel bbox from bounds_imcoords ───────────────────────────
        bounds_raw = props.get("bounds_imcoords", "")
        if not bounds_raw:
            no_bounds += 1
            continue
        try:
            parts = [float(v) for v in str(bounds_raw).split(",")]
            if len(parts) != 4:
                no_bounds += 1
                continue
            x1, y1, x2, y2 = parts
            if x2 <= x1 or y2 <= y1:
                no_bounds += 1
                continue
        except ValueError:
            no_bounds += 1
            continue

        # ── lon/lat from geometry (Point or Polygon = centroid) ──────────────────
        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])
        lon, lat = None, None
        
        if geom_type == "Point" and coords and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        elif geom_type == "Polygon" and coords and len(coords) > 0:
            # Extract centroid from polygon bounding box
            all_lons = []
            all_lats = []
            for ring in coords:
                for coord in ring:
                    if len(coord) >= 2:
                        all_lons.append(float(coord[0]))
                        all_lats.append(float(coord[1]))
            if all_lons and all_lats:
                lon = float(np.median(all_lons))
                lat = float(np.median(all_lats))

        by_image[image_id].append({
            "class_idx":  cls_idx,
            "class_name": cls_name,
            "lon":        lon,
            "lat":        lat,
            "bbox_px":    (x1, y1, x2, y2),
        })
        kept += 1

    print(f"  Total annotations : {total}")
    print(f"  In target classes : {kept}")
    print(f"  Skipped (no bbox) : {no_bounds}")
    print(f"  Skipped (no class): {no_class}")
    print(f"  Unique images     : {len(by_image)}")
    return by_image


# ──────────────────── image loading (rasterio) ──────────────────────────────

def load_image_rgb(img_path: str) -> np.ndarray | None:
    """
    Load a GeoTIFF and return (H, W, 3) uint8 RGB array.
    Handles 1-band (grayscale), 3-band (RGB), and 4-band (RGBIR) images.
    Normalises 16-bit → 8-bit using 2%/98% percentile stretch.
    """
    try:
        with rasterio.open(img_path) as src:
            n_bands = src.count
            if n_bands >= 3:
                arr = src.read([1, 2, 3])          # (3, H, W)
            else:
                band = src.read(1)                  # (H, W)
                arr  = np.stack([band, band, band]) # (3, H, W)

        arr = arr.astype(np.float32)
        # Percentile stretch per band
        for b in range(3):
            lo = np.percentile(arr[b], 2)
            hi = np.percentile(arr[b], 98)
            if hi > lo:
                arr[b] = np.clip((arr[b] - lo) / (hi - lo) * 255, 0, 255)
            else:
                arr[b] = 0
        arr = arr.astype(np.uint8)
        return np.moveaxis(arr, 0, -1)   # (H, W, 3)

    except Exception as e:
        print(f"    [WARN] Cannot read {img_path}: {e}")
        return None


# ──────────────────── tiling ────────────────────────────────────────────────

def tile_and_yield(img_array: np.ndarray,
                   labels:    list,
                   tile_size: int  = 512,
                   overlap:   float = 0.2):
    """
    Yield (tile_array, tile_labels, (x0, y0)) where:
      tile_array  = (tile_size, tile_size, 3) uint8
      tile_labels = list of [class_idx, cx, cy, bw, bh]  (normalised 0-1)
    """
    H, W = img_array.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))

    y_starts = list(range(0, H, stride))
    x_starts = list(range(0, W, stride))

    for y0 in y_starts:
        for x0 in x_starts:
            # Clamp to image boundary
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            # Shift origin so tile is always tile_size × tile_size
            y0c = y1 - tile_size
            x0c = x1 - tile_size
            if y0c < 0 or x0c < 0:
                # Image smaller than tile_size
                y0c, x0c = 0, 0
                y1 = min(tile_size, H)
                x1 = min(tile_size, W)

            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            patch = img_array[y0c:y1, x0c:x1]
            tile[:patch.shape[0], :patch.shape[1]] = patch

            tile_labels = []
            for cls, bx1, by1, bx2, by2 in labels:
                # Intersect bbox with tile
                cx1 = max(bx1, x0c)
                cy1 = max(by1, y0c)
                cx2 = min(bx2, x1)
                cy2 = min(by2, y1)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                # Require ≥40% of the bbox to be inside the tile
                orig_area = max((bx2 - bx1) * (by2 - by1), 1)
                clip_area = (cx2 - cx1) * (cy2 - cy1)
                if clip_area / orig_area < 0.4:
                    continue
                # Normalise to tile coords
                cx = ((cx1 + cx2) / 2 - x0c) / tile_size
                cy = ((cy1 + cy2) / 2 - y0c) / tile_size
                bw = (cx2 - cx1) / tile_size
                bh = (cy2 - cy1) / tile_size
                tile_labels.append([cls, cx, cy, bw, bh])

            if tile_labels:
                yield tile, tile_labels, (x0c, y0c)


def save_tile(tile, labels, img_path: Path, lbl_path: Path):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tile).save(str(img_path), quality=95)
    with open(lbl_path, "w") as f:
        for lbl in labels:
            cls, cx, cy, bw, bh = lbl
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ──────────────────── inspect mode ──────────────────────────────────────────

def inspect(geojson_path: str, img_dir: str, koppen_path: str):
    """Print stats about the dataset without writing anything."""
    print("\n=== INSPECT MODE (no files written) ===\n")

    img_dir_p = Path(img_dir)
    img_files = list(img_dir_p.glob("*.tif"))
    print(f"Images found in {img_dir}: {len(img_files)}")
    if img_files:
        print(f"  First 5: {[f.name for f in img_files[:5]]}")

    annotations = parse_geojson(geojson_path)

    # Class distribution
    class_counts = defaultdict(int)
    for img_id, anns in annotations.items():
        for a in anns:
            class_counts[a["class_name"]] += 1
    print("\nClass distribution across target classes:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<25} {cnt:>8}")

    # Check Köppen raster
    if Path(koppen_path).exists():
        with rasterio.open(koppen_path) as src:
            print(f"\nKöppen raster: {src.width}×{src.height}, CRS={src.crs}")
            zone_counts = {"CZ_A": 0, "CZ_B": 0, "CZ_C": 0, "Other": 0}
            sample_count = 0
            max_sample = len(annotations)  # Sample all images
            # Sample images (not annotations)
            for img_id, anns in tqdm(annotations.items(), desc="  Zone sampling", unit="img"):
                if sample_count >= max_sample:
                    break
                lons = [a["lon"] for a in anns if a["lon"] is not None]
                lats = [a["lat"] for a in anns if a["lat"] is not None]
                if not lons or not lats:
                    continue
                lon = float(np.median(lons))
                lat = float(np.median(lats))
                zone = get_zone_for_lonlat(lon, lat, src)
                if zone:
                    zone_counts[zone] += 1
                else:
                    zone_counts["Other"] += 1
                sample_count += 1
        print(f"\nZone distribution (sample of {sample_count} images):")
        for z, c in zone_counts.items():
            print(f"  {z}: {c}")
    else:
        print(f"\n[WARN] Köppen raster not found: {koppen_path}")


# ──────────────────── main conversion ───────────────────────────────────────

def main(args):
    img_dir = resolve_cli_path(args.xview_img_dir)
    geojson_path = resolve_cli_path(args.geojson)
    koppen_path = resolve_cli_path(args.koppen_raster)
    out_dir = resolve_cli_path(args.out_dir)

    # ── Validate paths ────────────────────────────────────────────────
    errors = []
    if not img_dir.exists():
        errors.append(f"xview_img_dir not found: {img_dir}")
    if not geojson_path.exists():
        errors.append(f"geojson not found: {geojson_path}")
    if not koppen_path.exists():
        errors.append(f"koppen_raster not found: {koppen_path}")
    if errors:
        print("\n[ERROR] Missing files:")
        for e in errors:
            print(f"  {e}")
        print(f"\n  Current directory : {Path.cwd()}")
        print(f"  Repository root   : {PROJECT_ROOT}")
        print("  Expected project paths:")
        print(f"    {PROJECT_ROOT / 'data_raw/xview/train_images'}")
        print(f"    {PROJECT_ROOT / 'data_raw/xview/xView_train.geojson'}")
        print(f"    {PROJECT_ROOT / 'data_raw/koppen/Beck_KG_V1_present_0p0083.tif'}")
        raise SystemExit(1)

    print("=" * 65)
    print("  xView → RWDS-CZ YOLO Conversion")
    print(f"  img_dir  : {img_dir}")
    print(f"  geojson  : {geojson_path}")
    print(f"  koppen   : {koppen_path}")
    print(f"  out_dir  : {out_dir}")
    print(f"  tile_size: {args.tile_size}   overlap: {args.overlap}")
    print(f"  classes  : {TARGET_CLASSES}")
    print("=" * 65)

    if args.inspect:
        inspect(str(geojson_path), str(img_dir), str(koppen_path))
        return

    # ── Parse annotations ─────────────────────────────────────────────
    annotations = parse_geojson(str(geojson_path))

    # ── Open Köppen raster (keep open for whole run) ──────────────────
    koppen_src = rasterio.open(str(koppen_path))
    print(f"  Köppen raster opened: {koppen_src.width}×{koppen_src.height}")

    # ── Per-image stats ───────────────────────────────────────────────
    zone_tile_counts  = defaultdict(int)
    zone_ann_counts   = defaultdict(int)
    skipped_no_image  = 0
    skipped_no_zone   = 0
    skipped_load_fail = 0

    image_ids = sorted(annotations.keys())
    print(f"\nProcessing {len(image_ids)} images ...\n")

    for image_id in tqdm(image_ids, desc="Images", unit="img"):
        anns = annotations[image_id]

        # ── Find image file ───────────────────────────────────────────
        img_path = img_dir / image_id
        if not img_path.exists():
            # Try without extension, or with .tif
            stem = Path(image_id).stem
            candidates = list(img_dir.glob(f"{stem}*"))
            if candidates:
                img_path = candidates[0]
            else:
                skipped_no_image += 1
                continue

        # ── Determine climate zone from annotation coordinates ────────
        lons = [a["lon"] for a in anns if a["lon"] is not None]
        lats = [a["lat"] for a in anns if a["lat"] is not None]
        if not lons:
            skipped_no_zone += 1
            continue

        lon  = float(np.median(lons))
        lat  = float(np.median(lats))
        zone = get_zone_for_lonlat(lon, lat, koppen_src)
        if zone is None:
            skipped_no_zone += 1
            continue

        # ── Load image ────────────────────────────────────────────────
        img_array = load_image_rgb(str(img_path))
        if img_array is None:
            skipped_load_fail += 1
            continue

        H, W = img_array.shape[:2]

        # ── Build label list ──────────────────────────────────────────
        labels = []
        for a in anns:
            x1, y1, x2, y2 = a["bbox_px"]
            # Clamp to image dimensions
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(0.0, min(x2, W))
            y2 = max(0.0, min(y2, H))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            labels.append([a["class_idx"], x1, y1, x2, y2])

        if not labels:
            continue

        # ── Tile, write ───────────────────────────────────────────────
        zone_img_dir = out_dir / zone / "images" / "all"
        zone_lbl_dir = out_dir / zone / "labels" / "all"
        stem = Path(image_id).stem

        for tile, tile_labels, (tx, ty) in tile_and_yield(
                img_array, labels, args.tile_size, args.overlap):
            tile_stem = f"{stem}_{tx}_{ty}"
            save_tile(
                tile, tile_labels,
                zone_img_dir / f"{tile_stem}.jpg",
                zone_lbl_dir / f"{tile_stem}.txt",
            )
            zone_tile_counts[zone] += 1
            zone_ann_counts[zone]  += len(tile_labels)

    koppen_src.close()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Conversion complete!")
    print(f"  Skipped (image not found) : {skipped_no_image}")
    print(f"  Skipped (no/outside zone) : {skipped_no_zone}")
    print(f"  Skipped (load failed)     : {skipped_load_fail}")
    print()
    print(f"  {'Zone':<10} {'Tiles':>8} {'Annotations':>14}")
    print(f"  {'-'*34}")
    for zone in ["CZ_A", "CZ_B", "CZ_C"]:
        print(f"  {zone:<10} {zone_tile_counts[zone]:>8} {zone_ann_counts[zone]:>14}")
    print()
    print("  Next step:")
    print(f"    python scripts/split_domain.py --data_dir {args.out_dir}")

    # Warn if any zone has very few tiles
    for zone in ["CZ_A", "CZ_B", "CZ_C"]:
        n = zone_tile_counts[zone]
        if n == 0:
            print(f"\n  [WARN] {zone} has 0 tiles! Check your Köppen raster coverage.")
        elif n < 50:
            print(f"\n  [WARN] {zone} has only {n} tiles — may be too few for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert xView + Köppen → RWDS-CZ YOLO dataset")

    parser.add_argument("--xview_img_dir", required=True,
                        help="Path to xView train_images/ folder (*.tif files)")
    parser.add_argument("--geojson",       required=True,
                        help="Path to xView_train.geojson")
    parser.add_argument("--koppen_raster", required=True,
                        help="Path to Beck_KG_V1_present_0p0083.tif")
    parser.add_argument("--out_dir",       default="data",
                        help="Output root directory (default: data/)")
    parser.add_argument("--tile_size",     type=int, default=512,
                        help="Tile size in pixels (default: 512)")
    parser.add_argument("--overlap",       type=float, default=0.2,
                        help="Tile overlap ratio (default: 0.2)")
    parser.add_argument("--inspect",       action="store_true",
                        help="Print stats only, write nothing (run this first!)")
    args = parser.parse_args()
    main(args)
