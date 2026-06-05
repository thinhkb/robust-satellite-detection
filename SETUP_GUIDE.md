# HƯỚNG DẪN CÀI ĐẶT & CHẠY — Windows + xView
# =============================================

## Cấu trúc dữ liệu cần có sau khi tải

Sau khi tải xView và Köppen raster, bạn cần tổ chức thư mục như sau:

```
E:\robust-satellite-detection\
│
├── data_raw\                      ← tạo thư mục này
│   ├── train_images\              ← giải nén từ xView .tar.gz
│   │   ├── 1.tif
│   │   ├── 2.tif
│   │   └── ...  (~847 file .tif)
│   ├── xView_train.geojson        ← file annotation từ xView
│   └── Beck_KG_V1_present_0p0083.tif  ← Köppen raster từ figshare
│
├── data\                          ← tự động tạo bởi script
├── configs\
├── scripts\
└── ...
```

---

## Bước 1: Cài đặt môi trường

```bash
# Tạo virtual environment (nếu chưa có)
python -m venv .venv
.venv\Scripts\activate

# Cài các package cơ bản
pip install ultralytics tqdm pyyaml pandas matplotlib Pillow numpy

# Cài rasterio (quan trọng — đọc GeoTIFF)
# Thử cách 1 (đơn giản nhất):
pip install rasterio

# Nếu thất bại trên Windows, thử cách 2 (wheel pre-built):
# Vào: https://github.com/cgohlke/geospatial-wheels/releases
# Tải file: rasterio-1.3.x-cp311-cp311-win_amd64.whl  (cp311 = Python 3.11)
# Rồi: pip install rasterio-1.3.x-cp311-cp311-win_amd64.whl

# Nếu dùng conda:
# conda install -c conda-forge rasterio
```

---

## Bước 2: Kiểm tra dữ liệu (chạy TRƯỚC khi convert)

```bash
python scripts/convert_to_yolo.py \
    --xview_img_dir data_raw/xview/train_images \
    --geojson       data_raw/xview/xView_train.geojson \
    --koppen_raster data_raw/koppen/Beck_KG_V1_present_0p0083.tif \
    --out_dir       data \
    --inspect
```

Output mong đợi:
```
Images found in data_raw/xview/train_images: 847
Total annotations : 601,937
In target classes : 162,841
...
Zone distribution (sample of 200 images):
  CZ_A: 45
  CZ_B: 38
  CZ_C: 71
  Other: 46
```

---

## Bước 3: Chạy conversion

```bash
python scripts/convert_to_yolo.py \
    --xview_img_dir data_raw/xview/train_images \
    --geojson       data_raw/xview/xView_train.geojson \
    --koppen_raster data_raw/koppen/Beck_KG_V1_present_0p0083.tif \
    --out_dir       data \
    --tile_size     512 \
    --overlap       0.2
```

⏱ Thời gian ước tính: 30–90 phút tùy CPU (847 ảnh × ~10 tiles/ảnh)

Output mong đợi sau khi xong:
```
Zone       Tiles   Annotations
CZ_A       ~3,000      ~45,000
CZ_B       ~1,500      ~22,000
CZ_C       ~4,000      ~60,000
```

---

## Bước 4: Tạo splits và YAML configs

```bash
python scripts/split_domain.py --data_dir data --cfg_dir configs
```

---

## Bước 5: Kiểm tra lần cuối

```bash
python scripts/check_data.py
```

Phải thấy: ✓ ALL CHECKS PASSED

---

## Bước 6: Train

```bash
# GPU yếu (< 6GB VRAM) hoặc CPU
python scripts/train_baseline.py \
    --cfg      configs/single_source_cz_a.yaml \
    --run_name baseline_CZA \
    --model    yolov8n.pt \
    --imgsz    512 \
    --epochs   50 \
    --batch    8

# GPU khá (8GB+ VRAM)
python scripts/train_baseline.py \
    --cfg      configs/single_source_cz_a.yaml \
    --run_name baseline_CZA \
    --model    yolov8s.pt \
    --imgsz    640 \
    --epochs   80 \
    --batch    16
```

---

## Xử lý lỗi thường gặp

### Lỗi: "No module named 'rasterio'"
```bash
# Thử theo thứ tự:
pip install rasterio
# Nếu lỗi, dùng conda:
conda install -c conda-forge rasterio
```

### Lỗi: "CPLE_AppDefined" hoặc rasterio CRS error
Đây là warning không ảnh hưởng — bỏ qua.

### Zone có 0 tiles
Kiểm tra coordinates trong GeoJSON có nằm trong phạm vi Köppen raster không.
Raster Beck et al. bao phủ toàn cầu (-180 to 180 lon, -90 to 90 lat).

### Lỗi khi đọc .tif "not a supported file format"
```bash
pip install rasterio --upgrade
# Hoặc tải wheel mới nhất từ:
# https://github.com/cgohlke/geospatial-wheels/releases
```

### Training quá chậm không có GPU
```bash
# Dùng yolov8n (nano) và ảnh nhỏ hơn:
python scripts/train_baseline.py \
    --model yolov8n.pt --imgsz 416 --epochs 30 --batch 4
```
