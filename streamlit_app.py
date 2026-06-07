"""Simple Streamlit interface for running satellite object detection on video."""

from __future__ import annotations

import tempfile
import time
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


def find_checkpoints() -> list[Path]:
    """Return trained best checkpoints, newest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted(
        RUNS_DIR.rglob("weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def checkpoint_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def device_options() -> list[str]:
    options = ["cpu"]
    if torch.cuda.is_available():
        options.insert(0, "0")
    return options


def convert_to_browser_video(input_path: Path, output_path: Path) -> None:
    """Convert OpenCV's MP4V output to browser-compatible H.264."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        details = completed.stderr.strip().splitlines()
        message = details[-1] if details else "Unknown FFmpeg error"
        raise RuntimeError(f"Không thể chuyển video sang H.264: {message}")


def process_video(
    input_path: Path,
    output_path: Path,
    model: YOLO,
    confidence: float,
    iou: float,
    image_size: int,
    device: str,
    progress_bar,
    status,
) -> tuple[dict, Counter]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Không thể đọc video đã tải lên.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Không thể tạo video kết quả bằng codec MP4V.")

    class_counts: Counter = Counter()
    processed_frames = 0
    started_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]
            writer.write(result.plot())

            if result.boxes is not None:
                for class_id in result.boxes.cls.int().cpu().tolist():
                    class_counts[result.names[class_id]] += 1

            processed_frames += 1
            if total_frames > 0:
                progress_bar.progress(
                    min(processed_frames / total_frames, 1.0),
                    text=f"Đang xử lý frame {processed_frames}/{total_frames}",
                )
            else:
                status.caption(f"Đã xử lý {processed_frames} frame...")
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started_at
    metadata = {
        "frames": processed_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "elapsed": elapsed,
    }
    return metadata, class_counts


def render_sidebar(checkpoints: list[Path]) -> tuple[Path | None, float, float, int, str]:
    st.sidebar.header("Cấu hình")

    selected_weights = None
    if checkpoints:
        selected_weights = st.sidebar.selectbox(
            "Model",
            checkpoints,
            format_func=checkpoint_label,
        )
    else:
        st.sidebar.warning("Chưa tìm thấy checkpoint `runs/**/weights/best.pt`.")

    custom_path = st.sidebar.text_input(
        "Hoặc nhập đường dẫn weights",
        placeholder="runs/my_model/weights/best.pt",
    ).strip()
    if custom_path:
        candidate = Path(custom_path).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        selected_weights = candidate.resolve()

    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.sidebar.select_slider(
        "Kích thước inference",
        options=[320, 416, 512, 640, 768, 960, 1280],
        value=640,
    )
    device = st.sidebar.selectbox("Thiết bị", device_options())
    return selected_weights, confidence, iou, image_size, device


def main() -> None:
    st.set_page_config(
        page_title="Satellite Object Detection",
        page_icon="🛰️",
        layout="wide",
    )
    st.title("Satellite Video Object Detection")
    st.caption("Phát hiện đối tượng trong video vệ tinh bằng các model YOLO đã huấn luyện.")

    checkpoints = find_checkpoints()
    weights, confidence, iou, image_size, device = render_sidebar(checkpoints)

    uploaded_video = st.file_uploader(
        "Tải video vệ tinh",
        type=VIDEO_TYPES,
        help="Hỗ trợ MP4, AVI, MOV, MKV và WebM.",
    )
    if uploaded_video is not None:
        st.video(uploaded_video)

    can_process = (
        uploaded_video is not None
        and weights is not None
        and weights.is_file()
    )
    if weights is not None and not weights.is_file():
        st.error(f"Không tìm thấy weights: `{weights}`")

    if st.button(
        "Bắt đầu detect",
        type="primary",
        disabled=not can_process,
        use_container_width=True,
    ):
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        progress_bar = st.progress(0.0, text="Đang chuẩn bị model...")
        status = st.empty()

        try:
            with st.spinner(f"Đang load `{checkpoint_label(weights)}`..."):
                model = load_model(str(weights))

            with tempfile.TemporaryDirectory(prefix="satellite_detection_") as temp_dir:
                input_path = Path(temp_dir) / f"input{suffix}"
                raw_output_path = Path(temp_dir) / "detected_raw.mp4"
                output_path = Path(temp_dir) / "detected_h264.mp4"
                input_path.write_bytes(uploaded_video.getvalue())

                metadata, class_counts = process_video(
                    input_path=input_path,
                    output_path=raw_output_path,
                    model=model,
                    confidence=confidence,
                    iou=iou,
                    image_size=image_size,
                    device=device,
                    progress_bar=progress_bar,
                    status=status,
                )
                progress_bar.progress(0.99, text="Đang tối ưu video cho trình duyệt...")
                convert_to_browser_video(raw_output_path, output_path)
                output_bytes = output_path.read_bytes()

            progress_bar.progress(1.0, text="Hoàn tất")
            status.success(
                f"Đã xử lý {metadata['frames']} frame trong "
                f"{metadata['elapsed']:.1f} giây."
            )

            st.subheader("Kết quả")
            st.video(output_bytes, format="video/mp4")
            st.download_button(
                "Tải video đã detect",
                data=output_bytes,
                file_name=f"{Path(uploaded_video.name).stem}_detected.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            metric_columns = st.columns(4)
            metric_columns[0].metric("Frames", metadata["frames"])
            metric_columns[1].metric("Độ phân giải", f"{metadata['width']}×{metadata['height']}")
            metric_columns[2].metric("Video FPS", f"{metadata['fps']:.1f}")
            metric_columns[3].metric("Detections", sum(class_counts.values()))

            if class_counts:
                counts_df = pd.DataFrame(
                    class_counts.most_common(),
                    columns=["Lớp", "Số detection"],
                )
                st.dataframe(counts_df, hide_index=True, use_container_width=True)
            else:
                st.info("Không phát hiện đối tượng nào với confidence hiện tại.")
        except Exception as exc:
            progress_bar.empty()
            status.empty()
            st.exception(exc)


if __name__ == "__main__":
    main()
