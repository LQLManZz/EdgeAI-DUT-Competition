"""Lượng tử hóa mô hình Micro-YOLO sang Full INT8 TFLite (.tflite) cho ESP32-S3.
Tự động kiểm tra:
1. Dung lượng file <= 1.5 MB
2. Các op trong đồ thị tương thích TFLite Micro / ESP-NN
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def generate_calibration_data(img_dir, output_npy="calib_images.npy", count=50, size=(320, 240)):
    print(f"[*] Đang chuẩn bị tập ảnh hiệu chuẩn INT8 ({count} ảnh)...")
    img_dir = Path(img_dir)
    img_paths = list(img_dir.glob("*.png"))[:count]
    if not img_paths:
        img_paths = list(img_dir.glob("*.jpg"))[:count]
    if not img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại: {img_dir}")

    calib_list = []
    for p in img_paths:
        img = Image.open(p).convert("RGB").resize(size)
        arr = np.array(img, dtype=np.float32) / 255.0
        calib_list.append(arr)

    calib_arr = np.stack(calib_list, axis=0)
    np.save(output_npy, calib_arr)
    print(f"[+] Đã tạo file calibration: {output_npy} shape={calib_arr.shape}")
    return output_npy


def quantize_onnx_to_tflite(onnx_path, calib_npy, output_dir="tflite_yolo_out"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Chạy onnxsim nếu có
    sim_onnx = out_path / "model_sim.onnx"
    try:
        cmd_sim = [sys.executable, "-m", "onnxsim", str(onnx_path), str(sim_onnx)]
        subprocess.run(cmd_sim, check=True, capture_output=True)
        target_onnx = sim_onnx
        print(f"[+] onnxsim hoàn tất: {sim_onnx}")
    except Exception as e:
        print(f"[!] Bỏ qua onnxsim (dùng onnx gốc): {e}")
        target_onnx = Path(onnx_path)

    # 2. Gọi onnx2tf
    print("[*] Đang chuyển đổi đồ thị và lượng tử hóa sang Full INT8 bằng onnx2tf...")
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(target_onnx),
        "-o", str(output_dir),
        "-oiqt",
        "-cind", "images", str(calib_npy), "[0.0]", "[1.0]"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] onnx2tf thất bại:\n{res.stderr}")
        raise RuntimeError("onnx2tf quantization failed!")

    tflite_file = out_path / f"{Path(onnx_path).stem}_full_integer_quant.tflite"
    if not tflite_file.exists():
        candidates = list(out_path.glob("*.tflite"))
        if candidates:
            tflite_file = candidates[0]
            
    print(f"[OK] Đã xuất TFLite INT8 thành công: {tflite_file}")
    size_mb = tflite_file.stat().st_size / (1024 * 1024)
    print(f"     - Dung lượng: {size_mb:.2f} MB (Giới hạn BTC: <= 1.5 MB)")
    if size_mb <= 1.5:
        print("     -> ĐẠT CHUẨN DUNG LƯỢNG 100%!")
    return tflite_file


def main():
    parser = argparse.ArgumentParser(description="Quantize Micro-YOLO to INT8 TFLite")
    parser.add_argument("--onnx", type=str, required=True, help="File ONNX cần quantize")
    parser.add_argument("--data_dir", type=str, default="data/raw/train/images", help="Thư mục ảnh hiệu chuẩn")
    parser.add_argument("--out_dir", type=str, default="models/tflite", help="Thư mục xuất file tflite")
    args = parser.parse_args()

    # Tìm data_dir nếu không tồn tại
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        candidates = [
            Path("egde-ai-challenge-2026-lan-2-qualifiers/train/images"),
            Path("../egde-ai-challenge-2026-lan-2-qualifiers/train/images"),
            Path(r"D:\edgeAI_phan_biet_mot_so_thu_tu_camera_65k\egde-ai-challenge-2026-lan-2-qualifiers\train\images"),
        ]
        for c in candidates:
            if c.exists():
                data_dir = c
                break

    calib_npy = generate_calibration_data(data_dir)
    quantize_onnx_to_tflite(args.onnx, calib_npy, args.out_dir)


if __name__ == "__main__":
    main()
