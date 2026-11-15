"""Huấn luyện Micro-YOLO trên máy cá nhân (hỗ trợ NVIDIA CUDA GPU).
Tự động kiểm tra số lượng tham số <= 200.000 theo quy định BTC.
"""
import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def resolve_dataset_dir(data_dir=None):
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    candidates.extend([
        Path("data/raw"),
        Path(__file__).resolve().parents[2] / "data" / "raw",
        Path(__file__).resolve().parents[3] / "egde-ai-challenge-2026-lan-2-qualifiers",
        Path("egde-ai-challenge-2026-lan-2-qualifiers"),
        Path("../egde-ai-challenge-2026-lan-2-qualifiers"),
        Path(r"D:\edgeAI_phan_biet_mot_so_thu_tu_camera_65k\egde-ai-challenge-2026-lan-2-qualifiers"),
    ])
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c
    return Path("data/raw")


def prepare_yolo_yaml(data_dir, output_yaml_path):
    data_dir = resolve_dataset_dir(data_dir).resolve()
    train_images = data_dir / "train" / "images"
    # Nếu có test thì dùng test, nếu không dùng train
    val_images = data_dir / "train" / "images"
    
    cfg = {
        "path": str(data_dir),
        "train": "train/images",
        "val": "train/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": 7
    }
    output_yaml_path = Path(output_yaml_path)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[+] Đã tạo file cấu hình dữ liệu YOLO: {output_yaml_path}")
    return str(output_yaml_path)


def main():
    parser = argparse.ArgumentParser(description="Train Micro-YOLO for Edge AI Challenge")
    parser.add_argument("--model_cfg", type=str, default="configs/micro_yolo_deep.yaml",
                        help="Đường dẫn file định nghĩa cấu trúc YOLO")
    parser.add_argument("--data", type=str, default=None, help="Thư mục dataset raw")
    parser.add_argument("--data_yaml", type=str, default="data/yolo_dataset.yaml",
                        help="Đường dẫn file yaml cho dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (16 hoặc 32 phù hợp 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[240, 320], help="Kích thước ảnh [H, W]")
    parser.add_argument("--project", type=str, default="experiments_yolo", help="Thư mục lưu kết quả")
    parser.add_argument("--name", type=str, default="micro_yolo_run", help="Tên lần chạy")
    parser.add_argument("--device", type=str, default=None, help="0 cho CUDA GPU, hoặc cpu")
    args = parser.parse_args()

    # 1. Kiểm tra phần cứng
    if args.device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 70)
    print(" HUẤN LUYỆN MICRO-YOLO CHO ESP32-S3 (PYTORCH + CUDA)")
    print("=" * 70)
    print(f"[+] Thiết bị huấn luyện: {device} ({torch.cuda.get_device_name(0) if device == '0' else 'CPU'})")
    
    # 2. Chuẩn bị file yaml
    data_yaml = args.data_yaml
    if not Path(data_yaml).exists():
        data_yaml = prepare_yolo_yaml(args.data, data_yaml)

    # 3. Khởi tạo mô hình từ file kiến trúc
    model_cfg = Path(args.model_cfg)
    if not model_cfg.exists():
        model_cfg = Path(__file__).resolve().parents[2] / "configs" / "micro_yolo_deep.yaml"
        
    print(f"[+] Khởi tạo kiến trúc từ: {model_cfg}")
    model = YOLO(str(model_cfg))

    # 4. Kiểm tra tham số
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"[+] Tổng số tham số: {total_params:,} (Giới hạn BTC: <= 200,000)")
    if total_params > 200_000:
        raise ValueError(f"Mô hình vượt quá 200.000 tham số ({total_params})! Vi phạm quy định cuộc thi.")
    else:
        print("[OK] Mô hình đạt chuẩn giới hạn tham số 100%!")

    # 5. Huấn luyện
    imgsz = args.imgsz if len(args.imgsz) == 2 else args.imgsz[0]
    print(f"[+] Bắt đầu huấn luyện {args.epochs} epochs với batch={args.batch}, imgsz={imgsz}...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=0.002,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        verbose=True
    )
    print(f"\n[OK] Huấn luyện hoàn tất! Kết quả lưu tại: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
