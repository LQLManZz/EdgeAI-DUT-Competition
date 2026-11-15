"""Trạm điều khiển huấn luyện & xuất mô hình Edge AI trên máy cá nhân (Local PC Runner).
Kết hợp trọn vẹn những điểm mạnh nhất của EdgeDet và Micro-YOLO cho Edge AI Challenge 2026.
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

# Cấu hình môi trường tự động
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR)
os.environ.setdefault("KERAS_HOME", str(ROOT_DIR / ".keras"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def print_banner():
    print("=" * 76)
    print("       EDGE AI CHALLENGE 2026 - TRẠM HUẤN LUYỆN V3 (LOCAL PC RUNNER)      ")
    print("           ESP32-S3 N16R8 + Camera OV7670 | <= 200k params | INT8        ")
    print("=" * 76)


def check_hardware():
    print("[*] Kiểm tra phần cứng & môi trường máy bạn:")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "Không có (CPU)"
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0
        print(f"    - PyTorch: {torch.__version__} | CUDA: {cuda_ok} | GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    except ImportError:
        print("    - PyTorch: Chưa cài đặt")

    try:
        import tensorflow as tf
        print(f"    - TensorFlow: {tf.__version__} | Keras backend: {os.environ.get('KERAS_BACKEND', 'tensorflow')}")
    except ImportError:
        print("    - TensorFlow: Chưa cài đặt")
    print("-" * 76)


def run_cmd(cmd_list, desc=""):
    if desc:
        print(f"\n[>>>] {desc}...")
    print(f"Lệnh: {' '.join(str(x) for x in cmd_list)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env["KERAS_HOME"] = str(ROOT_DIR / ".keras")
    res = subprocess.run(cmd_list, env=env, cwd=ROOT_DIR)
    if res.returncode != 0:
        print(f"[!] Lệnh kết thúc với mã lỗi {res.returncode}")
    else:
        print(f"[OK] Hoàn tất: {desc}")
    return res.returncode


def mode_train_edgedet(args):
    config = args.config or "configs/h1p_v2_240x320.json"
    out = args.out or (f"experiments/{args.name}" if getattr(args, "name", None) else "experiments/edgedet_run")
    epochs = str(args.epochs) if args.epochs else "80"
    batch = str(args.batch) if args.batch else "16"
    cmd = [
        sys.executable, "-m", "edgedet.train.train",
        "--config", str(config),
        "--out", str(out),
        "--epochs", epochs,
        "--batch", batch
    ]
    if args.fold is not None:
        cmd.extend(["--fold", str(args.fold)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    run_cmd(cmd, f"Huấn luyện EdgeDet CenterNet ({config})")


def mode_train_yolo(args):
    epochs = str(args.epochs) if args.epochs else "40"
    batch = str(args.batch) if args.batch else "16"
    cmd = [
        sys.executable, "-m", "yolo.train_yolo",
        "--model_cfg", "configs/micro_yolo_deep.yaml",
        "--epochs", epochs,
        "--batch", batch,
        "--name", args.name or "micro_yolo_local"
    ]
    run_cmd(cmd, "Huấn luyện Micro-YOLO (PyTorch + CUDA GPU)")


def mode_train_textcnn(args):
    config = args.config or "configs/textcnn_dual_v3.json"
    out = args.out or "experiments/textcnn_run"
    cmd = [
        sys.executable, "-m", "textcnn.train",
        "--config", str(config),
        "--out", str(out)
    ]
    run_cmd(cmd, "Huấn luyện TextCNN Intent Classifier (30% điểm thi)")


def mode_kfold(args):
    from edgedet.data.dataset import load_items
    from edgedet.data.kfold import multilabel_kfold_split
    print("[*] Đang đọc dữ liệu để chia K-Fold Stratified...")
    items = load_items(split="train")
    n_splits = args.n_splits or 3
    folds = multilabel_kfold_split(items, n_splits=n_splits)
    print(f"[OK] Đã chia thành công {n_splits} Folds cân bằng nhãn:")
    for i, (train_it, val_it) in enumerate(folds):
        print(f"     Fold {i+1}: Train {len(train_it)} ảnh | Val {len(val_it)} ảnh")


def mode_soup(args):
    if not args.models:
        print("[!] Vui lòng truyền danh sách model cần soup bằng --models m1 m2...")
        return
    out = args.out or "soup_model.keras"
    cmd = [sys.executable, "scripts/model_soup.py", "--models"] + args.models + ["--out", out]
    run_cmd(cmd, "Thực hiện Model Soup (Bình quân trọng số)")


def mode_quantize(args):
    if not args.model:
        # Tự tìm model tốt nhất
        cands = list(Path("experiments").glob("**/best.keras"))
        if not cands:
            cands = list(Path("models").glob("**/*.keras"))
        if cands:
            args.model = str(cands[0])
        else:
            print("[!] Không tìm thấy file .keras cần quantize. Hãy truyền --model <path>")
            return
    cmd = [sys.executable, "-m", "edgedet.quant.export_int8", "--model", args.model]
    run_cmd(cmd, f"Lượng tử hóa INT8 cho {args.model}")


def mode_submit(args):
    # Tìm tflite detection và intent
    tflite = args.tflite
    if not tflite:
        cands = list(Path("experiments").glob("**/model_int8.tflite"))
        if not cands:
            cands = list(Path("models").glob("**/*int8*.tflite"))
        tflite = str(cands[0]) if cands else "models/tflite/model_int8.tflite"

    intent = args.intent
    if not intent:
        cands = list(Path("experiments").glob("**/intent_submission.csv"))
        if cands:
            intent = str(cands[0])
        else:
            # Fallback sang intent của v2 nếu có
            intent = "submissions/submission_17_h1p_kd_full_240x320_bs35.csv"

    out = args.out or "submission_local.csv"
    cmd = [
        sys.executable, "-m", "edgedet.eval.make_submission",
        "--tflite", str(tflite),
        "--intent", str(intent),
        "--out", str(out)
    ]
    run_cmd(cmd, f"Tạo file nộp bài Kaggle: {out}")


def interactive_menu():
    print_banner()
    check_hardware()
    while True:
        print("\nCHỌN CHỨC NĂNG BẠN MUỐN THỰC HIỆN:")
        print("  [1] Huấn luyện EdgeDet CenterNet (Keras/TensorFlow - Champion Model 95.8%)")
        print("  [2] Huấn luyện Micro-YOLO (PyTorch + GPU Quadro T1000)")
        print("  [3] Huấn luyện TextCNN Intent Classifier (30% điểm thi - chỉ mất ~2 phút)")
        print("  [4] Chia 3-Fold / 5-Fold Stratified cân bằng 7 nhãn")
        print("  [5] Thực hiện Model Soup (Bình quân trọng số không tăng chi phí suy luận)")
        print("  [6] Lượng tử hóa sang Full INT8 TFLite & Kiểm tra luật thi BTC")
        print("  [7] Xuất file nộp bài Kaggle (submission.csv)")
        print("  [0] Thoát")
        
        choice = input("\nNhập lựa chọn của bạn (0-7): ").strip()
        if choice == "0":
            print("Tạm biệt!")
            break
        elif choice == "1":
            ep = input("Nhập số epochs (mặc định 60): ").strip() or "60"
            bs = input("Nhập batch size (mặc định 16): ").strip() or "16"
            class DummyArgs:
                config = "configs/h1p_v2_240x320.json"
                out = "experiments/edgedet_local"
                epochs = int(ep)
                batch = int(bs)
                fold = None
                limit = None
            mode_train_edgedet(DummyArgs())
        elif choice == "2":
            ep = input("Nhập số epochs (mặc định 30): ").strip() or "30"
            bs = input("Nhập batch size (mặc định 16): ").strip() or "16"
            class DummyArgs:
                epochs = int(ep)
                batch = int(bs)
                name = "micro_yolo_local"
            mode_train_yolo(DummyArgs())
        elif choice == "3":
            class DummyArgs:
                config = "configs/textcnn_dual_v3.json"
                out = "experiments/textcnn_local"
            mode_train_textcnn(DummyArgs())
        elif choice == "4":
            k = input("Nhập số Folds (mặc định 3): ").strip() or "3"
            class DummyArgs:
                n_splits = int(k)
            mode_kfold(DummyArgs())
        elif choice == "5":
            m_str = input("Nhập đường dẫn các file checkpoint (cách nhau bởi khoảng trắng): ").strip()
            if m_str:
                class DummyArgs:
                    models = m_str.split()
                    out = "models/soup_best.keras"
                mode_soup(DummyArgs())
            else:
                print("Chưa nhập danh sách model.")
        elif choice == "6":
            m_path = input("Nhập đường dẫn file model .keras (để trống tự tìm): ").strip()
            class DummyArgs:
                model = m_path if m_path else None
            mode_quantize(DummyArgs())
        elif choice == "7":
            class DummyArgs:
                tflite = None
                intent = None
                out = "submission_v3_local.csv"
            mode_submit(DummyArgs())
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")


def main():
    parser = argparse.ArgumentParser(description="EdgeAI V3 Local Runner")
    parser.add_argument("--mode", choices=["edgedet", "yolo", "textcnn", "kfold", "soup", "quantize", "submit"],
                        help="Chế độ chạy tự động không cần menu tương tác")
    parser.add_argument("--config", type=str, help="File cấu hình json")
    parser.add_argument("--out", type=str, help="Thư mục xuất kết quả")
    parser.add_argument("--epochs", type=int, help="Số epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--fold", type=int, help="Fold index")
    parser.add_argument("--n_splits", type=int, default=3, help="Số folds")
    parser.add_argument("--limit", type=int, help="Giới hạn số ảnh train thử nghiệm")
    parser.add_argument("--name", type=str, help="Tên lần chạy")
    parser.add_argument("--model", type=str, help="Đường dẫn model")
    parser.add_argument("--models", nargs="+", help="Danh sách models cho soup")
    parser.add_argument("--tflite", type=str, help="File tflite cho submission")
    parser.add_argument("--intent", type=str, help="File intent csv cho submission")
    parser.add_argument("--interactive", action="store_true", help="Bật menu tương tác")
    args = parser.parse_args()

    if args.mode == "edgedet":
        mode_train_edgedet(args)
    elif args.mode == "yolo":
        mode_train_yolo(args)
    elif args.mode == "textcnn":
        mode_train_textcnn(args)
    elif args.mode == "kfold":
        mode_kfold(args)
    elif args.mode == "soup":
        mode_soup(args)
    elif args.mode == "quantize":
        mode_quantize(args)
    elif args.mode == "submit":
        mode_submit(args)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
