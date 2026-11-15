"""
Script: predict_colab.py
Dự đoán 4.000 ảnh test trên Colab hoặc máy local, hỗ trợ cả 3 engine:
1. TFLite INT8 (micro_yolo_int8.tflite - ĐÚNG MÔ HÌNH CHIP ESP32-S3 THỰC TẾ)
2. ONNX (micro_yolo_split_240x320.onnx / micro_yolo_split_int8.onnx)
3. PyTorch FP32 (best_model_mAP92.pt)

Tự động ghép với file Intent của đồng đội (intent_predictions.csv)
để xuất submission.csv chuẩn 100% (8.000 dòng).
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

def decode_per_class_nms(boxes_xywh, scores, conf_thresh=0.01, iou_thresh=0.50, max_det=100):
    """
    Per-class NMS chuẩn xác theo đúng chuẩn benchmark (map50.py) & Kaggle evaluation.
    """
    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    xyxy = np.clip(xyxy, 0, [320, 240, 320, 240])

    dets = []
    for c in range(scores.shape[1]):
        m = scores[:, c] > conf_thresh
        if not m.any():
            continue
        b, s = xyxy[m], scores[m, c]
        boxes_cv = np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
        keep = cv2.dnn.NMSBoxes(boxes_cv.tolist(), s.tolist(), float(conf_thresh), float(iou_thresh))
        for k in np.array(keep).reshape(-1):
            dets.append({
                "class_id": int(c),
                "score": round(float(s[k]), 4),
                "bbox": [round(float(b[k, 0]), 1), round(float(b[k, 1]), 1),
                         round(float(b[k, 2]), 1), round(float(b[k, 3]), 1)]
            })
    dets.sort(key=lambda d: -d["score"])
    return dets[:max_det]

def predict_with_tflite(tflite_path, image_files, conf_thresh=0.01, iou_thresh=0.50, max_det=100):
    print(f"[+] Đang khởi tạo TFLite Interpreter từ: {tflite_path}...")
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            import tflite_runtime.interpreter as tflite

    try:
        interpreter = tflite.Interpreter(model_path=str(tflite_path), num_threads=8)
    except Exception:
        interpreter = tflite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp_info = interpreter.get_input_details()[0]
    out_infos = interpreter.get_output_details()
    
    # Định danh head boxes và scores
    box_idx, score_idx = None, None
    for o in out_infos:
        if "box" in o["name"].lower() or o["shape"][1] == 4:
            box_idx = o
        elif "score" in o["name"].lower() or o["shape"][1] == 7:
            score_idx = o

    in_scale, in_zero = inp_info["quantization"]
    b_scale, b_zero = box_idx["quantization"]
    s_scale, s_zero = score_idx["quantization"]

    rows = []
    total = len(image_files)
    print(f"[+] Bắt đầu suy luận {total} ảnh bằng TFLite Full INT8 (chuẩn ESP32-S3):")

    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path).convert("RGB").resize((320, 240))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Quantize int8
        arr_int8 = np.clip(np.round(arr / in_scale) + in_zero, -128, 127).astype(np.int8)
        arr_int8 = np.expand_dims(arr_int8, 0) # [1, 240, 320, 3]

        interpreter.set_tensor(inp_info["index"], arr_int8)
        interpreter.invoke()

        # Giải lượng tử hóa kết quả
        raw_b = interpreter.get_tensor(box_idx["index"])[0] # [4, 1500]
        raw_s = interpreter.get_tensor(score_idx["index"])[0] # [7, 1500]

        float_b = (raw_b.astype(np.float32) - b_zero) * b_scale # [4, 1500]
        float_s = (raw_s.astype(np.float32) - s_zero) * s_scale # [7, 1500]

        boxes_xywh = float_b.T # [1500, 4]
        scores = float_s.T     # [1500, 7]

        preds = decode_per_class_nms(boxes_xywh, scores, conf_thresh, iou_thresh, max_det)
        rows.append(["DETECTION", img_path.stem, json.dumps(preds) if preds else "[]"])

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"    - Đã xử lý {idx + 1} / {total} ảnh...")

    return rows

def predict_with_pytorch(weights_path, image_files, conf_thresh=0.01, iou_thresh=0.50, max_det=100):
    print(f"[+] Đang khởi tạo PyTorch YOLO từ: {weights_path}...")
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[+] Sử dụng thiết bị: {device} ({'GPU' if device == 0 else 'CPU'})")
    model = YOLO(str(weights_path))

    rows = []
    total = len(image_files)
    batch_size = 32

    for i in range(0, total, batch_size):
        batch = image_files[i:i + batch_size]
        results = model.predict(
            source=[str(f) for f in batch],
            imgsz=(240, 320),
            conf=conf_thresh,
            iou=iou_thresh,
            max_det=max_det,
            device=device,
            verbose=False
        )
        for img_path, res in zip(batch, results):
            preds = []
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                score = round(float(box.conf[0].item()), 4)
                bbox = [round(float(c), 1) for c in box.xyxy[0].tolist()]
                preds.append({"class_id": cls_id, "score": score, "bbox": bbox})
            rows.append(["DETECTION", img_path.stem, json.dumps(preds) if preds else "[]"])

        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total:
            print(f"    - Đã xử lý {min(i + batch_size, total)} / {total} ảnh...")

    return rows

def main():
    parser = argparse.ArgumentParser(description="Dự đoán tập test cho Edge AI Challenge")
    parser.add_argument("--engine", choices=["tflite", "pytorch", "auto"], default="auto",
                        help="Chọn engine suy luận: 'tflite' (INT8 chip thực), 'pytorch' (.pt), 'auto'")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số .pt hoặc .tflite")
    parser.add_argument("--out", type=str, default="submission.csv", help="Đường dẫn file submission output")
    parser.add_argument("--conf", type=float, default=0.01, help="Ngưỡng confidence (0.01 cho Kaggle mAP)")
    parser.add_argument("--iou", type=float, default=0.50, help="Ngưỡng NMS IoU")
    parser.add_argument("--test_dir", type=str, default="test/images", help="Đường dẫn thư mục ảnh test")
    args = parser.parse_args()

    print("=" * 70)
    print(" CHƯƠNG TRÌNH DỰ ĐOÁN TẬP TEST (INFERENCE) VÀ XUẤT SUBMISSION")
    print("=" * 70)

    # 1. Tìm thư mục ảnh test
    test_path = Path(args.test_dir)
    if not test_path.exists():
        test_path = Path("egde-ai-challenge-2026-lan-2-qualifiers/test/images")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_path}")

    image_files = sorted(
        [f for f in test_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=lambda x: x.stem
    )
    print(f"[+] Tìm thấy: {len(image_files)} ảnh test trong {test_path}")

    # 2. Lựa chọn engine
    if args.weights:
        w_path = Path(args.weights)
        if not w_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {w_path}")
        if w_path.suffix.lower() == ".pt":
            print(f"[+] CHẾ ĐỘ: PyTorch FP32 từ file chỉ định ({w_path})")
            rows = predict_with_pytorch(w_path, image_files, args.conf, args.iou)
        else:
            print(f"[+] CHẾ ĐỘ: TFLite Full INT8 từ file chỉ định ({w_path})")
            rows = predict_with_tflite(w_path, image_files, args.conf, args.iou)
    else:
        tflite_candidate = Path("Kaggle_Submission_Package/micro_yolo_int8.tflite")
        if not tflite_candidate.exists():
            tflite_candidate = Path("micro_yolo_int8.tflite")

        pt_candidate = Path("best_final_submission.pt")
        if not pt_candidate.exists():
            pt_candidate = Path("runs/detect/micro_yolo_final_sub/weights/best.pt")
        if not pt_candidate.exists():
            pt_candidate = Path("runs/detect/best_champion_kd.pt")
        if not pt_candidate.exists():
            pt_candidate = Path("runs/detect/micro_yolo_120epochs/weights/best.pt")
        if not pt_candidate.exists():
            pt_candidate = Path("best_model_mAP92.pt")
        if not pt_candidate.exists():
            pt_candidate = Path("Kaggle_Submission_Package/best_model_mAP92.pt")

        if args.engine == "tflite":
            print(f"[+] CHẾ ĐỘ: TFLite Full INT8 ({tflite_candidate})")
            rows = predict_with_tflite(tflite_candidate, image_files, args.conf, args.iou)
        elif args.engine == "pytorch" or (args.engine == "auto" and pt_candidate.exists()):
            print(f"[+] CHẾ ĐỘ: PyTorch FP32 ({pt_candidate})")
            rows = predict_with_pytorch(pt_candidate, image_files, args.conf, args.iou)
        else:
            print(f"[+] CHẾ ĐỘ: TFLite Full INT8 ({tflite_candidate})")
            rows = predict_with_tflite(tflite_candidate, image_files, args.conf, args.iou)

    df_det = pd.DataFrame(rows, columns=["task_type", "sample_id", "prediction"])
    print(f"[+] Đã hoàn thành suy luận: {len(df_det)} dòng DETECTION.")

    # 3. Ghép với phần Intent thực tế từ TextCNN INT8
    intent_paths = [
        Path("intent_submission.csv"),
        Path("Kaggle_Submission_Package/intent_submission.csv"),
        Path("EdgeAI-DUT-Competition/experiments/textcnn_word/intent_submission.csv"),
        Path("EdgeAI-DUT-Competition/experiments/textcnn_dual/intent_submission.csv"),
        Path("EdgeAI-DUT-Competition/intent_submission.csv"),
        Path("intent_predictions.csv"),
        Path("intent_test_preds.csv")
    ]
    df_intent = None
    for ip in intent_paths:
        if ip.exists():
            print(f"[+] [OK] Tìm thấy file Intent thực tế: {ip}")
            df_intent = pd.read_csv(ip)
            break

    if df_intent is None:
        print("\n" + "!" * 70)
        print(" [CHÚ Ý VỀ PHẦN INTENT / TEXTCNN]")
        print(" - Chưa tìm thấy intent_submission.csv. Tạm dùng khuôn mẫu từ sample_submission.csv.")
        print(" - Khuyến nghị: chạy TextCNN để lấy 100% điểm phần Intent (30% tổng điểm)!")
        print("!" * 70 + "\n")

        sample_sub = Path("sample_submission.csv")
        if not sample_sub.exists():
            sample_sub = Path("egde-ai-challenge-2026-lan-2-qualifiers/sample_submission.csv")
        df_sample = pd.read_csv(sample_sub)
        df_intent = df_sample[df_sample["task_type"] == "INTENT"].copy()

    # 4. Xuất file submission.csv hoàn chỉnh (8.000 dòng: 4.000 DETECTION + 4.000 INTENT)
    df_submission = pd.concat([df_det, df_intent], ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(out_path, index=False)
    print(f"[OK] Đã xuất thành công: {out_path} (Tổng cộng {len(df_submission)} dòng)")
    print("=" * 70)

if __name__ == "__main__":
    main()
