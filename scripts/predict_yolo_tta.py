"""Dự đoán Micro-YOLO trên tập test với tùy chọn Test-Time Augmentation (TTA).
Hỗ trợ lật ngang (Horizontal Flip) và hợp nhất box bằng Weighted Boxes Fusion (WBF).
Tự động giới hạn tối đa 100 boxes mỗi ảnh theo chuẩn Ban Tổ Chức.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    areaB = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def wbf_two_sets(preds_orig, preds_flip, iou_thr=0.55):
    """Hợp nhất predictions từ ảnh gốc và ảnh lật ngang."""
    all_boxes = {}
    for p in preds_orig:
        cls = p["class_id"]
        all_boxes.setdefault(cls, []).append({
            "bbox": np.array(p["bbox"], dtype=np.float32),
            "score": float(p["score"]),
            "weight": 1.0
        })
    for p in preds_flip:
        cls = p["class_id"]
        all_boxes.setdefault(cls, []).append({
            "bbox": np.array(p["bbox"], dtype=np.float32),
            "score": float(p["score"]),
            "weight": 1.0
        })

    fused = []
    for cls, b_list in all_boxes.items():
        b_list.sort(key=lambda x: -x["score"])
        clusters = []
        for b in b_list:
            matched = None
            for cl in clusters:
                c_box = np.mean([item["bbox"] for item in cl], axis=0)
                if compute_iou(b["bbox"], c_box) >= iou_thr:
                    matched = cl
                    break
            if matched:
                matched.append(b)
            else:
                clusters.append([b])

        for cl in clusters:
            weights = np.array([item["weight"] * item["score"] for item in cl])
            w_sum = np.sum(weights)
            if w_sum > 0:
                f_box = np.sum([item["bbox"] * (item["weight"] * item["score"]) for item in cl], axis=0) / w_sum
            else:
                f_box = np.mean([item["bbox"] for item in cl], axis=0)

            # Thưởng nếu cả ảnh gốc và lật đều nhận diện được
            agreement = 1.0 if len(cl) > 1 else 0.85
            score = float(np.mean([item["score"] for item in cl]) * agreement)
            fused.append({
                "class_id": int(cls),
                "score": round(float(np.clip(score, 0.0, 1.0)), 5),
                "bbox": [round(float(x), 2) for x in f_box]
            })

    fused.sort(key=lambda x: -x["score"])
    return fused[:100]


def run_inference(weights_path, test_dir, out_csv, base_sub_path, tta=True, conf=0.03, batch=32):
    print(f"[*] Đang tải mô hình Micro-YOLO: {weights_path}")
    model = YOLO(weights_path)

    test_dir = Path(test_dir)
    image_paths = sorted(test_dir.glob("*.png"))
    if not image_paths:
        image_paths = sorted(test_dir.glob("*.jpg"))
    print(f"[*] Tìm thấy {len(image_paths)} ảnh test trong {test_dir}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị dự đoán: {device} | Chế độ TTA: {tta}")

    predictions_dict = {}

    # Chạy theo từng batch để tối ưu tốc độ
    for i in range(0, len(image_paths), batch):
        batch_paths = image_paths[i:i + batch]
        batch_imgs = [cv2.imread(str(p)) for p in batch_paths]

        # 1. Dự đoán trên ảnh gốc
        results_orig = model.predict(batch_imgs, conf=conf, device=device, verbose=False, imgsz=[256, 320])

        if tta:
            # 2. Dự đoán trên ảnh lật ngang
            batch_imgs_flip = [cv2.flip(img, 1) for img in batch_imgs]
            results_flip = model.predict(batch_imgs_flip, conf=conf, device=device, verbose=False, imgsz=[256, 320])

        for idx, p in enumerate(batch_paths):
            stem = p.stem  # img_0001
            h, w = batch_imgs[idx].shape[:2]

            preds_orig = []
            for b in results_orig[idx].boxes:
                cls_id = int(b.cls.item())
                score = float(b.conf.item())
                xyxy = b.xyxy.cpu().numpy()[0].tolist()
                preds_orig.append({"class_id": cls_id, "score": score, "bbox": xyxy})

            if tta:
                preds_flip = []
                for b in results_flip[idx].boxes:
                    cls_id = int(b.cls.item())
                    score = float(b.conf.item())
                    x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0].tolist()
                    # Nghịch đảo tọa độ x do lật ngang
                    rev_x1 = float(w - x2)
                    rev_x2 = float(w - x1)
                    preds_flip.append({"class_id": cls_id, "score": score, "bbox": [rev_x1, y1, rev_x2, y2]})

                fused = wbf_two_sets(preds_orig, preds_flip)
            else:
                preds_orig.sort(key=lambda x: -x["score"])
                fused = preds_orig[:100]

            predictions_dict[stem] = fused

        if (i + batch) % 500 < batch or (i + batch) >= len(image_paths):
            print(f"    - Tiến độ: {min(i + batch, len(image_paths))}/{len(image_paths)} ảnh")

    # Ghép vào file submission chuẩn
    base_df = pd.read_csv(base_sub_path)
    output_rows = []
    for _, row in base_df.iterrows():
        s_id = row["sample_id"]
        t_type = row["task_type"]
        if t_type == "DETECTION":
            preds = predictions_dict.get(s_id, [])
            output_rows.append(json.dumps(preds, ensure_ascii=False))
        else:
            output_rows.append(row["prediction"])

    base_df["prediction"] = output_rows
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out, index=False)
    print(f"[OK] Đã lưu file dự đoán hoàn tất: {out} ({len(base_df)} dòng)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Đường dẫn file trọng số best.pt")
    ap.add_argument("--test_dir", default="egde-ai-challenge-2026-lan-2-qualifiers/test/images")
    ap.add_argument("--base_sub", default="submission_champion_ensemble.csv")
    ap.add_argument("--out", default="submissions/submission_yolo_tta.csv")
    ap.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (lật ngang)")
    ap.add_argument("--conf", type=float, default=0.03, help="Ngưỡng confidence")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    run_inference(args.weights, args.test_dir, args.out, args.base_sub, tta=args.tta, conf=args.conf, batch=args.batch)


if __name__ == "__main__":
    main()
