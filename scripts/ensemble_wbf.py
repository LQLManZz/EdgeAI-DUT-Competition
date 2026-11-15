"""Weighted Boxes Fusion (WBF) Ensemble cho Object Detection.
Kết hợp kết quả dự đoán của CenterNet và Micro-YOLO để tối đa hóa điểm số Kaggle Leaderboard.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_iou(box1, box2):
    """box = [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    boxBArea = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def wbf_single_image(predictions_list, weights, iou_thr=0.5, conf_type="avg"):
    """Thực hiện Weighted Boxes Fusion trên 1 ảnh từ nhiều mô hình dự đoán.
    predictions_list: list of list of dict(class_id, score, bbox=[x1, y1, x2, y2])
    weights: list of floats, e.g. [0.6, 0.4]
    """
    total_models = len(predictions_list)
    sum_weights = sum(weights)

    # Gom tất cả boxes theo từng class
    classes_boxes = {}
    for m_idx, preds in enumerate(predictions_list):
        w = weights[m_idx]
        for p in preds:
            cls = p["class_id"]
            if cls not in classes_boxes:
                classes_boxes[cls] = []
            classes_boxes[cls].append({
                "bbox": np.array(p["bbox"], dtype=np.float32),
                "score": float(p["score"]),
                "weight": w,
                "model_idx": m_idx
            })

    fused_predictions = []

    for cls, b_list in classes_boxes.items():
        # Sắp xếp theo score giảm dần
        b_list.sort(key=lambda x: -x["score"])
        clusters = []  # list of list of boxes

        for b in b_list:
            matched_cluster = None
            for cl in clusters:
                # Tính IoU với box trung bình của cluster
                cluster_box = np.mean([item["bbox"] for item in cl], axis=0)
                if compute_iou(b["bbox"], cluster_box) >= iou_thr:
                    matched_cluster = cl
                    break
            if matched_cluster is not None:
                matched_cluster.append(b)
            else:
                clusters.append([b])

        # Hợp nhất từng cluster
        for cl in clusters:
            weights_arr = np.array([item["weight"] * item["score"] for item in cl])
            w_sum = np.sum(weights_arr)

            if w_sum > 0:
                fused_bbox = np.sum([item["bbox"] * (item["weight"] * item["score"]) for item in cl], axis=0) / w_sum
            else:
                fused_bbox = np.mean([item["bbox"] for item in cl], axis=0)

            # Tính score hợp nhất
            models_in_cluster = {item["model_idx"] for item in cl}
            total_score = sum(item["weight"] * item["score"] for item in cl)
            # Thưởng cho box được phát hiện bởi nhiều model, phạt box chỉ 1 model thấy
            model_agreement_factor = len(models_in_cluster) / float(total_models)
            fused_score = (total_score / sum_weights) * (0.8 + 0.2 * model_agreement_factor)

            fused_predictions.append({
                "class_id": int(cls),
                "score": round(float(np.clip(fused_score, 0.0, 1.0)), 5),
                "bbox": [round(float(x), 2) for x in fused_bbox]
            })

    # Sắp xếp kết quả cuối cùng theo score giảm dần và giới hạn tối đa 100 box theo chuẩn BTC
    fused_predictions.sort(key=lambda x: -x["score"])
    return fused_predictions[:100]


def ensemble_submissions(sub_paths, weights, out_path, iou_thr=0.5):
    print(f"[*] Đang thực hiện WBF Ensemble trên {len(sub_paths)} file submission:")
    for p, w in zip(sub_paths, weights):
        print(f"    - {p} (Trọng số: {w})")

    dfs = [pd.read_csv(p) for p in sub_paths]
    base_df = dfs[0].copy()

    # Tối ưu hóa: tiền xử lý chuyển toàn bộ detection thành dict O(1) để tăng tốc 100 lần
    parsed_dfs = []
    for df in dfs:
        det_map = {}
        det_df = df[df["task_type"] == "DETECTION"]
        for s_id, pred_val in zip(det_df["sample_id"], det_df["prediction"]):
            try:
                det_map[s_id] = json.loads(pred_val) if isinstance(pred_val, str) else []
            except Exception:
                det_map[s_id] = []
        parsed_dfs.append(det_map)

    fused_rows = []
    for idx, row in base_df.iterrows():
        sample_id = row["sample_id"]
        task_type = row["task_type"]

        if task_type == "DETECTION":
            preds_list = [p_map.get(sample_id, []) for p_map in parsed_dfs]
            fused_preds = wbf_single_image(preds_list, weights, iou_thr=iou_thr)
            fused_rows.append(json.dumps(fused_preds, ensure_ascii=False))
        else:
            # INTENT: Giữ nguyên từ model đầu tiên (mô hình chuẩn/tốt nhất)
            fused_rows.append(row["prediction"])

    base_df["prediction"] = fused_rows
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(out, index=False)
    print(f"[OK] Đã xuất file Ensemble thành công: {out} ({len(base_df)} dòng)")


def main():
    ap = argparse.ArgumentParser(description="Weighted Boxes Fusion Ensemble")
    ap.add_argument("--subs", nargs="+", required=True, help="Danh sách các file submission.csv")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="Trọng số tương ứng (vd. 0.65 0.35)")
    ap.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU ghép box (mặc định 0.5)")
    ap.add_argument("--out", default="submissions/submission_ensemble.csv", help="File đầu ra")
    args = ap.parse_args()

    weights = args.weights
    if not weights:
        weights = [1.0 / len(args.subs)] * len(args.subs)
    assert len(weights) == len(args.subs), "Số lượng trọng số phải bằng số file submission"

    ensemble_submissions(args.subs, weights, args.out, iou_thr=args.iou)


if __name__ == "__main__":
    main()
