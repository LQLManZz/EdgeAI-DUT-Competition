"""mAP@0.5 – thước đo chính của vòng loại Kaggle.

Bài giảng: docs/03_training/learning_guide.md, Bài 1 (IoU) và Bài 2 (mAP).

Định dạng dùng chung trong cả dự án:
  preds[image_id] = [{"class_id": int, "score": float, "bbox": [x1, y1, x2, y2]}, ...]   # model đoán
  gts[image_id]   = [{"class_id": int, "bbox": [x1, y1, x2, y2]}, ...]                    # nhãn thật
Tọa độ theo pixel trên ảnh gốc 320x240 – đúng như file submission.

Chưa biết BTC tính AP kiểu nào (câu hỏi 6 trong docs/00_contest_constraints.md), nên có 2 kiểu:
  "all"     – nội suy mọi điểm (Pascal VOC 2010+), mặc định.
  "coco101" – lấy mẫu 101 mức recall (COCO). Hai kiểu thường lệch nhau < 1 điểm.
"""
import numpy as np


def iou(a, b):
    """IoU của 2 box [x1, y1, x2, y2]. Giao = hình chữ nhật chung; hợp = tổng diện tích - giao."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))     # chiều rộng phần giao (0 nếu không chạm)
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))     # chiều cao phần giao
    inter = ix * iy
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def _ap_from_pr(recall, precision, method):
    """Diện tích dưới đường precision-recall."""
    if method == "coco101":
        # với mỗi mức recall r = 0, 0.01, ..., 1: lấy precision lớn nhất đạt được khi recall >= r
        out = []
        for r in np.linspace(0, 1, 101):
            p = precision[recall >= r]
            out.append(p.max() if p.size else 0.0)
        return float(np.mean(out))
    # "all": làm đường precision thành bậc thang không tăng (nhìn từ phải sang), rồi cộng diện tích từng bậc
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def average_precision(preds, gts, class_id, iou_thr=0.5, method="all"):
    """AP của một lớp. Trả (ap, số box thật của lớp) – ap=None nếu lớp không có box thật nào."""
    # 1) gom box thật của lớp này theo ảnh; cờ "used" đánh dấu box thật đã bị một box đoán nhận
    gt_by_img = {k: [g["bbox"] for g in v if g["class_id"] == class_id] for k, v in gts.items()}
    used = {k: [False] * len(v) for k, v in gt_by_img.items()}
    n_gt = sum(len(v) for v in gt_by_img.values())
    if n_gt == 0:
        return None, 0

    # 2) gom mọi box đoán của lớp này (trên tất cả ảnh), xếp score từ cao xuống thấp
    dets = [(d["score"], k, d["bbox"]) for k, v in preds.items() for d in v if d["class_id"] == class_id]
    dets.sort(key=lambda t: -t[0])

    # 3) đi từng box đoán: khớp với box thật chưa dùng có IoU lớn nhất; >= ngưỡng -> TP, ngược lại FP
    tp = np.zeros(len(dets))
    for i, (_, k, box) in enumerate(dets):
        cands = gt_by_img.get(k, [])
        if not cands:
            continue                                       # ảnh không có vật lớp này -> FP
        ious = [iou(box, g) for g in cands]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not used[k][j]:
            tp[i], used[k][j] = 1, True                    # (nếu box thật đã bị nhận -> box này là FP trùng)

    # 4) precision/recall tích lũy theo thứ tự score
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(dets) + 1) if dets else np.zeros(0)
    return _ap_from_pr(recall, precision, method), n_gt


def map50(preds, gts, num_classes=7, method="all"):
    """Trả (mAP, {class_id: AP}). Lớp không có box thật nào trong gts thì bỏ qua khi lấy trung bình."""
    per_class = {}
    for c in range(num_classes):
        ap, _ = average_precision(preds, gts, c, 0.5, method)
        if ap is not None:
            per_class[c] = ap
    m = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return m, per_class
