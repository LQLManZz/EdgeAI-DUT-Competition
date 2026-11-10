"""Decode output EdgeDet (CenterNet head) -> danh sách box theo định dạng submission Kaggle.

Bản tham chiếu Python (dùng cho đánh giá trên PC). Bản C trên ESP32-S3 phải cho cùng kết quả (GĐ5).
Quy ước output (xem src/edgedet/models/edgedet.py::head_center):
  hm  (Hf, Wf, C): logits heatmap, stride 8
  box (Hf, Wf, 4): [log(w/8), log(h/8), off_x, off_y], w,h theo pixel của ảnh input model
"""
import numpy as np

STRIDE = 8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_center(hm, box, input_hw, orig_hw=(240, 320), score_thr=0.05, topk=100):
    """hm, box: float (đã dequantize), không có batch. Trả list dict {class_id, score, bbox[x1,y1,x2,y2]} trên ảnh gốc."""
    p = _sigmoid(hm)
    # NMS kiểu CenterNet: giữ điểm là cực đại trong lân cận 3x3 (cùng lớp)
    hf, wf, c = p.shape
    pad = np.pad(p, ((1, 1), (1, 1), (0, 0)), constant_values=-1)
    nb = np.max(np.stack([pad[dy:dy + hf, dx:dx + wf] for dy in range(3) for dx in range(3)]), axis=0)
    peak = (p >= nb) & (p >= score_thr)
    ys, xs, cs = np.nonzero(peak)
    scores = p[ys, xs, cs]
    order = np.argsort(-scores)[:topk]
    sy, sx = orig_hw[0] / input_hw[0], orig_hw[1] / input_hw[1]
    out = []
    for i in order:
        y, x, k = ys[i], xs[i], cs[i]
        lw, lh, ox, oy = box[y, x]
        cx, cy = (x + ox) * STRIDE, (y + oy) * STRIDE
        w, h = np.exp(lw) * STRIDE, np.exp(lh) * STRIDE
        x1 = float(np.clip((cx - w / 2) * sx, 0, orig_hw[1])); x2 = float(np.clip((cx + w / 2) * sx, 0, orig_hw[1]))
        y1 = float(np.clip((cy - h / 2) * sy, 0, orig_hw[0])); y2 = float(np.clip((cy + h / 2) * sy, 0, orig_hw[0]))
        out.append({"class_id": int(k), "score": round(float(scores[i]), 4), "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return out


def _nms(boxes, scores, iou_thr):
    order, keep = np.argsort(-scores), []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def decode_yolo(out, anchors, input_hw, orig_hw=(240, 320), score_thr=0.05, iou_thr=0.5, topk=100):
    """H2: out (Hf, Wf, A*(5+C)) float; anchor (w, h) theo pixel input. Mỗi anchor: [tx, ty, tw, th, obj, cls...].
    NMS theo từng lớp. Trả cùng định dạng decode_center."""
    hf, wf, ch = out.shape
    a = len(anchors)
    c = ch // a - 5
    o = out.reshape(hf, wf, a, 5 + c)
    gy, gx = np.meshgrid(np.arange(hf), np.arange(wf), indexing="ij")
    cx = (_sigmoid(o[..., 0]) + gx[..., None]) * STRIDE
    cy = (_sigmoid(o[..., 1]) + gy[..., None]) * STRIDE
    an = np.asarray(anchors, dtype=np.float32)
    w, h = np.exp(np.clip(o[..., 2], -8, 8)) * an[:, 0], np.exp(np.clip(o[..., 3], -8, 8)) * an[:, 1]
    score = _sigmoid(o[..., 4:5]) * _sigmoid(o[..., 5:])                       # (hf, wf, a, c)
    sy, sx = orig_hw[0] / input_hw[0], orig_hw[1] / input_hw[1]
    b = np.stack([(cx - w / 2) * sx, (cy - h / 2) * sy, (cx + w / 2) * sx, (cy + h / 2) * sy], -1)
    b = np.clip(b, 0, [orig_hw[1], orig_hw[0], orig_hw[1], orig_hw[0]])
    res = []
    for k in range(c):
        m = score[..., k] >= score_thr
        if not m.any():
            continue
        bk, sk = b[m], score[..., k][m]
        for i in _nms(bk, sk, iou_thr):
            res.append({"class_id": k, "score": round(float(sk[i]), 4), "bbox": [round(float(v), 1) for v in bk[i]]})
    return sorted(res, key=lambda d: -d["score"])[:topk]
