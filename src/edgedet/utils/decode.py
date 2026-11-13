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


def _decode_coords(box, ys, xs, sy, sx, orig_hw):
    lw, lh, ox, oy = box[ys, xs].T
    cx, cy = (xs + ox) * STRIDE, (ys + oy) * STRIDE
    w, h = np.exp(np.clip(lw, -8.0, 8.0)) * STRIDE, np.exp(np.clip(lh, -8.0, 8.0)) * STRIDE
    return np.stack([
        np.clip((cx - w * 0.5) * sx, 0.0, orig_hw[1]),
        np.clip((cy - h * 0.5) * sy, 0.0, orig_hw[0]),
        np.clip((cx + w * 0.5) * sx, 0.0, orig_hw[1]),
        np.clip((cy + h * 0.5) * sy, 0.0, orig_hw[0]),
    ], axis=1)


def _pair_iou(b1, b2):
    xx1 = np.maximum(b1[:, 0], b2[:, 0]); yy1 = np.maximum(b1[:, 1], b2[:, 1])
    xx2 = np.minimum(b1[:, 2], b2[:, 2]); yy2 = np.minimum(b1[:, 3], b2[:, 3])
    inter = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def decode_center(hm, box, input_hw, orig_hw=(240, 320), score_thr=0.05, topk=100,
                  tie_break=True, nms_iou=0.65, score_smooth=0.1,
                  box_smooth=0.35, cross_nms_iou=None):
    """hm, box: float (đã dequantize), không có batch. Trả list dict {class_id, score, bbox[x1,y1,x2,y2]} trên ảnh gốc.

    Các cải tiến cho INT8 (mặc định bật):
      tie_break     : khi lượng tử hóa INT8 làm 2 ô cạnh nhau có cùng giá trị đỉnh (p == nb), chỉ giữ đúng 1 ô có
                      trung bình lân cận 3x3 lớn nhất (tâm đồi Gauss thật, nơi box được giám sát khi train).
      score_smooth  : trộn nhẹ (1 - α)*p + α*mean_3x3 để phá vỡ hiện tượng hàng nghìn box có cùng mức INT8 (1/256)
                      khi xếp hạng tính AP toàn cục.
      nms_iou       : lọc trùng theo từng lớp (None = tắt) khi vật lớn sinh ra 2 đỉnh cách nhau >= 2 ô lưới.
      box_smooth    : trộn thêm dự đoán box của 4 ô sát tâm (đã được dạy qua neighbor_weight/KDBoxLoss) để giảm nhiễu
                      lượng tử hóa INT8 và tinh chỉnh vị trí sub-pixel (0 = tắt).
      cross_nms_iou : giảm điểm box khác lớp trùng khít (IoU >= ngưỡng) với 1 đỉnh mạnh hơn (mặc định None = tắt
                      vì đánh giá AP độc lập từng lớp cần giữ lại ứng viên lớp thứ cấp).
    """
    p = _sigmoid(hm)
    hf, wf, c = p.shape
    pad = np.pad(p, ((1, 1), (1, 1), (0, 0)), constant_values=-1.0)
    win = np.stack([pad[dy:dy + hf, dx:dx + wf] for dy in range(3) for dx in range(3)])
    nb = np.max(win, axis=0)
    mean_3x3 = np.mean(np.maximum(win, 0.0), axis=0)
    if tie_break:
        idx_bias = (np.arange(hf)[:, None, None] * wf + np.arange(wf)[None, :, None]) * 1e-7
        ts = mean_3x3 + idx_bias
        pad_ts = np.pad(ts, ((1, 1), (1, 1), (0, 0)), constant_values=-1.0)
        win_ts = np.stack([pad_ts[dy:dy + hf, dx:dx + wf] for dy in range(3) for dx in range(3)])
        best_ts = np.max(np.where(win == nb, win_ts, -1.0), axis=0)
        peak = (p >= nb) & (ts >= best_ts) & (p >= score_thr)
    else:
        peak = (p >= nb) & (p >= score_thr)

    ys, xs, cs = np.nonzero(peak)
    if not len(ys):
        return []
    raw_s = p[ys, xs, cs]
    scores = (1.0 - score_smooth) * raw_s + score_smooth * mean_3x3[ys, xs, cs] if score_smooth else raw_s
    sy, sx = orig_hw[0] / input_hw[0], orig_hw[1] / input_hw[1]

    boxes = _decode_coords(box, ys, xs, sy, sx, orig_hw)
    if box_smooth and box_smooth > 0:
        b_sum = boxes.copy()
        w_sum = np.ones((len(ys), 1), dtype=np.float32)
        p_peak = raw_s
        thr_nb = np.maximum(score_thr, 0.20 * p_peak)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inb = (ny >= 0) & (ny < hf) & (nx >= 0) & (nx < wf)
            ny_c, nx_c = np.clip(ny, 0, hf - 1), np.clip(nx, 0, wf - 1)
            p_nb = p[ny_c, nx_c, cs]
            valid = inb & (p_nb >= thr_nb)
            if np.any(valid):
                nb_boxes = _decode_coords(box, ny_c[valid], nx_c[valid], sy, sx, orig_hw)
                same_obj = _pair_iou(nb_boxes, boxes[valid]) >= 0.30
                if np.any(same_obj):
                    v_idx = np.nonzero(valid)[0][same_obj]
                    w_nb = (box_smooth * (p_nb[v_idx] / (p_peak[v_idx] + 1e-7)))[:, None]
                    b_sum[v_idx] += w_nb * nb_boxes[same_obj]
                    w_sum[v_idx] += w_nb
        boxes = b_sum / w_sum

    if nms_iou is not None:
        keep = []
        for k in np.unique(cs):
            idx = np.where(cs == k)[0]
            for j in _nms(boxes[idx], scores[idx], nms_iou):
                keep.append(idx[j])
        keep = np.asarray(keep, dtype=np.int64)
        ys, xs, cs, scores, boxes = ys[keep], xs[keep], cs[keep], scores[keep], boxes[keep]

    if cross_nms_iou is not None and len(scores) > 1:
        ord_idx = np.argsort(-scores)
        boxes, scores, cs = boxes[ord_idx], scores[ord_idx], cs[ord_idx]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for i in range(len(scores)):
            if scores[i] < 0.30:
                break
            for j in range(i + 1, len(scores)):
                if cs[j] == cs[i] or scores[j] >= 0.75 * scores[i]:
                    continue
                xx1 = max(boxes[i, 0], boxes[j, 0]); yy1 = max(boxes[i, 1], boxes[j, 1])
                xx2 = min(boxes[i, 2], boxes[j, 2]); yy2 = min(boxes[i, 3], boxes[j, 3])
                inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                if inter / (areas[i] + areas[j] - inter + 1e-9) >= cross_nms_iou:
                    scores[j] *= 0.25
        keep_thr = scores >= score_thr
        boxes, scores, cs = boxes[keep_thr], scores[keep_thr], cs[keep_thr]

    order = np.argsort(-scores)[:topk]
    return [{"class_id": int(cs[i]), "score": round(float(scores[i]), 5),
             "bbox": [round(float(v), 1) for v in boxes[i]]} for i in order]


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
