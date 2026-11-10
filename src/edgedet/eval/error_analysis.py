"""Phân tích lỗi: phần mAP50 còn thiếu nằm ở đâu? (bài 10)

  python -m edgedet.eval.error_analysis --tflite experiments/h1p_pen_192x256/model_int8.tflite --data data/raw
-> error_analysis.txt + errors.png cạnh file model, và in kết quả ra màn hình.

Dùng CHÍNH file INT8 (model nộp). Ba câu hỏi, mỗi câu trả lời bằng một con số:
  1. Box LỆCH?   mAP khi nới IoU xuống 0.1. Tăng nhiều -> model tìm đúng vật nhưng box chưa khít
                 (sửa: head box, loss box, độ phân giải).
  2. NHẦM LỚP?   mAP khi bỏ qua lớp (gộp 7 lớp làm 1). Tăng nhiều -> vật được tìm thấy nhưng gán sai lớp
                 (sửa: backbone/đặc trưng, augmentation màu, dữ liệu lớp hay bị nhầm).
  3. BỎ SÓT?     % vật thật không có box đoán CÙNG LỚP nào gần (IoU ≥ 0.1) trong top 100: model không thấy vật,
                 hoặc chỉ thấy với lớp sai (đối chiếu bảng nhầm lớp). Sửa: độ phân giải, vật nhỏ/bị che.
Thêm: recall theo cỡ vật, bảng lớp hay bị nhầm thành lớp nào, và ảnh các lỗi tệ nhất.
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from edgedet.data.dataset import CLASS_NAMES, gts_of, load_items, read_image, split_train_val
from edgedet.eval.evaluate import TFLiteDetector, predict_items
from edgedet.eval.map50 import average_precision, iou
from edgedet.utils.viz import draw, grid

SIZES = [("nhỏ  (<28 px)", 0, 28), ("vừa (28–56)", 28, 56), ("lớn  (≥56)", 56, 1e9)]   # cạnh = sqrt(diện tích)


def mean_ap(preds, gts, iou_thr=0.5, num_classes=7):
    aps = [average_precision(preds, gts, c, iou_thr)[0] for c in range(num_classes)]
    return float(np.mean([a for a in aps if a is not None]))


def agnostic(d):
    return {k: [{**b, "class_id": 0} for b in v] for k, v in d.items()}


def side(b):
    return float(np.sqrt(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])))


def analyze(preds, gts, score_thr=0.3):
    lines = []
    out = lines.append
    base = mean_ap(preds, gts)
    loose = mean_ap(preds, gts, 0.1)
    agn = mean_ap(agnostic(preds), agnostic(gts), 0.5, num_classes=1)
    out(f"mAP50 (INT8)                       = {base:.4f}")
    out(f"1. nới IoU xuống 0.1 (box lệch)     = {loose:.4f}   (+{loose - base:.4f})")
    out(f"2. bỏ qua lớp (nhầm lớp)            = {agn:.4f}   (+{agn - base:.4f})")

    # 3. bỏ sót + recall theo cỡ + bảng nhầm lớp (với box đoán score >= score_thr)
    miss = Counter(); n_cls = Counter(); size_hit = Counter(); size_n = Counter()
    confuse = Counter(); per_img_err = {}
    for k, gl in gts.items():
        pl = preds.get(k, [])
        strong = [p for p in pl if p["score"] >= score_thr]
        err = 0
        for g in gl:
            c = g["class_id"]; n_cls[c] += 1
            same = [iou(p["bbox"], g["bbox"]) for p in pl if p["class_id"] == c]
            if not same or max(same) < 0.1:
                miss[c] += 1
            s = next(i for i, (_, lo, hi) in enumerate(SIZES) if lo <= side(g["bbox"]) < hi)
            size_n[s] += 1
            hit = any(p["class_id"] == c and iou(p["bbox"], g["bbox"]) >= 0.5 for p in strong)
            size_hit[s] += hit
            if not hit:
                err += 1
                # box đoán mạnh nhất chồng lên vật này nhưng sai lớp?
                other = [(iou(p["bbox"], g["bbox"]), p["class_id"]) for p in strong if p["class_id"] != c]
                other = [o for o in other if o[0] >= 0.5]
                if other:
                    confuse[(c, max(other)[1])] += 1
        # box đoán mạnh không khớp vật thật nào (báo động giả)
        err += sum(1 for p in strong
                   if not any(p["class_id"] == g["class_id"] and iou(p["bbox"], g["bbox"]) >= 0.5 for g in gl))
        per_img_err[k] = err
    tot_miss, tot = sum(miss.values()), sum(n_cls.values())
    out(f"3. vật không có box ĐÚNG LỚP nào gần = {tot_miss}/{tot} ({100 * tot_miss / tot:.1f}%)")
    out("   theo lớp: " + "  ".join(f"{CLASS_NAMES[c]} {miss[c]}/{n_cls[c]}" for c in sorted(n_cls)))
    out(f"\nRecall@IoU0.5 với box score ≥ {score_thr}, theo cỡ vật:")
    for i, (name, _, _) in enumerate(SIZES):
        if size_n[i]:
            out(f"   {name}: {size_hit[i] / size_n[i]:.3f}   ({size_n[i]} vật)")
    out(f"\nNhầm lớp hay gặp nhất (vật thật -> bị đoán thành, score ≥ {score_thr}):")
    for (a, b), n in confuse.most_common(8):
        out(f"   {CLASS_NAMES[a]:>8} -> {CLASS_NAMES[b]:<8} {n}")
    if not confuse:
        out("   (không có)")
    return "\n".join(lines), per_img_err


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--score_thr", type=float, default=0.3, help="ngưỡng để coi là 'model khẳng định' (cho phần 3, ảnh)")
    a = ap.parse_args(argv)

    det = TFLiteDetector(a.tflite)
    _, val_items = split_train_val(load_items(a.data, "train"))
    preds = predict_items(det, val_items, det.input_hw)
    gts = gts_of(val_items)
    report, per_img_err = analyze(preds, gts, a.score_thr)
    out = Path(a.tflite).with_name("error_analysis.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(report)

    # 8 ảnh nhiều lỗi nhất: trái = nhãn thật, phải = model đoán (score >= ngưỡng)
    worst = sorted(val_items, key=lambda it: -per_img_err[it["id"]])[:8]
    tiles = []
    for it in worst:
        img = read_image(it["path"]).numpy()
        tiles += [draw(img, gts[it["id"]], 2), draw(img, [p for p in preds[it["id"]] if p["score"] >= a.score_thr], 2)]
    png = out.with_name("errors.png")
    cv2.imwrite(str(png), cv2.cvtColor(grid(tiles, 4), cv2.COLOR_RGB2BGR))
    print(f"\n-> {out}\n-> {png}  (mỗi cặp: trái = nhãn thật, phải = model đoán)")


if __name__ == "__main__":
    main()
