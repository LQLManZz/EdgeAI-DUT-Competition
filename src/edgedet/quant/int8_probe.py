"""Vì sao cùng một config mà INT8 lúc mất 1,6 điểm, lúc mất 3,4 điểm? (#9 so với #10)

  python -m edgedet.quant.int8_probe --model experiments/h1p_pen_240x320_bw2/best.keras --data data/raw
-> int8_probe.txt cạnh file model (không ghi đè model_int8.tflite).

Hai câu hỏi, mỗi câu trả lời bằng vài con số mAP50 trên val:
  A. Mất ở ĐẦU RA NÀO? Ghép heatmap (lớp + điểm) và box (kích thước) lấy từ FP32 hoặc INT8:
       hm32+box8 ≈ FP32  -> box INT8 không sao, phần mất nằm ở heatmap (và ngược lại).
  B. Có phải do CALIBRATION? Convert lại với tập ảnh calibration khác (200 ảnh khác, và 1000 ảnh).
       mAP INT8 đổi nhiều -> scale phụ thuộc vài ảnh "lạ" (min/max nhạy với ngoại lai) -> sửa ở bước export.
       Gần như không đổi  -> lỗi nằm ở chính trọng số của lần train đó -> phải sửa lúc train.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgedet.data.dataset import ORIG_HW, gts_of, image_batches, load_items, split_train_val
from edgedet.eval.evaluate import TFLiteDetector, evaluate, keras_predict_fn
from edgedet.eval.map50 import map50
from edgedet.quant.export_int8 import convert
from edgedet.utils.decode import decode_center


def raw_outputs(fn, items, hw):
    hm, box = [], []
    for x in image_batches(items, hw, 32):
        h, b = fn(x.numpy())
        hm += list(h); box += list(b)
    return np.stack(hm), np.stack(box)


def map_of(hm, box, items, hw, **decode_kw):
    preds = {it["id"]: decode_center(hm[i], box[i], hw, ORIG_HW, 0.05, 100, **decode_kw) for i, it in enumerate(items)}
    return map50(preds, gts_of(items))[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/raw")
    a = ap.parse_args(argv)

    model = tf.keras.models.load_model(a.model, compile=False)
    hw = tuple(model.input_shape[1:3])
    train_items, val = split_train_val(load_items(a.data, "train"))
    tfl = Path(a.model).with_name("model_int8.tflite")
    lines = []
    out = lambda s: (lines.append(s), print(s, flush=True))

    # A. ghép đầu ra FP32 / INT8
    h32, b32 = raw_outputs(keras_predict_fn(model), val, hw)
    h8, b8 = raw_outputs(TFLiteDetector(tfl), val, hw)
    out("A. Ghép đầu ra (mAP50 val, decode mới):")
    out(f"   hm32+box32 = {map_of(h32, b32, val, hw):.4f}   (FP32)")
    out(f"   hm32+box8  = {map_of(h32, b8, val, hw):.4f}   (chỉ box là INT8)")
    out(f"   hm8+box32  = {map_of(h8, b32, val, hw):.4f}   (chỉ heatmap là INT8)")
    out(f"   hm8+box8   = {map_of(h8, b8, val, hw):.4f}   (INT8, file nộp)")
    d = np.abs(h8 - h32)
    peak = h32 > 0.3
    out(f"   |hm8-hm32|: trung bình {d.mean():.4f}, p99 {np.percentile(d, 99):.4f}, lớn nhất {d.max():.3f}")
    out(f"   tại các đỉnh FP32 > 0.3 ({peak.sum()} điểm): hm32 trung bình {h32[peak].mean():.3f}, hm8 {h8[peak].mean():.3f}")
    out(f"   |box8-box32| trung bình {np.abs(b8 - b32).mean():.3f}")

    # B. tác dụng của từng bước sửa decode trên INT8 (không cần train lại)
    out("\nB. Tối ưu Decode trên INT8 (hm8+box8):")
    m_old = map_of(h8, b8, val, hw, tie_break=False, nms_iou=None, score_smooth=0.0, box_smooth=0.0, cross_nms_iou=None)
    m_tie = map_of(h8, b8, val, hw, tie_break=True, nms_iou=None, score_smooth=0.0, box_smooth=0.0, cross_nms_iou=None)
    m_sm = map_of(h8, b8, val, hw, tie_break=True, nms_iou=None, score_smooth=0.1, box_smooth=0.0, cross_nms_iou=None)
    m_nms = map_of(h8, b8, val, hw, tie_break=True, nms_iou=0.65, score_smooth=0.1, box_smooth=0.0, cross_nms_iou=None)
    m_new = map_of(h8, b8, val, hw)
    out(f"   decode cũ (p >= nb, không NMS)          = {m_old:.4f}")
    out(f"   + tie_break 3x3 (chọn đúng 1 đỉnh INT8) = {m_tie:.4f}   ({m_tie - m_old:+.4f})")
    out(f"   + score_smooth 3x3 (phá đồng hạng AP)   = {m_sm:.4f}   ({m_sm - m_old:+.4f})")
    out(f"   decode GĐ5 (+ NMS cùng lớp 0.65)        = {m_nms:.4f}   ({m_nms - m_old:+.4f})")
    out(f"   decode GĐ6 (+ box_smooth 4-lân cận)     = {m_new:.4f}   ({m_new - m_old:+.4f})")
    out(f"   Tóm tắt: decode cũ -> {m_old:.4f} | decode GĐ5 -> {m_nms:.4f} | decode GĐ6 (box_smooth) -> {m_new:.4f}")

    # C. calibration khác (file tạm, không đụng model_int8.tflite)
    out("\nC. Convert lại với calibration khác (mAP50 INT8 val, decode mới):")
    out(f"   200 ảnh, seed 0 (file nộp hiện tại) = {m_new:.4f}")
    for name, seed, n in [("200 ảnh khác, seed 1", 1, 200), ("500 ảnh, seed 2", 2, 500)]:
        idx = np.random.default_rng(seed).choice(len(train_items), n, replace=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.tflite"
            p.write_bytes(convert(model, [train_items[i] for i in idx], n_rep=n))
            out(f"   {name:<22} = {evaluate(TFLiteDetector(p), val, hw)[0]:.4f}")

    txt = Path(a.model).with_name("int8_probe.txt")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> {txt}")


if __name__ == "__main__":
    main()
