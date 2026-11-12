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


def map_of(hm, box, items, hw):
    preds = {it["id"]: decode_center(hm[i], box[i], hw, ORIG_HW, 0.05, 100) for i, it in enumerate(items)}
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
    out("A. Ghép đầu ra (mAP50 val):")
    out(f"   hm32+box32 = {map_of(h32, b32, val, hw):.4f}   (FP32)")
    out(f"   hm32+box8  = {map_of(h32, b8, val, hw):.4f}   (chỉ box là INT8)")
    out(f"   hm8+box32  = {map_of(h8, b32, val, hw):.4f}   (chỉ heatmap là INT8)")
    out(f"   hm8+box8   = {map_of(h8, b8, val, hw):.4f}   (INT8, file nộp)")
    d = np.abs(h8 - h32)
    peak = h32 > 0.3
    out(f"   |hm8-hm32|: trung bình {d.mean():.4f}, p99 {np.percentile(d, 99):.4f}, lớn nhất {d.max():.3f}")
    out(f"   tại các đỉnh FP32 > 0.3 ({peak.sum()} điểm): hm32 trung bình {h32[peak].mean():.3f}, hm8 {h8[peak].mean():.3f}")
    out(f"   |box8-box32| trung bình {np.abs(b8 - b32).mean():.3f}")

    # B. calibration khác (file tạm, không đụng model_int8.tflite)
    out("\nB. Convert lại với calibration khác (mAP50 INT8 val):")
    out("   200 ảnh, seed 0 (file nộp hiện tại) = xem dòng hm8+box8 ở trên")
    for name, seed, n in [("200 ảnh khác, seed 1", 1, 200), ("1000 ảnh, seed 2", 2, 1000)]:
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
