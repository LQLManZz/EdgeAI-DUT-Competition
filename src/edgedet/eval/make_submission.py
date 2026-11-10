"""Tạo submission.csv: dòng DETECTION từ file INT8 + dòng INTENT của nhóm TextCNN.

Bài giảng: docs/03_training/learning_guide.md, Bài 9.

  python -m edgedet.eval.make_submission --tflite experiments/h1c_144x192/model_int8.tflite \
      --data data/raw --intent path/to/intent_submission.csv --out submission.csv

--intent: file CSV của nhóm TextCNN, có cột task_type,sample_id,prediction (chỉ cần các dòng INTENT).
Thiếu --intent: dòng INTENT lấy NGUYÊN từ sample_submission.csv (nhãn mẫu, gần như chắc chắn sai)
-> file vẫn hợp lệ để thử nộp detection, nhưng mất phần lớn 30 điểm TextCNN.

Dự đoán dùng CHÍNH file .tflite INT8 (không dùng Keras FP32): đó là model sẽ nộp cho BTC kiểm tra,
nên điểm Kaggle phải đến từ nó.
"""
import argparse
import csv
import json
from pathlib import Path

from edgedet.data.dataset import ORIG_HW, load_items
from edgedet.eval.evaluate import TFLiteDetector, keras_predict_fn, predict_items

INTENTS = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(rows, sample_rows):
    """Kiểm tra định dạng theo thể lệ. Trả list lỗi (rỗng = hợp lệ)."""
    err = []
    want = [(r["task_type"], r["sample_id"]) for r in sample_rows]
    got = [(r["task_type"], r["sample_id"]) for r in rows]
    if len(set(got)) != len(got):
        err.append("có sample_id bị lặp")
    if set(got) != set(want):
        err.append(f"tập (task_type, sample_id) khác sample_submission: thiếu {len(set(want) - set(got))}, thừa {len(set(got) - set(want))}")
    H, W = ORIG_HW
    for r in rows:
        if r["task_type"] == "DETECTION":
            try:
                dets = json.loads(r["prediction"])
            except json.JSONDecodeError:
                err.append(f"{r['sample_id']}: JSON lỗi"); continue
            if len(dets) > 100:
                err.append(f"{r['sample_id']}: {len(dets)} box > 100")
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                if not (0 <= d["class_id"] <= 6 and 0 <= d["score"] <= 1 and 0 <= x1 <= x2 <= W and 0 <= y1 <= y2 <= H):
                    err.append(f"{r['sample_id']}: box sai {d}"); break
        elif r["prediction"] not in INTENTS:
            err.append(f"{r['sample_id']}: intent '{r['prediction']}' không hợp lệ")
        if len(err) > 20:
            err.append("..."); break
    return err


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tflite", help="model INT8 (khuyến nghị – đúng model sẽ nộp)")
    g.add_argument("--keras", help="model .keras FP32 (chỉ để thử nhanh)")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--intent", help="CSV của nhóm TextCNN")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--score_thr", type=float, default=0.05)
    a = ap.parse_args(argv)

    if a.tflite:
        fn = TFLiteDetector(a.tflite); hw = fn.input_hw
    else:
        import tensorflow as tf
        m = tf.keras.models.load_model(a.keras, compile=False)
        fn, hw = keras_predict_fn(m), tuple(m.input_shape[1:3])

    test_items = load_items(a.data, "test")
    print(f"Dự đoán {len(test_items)} ảnh test, input {hw} ...")
    preds = predict_items(fn, test_items, hw, batch=64, score_thr=a.score_thr, topk=100)

    sample = read_rows(Path(a.data) / "sample_submission.csv")
    intent = {}
    if a.intent:
        intent = {r["sample_id"]: r["prediction"] for r in read_rows(a.intent) if r["task_type"] == "INTENT"}
    else:
        print("!! Không có --intent: dòng INTENT lấy từ sample_submission (nhãn mẫu).")

    rows = []
    for r in sample:                                  # giữ nguyên thứ tự và sample_id của file mẫu
        if r["task_type"] == "DETECTION":
            p = json.dumps(preds.get(r["sample_id"], []), separators=(",", ":"))
        else:
            p = intent.get(r["sample_id"], r["prediction"])
        rows.append({"task_type": r["task_type"], "sample_id": r["sample_id"], "prediction": p})

    err = validate(rows, sample)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_type", "sample_id", "prediction"])
        w.writeheader(); w.writerows(rows)
    n_box = sum(len(v) for v in preds.values())
    print(f"Đã ghi {a.out}: {len(rows)} dòng, {n_box} box ({n_box / max(1, len(preds)):.1f} box/ảnh)")
    print("HỢP LỆ" if not err else "LỖI ĐỊNH DẠNG:\n  " + "\n  ".join(err))


if __name__ == "__main__":
    main()
