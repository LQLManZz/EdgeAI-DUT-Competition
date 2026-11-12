"""Keras FP32 (best.keras) -> TFLite INT8, rồi kiểm tra tuân thủ và đo mAP50 val của chính file INT8.

Bài giảng: docs/03_training/learning_guide.md, Bài 8.

  python -m edgedet.quant.export_int8 --model experiments/h1c_144x192/best.keras --data data/raw
-> experiments/h1c_144x192/model_int8.tflite + int8_report.json

Lượng tử hóa sau train (PTQ): converter cho ~200 ảnh THẬT chạy qua model để đo khoảng giá trị (min/max)
của từng tensor, rồi chọn scale/zero_point sao cho khoảng đó vừa 256 mức của int8.
Ảnh giả/ngẫu nhiên cho khoảng giá trị sai -> mAP INT8 giảm mạnh.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgedet.data.dataset import image_batches, load_items, split_train_val
from edgedet.eval.evaluate import TFLiteDetector, evaluate, keras_predict_fn

ALLOWED_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "RESIZE_NEAREST_NEIGHBOR"}   # đã đo chạy được trên kit (GĐ1-2)
LIMITS = {"params": 200_000, "file_bytes": int(1.5 * 1024 * 1024)}


def convert(model, rep_items, n_rep=200):
    hw = tuple(model.input_shape[1:3])

    def rep():
        n = 0
        for x in image_batches(rep_items, hw, batch=1):
            yield [x.numpy()]
            n += 1
            if n >= n_rep:
                return

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def op_names(path):
    it = tf.lite.Interpreter(model_path=str(path))
    return sorted({op["op_name"] for op in it._get_ops_details()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="file .keras (thường là best.keras)")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", help="mặc định: model_int8.tflite cạnh file .keras")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_rep", type=int, default=300, help="số ảnh train dùng để calibration INT8")
    ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args(argv)

    model = tf.keras.models.load_model(a.model, compile=False)
    hw = tuple(model.input_shape[1:3])
    train_items, val_items = split_train_val(load_items(a.data, "train"), a.val_frac, a.seed)
    rng = np.random.default_rng(0)
    n_rep = min(a.n_rep, len(train_items))
    rep_items = [train_items[i] for i in rng.choice(len(train_items), n_rep, replace=False)]   # ảnh train, không dùng val

    out = Path(a.out) if a.out else Path(a.model).with_name("model_int8.tflite")
    out.write_bytes(convert(model, rep_items, n_rep=n_rep))

    ops = op_names(out)
    rep = {"tflite": str(out), "input_hw": list(hw), "params_keras": int(model.count_params()),
           "file_bytes": out.stat().st_size, "ops": ops}
    rep["checks"] = {
        "params<=200K": rep["params_keras"] <= LIMITS["params"],
        "file<=1.5MB": rep["file_bytes"] <= LIMITS["file_bytes"],
        "ops_subset_of_tested": set(ops) <= ALLOWED_OPS,
    }
    if not a.no_eval:
        m32, _, _ = evaluate(keras_predict_fn(model), val_items, hw)
        m8, per8, _ = evaluate(TFLiteDetector(out), val_items, hw)
        rep.update({"val_map50_fp32": round(m32, 4), "val_map50_int8": round(m8, 4),
                    "int8_drop": round(m32 - m8, 4), "val_ap_int8_per_class": {k: round(v, 4) for k, v in per8.items()}})
    Path(out).with_name("int8_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if not all(rep["checks"].values()):
        print("!! CÓ KIỂM TRA KHÔNG ĐẠT – xem 'checks' ở trên")


if __name__ == "__main__":
    main()
