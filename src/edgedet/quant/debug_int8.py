"""Đo sai số INT8 TỪNG LỚP: tìm xem mAP bị mất khi lượng tử hóa nằm ở lớp nào.

Bài giảng: docs/03_training/learning_guide.md, Bài 8.

  python -m edgedet.quant.debug_int8 --model experiments/h1p_144x192/best.keras --data data/raw
-> int8_layer_debug.csv cạnh file .keras + in ra 12 lớp tệ nhất.

Dùng tf.lite.experimental.QuantizationDebugger: chạy cùng ảnh qua model float và model INT8,
so sánh output của TỪNG lớp. Hai chỉ số chính:
  range      = 255 * scale: khoảng giá trị mà 256 mức int8 phải phủ. Range càng rộng, mỗi mức càng thô.
  rmse/scale = sai số trung bình tính theo "số mức int8". Lý tưởng ≈ 0.29 (chỉ do làm tròn);
               >> 0.29 nghĩa là lớp đó có giá trị bị kẹp (outlier) hoặc sai số dồn từ các lớp trước.
CHÚ Ý: rmse/scale đã chia cho scale nên KHÔNG thấy vấn đề "range quá rộng". Phải xem cả range và max_abs_error.
  Đã đo: H1-C range ≤ ~35, max_abs_error ≤ ~0.2; H1-P (#2) range tới 309, max_abs_error tới 2.3 -> mất 4 điểm mAP.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgedet.data.dataset import image_batches, load_items, split_train_val


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--n_rep", type=int, default=200, help="số ảnh train để chọn scale (giống export_int8)")
    ap.add_argument("--n_debug", type=int, default=32, help="số ảnh val để đo sai số")
    a = ap.parse_args(argv)

    model = tf.keras.models.load_model(a.model, compile=False)
    hw = tuple(model.input_shape[1:3])
    train_items, val_items = split_train_val(load_items(a.data, "train"))
    rng = np.random.default_rng(0)
    rep_items = [train_items[i] for i in rng.choice(len(train_items), a.n_rep, replace=False)]   # giống export_int8

    def gen(items):
        def g():
            for x in image_batches(items, hw, batch=1):
                yield [x.numpy()]
        return g

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen(rep_items)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    dbg = tf.lite.experimental.QuantizationDebugger(converter=conv, debug_dataset=gen(val_items[:a.n_debug]))
    dbg.run()
    out = Path(a.model).with_name("int8_layer_debug.csv")
    with open(out, "w", encoding="utf-8") as f:
        dbg.layer_statistics_dump(f)

    import pandas as pd
    df = pd.read_csv(out)
    df["range"] = 255.0 * df["scale"]
    df["rmse/scale"] = np.sqrt(df["mean_squared_error"]) / df["scale"]
    # "model/b3_proj_bn_1/batchnorm/add_1;..." -> "b3_proj_bn"   (tên lớp Keras, dễ đối chiếu với edgedet.py)
    short = lambda t: (t.split(";")[0].split("/")[1] if "/" in t else t).removesuffix("_1")
    df.insert(1, "layer", df["tensor_name"].map(short))
    df.to_csv(out, index=False)
    cols = ["op_name", "layer", "range", "rmse/scale", "max_abs_error"]
    pd.set_option("display.width", 160)
    print(f"{len(df)} lớp. rmse/scale trung vị = {df['rmse/scale'].median():.3f} (lý tưởng ≈ 0.29)\n")
    print("== 12 lớp có rmse/scale lớn nhất (sai số lớn nhất tính theo mức int8):")
    print(df.sort_values("rmse/scale", ascending=False)[cols].head(12).to_string(index=False))
    print("\n== 12 lớp có range lớn nhất (activation trải rộng nhất):")
    print(df.sort_values("range", ascending=False)[cols].head(12).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
