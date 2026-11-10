"""Chạy model trên một tập ảnh -> danh sách box (tọa độ ảnh gốc) -> mAP50.

Dùng chung cho model Keras (FP32) và file .tflite INT8: chỉ cần một hàm `predict_fn(x) -> (hm, box)`.
Nhờ vậy số mAP của FP32 và INT8 được tính đúng cùng một cách và so sánh trực tiếp được.
"""
import numpy as np
import tensorflow as tf

from edgedet.data.dataset import ORIG_HW, gts_of, image_batches
from edgedet.eval.map50 import map50
from edgedet.utils.decode import decode_center


def keras_predict_fn(model):
    """Model Keras -> hàm dự đoán theo batch. training=False: BatchNorm dùng thống kê trượt (giống trên chip)."""
    @tf.function(reduce_retracing=True)
    def f(x):
        hm, box = model(x, training=False)
        return hm, box
    return lambda x: [t.numpy() for t in f(x)]


class TFLiteDetector:
    """Chạy file .tflite INT8 trên PC – giống hệt phép tính chip sẽ làm (trừ khác biệt làm tròn rất nhỏ).

    Input: float [-1,1] -> lượng tử hóa: q = round(x / scale + zero_point), kẹp vào [-128, 127].
    Output: int8 -> giải lượng tử: x = (q - zero_point) * scale. Hai output phân biệt bằng số kênh (7 = hm, 4 = box).
    """

    def __init__(self, path, num_classes=7):
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.hm = next(o for o in outs if o["shape"][-1] == num_classes)
        self.box = next(o for o in outs if o["shape"][-1] == 4)
        self.input_hw = tuple(int(v) for v in self.inp["shape"][1:3])

    def _q(self, x, d):
        s, z = d["quantization"]
        return np.clip(np.round(x / s + z), -128, 127).astype(np.int8) if d["dtype"] == np.int8 else x.astype(d["dtype"])

    def _dq(self, q, d):
        s, z = d["quantization"]
        return (q.astype(np.float32) - z) * s if d["dtype"] == np.int8 else q

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        hms, boxes = [], []
        for xi in x:                                   # interpreter chạy từng ảnh (batch = 1, như trên chip)
            self.it.set_tensor(self.inp["index"], self._q(xi[None], self.inp))
            self.it.invoke()
            hms.append(self._dq(self.it.get_tensor(self.hm["index"]), self.hm)[0])
            boxes.append(self._dq(self.it.get_tensor(self.box["index"]), self.box)[0])
        return np.stack(hms), np.stack(boxes)


def predict_items(predict_fn, items, input_hw, batch=64, score_thr=0.05, topk=100):
    """-> {image_id: [ {class_id, score, bbox}, ... ]} theo tọa độ ảnh gốc 320x240."""
    preds, i = {}, 0
    for x in image_batches(items, input_hw, batch):
        hm, box = predict_fn(x.numpy())
        for j in range(len(hm)):
            preds[items[i]["id"]] = decode_center(hm[j], box[j], input_hw, ORIG_HW, score_thr, topk)
            i += 1
    return preds


def evaluate(predict_fn, items, input_hw, batch=64, method="all"):
    """-> (mAP50, {class_id: AP}, preds)."""
    preds = predict_items(predict_fn, items, input_hw, batch)
    m, per = map50(preds, gts_of(items), method=method)
    return m, per, preds
