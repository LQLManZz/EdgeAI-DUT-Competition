"""Đo mAP50 của model YOLO của nhóm khác (ANOTHER_WORK/best_model_mAP87.onnx): FP32 ONNX và TFLite INT8.

Mục đích: kiểm tra con số "87%" (FP32 PyTorch) còn lại bao nhiêu sau khi lượng tử INT8 (điều kiện thật của cuộc thi).

  .venv/Scripts/python.exe scripts/analysis/eval_other_yolo.py --model ANOTHER_WORK/best_model_mAP87.onnx
  .venv/Scripts/python.exe scripts/analysis/eval_other_yolo.py --model ANOTHER_WORK/int8/xxx_full_integer_quant.tflite

Lưu ý quan trọng:
- Đo trên TẬP VAL CỦA HỌ (ANOTHER_WORK/val.txt). Tập train của họ chứa phần lớn val của mình, đo trên val của mình sẽ bị rò.
- Tiền xử lý giống Ultralytics: ảnh 320x240 -> letterbox thành 320x320 (đệm 40 px trên + dưới, màu 114), RGB, /255.
- Output [1, 11, 2000] = 2000 anchor (40x40 + 20x20) x (4 số box cx,cy,w,h theo pixel input + 7 xác suất lớp đã sigmoid).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from edgedet.data.dataset import read_yolo_label  # noqa: E402
from edgedet.eval.map50 import map50  # noqa: E402

CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
PAD = 40                                 # 240 + 2*40 = 320; đặt = 0 nếu model nhận thẳng 240x320 (tự dò trong main)


def load_list(txt, data_root):
    items = []
    for line in Path(txt).read_text().split():
        stem = Path(line).stem
        img = data_root / "train" / "images" / f"{stem}.png"
        b, c = read_yolo_label(data_root / "train" / "labels" / f"{stem}.txt")
        items.append({"id": stem, "path": str(img), "boxes": b, "classes": c})
    return items


def letterbox(path):
    """-> float32 (240+2*PAD, 320, 3) RGB trong [0, 1]."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert img.shape[:2] == (240, 320), img.shape
    img = cv2.copyMakeBorder(img, PAD, PAD, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img.astype(np.float32) / 255.0


class OnnxRunner:
    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def __call__(self, x):                              # x (H,320,3) -> (N anchor, 11)
        outs = [o[0].T for o in self.s.run(None, {self.name: x.transpose(2, 0, 1)[None]})]
        outs.sort(key=lambda o: o.shape[1] != 4)        # nếu output đã tách: box trước
        return np.concatenate(outs, 1)


class TFLiteRunner:
    def __init__(self, path):
        import tensorflow as tf
        # tắt XNNPACK: delegate này không nhận các node TRANSPOSE int8 mà onnx2tf sinh ra
        self.it = tf.lite.Interpreter(
            model_path=str(path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()
        print("input", self.i["shape"], self.i["dtype"].__name__, self.i["quantization"])
        for o in self.o:
            print("output", o["name"], o["shape"], o["dtype"].__name__, o["quantization"])

    def _q(self, x, d):
        s, z = d["quantization"]
        if d["dtype"] == np.int8:
            return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        return x.astype(d["dtype"])

    def _dq(self, q, d):
        s, z = d["quantization"]
        return (q.astype(np.float32) - z) * s if d["dtype"] == np.int8 else q.astype(np.float32)

    def __call__(self, x):
        x = x[None]
        if self.i["shape"][1] == 3:                     # model giữ NCHW
            x = x.transpose(0, 3, 1, 2)
        self.it.set_tensor(self.i["index"], self._q(x, self.i))
        self.it.invoke()
        outs = [self._dq(self.it.get_tensor(o["index"]), o)[0] for o in self.o]
        # chuẩn hóa mọi output về (2000, k) rồi ghép theo kênh: box (4) trước, lớp (7) sau
        outs = [o if o.shape[0] > o.shape[1] else o.T for o in outs]
        outs.sort(key=lambda o: o.shape[1] != 4)        # nếu output đã tách: box trước
        return np.concatenate(outs, 1) if len(outs) > 1 else outs[0]


def decode(out, conf, iou, max_det):
    """(2000, 11) -> list box trên ảnh gốc 320x240. NMS theo từng lớp; một anchor có thể cho nhiều lớp."""
    cxcywh, scores = out[:, :4], out[:, 4:]
    x1y1 = cxcywh[:, :2] - cxcywh[:, 2:] / 2
    xyxy = np.concatenate([x1y1, x1y1 + cxcywh[:, 2:]], 1)
    xyxy[:, [1, 3]] -= PAD                              # bỏ phần đệm letterbox
    xyxy = np.clip(xyxy, 0, [320, 240, 320, 240])
    dets = []
    for c in range(scores.shape[1]):
        m = scores[:, c] > conf
        if not m.any():
            continue
        b, s = xyxy[m], scores[m, c]
        keep = cv2.dnn.NMSBoxes(np.c_[b[:, :2], b[:, 2:] - b[:, :2]].tolist(), s.tolist(), conf, iou)
        for k in np.array(keep).reshape(-1):
            dets.append({"class_id": c, "score": float(s[k]), "bbox": b[k].tolist()})
    dets.sort(key=lambda d: -d["score"])
    return dets[:max_det]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", default=str(ROOT / "ANOTHER_WORK" / "val.txt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump_calib", type=int, default=0, help="ghi N ảnh (lấy từ train.txt của họ) ra calib.npy rồi thoát")
    a = ap.parse_args()

    global PAD
    if a.model.endswith(".onnx"):
        import onnx
        PAD = (onnx.load(a.model).graph.input[0].type.tensor_type.shape.dim[2].dim_value - 240) // 2
    else:
        import tensorflow as tf
        shp = tf.lite.Interpreter(model_path=a.model).get_input_details()[0]["shape"]
        PAD = ((shp[2] if shp[1] == 3 else shp[1]) - 240) // 2
    print("PAD =", PAD)
    if a.dump_calib:
        items = load_list(ROOT / "ANOTHER_WORK" / "train.txt", Path(a.data))[: a.dump_calib]
        out = Path(a.model).parent / "calib.npy"
        np.save(out, np.stack([letterbox(it["path"]) for it in items]))
        print("calib ->", out)
        return

    items = load_list(a.list, Path(a.data))
    if a.limit:
        items = items[: a.limit]
    run = OnnxRunner(a.model) if a.model.endswith(".onnx") else TFLiteRunner(a.model)
    raw = {it["id"]: run(letterbox(it["path"])) for it in items}
    gts = {it["id"]: [{"class_id": int(c), "bbox": b.tolist()} for b, c in zip(it["boxes"], it["classes"])]
           for it in items}

    # 2 cấu hình: (1) giống Ultralytics val để đối chiếu số 0,869; (2) giống file nộp bài của dự án mình
    for tag, conf, iou, max_det in [("ultralytics-val conf.001 iou.7 max300", 0.001, 0.7, 300),
                                    ("submission conf.05 iou.5 max100", 0.05, 0.5, 100)]:
        preds = {k: decode(v, conf, iou, max_det) for k, v in raw.items()}
        for method in ("all", "coco101"):
            m, per = map50(preds, gts, method=method)
            print(f"[{tag}] mAP50({method}) = {m:.4f} | "
                  + " ".join(f"{CLASS_NAMES[c]} {v:.2f}" for c, v in per.items()))
    print(f"{len(items)} ảnh val")


if __name__ == "__main__":
    main()
