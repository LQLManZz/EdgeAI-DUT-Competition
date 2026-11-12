"""Đọc dataset Kaggle -> tf.data cho training và đánh giá.

Bài giảng: docs/03_training/learning_guide.md, Bài 6.

Cấu trúc dataset (data/raw/):
  train/images/00001.png   ảnh RGB 320x240
  train/labels/00001.txt   mỗi dòng "lớp cx cy w h", 4 số cuối đã chia cho kích thước ảnh (0..1) – định dạng YOLO
  test/images/img_0001.png ảnh để nộp bài (không có nhãn)

Trong code, box luôn ở dạng [x1, y1, x2, y2] theo pixel. Quy ước tọa độ:
  - "ảnh gốc"  = 320x240, dùng cho nhãn, mAP và file submission
  - "ảnh input" = kích thước model nhận (vd. 192x144), dùng cho encode/decode
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgedet.data.targets import encode

ORIG_HW = (240, 320)
MAX_BOXES = 32          # đệm mỗi ảnh lên đúng 32 box (lớp -1 = ô trống) để gộp batch được; dataset tối đa 12 box/ảnh
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


# ---------------------------------------------------------------- đọc nhãn, chia train/val

def read_yolo_label(path, orig_hw=ORIG_HW):
    """File nhãn YOLO -> (boxes (N,4) [x1,y1,x2,y2] pixel ảnh gốc, classes (N,) int)."""
    H, W = orig_hw
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int32)
    a = np.array(rows, np.float32)
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return np.clip(boxes, 0, [W, H, W, H]).astype(np.float32), a[:, 0].astype(np.int32)


def load_items(root, split="train"):
    """Danh sách item = dict(id, path, boxes, classes). Tập test không có nhãn -> boxes rỗng."""
    root = Path(root)
    items = []
    for p in sorted((root / split / "images").glob("*.png")):
        lab = root / split / "labels" / (p.stem + ".txt")
        b, c = read_yolo_label(lab) if lab.exists() else (np.zeros((0, 4), np.float32), np.zeros((0,), np.int32))
        items.append({"id": p.stem, "path": str(p), "boxes": b, "classes": c})
    return items


def split_train_val(items, val_frac=0.1, seed=0):
    """Chia ngẫu nhiên nhưng CỐ ĐỊNH (seed) – mọi thí nghiệm dùng cùng tập val thì mới so sánh được."""
    idx = np.random.default_rng(seed).permutation(len(items))
    n_val = int(round(len(items) * val_frac))
    val = [items[i] for i in sorted(idx[:n_val])]
    train = [items[i] for i in sorted(idx[n_val:])]
    return train, val


def gts_of(items):
    """Nhãn thật theo định dạng map50.py (tọa độ ảnh gốc)."""
    return {it["id"]: [{"class_id": int(k), "bbox": [float(v) for v in b]}
                       for b, k in zip(it["boxes"], it["classes"])] for it in items}


def stats(items):
    """Thống kê nhanh: số ảnh, số box theo lớp, kích thước box (pixel ảnh gốc)."""
    cls = np.concatenate([it["classes"] for it in items])
    bx = np.concatenate([it["boxes"] for it in items])
    w, h = bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]
    per = {CLASS_NAMES[k]: int((cls == k).sum()) for k in range(len(CLASS_NAMES))}
    n_box = [len(it["classes"]) for it in items]
    return {"images": len(items), "boxes": int(len(cls)), "boxes_per_image_mean": round(float(np.mean(n_box)), 2),
            "boxes_per_image_max": int(np.max(n_box)), "empty_images": int(sum(n == 0 for n in n_box)),
            "per_class": per,
            "box_w_px_p5_p50_p95": [round(float(v), 1) for v in np.percentile(w, [5, 50, 95])],
            "box_h_px_p5_p50_p95": [round(float(v), 1) for v in np.percentile(h, [5, 50, 95])]}


# ---------------------------------------------------------------- tiền xử lý (dùng chung train / val / test / INT8)

def preprocess(img_uint8, input_hw):
    """Ảnh uint8 (H,W,3) -> float32 input model trong [-1, 1].

    QUAN TRỌNG: firmware ESP32 (GĐ5) phải làm ĐÚNG phép này (resize bilinear + x/127.5-1), nếu không
    model trên chip sẽ thấy ảnh "khác" lúc train và đoán sai.
    """
    x = tf.image.resize(tf.cast(img_uint8, tf.float32), input_hw, method="bilinear")
    return x / 127.5 - 1.0


def read_image(path):
    return tf.io.decode_png(tf.io.read_file(path), channels=3)


def _pad(items):
    b = np.zeros((len(items), MAX_BOXES, 4), np.float32)
    c = -np.ones((len(items), MAX_BOXES), np.int32)
    for i, it in enumerate(items):
        n = min(len(it["classes"]), MAX_BOXES)
        b[i, :n], c[i, :n] = it["boxes"][:n], it["classes"][:n]
    return b, c


# ---------------------------------------------------------------- augmentation

DEFAULT_AUG = {"zoom": [0.7, 1.3], "flip": 0.5, "brightness": 0.15, "contrast": [0.7, 1.3],
               "saturation": [0.6, 1.4], "hue": 0.03, "noise": 0.03, "min_visible": 0.4}


def _augment(img, boxes, classes, input_hw, aug):
    """Biến đổi ngẫu nhiên ảnh + box cùng lúc (box phải đi theo ảnh!).

    1) Zoom/dịch: cắt một "cửa sổ" có kích thước z * ảnh gốc ở vị trí ngẫu nhiên rồi resize về input.
       z < 1: phóng to (vật to ra, có thể bị cắt mép); z > 1: thu nhỏ (thêm viền xám) -> model gặp vật đủ cỡ.
    2) Lật ngang.  3) Màu: sáng/tương phản/bão hòa/sắc độ + nhiễu – mô phỏng camera OV7670 rẻ tiền.
    """
    H, W = ORIG_HW
    h_in, w_in = input_hw
    img = tf.cast(img, tf.float32) / 255.0

    z = tf.random.uniform([], aug["zoom"][0], aug["zoom"][1])
    wh, ww = H * z, W * z
    y0 = tf.random.uniform([], tf.minimum(0.0, H - wh), tf.maximum(0.0, H - wh) + 1e-6)
    x0 = tf.random.uniform([], tf.minimum(0.0, W - ww), tf.maximum(0.0, W - ww) + 1e-6)
    crop = tf.stack([y0 / H, x0 / W, (y0 + wh) / H, (x0 + ww) / W])[None]
    img = tf.image.crop_and_resize(img[None], crop, [0], input_hw, extrapolation_value=0.5)[0]

    sx, sy = w_in / ww, h_in / wh
    b = (boxes - tf.stack([x0, y0, x0, y0])) * tf.stack([sx, sy, sx, sy])
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = tf.clip_by_value(b, 0.0, tf.constant([w_in, h_in, w_in, h_in], tf.float32))
    area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = (area1 >= aug["min_visible"] * area0) & (b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)
    classes = tf.where(keep, classes, -1)                       # vật bị cắt gần hết -> bỏ khỏi nhãn

    if tf.random.uniform([]) < aug["flip"]:
        img = tf.image.flip_left_right(img)
        b = tf.stack([w_in - b[:, 2], b[:, 1], w_in - b[:, 0], b[:, 3]], 1)

    img = tf.image.random_brightness(img, aug["brightness"])
    img = tf.image.random_contrast(img, *aug["contrast"])
    img = tf.image.random_saturation(img, *aug["saturation"])
    img = tf.image.random_hue(img, aug["hue"])
    img = img + tf.random.normal(tf.shape(img), 0.0, tf.random.uniform([], 0.0, aug["noise"]))
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img * 2.0 - 1.0, b, classes                          # [0,1] -> [-1,1], khớp preprocess()


# ---------------------------------------------------------------- tf.data

def _encode_tf(x, boxes, classes, input_hw, num_classes, neighbor_weight=0.25):
    def _np(b, c):
        m = c >= 0
        return encode(b[m], c[m], input_hw, num_classes, neighbor_weight=neighbor_weight)
    hm, bt = tf.numpy_function(_np, [boxes, classes], [tf.float32, tf.float32])
    hf, wf = input_hw[0] // 8, input_hw[1] // 8
    hm.set_shape((hf, wf, num_classes)); bt.set_shape((hf, wf, 5))
    return x, {"head_hm": hm, "head_box": bt}


def make_dataset(items, input_hw, batch, training, aug=None, num_classes=7, cache=True, neighbor_weight=0.25):
    """Dataset cho model.fit: phần tử = (ảnh (B,H,W,3) [-1,1], {"head_hm": ..., "head_box": ...}).

    cache=True giữ ảnh đã giải nén (uint8) trong RAM: 6000 ảnh ~ 1,4 GB, đọc lại từ đĩa mỗi epoch thì rất chậm.
    """
    aug = {**DEFAULT_AUG, **(aug or {})}
    b, c = _pad(items)
    ds = tf.data.Dataset.from_tensor_slices(([it["path"] for it in items], b, c))
    ds = ds.map(lambda p, b, c: (read_image(p), b, c), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(items), reshuffle_each_iteration=True)
        ds = ds.map(lambda im, b, c: _augment(im, b, c, input_hw, aug), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        s = tf.constant([input_hw[1] / ORIG_HW[1], input_hw[0] / ORIG_HW[0]] * 2, tf.float32)
        ds = ds.map(lambda im, b, c: (preprocess(im, input_hw), b * s, c), num_parallel_calls=tf.data.AUTOTUNE)
    nw = neighbor_weight if training else 0.0
    ds = ds.map(lambda x, b, c: _encode_tf(x, b, c, input_hw, num_classes, nw), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch, drop_remainder=training).prefetch(tf.data.AUTOTUNE)


def image_batches(items, input_hw, batch=64):
    """Chỉ ảnh (không nhãn), đã preprocess – dùng để dự đoán khi đánh giá / tạo submission."""
    ds = tf.data.Dataset.from_tensor_slices([it["path"] for it in items])
    ds = ds.map(lambda p: preprocess(read_image(p), input_hw), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
