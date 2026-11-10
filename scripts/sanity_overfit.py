"""Kiểm tra "học thuộc" (overfit): model + encode + loss + decode có ăn khớp với nhau không.

Bài giảng: docs/03_training/learning_guide.md, Bài 5.
Tạo 8 ảnh giả (nền nhiễu + hình chữ nhật màu = "vật"), train vài trăm bước, rồi đo mAP50 trên CHÍNH 8 ảnh đó.
Pipeline đúng -> mAP50 phải gần 1. Nếu không đạt thì có lỗi ở đâu đó, đừng train dataset thật vội.
Chạy: .venv/Scripts/python scripts/sanity_overfit.py
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edgedet.data.targets import encode
from edgedet.eval.map50 import map50
from edgedet.losses.center_loss import box_l1_loss, heatmap_focal_loss
from edgedet.models.edgedet import build_edgedet
from edgedet.utils.decode import decode_center

HW, ORIG, N, C = (144, 192), (240, 320), 8, 7
rng = np.random.default_rng(0)
COLORS = rng.uniform(-1, 1, (C, 3))                       # mỗi lớp một màu -> model phân biệt lớp qua màu

imgs, hms, boxes, gts = [], [], [], {}
for i in range(N):
    img = rng.normal(0, 0.1, (*HW, 3)).astype(np.float32)
    bb, cl = [], []
    for _ in range(rng.integers(1, 4)):
        k = int(rng.integers(C))
        w, h = rng.uniform(20, 80), rng.uniform(20, 70)
        x1, y1 = rng.uniform(0, HW[1] - w), rng.uniform(0, HW[0] - h)
        if any(x1 < b[2] and b[0] < x1 + w and y1 < b[3] and b[1] < y1 + h for b in bb):
            continue                                      # không cho chồng nhau: vật bị che hết thì không thể học
        img[int(y1):int(y1 + h), int(x1):int(x1 + w)] = COLORS[k]
        bb.append([x1, y1, x1 + w, y1 + h]); cl.append(k)
    hm, bt = encode(np.array(bb), cl, HW)
    imgs.append(img); hms.append(hm); boxes.append(bt)
    s = np.array([ORIG[1] / HW[1], ORIG[0] / HW[0]] * 2)  # đổi nhãn về tọa độ ảnh gốc để chấm như Kaggle
    gts[i] = [{"class_id": k, "bbox": list(np.array(b) * s)} for b, k in zip(bb, cl)]

x = np.stack(imgs); y = {"head_hm": np.stack(hms), "head_box": np.stack(boxes)}
model = build_edgedet("custom", "center", HW)
EPOCHS = 400
lr = tf.keras.optimizers.schedules.CosineDecay(2e-3, EPOCHS)   # learning rate giảm dần theo hình cos về 0
model.compile(tf.keras.optimizers.Adam(lr),
              loss={"head_hm": heatmap_focal_loss, "head_box": box_l1_loss},
              jit_compile=False)   # XLA trên GPU không có đạo hàm của RESIZE_NEAREST_NEIGHBOR
hist = model.fit(x, y, batch_size=8, epochs=EPOCHS, verbose=0)
l = hist.history["loss"]
print(f"loss: đầu {l[0]:.3f} -> cuối {l[-1]:.4f}")

# BatchNorm: lúc train dùng thống kê (mean/var) của batch hiện tại; lúc dự đoán dùng TRUNG BÌNH TRƯỢT
# (momentum 0.99, tức ~100 bước mới "quên" giá trị cũ). Chỉ 400 bước thì trung bình trượt còn mang dấu vết
# của trọng số lúc đầu -> đã đo: score dự đoán 0.02 dù ở chế độ train là 0.90, mAP = 0.
# Sửa: đặt momentum = 0 rồi chạy forward 1 lần ở chế độ train (không cập nhật trọng số)
# -> trung bình trượt = đúng thống kê của 8 ảnh. Đã đo: mAP chế độ train = 1.0, sau bước này chế độ dự đoán = 1.0.
# Train thật có hàng nghìn bước nên không cần bước này.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = 0.0
model(x, training=True)

hm_p, box_p = model.predict(x, verbose=0)
preds = {i: decode_center(hm_p[i], box_p[i], HW, ORIG) for i in range(N)}
m, per = map50(preds, gts)
print(f"mAP50 trên chính tập train: {m:.3f}  theo lớp: { {k: round(v, 2) for k, v in per.items()} }")
print("OK – pipeline học được" if m > 0.9 else "CHƯA ĐẠT – kiểm tra lại pipeline")
