"""Vẽ box lên ảnh để KIỂM TRA BẰNG MẮT (nhãn, augmentation, dự đoán). Dùng OpenCV, có sẵn trên Colab."""
import cv2
import numpy as np

from edgedet.data.dataset import CLASS_NAMES

_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]


def draw(img, dets, scale=1):
    """img: uint8 RGB hoặc float [-1,1]; dets: list dict {class_id, bbox, [score]} theo pixel của img.
    Trả ảnh uint8 RGB đã vẽ (phóng to `scale` lần cho dễ nhìn)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).copy()
    for d in dets:
        k = int(d["class_id"])
        x1, y1, x2, y2 = [int(round(v * scale)) for v in d["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), _COLORS[k % 7], 1)
        txt = CLASS_NAMES[k] + (f" {d['score']:.2f}" if "score" in d else "")
        cv2.putText(img, txt, (x1, max(8, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, _COLORS[k % 7], 1)
    return img


def grid(imgs, cols=4):
    """Ghép nhiều ảnh cùng kích thước thành 1 lưới."""
    rows = [np.concatenate(imgs[i:i + cols] + [np.zeros_like(imgs[0])] * (cols - len(imgs[i:i + cols])), 1)
            for i in range(0, len(imgs), cols)]
    return np.concatenate(rows, 0)
