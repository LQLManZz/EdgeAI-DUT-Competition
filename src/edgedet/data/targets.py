"""Mã hóa nhãn (encode): danh sách box thật -> "đáp án" có cùng hình dạng với output của model.

Bài giảng: docs/03_training/learning_guide.md, Bài 3.

Model không xuất ra danh sách box. Nó xuất 2 tensor dạng lưới (xem head_center trong models/edgedet.py):
  hm  (Hf, Wf, C)  – heatmap: ô chứa tâm vật lớp k có giá trị 1, xung quanh giảm dần theo hình Gauss
  box (Hf, Wf, 4)  – [log(w/8), log(h/8), off_x, off_y]; chỉ có ý nghĩa tại ô chứa tâm vật
Muốn tính loss thì phải biến box thật thành 2 tensor cùng hình dạng như vậy. Hàm encode() làm việc đó.
decode() trong utils/decode.py làm chiều ngược lại; kiểm tra decode(encode(box)) ≈ box ở tests/.
"""
import numpy as np

STRIDE = 8   # mỗi ô lưới = 8x8 pixel ảnh input (input 144x192 -> lưới 18x24)


def gaussian_radius(h, w, min_overlap=0.7):
    """Bán kính "đồi" Gauss quanh tâm, theo CenterNet (Zhou et al., 2019).

    Ý tưởng: nếu dịch tâm đi r ô mà box vẫn có IoU >= min_overlap với box thật, thì dự đoán ở ô đó
    "gần đúng", nên không bị phạt nặng. r = nghiệm nhỏ nhất của 3 trường hợp đặt góc box. (h, w theo ô lưới.)
    """
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def _draw_gaussian(hm, cx, cy, radius):
    """Vẽ đồi Gauss đỉnh = 1 tại ô (cx, cy) lên kênh hm (2D). Chỗ chồng nhau thì lấy max (không cộng)."""
    r = max(0, int(radius))
    sigma = (2 * r + 1) / 6
    ys, xs = np.ogrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    H, W = hm.shape
    l, ri = min(cx, r), min(W - cx, r + 1)        # cắt phần đồi rơi ra ngoài lưới
    t, b = min(cy, r), min(H - cy, r + 1)
    np.maximum(hm[cy - t:cy + b, cx - l:cx + ri], g[r - t:r + b, r - l:r + ri],
               out=hm[cy - t:cy + b, cx - l:cx + ri])


def encode(boxes, classes, input_hw, num_classes=7):
    """boxes: (N, 4) [x1,y1,x2,y2] theo pixel ẢNH INPUT (đã resize); classes: (N,) int.

    Trả:
      hm      (Hf, Wf, C) float32 – đáp án heatmap
      box_t   (Hf, Wf, 5) float32 – 4 kênh đáp án box + kênh 5 là mask (1 tại ô có tâm vật, 0 nơi khác)
    Mask được gói chung vào box_t để Keras đưa nó vào hàm loss cùng với đáp án (xem losses/center_loss.py).
    """
    H, W = input_hw
    hf, wf = H // STRIDE, W // STRIDE
    hm = np.zeros((hf, wf, num_classes), np.float32)
    box_t = np.zeros((hf, wf, 5), np.float32)
    for (x1, y1, x2, y2), k in zip(np.asarray(boxes, np.float32).reshape(-1, 4), classes):
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            continue                                        # box suy biến (vd. bị cắt hết khi augment)
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE   # tâm, đơn vị ô lưới (số thực)
        ix, iy = min(int(cx), wf - 1), min(int(cy), hf - 1)         # ô chứa tâm
        _draw_gaussian(hm[:, :, int(k)], ix, iy, gaussian_radius(h / STRIDE, w / STRIDE))
        box_t[iy, ix] = [np.log(w / STRIDE), np.log(h / STRIDE), cx - ix, cy - iy, 1.0]
    return hm, box_t
