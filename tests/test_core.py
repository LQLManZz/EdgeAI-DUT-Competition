"""Kiểm tra các khối lõi bằng ví dụ tính tay. Chạy: .venv/Scripts/python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgedet.data.targets import encode
from edgedet.eval.map50 import average_precision, iou, map50
from edgedet.utils.decode import decode_center


def test_iou_by_hand():
    # 2 hình vuông 10x10 lệch nhau 5 px: giao 5x10 = 50, hợp 100+100-50 = 150 -> 1/3
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_ap_by_hand():
    gts = {"a": [{"class_id": 0, "bbox": [0, 0, 10, 10]}],
           "b": [{"class_id": 0, "bbox": [50, 50, 60, 60]}]}
    preds = {"a": [{"class_id": 0, "score": 0.9, "bbox": [0, 0, 10, 10]},     # TP
                   {"class_id": 0, "score": 0.8, "bbox": [0, 0, 10, 10]}],    # FP: box thật đã bị nhận
             "b": [{"class_id": 0, "score": 0.7, "bbox": [51, 51, 61, 61]}]}  # TP (IoU 81/119 = 0.68)
    # TP/FP theo score: [1, 0, 1] -> recall [.5, .5, 1], precision [1, .5, .667]
    # đường bậc thang: recall 0->.5 có p=1, .5->1 có p=.667 -> AP = .5*1 + .5*.667 = .8333
    ap, n = average_precision(preds, gts, 0)
    assert n == 2 and abs(ap - (0.5 + 0.5 * 2 / 3)) < 1e-6


def test_map_edge_cases():
    gts = {"a": [{"class_id": 0, "bbox": [0, 0, 10, 10]}, {"class_id": 0, "bbox": [20, 0, 30, 10]}]}
    one = {"a": [{"class_id": 0, "score": 0.9, "bbox": [0, 0, 10, 10]}]}
    assert abs(average_precision(one, gts, 0)[0] - 0.5) < 1e-9                  # bỏ sót 1/2 vật
    wrong = {"a": [{"class_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]}]}
    m, per = map50(wrong, gts)
    assert m == 0.0 and list(per) == [0]                                         # lớp 1 không có box thật -> bỏ qua


def test_encode_then_decode_recovers_box():
    # box trên ảnh gốc 320x240 -> scale về input 144x192 -> encode -> giả làm output model -> decode
    input_hw, orig = (144, 192), (240, 320)
    gt = np.array([[40.0, 60.0, 120.0, 180.0], [200.0, 20.0, 300.0, 90.0]])
    s = np.array([input_hw[1] / orig[1], input_hw[0] / orig[0]] * 2)
    hm, box_t = encode(gt * s, [1, 4], input_hw)
    logits = np.where(hm >= 1.0, 10.0, -10.0 + 5 * hm)                         # sigmoid ~1 tại tâm, nhỏ nơi khác
    dets = decode_center(logits, box_t[..., :4], input_hw, orig, score_thr=0.5)
    assert sorted(d["class_id"] for d in dets) == [1, 4]
    for d in dets:
        ref = gt[0] if d["class_id"] == 1 else gt[1]
        assert iou(d["bbox"], ref) > 0.99, (d, ref)


def test_decode_int8_plateau_tie_and_nms():
    # Khi INT8 làm 2 ô cạnh nhau (5,5) và (5,6) có cùng giá trị đỉnh, tie_break chỉ giữ ô tâm (5,5)
    input_hw, orig = (144, 192), (240, 320)
    gt = np.array([[60.0, 60.0, 140.0, 140.0]])
    s = np.array([input_hw[1] / orig[1], input_hw[0] / orig[0]] * 2)
    hm, box_t = encode(gt * s, [3], input_hw)
    logits = -10.0 + 12.0 * hm
    iy, ix = np.unravel_index(np.argmax(hm[:, :, 3]), hm[:, :, 3].shape)
    logits[iy, ix + 1, 3] = logits[iy, ix, 3]   # mô phỏng 2 ô cạnh nhau cùng mức lượng tử INT8

    old_dets = decode_center(logits, box_t[..., :4], input_hw, orig, score_thr=0.5,
                             tie_break=False, nms_iou=None, score_smooth=0.0)
    new_dets = decode_center(logits, box_t[..., :4], input_hw, orig, score_thr=0.5)
    assert len(old_dets) == 2                   # decode cũ sinh 2 box cho 1 vật (ô bên cạnh có box chưa train)
    assert len(new_dets) == 1 and iou(new_dets[0]["bbox"], gt[0]) > 0.99


def test_neighbor_box_supervision_recovers_shifted_peak():
    # Khi đỉnh heatmap bị lệch hẳn 1 ô sang (iy, ix+1), ô sát tâm đã được gán nhãn box nên vẫn khôi phục đúng box
    input_hw, orig = (144, 192), (240, 320)
    gt = np.array([[60.0, 60.0, 140.0, 140.0]])
    s = np.array([input_hw[1] / orig[1], input_hw[0] / orig[0]] * 2)
    hm, box_t = encode(gt * s, [2], input_hw, neighbor_weight=0.25)
    iy, ix = np.unravel_index(np.argmax(hm[:, :, 2]), hm[:, :, 2].shape)
    assert box_t[iy, ix, 4] == 1.0 and box_t[iy, ix + 1, 4] == 0.25

    logits = np.full_like(hm, -10.0)
    logits[iy, ix + 1, 2] = 8.0                 # giả sử đỉnh rơi vào ô bên cạnh (iy, ix+1)
    dets = decode_center(logits, box_t[..., :4], input_hw, orig, score_thr=0.5)
    assert len(dets) == 1 and iou(dets[0]["bbox"], gt[0]) > 0.99


def test_decoupled_head_and_mosaic():
    from edgedet.data.dataset import _mosaic4_np
    from edgedet.models.edgedet import build_edgedet
    m = build_edgedet("mnv3s", "center", (240, 320), neck_c=80, head_c=80, pretrained=False, decoupled_head=True)
    assert m.count_params() == 197_139 <= 200_000

    imgs = np.zeros((4, 144, 192, 3), np.float32)
    boxes = np.zeros((4, 32, 4), np.float32)
    classes = -np.ones((4, 32), np.int32)
    for i in range(4):
        boxes[i, 0] = [60, 40, 120, 100]        # nằm ở vùng trung tâm -> phải còn lại sau khi ghép 4 góc
        classes[i, 0] = i
    im_m, b_m, c_m = _mosaic4_np(imgs, boxes, classes, 144, 192, prob=1.0, min_vis=0.4)
    assert im_m.shape == (4, 144, 192, 3) and (c_m[0] >= 0).sum() == 4
