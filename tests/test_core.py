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
