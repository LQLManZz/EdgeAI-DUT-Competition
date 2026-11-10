"""Test nhanh TextCNN (chạy trên Colab ở Ô 2):  PYTHONPATH=src python -m pytest tests/test_textcnn.py -q"""
import json
from pathlib import Path

import numpy as np
import pytest

from textcnn import text as T
from textcnn.model import build
from textcnn.train import Int8Model, compliance, encode_all, to_int8

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("textcnn_*.json"))
CONFIGS = [p for p in CONFIGS if p.stem != "textcnn_queue"]


def test_norm_strips_vietnamese_marks():
    assert T.norm("Cuốn SÁCH ở đâu? Đếm đi!") == "cuon sach o dau dem di"


def test_encode_pads_truncates_and_marks_unknown():
    vocab = T.build_vocab(["co bao nhieu chai"], "word")
    ids = T.encode(["co bao nhieu laptop", "co " * 40], "word", vocab, 6)
    assert ids.shape == (2, 6) and ids.dtype == np.int32
    assert list(ids[0]) == [vocab.index("co"), vocab.index("bao"), vocab.index("nhieu"), T.UNK, T.PAD, T.PAD]
    assert (ids[1] == vocab.index("co")).all()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_fits_contest_limits_as_int8(path):
    """Mọi config: <=200K tham số, INT8 hoàn toàn, chỉ op TFLM cơ bản, file INT8 chạy được."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    X = ["co bao nhieu chai", "cuon sach o dau", "trong anh co gi", "hom nay troi mua khong"] * 4
    vocabs = {k: T.build_vocab(X, k) for k in cfg["branches"]}
    m = build(cfg, {k: len(v) for k, v in vocabs.items()})
    blob = to_int8(m, cfg, vocabs, X)
    rep = compliance(m, blob)
    assert all(rep["checks"].values()), rep
    assert Int8Model(blob).predict(encode_all(cfg, vocabs, X)).shape == (len(X),)
