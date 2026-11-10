"""Xử lý câu hỏi: chuẩn hóa -> tách token -> dãy id có độ dài cố định.

Firmware ESP32 phải làm Y HỆT các bước này (cùng norm, cùng vocab.json, cùng độ dài L),
nếu không model nhận đầu vào khác lúc train và đoán sai dù accuracy trên máy rất cao.

Gốc: code TextCNN của nhóm intent (textcnn_colab.zip, PyTorch). Giữ nguyên logic tiền xử lý,
template_key, augmentation OUT_OF_SCOPE và danh sách cue của stress test.
"""
import collections
import random
import re
import unicodedata

import numpy as np

PAD, UNK = 0, 1          # id 0 = đệm cho đủ độ dài L, id 1 = token chưa thấy lúc train


def norm(s):
    """'Cuốn SÁCH ở đâu?' -> 'cuon sach o dau'. Dữ liệu Kaggle không dấu; bỏ dấu để chung kết gõ có dấu vẫn chạy."""
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")          # Mn = dấu thanh, dấu mũ
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def tokens(s, kind):
    """kind='word': danh sách từ; kind='char': danh sách ký tự (kể cả dấu cách)."""
    return norm(s).split() if kind == "word" else list(norm(s))


def build_vocab(texts, kind):
    """Chỉ xây từ dữ liệu TRAIN (của fold đang train) -> không rò rỉ từ vựng của tập kiểm tra."""
    cnt = collections.Counter(t for s in texts for t in tokens(s, kind))
    return ["<pad>", "<unk>"] + sorted(cnt)


def encode(texts, kind, vocab, L):
    """-> mảng int32 (N, L): id từng token, cắt nếu dài hơn L, đệm PAD nếu ngắn hơn."""
    idx = {t: i for i, t in enumerate(vocab)}
    out = np.full((len(texts), L), PAD, dtype=np.int32)
    for i, s in enumerate(texts):
        ids = [idx.get(t, UNK) for t in tokens(s, kind)][:L]
        out[i, :len(ids)] = ids
    return out


# ---------- "khuôn câu": dùng để chia CV sao cho khuôn câu ở fold kiểm tra chưa từng thấy lúc train ----------
PREFIX = re.compile(r"^(xin hoi|ban oi|cho toi hoi|cho minh hoi)\s+")
OBJ = set("chai nuoc nhua binh coc uong tach ly dien thoai di dong smartphone quyen cuon sach "
          "chuot may tinh thung giay hop xe oto do choi con cai chiec".split())


def template_key(s):
    """Bỏ lời mở đầu lịch sự + thay tên vật bằng <O>: 'ban oi co bao nhieu chai' -> 'co bao nhieu <O>'."""
    out = []
    for w in PREFIX.sub("", norm(s)).split():
        w = "<O>" if w in OBJ else w
        if not (out and out[-1] == "<O>" and w == "<O>"):
            out.append(w)
    return " ".join(out)


# ---------- augmentation chỉ cho phần train ----------
# Quy luật trong intent_train.csv: hỏi về vật KHÔNG thuộc 7 lớp (laptop, dong ho, ...) luôn là OUT_OF_SCOPE.
# Nhưng câu OOS trong train chỉ có vài khuôn đơn giản -> model dễ tin nhầm khuôn LOCATION/COUNT mạnh.
# Cách bù: lấy câu COUNT/EXIST/LOCATION, đổi tên vật trong lớp thành vật ngoài lớp, gán nhãn OUT_OF_SCOPE.
IN_OBJ = sorted("""chai nuoc|chai nhua|binh nuoc|coc uong nuoc|chiec coc|coc|tach|ly|dien thoai di dong|
dien thoai|smartphone|quyen sach|cuon sach|sach|chuot may tinh|con chuot|chuot|thung giay|hop giay|cai hop|hop|
xe oto do choi|xe do choi|oto do choi|o to do choi|chai""".replace("\n", "").split("|"), key=len, reverse=True)
OUT_OBJ = ("laptop|cay but|dong ho|con chim|cai mu|cai remote|cai o|con meo|cai ghe|cai keo|cai thia|"
           "con gau bong|qua tao|qua chuoi|cai quat|kinh mat|cai dia").split("|")
IN_RE = re.compile(r"\b(" + "|".join(re.escape(o) for o in IN_OBJ) + r")\b")


def augment_oos(X, y, classes, ratio, seed):
    """Thêm câu OOS tổng hợp (tỉ lệ `ratio` của các câu COUNT/EXIST/LOCATION có tên vật)."""
    if ratio <= 0:
        return list(X), np.asarray(y)
    rng = random.Random(seed)
    oos = classes.index("OUT_OF_SCOPE")
    src = {classes.index(c) for c in ("COUNT", "EXIST", "LOCATION")}
    newX, newy = [], []
    for s, lab in zip(X, y):
        if lab in src and rng.random() < ratio:
            ns = norm(s)
            if IN_RE.search(ns):
                newX.append(IN_RE.sub(rng.choice(OUT_OBJ), ns, count=1))
                newy.append(oos)
    return list(X) + newX, np.concatenate([np.asarray(y), np.array(newy, dtype=np.int64)])


# ---------- stress test: bỏ hẳn 1 cụm từ khỏi train, kiểm tra trên các câu chứa nó ----------
# Mô phỏng tập test có cách hỏi chưa từng gặp (test có 'lech', 'kiem' không hề có trong train).
CUES = {
    "LOCATION": ["vi tri", "phia", "mep", "ngang", "xac dinh"],
    "COUNT": ["dem", "so luong", "bao nhieu cai", "tat ca"],
    "EXIST": ["xuat hien", "nhin thay", "ton tai", "hay khong"],
    "LIST": ["loai", "kieu", "nhom", "lop", "khong trung"],
    "OUT_OF_SCOPE": ["ai", "lam bang", "dung de", "the nao"],
}
