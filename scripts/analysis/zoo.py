"""Model đại diện (probe) cho từng họ kiến trúc, thu nhỏ về <= 200K params.

Đây là bản PROBE để đo params / MACs / op / RAM / latency – KHÔNG phải kiến trúc đề xuất cuối.
Trọng số ngẫu nhiên. 7 lớp (bottle, cup, phone, book, mouse, box, toy_car).
"""
import tensorflow as tf
from tensorflow.keras import layers as L

NC = 7


# ---------------- khối cơ bản ----------------
def cbr(x, c, k=3, s=1, act="relu"):
    x = L.Conv2D(c, k, s, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x) if act == "relu" else x


def dw(x, k=3, s=1, act="relu"):
    x = L.DepthwiseConv2D(k, s, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x) if act == "relu" else x


def dwsep(x, c, s=1):
    return cbr(dw(x, 3, s), c, 1)


def ghost_module(x, c, relu=True):
    # GhostNet: ratio s=2, cheap op = depthwise 3x3
    p = cbr(x, c // 2, 1, act="relu" if relu else None)
    g = dw(p, 3, act="relu" if relu else None)
    return L.Concatenate()([p, g])


def ghost_bneck(x, mid, out, s=1):
    cin = x.shape[-1]
    y = ghost_module(x, mid, True)
    if s == 2:
        y = dw(y, 3, 2, act=None)
    y = ghost_module(y, out, False)
    if s == 1 and cin == out:
        return L.Add()([x, y])
    sc = cbr(dw(x, 3, s, act=None), out, 1, act=None)
    return L.Add()([sc, y])


def channel_shuffle(x):
    h, w, c = x.shape[1:]
    x = L.Reshape((h, w, 2, c // 2))(x)
    x = L.Permute((1, 2, 4, 3))(x)
    return L.Reshape((h, w, c))(x)


def shuffle_unit(x, out, s=1):
    # ShuffleNetV2 unit (backbone của YOLO-FastestV2 / NanoDet)
    if s == 1:
        c = x.shape[-1] // 2
        a = L.Lambda(lambda t: t[..., :c])(x)
        b = L.Lambda(lambda t: t[..., c:])(x)
        b = cbr(dw(cbr(b, c, 1), 3, 1, act=None), c, 1)
    else:
        c = out // 2
        a = cbr(dw(x, 3, 2, act=None), c, 1)
        b = cbr(dw(cbr(x, c, 1), 3, 2, act=None), c, 1)
    return channel_shuffle(L.Concatenate()([a, b]))


def inv_res(x, out, s, t):
    cin = x.shape[-1]
    y = cbr(x, cin * t, 1) if t != 1 else x
    y = dw(y, 3, s)
    y = cbr(y, out, 1, act=None)
    return L.Add()([x, y]) if s == 1 and cin == out else y


# ---------------- backbone: trả về dict {stride: feature} ----------------
def _cut(model, strides, size):
    """Lấy output cuối cùng có stride tương ứng trong model keras.applications."""
    feats = {}
    for lyr in model.layers:
        shp = lyr.output.shape
        if len(shp) == 4 and shp[1]:
            for st in strides:  # khớp chính xác ceil(H/st), ceil(W/st) – loại các layer ZeroPadding
                if (shp[1], shp[2]) == (-(-size[0] // st), -(-size[1] // st)):
                    feats[st] = lyr.output
    return feats


def bb_mnv2(inp, size, alpha=0.35, strides=(8,)):
    m = tf.keras.applications.MobileNetV2(input_tensor=inp, alpha=alpha, include_top=False, weights=None)
    return _cut(m, strides, size)


def bb_mnv1(inp, size, alpha=0.25, strides=(8,)):
    m = tf.keras.applications.MobileNet(input_tensor=inp, alpha=alpha, include_top=False, weights=None)
    return _cut(m, strides, size)


def bb_mnv3s(inp, size, alpha=0.75, minimalistic=True, strides=(16,)):
    m = tf.keras.applications.MobileNetV3Small(input_tensor=inp, alpha=alpha, minimalistic=minimalistic,
                                               include_top=False, weights=None, include_preprocessing=False)
    return _cut(m, strides, size)


def bb_ghost(inp, size, w=1.0, strides=(8,)):
    c = lambda n: max(8, int(n * w) // 4 * 4)
    x = cbr(inp, c(16), 3, 2)                              # /2
    x = ghost_bneck(x, c(16), c(16))
    x = ghost_bneck(x, c(48), c(24), 2)                    # /4
    x = ghost_bneck(x, c(72), c(24))
    x = ghost_bneck(x, c(72), c(40), 2); f8 = ghost_bneck(x, c(120), c(40))   # /8
    x = ghost_bneck(f8, c(240), c(80), 2); f16 = ghost_bneck(x, c(200), c(80))  # /16
    return {8: f8, 16: f16}


def bb_shuffle(inp, size, w=1.0, strides=(8,)):
    c = lambda n: max(8, int(n * w) // 4 * 4)
    x = cbr(inp, c(24), 3, 2)                              # /2
    x = L.MaxPooling2D(3, 2, padding="same")(x)            # /4
    x = shuffle_unit(x, c(48), 2)                          # /8
    for _ in range(3):
        x = shuffle_unit(x, c(48))
    f8 = x
    x = shuffle_unit(x, c(96), 2)                          # /16
    for _ in range(3):
        x = shuffle_unit(x, c(96))
    return {8: f8, 16: x}


# ---------------- head ----------------
def head_center(f, c=48):
    """CenterNet / FOMO+box: heatmap(7) + wh(2) + offset(2), 1 scale."""
    x = dwsep(f, c)
    return L.Conv2D(NC + 4, 1)(x)


def head_yolo(f, c=48, A=3):
    x = dwsep(f, c)
    return L.Conv2D(A * (5 + NC), 1)(x)


def head_gfl(f, c=48, reg_max=7):
    """NanoDet/PicoDet-style: cls(7) + 4*(reg_max+1) bins phân phối khoảng cách."""
    x = dwsep(f, c)
    return L.Conv2D(NC + 4 * (reg_max + 1), 1)(x)


def head_fomo(f, c=32):
    """FOMO gốc: chỉ heatmap (NC + background), không có box."""
    return L.Conv2D(NC + 1, 1)(cbr(f, c, 1))


HEADS = {"center": head_center, "yolo": head_yolo, "gfl": head_gfl, "fomo": head_fomo}
BACKBONES = {"mnv2": bb_mnv2, "mnv1": bb_mnv1, "mnv3s": bb_mnv3s, "ghost": bb_ghost, "shuffle": bb_shuffle}


def build(backbone, head="center", size=(128, 128), stride=8, bb_kw=None, head_c=48, fpn=False, up="resize"):
    """size = (H, W). fpn=True: thêm nhánh stride 16 upsample + add vào stride 8 (PAN-lite).
    up="upsampling": keras UpSampling2D (Keras 3 convert ra TILE – KHÔNG có trong TFLM);
    up="resize": keras Resizing nearest -> RESIZE_NEAREST_NEIGHBOR."""
    inp = L.Input((size[0], size[1], 3))
    bb_kw = dict(bb_kw or {})
    strides = (8, 16) if fpn else (stride,)
    feats = BACKBONES[backbone](inp, size, strides=strides, **bb_kw)
    if fpn:
        f8, f16 = feats[8], feats[16]
        c = head_c
        x16 = cbr(f16, c, 1)
        if up == "upsampling":
            up_ = L.UpSampling2D(2, interpolation="nearest")(x16)
        else:
            up_ = L.Resizing(f8.shape[1], f8.shape[2], interpolation="nearest")(x16)
        f = L.Add()([cbr(f8, c, 1), up_])
    else:
        f = feats[stride]
    hk = {} if head == "fomo" else {"c": head_c}
    out = HEADS[head](f, **hk)
    tag = "".join(f"{k[0]}{v}" if not isinstance(v, bool) else ("min" if v else "full") for k, v in bb_kw.items())
    name = f"{backbone}-{tag}_{head}_{'fpn8' if fpn else f's{stride}'}{'-tile' if fpn and up == 'upsampling' else ''}_{size[0]}x{size[1]}"
    return tf.keras.Model(inp, out, name=name)


# ---------------- cấu hình probe (Giai đoạn 1) ----------------
# Width của mỗi họ được chọn (sweep, xem docs/01_survey) để toàn detector ~130–200K params,
# tức là so sánh các họ ở CÙNG ngân sách tham số.
FAMILY = {
    "mnv2": {"alpha": 0.5},
    "mnv1": {"alpha": 0.3},
    "ghost": {"w": 1.75},
    "shuffle": {"w": 2.25},
    "mnv3s": {"alpha": 1.0, "minimalistic": True},
}
CONFIGS = [
    # 1) Họ backbone, cùng ngân sách, input 128x128: head 1 scale stride 16 vs PAN-lite về stride 8
    *[dict(backbone=b, bb_kw=kw, stride=16) for b, kw in FAMILY.items()],
    *[dict(backbone=b, bb_kw=kw, fpn=True) for b, kw in FAMILY.items()],
    # 2) MobileNetV3-Small đầy đủ (SE + h-swish) để đo chi phí op lạ
    dict(backbone="mnv3s", bb_kw={"alpha": 0.75, "minimalistic": False}, stride=16),
    # 3) Head: cùng backbone mnv2-0.5 + PAN-lite stride 8
    *[dict(backbone="mnv2", bb_kw=FAMILY["mnv2"], fpn=True, head=h) for h in ["yolo", "gfl", "fomo"]],
    # 4) Input size
    *[dict(backbone=b, bb_kw=FAMILY[b], fpn=True, size=s)
      for b in ["mnv2", "ghost", "shuffle"] for s in [(96, 96), (160, 160), (96, 128)]],
    dict(backbone="mnv2", bb_kw=FAMILY["mnv2"], fpn=True, size=(240, 320)),
    # 5) Bẫy convert: UpSampling2D của Keras 3 -> TILE
    dict(backbone="mnv2", bb_kw=FAMILY["mnv2"], fpn=True, up="upsampling"),
    # 6) Model rất nhỏ kiểu FOMO (backbone cắt ở stride 8, ~20K params) – để thử arena trong SRAM nội
    dict(backbone="mnv2", bb_kw={"alpha": 0.35}, stride=8, head="fomo", size=(96, 96)),
    dict(backbone="mnv2", bb_kw={"alpha": 0.35}, stride=8, size=(96, 96)),
    dict(backbone="mnv1", bb_kw={"alpha": 0.25}, stride=8, size=(96, 96)),
    dict(backbone="mnv1", bb_kw={"alpha": 0.25}, stride=8, size=(128, 128)),
]
