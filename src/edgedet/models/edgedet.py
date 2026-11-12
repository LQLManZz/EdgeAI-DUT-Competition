"""EdgeDet – Tiny Object Detector cho ESP32-S3 (TFLite Micro, INT8, <= 200K params).

Các biến thể (Giai đoạn 2, xem docs/02_architecture/):
  backbone="custom"  (H1-C): inverted residual không SE/h-swish, số kênh bội 16, train từ đầu.
  backbone="mnv3s"   (H1-P): MobileNetV3-Small minimalistic α=1.0 (có thể nạp ImageNet), cắt ở stride 16.
  head="center"      (H1)  : CenterNet – heatmap(C) + wh(2) + offset(2) ở stride 8.
  head="yolo"        (H2)  : anchor-based – A*(5+C) ở stride 8.

Quy tắc thiết kế (từ GĐ1, docs/01_survey/survey_report.md §6):
  L1 kênh bội 8/16 · L2 không SE/h-swish/Ghost/shuffle · L3 downsample sớm · L4 PAN-lite stride16->8 bằng Resizing
  L5 head anchor-free · L9 không dùng UpSampling2D (ra TILE), không Lambda.
Op sinh ra khi convert INT8: CONV_2D, DEPTHWISE_CONV_2D, ADD, RESIZE_NEAREST_NEIGHBOR (đã kiểm trên kit).
"""
import math
import tensorflow as tf
from tensorflow.keras import layers as L

NUM_CLASSES = 7

# (tên, expand, out, stride) cho backbone custom. f8 = output cuối ở stride 8, f16 = output cuối ở stride 16.
CUSTOM_BLOCKS = [
    ("b1", None, 16, 2),   # dw-separable /4
    ("b2", 64, 32, 2),     # /8
    ("b3", 128, 32, 1),    # /8
    ("b3b", 128, 32, 1),   # /8  -> f8 (thêm năng lực ở stride 8 cho vật nhỏ)
    ("b4", 128, 48, 2),    # /16
    ("b5", 192, 48, 1),
    ("b6", 192, 64, 1),
    ("b7", 256, 64, 1),
    ("b8", 256, 64, 1),    # /16 -> f16
]


def _cbr(x, c, k, s, name, act=True):
    x = L.Conv2D(c, k, s, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = L.BatchNormalization(name=f"{name}_bn")(x)
    return L.ReLU(6.0, name=f"{name}_relu")(x) if act else x


def _dw(x, s, name, act=True):
    x = L.DepthwiseConv2D(3, s, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = L.BatchNormalization(name=f"{name}_dwbn")(x)
    return L.ReLU(6.0, name=f"{name}_dwrelu")(x) if act else x


def _ir(x, expand, out, s, name):
    """Inverted residual (MobileNetV2/V3-minimalistic): 1x1 expand -> dw3x3 -> 1x1 linear (+skip)."""
    cin = x.shape[-1]
    y = _cbr(x, expand, 1, 1, f"{name}_exp") if expand else x
    y = _dw(y, s, name)
    y = _cbr(y, out, 1, 1, f"{name}_proj", act=False)
    if s == 1 and cin == out:
        y = L.Add(name=f"{name}_add")([x, y])
    return y


def backbone_custom(inp, blocks=CUSTOM_BLOCKS, stem=16):
    x = _cbr(inp, stem, 3, 2, "stem")                       # /2
    feats = {}
    stride = 2
    for name, e, o, s in blocks:
        x = _ir(x, e, o, s, name)
        stride *= s
        feats[stride] = x
    return feats[8], feats[16]


# Khối thêm (train từ đầu) sau phần pretrained của H1-P: (tên, expand, out, stride)
MNV3S_EXTRA8 = [("x8a", 96, 32, 1)]
MNV3S_EXTRA16 = [("x16a", 192, 64, 1), ("x16b", 256, 64, 1)]


def backbone_mnv3s(inp, pretrained=False, extra8=MNV3S_EXTRA8, extra16=MNV3S_EXTRA16, relu6=False):
    """MobileNetV3-Small minimalistic α=1.0, cắt ở stride 16 (expanded_conv_7_add, 48 kênh).
    pretrained=True nạp trọng số ImageNet của Keras (phải khai báo theo thể lệ). Input mong đợi trong [-1, 1].
    relu6=True: chặn mọi ReLU của Keras MobileNetV3 (vốn KHÔNG chặn trên, keras/.../mobilenet_v3.py::relu) ở 6.
      Lý do: thí nghiệm #2 mất 4 điểm mAP khi lên INT8; activation không chặn -> khoảng giá trị rộng -> INT8 chia thô."""
    m = tf.keras.applications.MobileNetV3Small(
        input_tensor=inp, alpha=1.0, minimalistic=True, include_top=False,
        weights="imagenet" if pretrained else None, include_preprocessing=False)
    if relu6:
        for layer in m.layers:   # max_value được đọc lúc chạy và lưu trong get_config -> .keras/TFLite đều thấy ReLU6
            if isinstance(layer, L.ReLU) and layer.max_value is None:
                layer.max_value = 6.0
    f8, f16 = m.get_layer("expanded_conv_2_add").output, m.get_layer("expanded_conv_7_add").output
    for name, e, o, st in extra8:
        f8 = _ir(f8, e, o, st, name)
    for name, e, o, st in extra16:
        f16 = _ir(f16, e, o, st, name)
    return f8, f16


def neck_pan_lite(f8, f16, c=64):
    """stride 16 -> 1x1 -> nearest x2 (RESIZE_NEAREST_NEIGHBOR) + 1x1(stride 8) -> dw-sep 3x3."""
    a = _cbr(f16, c, 1, 1, "neck_l16")
    a = L.Resizing(f8.shape[1], f8.shape[2], interpolation="nearest", name="neck_up")(a)
    b = _cbr(f8, c, 1, 1, "neck_l8")
    x = L.Add(name="neck_add")([a, b])
    x = _dw(x, 1, "neck_fuse")
    return _cbr(x, c, 1, 1, "neck_fuse_pw")


def head_center(x, num_classes=NUM_CLASSES, c=64, decoupled=False):
    """CenterNet head, 2 output tách riêng để mỗi output có scale INT8 riêng:
      hm  : (H/8, W/8, C)  logits heatmap (sigmoid khi decode)
      box : (H/8, W/8, 4)  [log(w/8), log(h/8), off_x, off_y] – w,h tính theo pixel input, offset trong ô (0..1)
    decoupled=True: tách riêng nhánh dw3x3+pw1x1 cho hm (phân loại) và box (căn khung), tận dụng ngân sách <=200K params.
    """
    xh = _dw(x, 1, "head")
    xh = _cbr(xh, c, 1, 1, "head_pw")
    xb = _cbr(_dw(x, 1, "head_b"), c, 1, 1, "head_b_pw") if decoupled else xh
    prior = -math.log((1 - 0.01) / 0.01)   # focal-loss prior p=0.01 cho heatmap
    hm = L.Conv2D(num_classes, 1, name="head_hm", bias_initializer=tf.keras.initializers.Constant(prior))(xh)
    box = L.Conv2D(4, 1, name="head_box")(xb)
    return [hm, box]


def head_yolo(x, num_classes=NUM_CLASSES, c=64, anchors=3):
    """Output thô [A x (tx, ty, tw, th, obj, cls...)] ở stride 8."""
    x = _dw(x, 1, "head")
    x = _cbr(x, c, 1, 1, "head_pw")
    return L.Conv2D(anchors * (5 + num_classes), 1, name="head_out")(x)


def build_edgedet(backbone="custom", head="center", input_hw=(128, 160), num_classes=NUM_CLASSES,
                  neck_c=80, head_c=80, pretrained=False, blocks=CUSTOM_BLOCKS, mnv3_relu6=False,
                  decoupled_head=False):
    h, w = input_hw
    assert h % 16 == 0 and w % 16 == 0, "input phải chia hết cho 16 (PAN-lite stride 16 -> 8)"
    inp = L.Input((h, w, 3), name="image")
    if backbone == "custom":
        f8, f16 = backbone_custom(inp, blocks)
    elif backbone == "mnv3s":
        f8, f16 = backbone_mnv3s(inp, pretrained, relu6=mnv3_relu6)
    else:
        raise ValueError(backbone)
    x = neck_pan_lite(f8, f16, neck_c)
    out = head_center(x, num_classes, head_c, decoupled=decoupled_head) if head == "center" else head_yolo(x, num_classes, head_c)
    name = f"edgedet_{'c' if backbone == 'custom' else 'p'}_{head}_{h}x{w}"
    return tf.keras.Model(inp, out, name=name)


def folded_params(model):
    """Số tham số sau khi gộp BatchNorm vào conv (= số hằng số weights+bias trong .tflite)."""
    n = 0
    for l in model.layers:
        if isinstance(l, L.DepthwiseConv2D):   # Keras 3: kernel (kh, kw, C, mult)
            k = l.kernel.shape
            n += math.prod(k) + k[2] * k[3]
        elif isinstance(l, L.Conv2D):
            n += math.prod(l.kernel.shape) + l.filters
    return n
