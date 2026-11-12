"""Hàm loss cho head CenterNet.

Bài giảng: docs/03_training/learning_guide.md, Bài 4.

Loss là con số đo "model sai bao nhiêu". Gradient descent chỉnh trọng số để con số này giảm dần.
Model có 2 output nên có 2 loss; Keras cộng chúng lại theo loss_weights (xem train/train.py):
  heatmap_focal_loss – cho output "head_hm"
  box_l1_loss        – cho output "head_box"
"""
import tensorflow as tf


def _ste_fake_quant8(x):
    """Mô phỏng lượng tử hóa INT8 (256 mức) bằng Straight-Through Estimator (STE) ngay trong hàm loss.
    Forward đi qua lưới 256 mức; backward giữ nguyên đạo hàm của x -> không thêm bất kỳ op nào vào file .tflite."""
    lo = tf.stop_gradient(tf.reduce_min(x))
    hi = tf.stop_gradient(tf.reduce_max(x))
    scale = tf.maximum((hi - lo) / 255.0, 1e-4)
    xq = tf.round((x - lo) / scale) * scale + lo
    return x + tf.stop_gradient(xq - x)


def _focal_core(y_true, logits, pos, n_pos, alpha=2.0, beta=4.0):
    p = tf.sigmoid(logits)
    log_p, log_1mp = tf.math.log_sigmoid(logits), tf.math.log_sigmoid(-logits)
    pos_loss = -pos * tf.pow(1 - p, alpha) * log_p
    neg_loss = -(1 - pos) * tf.pow(1 - y_true, beta) * tf.pow(p, alpha) * log_1mp
    return (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / n_pos


def heatmap_focal_loss(y_true, logits, alpha=2.0, beta=4.0, logit_clip=8.0, logit_reg=0.02):
    """Focal loss biến thể CenterNet + STE FakeQuant INT8 + phạt nhẹ logits quá lớn để thu hẹp dải lượng tử INT8."""
    y_true = tf.cast(y_true, logits.dtype)
    pos = tf.cast(y_true >= 1.0 - 1e-4, logits.dtype)                             # ô chứa tâm vật (đỉnh đồi = 1)
    n_pos = tf.maximum(tf.reduce_sum(pos), 1.0)
    focal_fp32 = _focal_core(y_true, logits, pos, n_pos, alpha, beta)
    focal_int8 = _focal_core(y_true, _ste_fake_quant8(logits), pos, n_pos, alpha, beta)
    focal = 0.7 * focal_fp32 + 0.3 * focal_int8
    over = tf.nn.relu(tf.abs(logits) - logit_clip)
    return focal + logit_reg * tf.reduce_mean(tf.square(over))


@tf.keras.utils.register_keras_serializable("edgedet")
class ActRangePenalty(tf.keras.losses.Loss):
    """Phạt activation có |x| vượt ngưỡng – để khoảng giá trị (range) hẹp lại, INT8 bớt thô.

    Dùng bậc hai x^2 cho toàn bộ dải vượt ngưỡng <= 48 (tức |y_pred| <= 64, giống hệt h1p_v2 chỉ mất 1,01% INT8),
    và chỉ chặn tuyến tính khi vượt quá 48 để tránh nổ đạo hàm ở vài epoch đầu tiên khi activation ~180–250.
    """

    def __init__(self, threshold=16.0, name="act_range_penalty", **kw):
        super().__init__(name=name, **kw)
        self.threshold = float(threshold)

    def call(self, y_true, y_pred):
        x = tf.nn.relu(tf.abs(y_pred) - self.threshold)
        quad_capped = tf.where(x <= 48.0, tf.square(x), 96.0 * x - 2304.0)
        return tf.reduce_mean(quad_capped, axis=list(range(1, len(y_pred.shape))))

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


@tf.keras.utils.register_keras_serializable("edgedet")
class KDHmLoss(tf.keras.losses.Loss):
    """Chưng cất tri thức (Knowledge Distillation) cho Heatmap từ Teacher FP32 sang Student (có STE INT8).
    y_pred = concat([hm_student (C), hm_teacher (C)], axis=-1)."""

    def __init__(self, num_classes=7, name="kd_hm_loss", **kw):
        super().__init__(name=name, **kw)
        self.num_classes = int(num_classes)

    def call(self, y_true, y_pred):
        c = self.num_classes
        s_logits = y_pred[..., :c]
        t_logits = tf.stop_gradient(y_pred[..., c:2 * c])
        s_q = _ste_fake_quant8(s_logits)
        p_t = tf.sigmoid(t_logits)
        p_s = tf.sigmoid(s_q)
        # Trọng số Quality-Focal: tập trung mạnh vào các đỉnh vật của Teacher (p_t cao) và nơi Student đoán lệch
        w = tf.square(tf.abs(p_t - p_s)) * (1.0 + 4.0 * p_t)
        kl = (tf.nn.sigmoid_cross_entropy_with_logits(labels=p_t, logits=s_q)
              - tf.nn.sigmoid_cross_entropy_with_logits(labels=p_t, logits=t_logits))
        n_pos = tf.maximum(tf.reduce_sum(tf.cast(p_t > 0.3, y_pred.dtype), axis=[1, 2, 3]), 1.0)
        return tf.reduce_sum(w * tf.maximum(kl, 0.0), axis=[1, 2, 3]) / n_pos

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}


@tf.keras.utils.register_keras_serializable("edgedet")
class KDBoxLoss(tf.keras.losses.Loss):
    """Chưng cất tri thức Box (L1 + GIoU) từ Teacher sang Student tại mọi ô mà Teacher tự tin (max(p_t) >= 0.25).
    y_pred = concat([box_student (4), box_teacher (4), hm_teacher (C)], axis=-1)."""

    def __init__(self, thr=0.25, name="kd_box_loss", **kw):
        super().__init__(name=name, **kw)
        self.thr = float(thr)

    def call(self, y_true, y_pred):
        s_box = y_pred[..., :4]
        t_box = tf.stop_gradient(y_pred[..., 4:8])
        t_hm = tf.stop_gradient(y_pred[..., 8:])
        t_obj = tf.reduce_max(tf.sigmoid(t_hm), axis=-1, keepdims=True)
        mask = tf.where(t_obj >= self.thr, t_obj, tf.zeros_like(t_obj))
        pseudo_target = tf.concat([t_box, mask], axis=-1)
        return box_l1_loss(pseudo_target, s_box, giou_weight=1.0)

    def get_config(self):
        return {**super().get_config(), "thr": self.thr}


def box_l1_loss(y_true, pred, giou_weight=1.0):
    """Loss hộp giới hạn = L1([log(w/8), log(h/8), ox, oy]) + giou_weight * (1 - GIoU), có trọng số theo mask.

    Ở các ô có mask > 0 (tâm vật = 1.0, 4 ô sát tâm = 0.25):
      - L1 kéo từng thành phần về đáp án ổn định lúc đầu.
      - GIoU tối ưu trực tiếp độ trùng khít hình chữ nhật (thước đo IoU >= 0.5 của Kaggle).
    """
    y_true = tf.cast(y_true, pred.dtype)
    target, mask = y_true[..., :4], y_true[..., 4:5]
    n = tf.maximum(tf.reduce_sum(mask), 1.0)
    l1 = tf.reduce_sum(tf.abs(pred - target) * mask) / n
    if giou_weight <= 0:
        return l1

    wt, ht = tf.exp(tf.clip_by_value(target[..., 0], -6.0, 6.0)), tf.exp(tf.clip_by_value(target[..., 1], -6.0, 6.0))
    wp, hp = tf.exp(tf.clip_by_value(pred[..., 0], -6.0, 6.0)), tf.exp(tf.clip_by_value(pred[..., 1], -6.0, 6.0))
    oxt, oyt, oxp, oyp = target[..., 2], target[..., 3], pred[..., 2], pred[..., 3]

    x1t, y1t, x2t, y2t = oxt - wt * 0.5, oyt - ht * 0.5, oxt + wt * 0.5, oyt + ht * 0.5
    x1p, y1p, x2p, y2p = oxp - wp * 0.5, oyp - hp * 0.5, oxp + wp * 0.5, oyp + hp * 0.5

    inter_w = tf.maximum(0.0, tf.minimum(x2p, x2t) - tf.maximum(x1p, x1t))
    inter_h = tf.maximum(0.0, tf.minimum(y2p, y2t) - tf.maximum(y1p, y1t))
    inter = inter_w * inter_h
    union = wp * hp + wt * ht - inter + 1e-7
    iou = inter / union

    encl_w = tf.maximum(0.0, tf.maximum(x2p, x2t) - tf.minimum(x1p, x1t))
    encl_h = tf.maximum(0.0, tf.maximum(y2p, y2t) - tf.minimum(y1p, y1t))
    encl = encl_w * encl_h + 1e-7
    giou = iou - (encl - union) / encl
    giou_loss = tf.reduce_sum((1.0 - giou)[..., None] * mask) / n
    return l1 + giou_weight * giou_loss
