"""Hàm loss cho head CenterNet.

Bài giảng: docs/03_training/learning_guide.md, Bài 4.

Loss là con số đo "model sai bao nhiêu". Gradient descent chỉnh trọng số để con số này giảm dần.
Model có 2 output nên có 2 loss; Keras cộng chúng lại theo loss_weights (xem train/train.py):
  heatmap_focal_loss – cho output "head_hm"
  box_l1_loss        – cho output "head_box"
"""
import tensorflow as tf


def heatmap_focal_loss(y_true, logits, alpha=2.0, beta=4.0, logit_clip=8.0, logit_reg=0.02):
    """Focal loss biến thể CenterNet + phạt nhẹ logits quá lớn để thu hẹp dải lượng tử INT8 của head_hm.

    Vấn đề: lưới 18x24x7 có ~3000 ô, nhưng mỗi vật chỉ có 1 ô "đúng". Nếu dùng cross-entropy thường,
    hàng nghìn ô nền (dễ đoán = 0) sẽ áp đảo loss, model học cách đoán "không có gì" ở mọi nơi.
    Focal loss sửa bằng 2 hệ số:
      (1-p)^alpha / p^alpha – ô đã đoán đúng (dễ) gần như không đóng góp loss -> tập trung vào ô khó
      (1-y)^beta            – ô nền nằm gần tâm vật (y gần 1) bị phạt nhẹ, vì đoán cao ở đó cũng "gần đúng"
    Ngoài [-logit_clip, +logit_clip] (p < 0.0003 hoặc p > 0.9997), phạt nhẹ phần vượt ngưỡng để head_hm không
    trôi xuống -15..-20 gây lãng phí 80% số mức INT8 ở vùng nền.
    """
    y_true = tf.cast(y_true, logits.dtype)
    p = tf.sigmoid(logits)
    log_p, log_1mp = tf.math.log_sigmoid(logits), tf.math.log_sigmoid(-logits)   # ổn định số học hơn log(sigmoid)
    pos = tf.cast(y_true >= 1.0 - 1e-4, logits.dtype)                             # ô chứa tâm vật (đỉnh đồi = 1)
    pos_loss = -pos * tf.pow(1 - p, alpha) * log_p
    neg_loss = -(1 - pos) * tf.pow(1 - y_true, beta) * tf.pow(p, alpha) * log_1mp
    n_pos = tf.maximum(tf.reduce_sum(pos), 1.0)
    focal = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / n_pos
    over = tf.nn.relu(tf.abs(logits) - logit_clip)
    return focal + logit_reg * tf.reduce_mean(tf.square(over))


@tf.keras.utils.register_keras_serializable("edgedet")
class ActRangePenalty(tf.keras.losses.Loss):
    """Phạt activation có |x| vượt ngưỡng – để khoảng giá trị (range) hẹp lại, INT8 bớt thô.

    Vì sao: thí nghiệm #2 (backbone pretrained) có activation tới ~309 -> mất 4 điểm mAP khi lên INT8.
    Thí nghiệm #3 CẮT cứng ở 6 -> đặc trưng pretrained bị phá (FP32 0.67). Ở đây chỉ PHẠT nhẹ phần vượt ngưỡng:
    loss = mean( max(|x| - threshold, 0)^2 ). Giá trị dưới ngưỡng không bị phạt; giá trị càng vượt xa càng bị phạt
    nặng, và luôn có gradient -> model tự học cách thu nhỏ outlier mà vẫn giữ đặc trưng.
    y_true không dùng (chỉ là số 0 giả để Keras chấp nhận output này có loss).
    """

    def __init__(self, threshold=16.0, name="act_range_penalty", **kw):
        super().__init__(name=name, **kw)
        self.threshold = float(threshold)

    def call(self, y_true, y_pred):
        x = tf.nn.relu(tf.abs(y_pred) - self.threshold)
        # Dạng Huber: bậc hai khi vượt ngưỡng <= 8, tuyến tính khi > 8 để tránh nổ đạo hàm ở 40 epoch đầu
        huber = tf.where(x <= 8.0, tf.square(x), 16.0 * x - 64.0)
        return tf.reduce_mean(huber, axis=list(range(1, len(y_pred.shape))))

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


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
