"""Hàm loss cho head CenterNet.

Bài giảng: docs/03_training/learning_guide.md, Bài 4.

Loss là con số đo "model sai bao nhiêu". Gradient descent chỉnh trọng số để con số này giảm dần.
Model có 2 output nên có 2 loss; Keras cộng chúng lại theo loss_weights (xem train/train.py):
  heatmap_focal_loss – cho output "head_hm"
  box_l1_loss        – cho output "head_box"
"""
import tensorflow as tf


def heatmap_focal_loss(y_true, logits, alpha=2.0, beta=4.0):
    """Focal loss biến thể CenterNet.

    Vấn đề: lưới 18x24x7 có ~3000 ô, nhưng mỗi vật chỉ có 1 ô "đúng". Nếu dùng cross-entropy thường,
    hàng nghìn ô nền (dễ đoán = 0) sẽ áp đảo loss, model học cách đoán "không có gì" ở mọi nơi.
    Focal loss sửa bằng 2 hệ số:
      (1-p)^alpha / p^alpha – ô đã đoán đúng (dễ) gần như không đóng góp loss -> tập trung vào ô khó
      (1-y)^beta            – ô nền nằm gần tâm vật (y gần 1) bị phạt nhẹ, vì đoán cao ở đó cũng "gần đúng"
    Chia cho số vật trong batch để loss không phụ thuộc vào việc ảnh có nhiều hay ít vật.
    """
    y_true = tf.cast(y_true, logits.dtype)
    p = tf.sigmoid(logits)
    log_p, log_1mp = tf.math.log_sigmoid(logits), tf.math.log_sigmoid(-logits)   # ổn định số học hơn log(sigmoid)
    pos = tf.cast(y_true >= 1.0 - 1e-4, logits.dtype)                             # ô chứa tâm vật (đỉnh đồi = 1)
    pos_loss = -pos * tf.pow(1 - p, alpha) * log_p
    neg_loss = -(1 - pos) * tf.pow(1 - y_true, beta) * tf.pow(p, alpha) * log_1mp
    n_pos = tf.maximum(tf.reduce_sum(pos), 1.0)
    return (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / n_pos


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
        return tf.reduce_mean(tf.square(x), axis=list(range(1, len(y_pred.shape))))

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}


def box_l1_loss(y_true, pred):
    """L1 = |đáp án - dự đoán|, chỉ tính tại ô có tâm vật (mask = kênh thứ 5 của y_true).

    Ở các ô nền, 4 số box không có nghĩa gì (không có vật thì không có kích thước), nên bỏ qua.
    """
    y_true = tf.cast(y_true, pred.dtype)
    target, mask = y_true[..., :4], y_true[..., 4:5]
    n = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(tf.abs(pred - target) * mask) / n
