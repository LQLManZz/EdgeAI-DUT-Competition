"""TextCNN (Kim 2014) bằng Keras, viết sao cho đổi sang TFLite INT8 chỉ ra các op cơ bản của TFLite Micro.

  ids (L,) --Embedding--> (L,E) --Reshape--> (L,1,E)
     --Conv2D (k×1) + ReLU, k = 1..4--> --MaxPool trên toàn câu--> mỗi kernel 1 vector nf số
  nối tất cả --Dropout--> Dense(5) = logits 5 intent

Op sau khi đổi: GATHER (embedding), RESHAPE, CONV_2D, MAX_POOL_2D, CONCATENATION, FULLY_CONNECTED.
So với bản PyTorch gốc: Conv1d -> Conv2D(k×1) và max-over-time -> MaxPool2D (cùng phép toán, nhưng là op
quen thuộc nhất của TFLM); bỏ phần che vị trí PAD khi max-pool (cần thêm op NOT_EQUAL/CAST, dễ vỡ khi INT8)
-> model tự học vector cho <pad>, giống bản TextCNN tham chiếu mà nhóm intent dẫn link.

Config (configs/textcnn_*.json), khóa "branches":
  {"word": {"E": 64, "nf": 96, "ks": [1,2,3,4]}}             -> 1 nhánh theo từ
  {"char": {...}}                                             -> 1 nhánh theo ký tự (chịu được từ lạ)
  {"word": {...}, "char": {...}}                              -> 2 nhánh, nối trước lớp Dense
"""
import tensorflow as tf

keras = tf.keras
L_DEFAULT = {"word": 28, "char": 128}       # câu dài nhất: 21 từ / 85 ký tự (train và test)


@keras.utils.register_keras_serializable(package="textcnn")
class GatherEmbedding(keras.layers.Embedding):
    """Embedding = tra bảng: lấy hàng thứ id. Keras 3 tự thêm phần xử lý id âm (LESS/ADD/SELECT_V2 khi đổi TFLite)
    -> gọi thẳng tf.gather để file INT8 chỉ còn 1 op GATHER."""

    def call(self, inputs):
        return tf.gather(self.embeddings, inputs)


def build(cfg, vocab_sizes, n_classes=5, batch_size=None):
    """batch_size=1 khi đổi TFLite: shape cố định -> Reshape/Flatten không sinh op tính shape."""
    inputs, feats = [], []
    for kind, b in cfg["branches"].items():
        L = cfg.get("L", L_DEFAULT)[kind]
        ids = keras.Input((L,), batch_size=batch_size, dtype="int32", name=f"{kind}_ids")
        e = GatherEmbedding(vocab_sizes[kind], b["E"], name=f"{kind}_emb")(ids)
        e = keras.layers.Reshape((L, 1, b["E"]))(e)
        for k in b["ks"]:
            h = keras.layers.Conv2D(b["nf"], (k, 1), activation="relu", name=f"{kind}_conv{k}")(e)
            feats.append(keras.layers.MaxPooling2D((L - k + 1, 1), name=f"{kind}_max{k}")(h))   # -> (1,1,nf)
        inputs.append(ids)
    x = keras.layers.Concatenate()(feats) if len(feats) > 1 else feats[0]
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(cfg.get("dropout", 0.5))(x)
    return keras.Model(inputs, keras.layers.Dense(n_classes, name="logits")(x), name="textcnn")
