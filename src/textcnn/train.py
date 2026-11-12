"""Pipeline TextCNN intent: CV -> stress test -> train 100% -> TFLite INT8 -> kiểm tra luật -> intent_submission.csv

  python -m textcnn.train --config configs/textcnn_word.json --data data/raw --out experiments/textcnn_word
  (trên Colab chạy qua scripts/colab_intent.py, không gọi trực tiếp)

Tập test KHÔNG có nhãn -> accuracy chỉ đo được trên train, theo 2 cách:
  cv.json      Chia 5 fold THEO KHUÔN CÂU (GroupKFold): khuôn câu ở fold kiểm tra chưa từng thấy lúc train.
  stress.json  Bỏ hẳn 1 cụm từ (vd 'vi tri') khỏi train rồi kiểm tra trên các câu chứa nó.
               Test có 99 câu chứa từ chưa từng gặp ('lech', 'kiem'), nên đây là thước đo phân biệt các model;
               CV thường đạt ~100% cho mọi cấu hình.
Mỗi model được đo cả FP32 (Keras) lẫn INT8 (file .tflite sẽ nộp). Điểm thật là cột INT8.

Kết quả trong --out:
  cv.json, cv_errors.csv, stress.json, stress.csv     đánh giá
  model.keras, model_int8.tflite, vocab.json, log.csv model cuối (train trên 100% dữ liệu, 1 model, không ensemble)
  intent_submission.csv                               4000 dòng INTENT, dự đoán bằng file INT8
  report.json                                         tóm tắt + các kiểm tra luật thi
Mỗi bước bỏ qua nếu đã có file kết quả của nó -> chạy lại bao nhiêu lần cũng được.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from textcnn import text as T
from textcnn.model import L_DEFAULT, build, keras

LIMITS = {"params": 200_000, "file_bytes": int(1.5 * 1024 * 1024)}       # Thể lệ mục V, dòng "Intent model"
TFLM_BASIC_OPS = {"GATHER", "RESHAPE", "CONV_2D", "MAX_POOL_2D", "CONCATENATION", "FULLY_CONNECTED"}


# ------------------------------------------------------------------ train 1 model
def encode_all(cfg, vocabs, X):
    """-> {'word_ids': (N, 28) int32, 'char_ids': (N, 128) int32}, khóa trùng tên Input của model."""
    L = cfg.get("L", L_DEFAULT)
    return {f"{k}_ids": T.encode(X, k, vocabs[k], L[k]) for k in cfg["branches"]}


def fit(cfg, X, y, classes, seed=0, verbose=False):
    keras.utils.set_random_seed(seed)
    X, y = T.augment_oos(X, y, classes, cfg.get("aug_ratio", 0.5), seed)
    vocabs = {k: T.build_vocab(X, k) for k in cfg["branches"]}
    model = build(cfg, {k: len(v) for k, v in vocabs.items()}, len(classes))

    p, bs, epochs, lr = cfg.get("word_dropout", 0.1), cfg.get("bs", 64), cfg.get("epochs", 25), cfg.get("lr", 2e-3)
    wb_drop = cfg.get("word_branch_dropout", 0.0)

    def token_dropout(x, yy):   # thay ngẫu nhiên p% token bằng <unk> -> model quen với từ lạ
        out = {n: tf.where((tf.random.uniform(tf.shape(v)) > p) | (v == T.PAD), v, tf.ones_like(v) * T.UNK)
               for n, v in x.items()}
        if wb_drop > 0 and "word_ids" in out and len(out) > 1:
            # Với xác suất wb_drop trên từng câu, che toàn bộ word_ids thành <unk> để ép nhánh char độc lập đoán đúng
            v = out["word_ids"]
            keep = tf.random.uniform((tf.shape(v)[0], 1)) > wb_drop
            out["word_ids"] = tf.where(keep | (v == T.PAD), v, tf.ones_like(v) * T.UNK)
        return out, yy

    ds = (tf.data.Dataset.from_tensor_slices((encode_all(cfg, vocabs, X), keras.utils.to_categorical(y, len(classes))))
          .shuffle(len(y), seed=seed).batch(bs).map(token_dropout).prefetch(2))
    steps = epochs * math.ceil(len(y) / bs)
    warm = int(0.3 * steps)                 # lr tăng dần 30% đầu rồi giảm theo cos (tương đương OneCycle bản gốc)
    sched = keras.optimizers.schedules.CosineDecay(lr / 25, steps - warm, warmup_target=lr, warmup_steps=warm)
    model.compile(keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=cfg.get("label_smoothing", 0.05)),
                  metrics=["accuracy"])
    cb = [keras.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: print(
        f"   epoch {e + 1:2d}/{epochs}  loss {logs['loss']:.4f}  acc {logs['accuracy']:.4f}", flush=True)
        if (e + 1) % 5 == 0 else None)] if verbose else []
    hist = model.fit(ds, epochs=epochs, verbose=0, callbacks=cb)
    return model, vocabs, hist.history


# ------------------------------------------------------------------ INT8
def to_int8(model, cfg, vocabs, X_rep, n_rep=300):
    """PTQ: cho ~300 câu TRAIN chạy qua model để đo khoảng giá trị từng tensor, rồi lượng tử hóa INT8.
    Đầu vào giữ int32 (id token, không cần lượng tử); trọng số + activation + đầu ra là int8.
    Batch cố định = 1 (ESP32 xử lý từng câu): Reshape/Flatten thành hằng số, không sinh SHAPE/STRIDED_SLICE/PACK."""
    feats = encode_all(cfg, vocabs, list(X_rep)[:n_rep])
    fixed = build(cfg, {k: len(v) for k, v in vocabs.items()}, model.output_shape[-1], batch_size=1)
    fixed.set_weights(model.get_weights())

    def rep():   # theo TÊN input: model 2 nhánh bị converter xếp lại thứ tự input
        for i in range(len(next(iter(feats.values())))):
            yield {n: v[i:i + 1] for n, v in feats.items()}

    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_output_type = tf.int8
    return conv.convert()


class Int8Model:
    """Chạy file .tflite INT8 từng câu một, như trên ESP32. Đầu ra int8 là logits đã lượng tử -> argmax vẫn đúng."""

    def __init__(self, blob):
        self.it = tf.lite.Interpreter(model_content=blob)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details(), self.it.get_output_details()[0]

    def predict(self, feats):
        keys = [next(k for k in feats if k in d["name"]) for d in self.inp]   # input tflite tên 'word_ids'/'char_ids'
        preds = []
        for i in range(len(feats[keys[0]])):
            for d, k in zip(self.inp, keys):
                self.it.set_tensor(d["index"], feats[k][i:i + 1])
            self.it.invoke()
            preds.append(int(self.it.get_tensor(self.out["index"])[0].argmax()))
        return np.array(preds)


def compliance(model, blob):
    it = tf.lite.Interpreter(model_content=blob)
    ops = sorted({o["op_name"] for o in it._get_ops_details()})
    floats = [t["name"] for t in it.get_tensor_details() if t["dtype"] == np.float32]
    return {"params": int(model.count_params()), "file_bytes": len(blob), "ops": ops, "float32_tensors": floats[:5],
            "checks": {"params<=200K": model.count_params() <= LIMITS["params"],
                       "file<=1.5MB": len(blob) <= LIMITS["file_bytes"],
                       "full_int8 (no float32 tensor)": not floats,
                       "ops_basic_TFLM (chưa chạy thử trên kit)": set(ops) <= TFLM_BASIC_OPS}}


# ------------------------------------------------------------------ đánh giá
def eval_split(cfg, X, y, classes, tr, va):
    m, vocabs, _ = fit(cfg, X[tr], y[tr], classes)
    f = encode_all(cfg, vocabs, X[va])
    return m.predict(f, batch_size=512, verbose=0).argmax(1), Int8Model(to_int8(m, cfg, vocabs, X[tr])).predict(f)


def run_cv(cfg, X, y, classes, out, smoke):
    folds = list(GroupKFold(5).split(X, y, [T.template_key(s) for s in X]))[:1 if smoke else 5]
    oof32, oof8, acc8 = np.full(len(y), -1), np.full(len(y), -1), []
    for i, (tr, va) in enumerate(folds, 1):
        oof32[va], oof8[va] = eval_split(cfg, X, y, classes, tr, va)
        acc8.append(accuracy_score(y[va], oof8[va]))
        print(f"  CV fold {i}/5: FP32 {accuracy_score(y[va], oof32[va]):.4f}  INT8 {acc8[-1]:.4f}", flush=True)
    m = oof8 >= 0
    res = {"acc_fp32": accuracy_score(y[m], oof32[m]), "acc_int8": accuracy_score(y[m], oof8[m]),
           "macro_f1_int8": f1_score(y[m], oof8[m], average="macro"), "fold_acc_int8": [round(a, 4) for a in acc8],
           "classes": classes, "confusion_int8 (hàng=đúng, cột=đoán)": confusion_matrix(y[m], oof8[m]).tolist()}
    wrong = m & ((oof8 != y) | (oof32 != y))
    pd.DataFrame({"question": X[wrong], "intent": [classes[i] for i in y[wrong]],
                  "pred_int8": [classes[i] for i in oof8[wrong]], "pred_fp32": [classes[i] for i in oof32[wrong]]}
                 ).to_csv(out / "cv_errors.csv", index=False)
    return res


def run_stress(cfg, X, y, classes, out, smoke):
    nq = [f" {T.norm(s)} " for s in X]
    cues = [(it, c) for it, cs in T.CUES.items() for c in cs][:2 if smoke else None]
    rows = []
    for intent, cue in cues:
        held = np.array([f" {cue} " in s for s in nq])
        if held.sum() < 10 or (~held).sum() < 1000:
            continue
        va = np.where(held)[0]
        p32, p8 = eval_split(cfg, X, y, classes, np.where(~held)[0], va)
        rows.append({"cue": cue, "intent": intent, "n": len(va), "acc_fp32": round(accuracy_score(y[va], p32), 4),
                     "acc_int8": round(accuracy_score(y[va], p8), 4),
                     "errors_int8": "; ".join(f"{X[va][j]} -> {classes[p8[j]]}" for j in np.where(p8 != y[va])[0][:3])})
        r = rows[-1]
        print(f"  stress '{cue}' (n={r['n']}): FP32 {r['acc_fp32']:.4f}  INT8 {r['acc_int8']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stress.csv", index=False)
    pooled = lambda c: float((df[c] * df.n).sum() / df.n.sum())
    return {"pooled_acc_fp32": pooled("acc_fp32"), "pooled_acc_int8": pooled("acc_int8"), "n_cues": len(df),
            "n_sentences": int(df.n.sum()), "worst_int8": df.nsmallest(3, "acc_int8")[["cue", "acc_int8"]].to_dict("records")}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="chạy thử nhanh: 2 epoch, 1 fold, 2 cue")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.smoke:
        cfg["epochs"] = 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    df = pd.read_csv(data / "intent_train.csv")
    classes = sorted(df.intent.unique())
    X, y = df.question.to_numpy(dtype=object), df.intent.map({c: i for i, c in enumerate(classes)}).to_numpy()
    print(f"GPU: {[d.name for d in tf.config.list_physical_devices('GPU')] or 'không (CPU)'}   TF {tf.__version__}")
    print(f"config: {cfg}\ntrain={len(df)}  classes={classes}", flush=True)

    for name, step in (("cv", run_cv), ("stress", run_stress)):
        f = out / f"{name}.json"
        if f.exists():
            print(f"{name}: đã có {f.name} -> bỏ qua")
            continue
        print(f"\n== {name} ==", flush=True)
        f.write_text(json.dumps(step(cfg, X, y, classes, out, a.smoke), indent=2, ensure_ascii=False), encoding="utf-8")

    if (out / "report.json").exists():
        print("model cuối: đã có report.json -> bỏ qua")
        print((out / "report.json").read_text(encoding="utf-8"))
        return

    print("\n== model cuối: train trên 100% dữ liệu train ==", flush=True)
    m, vocabs, hist = fit(cfg, X, y, classes, verbose=True)
    m.save(out / "model.keras")
    blob = to_int8(m, cfg, vocabs, X)
    (out / "model_int8.tflite").write_bytes(blob)
    pd.DataFrame(hist).rename_axis("epoch").to_csv(out / "log.csv")
    (out / "vocab.json").write_text(json.dumps({     # firmware cần file này để biến câu hỏi thành id
        "classes": classes, "L": {k: cfg.get("L", L_DEFAULT)[k] for k in vocabs}, "pad_id": T.PAD, "unk_id": T.UNK,
        "norm": "lowercase; đ->d; bỏ dấu (NFD, xóa Mn); ký tự ngoài [a-z0-9 ] -> cách; gộp khoảng trắng",
        "vocab": vocabs}, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = {"config": cfg, **compliance(m, blob)}
    int8 = Int8Model(blob)
    rep["train_acc_int8 (dữ liệu đã thấy, chỉ để kiểm tra)"] = accuracy_score(y, int8.predict(encode_all(cfg, vocabs, X)))

    te = pd.read_csv(data / "intent_test.csv")
    ft = encode_all(cfg, vocabs, te.question.tolist())
    p8, p32 = int8.predict(ft), m.predict(ft, batch_size=512, verbose=0).argmax(1)
    pred = dict(zip(te.question_id, (classes[i] for i in p8)))
    sample = pd.read_csv(data / "sample_submission.csv")
    sub = sample[sample.task_type == "INTENT"][["task_type", "sample_id"]].copy()
    assert set(sub.sample_id) == set(pred) and len(sub) == len(pred), "ID test không khớp sample_submission"
    sub["prediction"] = sub.sample_id.map(pred)
    sub.to_csv(out / "intent_submission.csv", index=False)

    known = set(T.build_vocab(X, "word"))
    oov = np.array([any(w not in known for w in T.tokens(q, "word")) for q in te.question])
    rep["test (không có nhãn -> KHÔNG phải accuracy)"] = {
        "pred_counts": sub.prediction.value_counts().to_dict(),
        "fp32_int8_agreement": float((p8 == p32).mean()),
        "sentences_with_unseen_words": int(oov.sum()),
        "their_pred_counts": pd.Series([classes[i] for i in p8[oov]]).value_counts().to_dict()}
    for name in ("cv", "stress"):
        rep[name] = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("params", "file_bytes", "ops", "checks")}, indent=2, ensure_ascii=False))
    print(f"CV acc INT8 {rep['cv']['acc_int8']:.4f} | stress INT8 {rep['stress']['pooled_acc_int8']:.4f} | "
          f"test FP32/INT8 khớp {rep['test (không có nhãn -> KHÔNG phải accuracy)']['fp32_int8_agreement']:.4f}")
    if not all(rep["checks"].values()):
        print("!! CÓ KIỂM TRA KHÔNG ĐẠT – xem 'checks'")


if __name__ == "__main__":
    main()
