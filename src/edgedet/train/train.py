"""Train EdgeDet.

Bài giảng: docs/03_training/learning_guide.md, Bài 7.

Chạy (từ thư mục gốc dự án, có src/ trong PYTHONPATH):
  python -m edgedet.train.train --config configs/h1c_144x192.json --data data/raw --out experiments/h1c_144x192
Tùy chọn:
  --epochs N   ghi đè số epoch trong config      --limit N   chỉ dùng N ảnh train (chạy thử nhanh)
Bị ngắt giữa chừng (Colab hết phiên)? Chạy lại ĐÚNG lệnh cũ: script tự đọc last.keras và train tiếp.

Kết quả trong --out:
  last.keras   model sau epoch gần nhất (kèm trạng thái optimizer, để train tiếp)
  best.keras   model có mAP50 val cao nhất  -> dùng file này để xuất INT8
  log.csv      loss + val_map50 từng epoch  -> vẽ đường cong
  config.json  bản sao config (để biết model này được train thế nào)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgedet.data.dataset import CLASS_NAMES, load_items, make_dataset, split_train_val
from edgedet.eval.evaluate import evaluate, keras_predict_fn
from edgedet.losses.center_loss import ActRangePenalty, box_l1_loss, heatmap_focal_loss
from edgedet.models.edgedet import build_edgedet

DEFAULTS = {
    "backbone": "custom",       # "custom" (H1-C, train từ đầu) | "mnv3s" (H1-P, backbone pretrained ImageNet)
    "pretrained": False,
    "mnv3_relu6": False,        # chỉ với backbone mnv3s: chặn ReLU ở 6 (giảm mất mát INT8, xem experiments.md #3)
    "decoupled_head": False,    # tách riêng nhánh conv cho heatmap và box trong head_center
    "input_hw": [144, 192],
    "neck_c": 80, "head_c": 80,
    "epochs": 80, "batch": 64,
    "lr": 2e-3, "warmup_epochs": 3, "weight_decay": 0.01,
    "box_weight": 1.0,          # loss tổng = focal(heatmap) + box_weight * (L1 + GIoU)(box)
    "neighbor_weight": 0.25,    # trọng số giám sát box ở 4 ô sát tâm (0 = chỉ giám sát 1 ô tâm)
    "ema_decay": 0.9,           # trung bình trượt trọng số theo epoch ở nửa sau quá trình train (0 = tắt)
    "full_train": False,        # True = train trên 100% dữ liệu (6000 ảnh) để nộp bài cuối
    "aug": {},                  # ghi đè DEFAULT_AUG trong data/dataset.py
    "val_frac": 0.1, "seed": 0,
    "eval_every": 2,            # tính mAP val mỗi N epoch (tốn ~vài giây trên GPU)
    # Phạt activation vượt ngưỡng ở backbone MobileNetV3 (giảm range -> INT8 bớt mất điểm), vd.
    #   {"threshold": 16, "weight": 0.01}. null = tắt. Xem losses/center_loss.py::ActRangePenalty.
    "act_penalty": None,
}
CUSTOM_OBJECTS = {"heatmap_focal_loss": heatmap_focal_loss, "box_l1_loss": box_l1_loss,
                  "ActRangePenalty": ActRangePenalty}
_OUR_PREFIXES = ("x8a", "x16a", "x16b", "neck", "head", "image")


def penalty_layers(model):
    """Các tensor bị lượng tử hóa trong phần MobileNetV3: output ReLU, Add và BN 'project' (tuyến tính).
    (BN có ReLU phía sau được gộp vào conv khi convert -> tensor INT8 là output của ReLU.)"""
    L = tf.keras.layers
    return [l.name for l in model.layers
            if not l.name.startswith(_OUR_PREFIXES)
            and (isinstance(l, (L.ReLU, L.Add)) or (isinstance(l, L.BatchNormalization) and "project" in l.name))]


def load_config(path, **override):
    cfg = {**DEFAULTS, **json.loads(Path(path).read_text(encoding="utf-8"))}
    cfg.update({k: v for k, v in override.items() if v is not None})
    cfg["input_hw"] = tuple(cfg["input_hw"])
    return cfg


def build_and_compile(cfg, steps_per_epoch):
    model = build_edgedet(cfg["backbone"], "center", cfg["input_hw"], neck_c=cfg["neck_c"], head_c=cfg["head_c"],
                          pretrained=cfg["pretrained"], mnv3_relu6=cfg["mnv3_relu6"],
                          decoupled_head=cfg.get("decoupled_head", False))
    total = cfg["epochs"] * steps_per_epoch
    warm = min(cfg["warmup_epochs"], cfg["epochs"] // 4) * steps_per_epoch     # chạy thử ít epoch -> warmup ngắn lại
    # Learning rate: tăng dần trong vài epoch đầu (warmup – tránh "sốc" khi trọng số còn ngẫu nhiên),
    # sau đó giảm theo hình cos về ~0 ở epoch cuối (bước nhỏ dần để "tinh chỉnh").
    lr = tf.keras.optimizers.schedules.CosineDecay(cfg["lr"] * 0.01, total - warm, alpha=0.01,
                                                   warmup_target=cfg["lr"], warmup_steps=warm)
    opt = tf.keras.optimizers.AdamW(lr, weight_decay=cfg["weight_decay"])
    opt.exclude_from_weight_decay(var_names=["bn", "bias"])    # không "co" BatchNorm và bias về 0
    loss = {"head_hm": heatmap_focal_loss, "head_box": box_l1_loss}
    weights = {"head_hm": 1.0, "head_box": cfg["box_weight"]}
    pen = cfg["act_penalty"]
    if pen:
        # Model train = model detect + thêm output là các activation cần phạt (dùng chung trọng số).
        # Khi lưu best.keras / xuất INT8 chỉ lấy 2 output đầu (xem detector_of) -> file nộp không đổi.
        assert cfg["backbone"] == "mnv3s", "act_penalty chỉ dành cho backbone mnv3s"
        names = penalty_layers(model)
        model = tf.keras.Model(model.input, list(model.outputs) + [model.get_layer(n).output for n in names],
                               name=model.name)
        for n in names:
            loss[n] = ActRangePenalty(pen["threshold"])
            weights[n] = pen["weight"]
        print(f"act_penalty: {len(names)} lớp, threshold={pen['threshold']}, weight={pen['weight']}")
    model.compile(opt, loss=loss, loss_weights=weights,
                  jit_compile=False)   # XLA trên GPU không có đạo hàm của RESIZE_NEAREST_NEIGHBOR (lỗi thật trên Colab T4)
    return model


def detector_of(model):
    """Model detect thuần (2 output hm, box). Nếu model train có thêm output phạt thì cắt bớt (chung trọng số)."""
    if len(model.outputs) == 2:
        return model
    return tf.keras.Model(model.input, model.outputs[:2], name=model.name)


def add_dummy_targets(ds, model):
    """Output phạt cần "đáp án" để Keras chịu tính loss -> thêm số 0 giả (cùng số chiều với output)."""
    extra = model.output_names[2:]
    if not extra:
        return ds
    return ds.map(lambda x, y: (x, {**y, **{n: tf.zeros([tf.shape(x)[0], 1, 1, 1]) for n in extra}}))


class ValMap(tf.keras.callbacks.Callback):
    """Mỗi `every` epoch: đo mAP50 trên tập val, lưu best.keras nếu tốt hơn. Hỗ trợ EMA trọng số ở nửa sau."""

    def __init__(self, items, input_hw, out, every, detector, ema_decay=0.9, full_train=False):
        super().__init__()
        self.items, self.hw, self.out, self.every = items, input_hw, Path(out), every
        self.detector = detector          # model 2 output (chung trọng số với model đang train)
        self.ema_decay = float(ema_decay)
        self.full_train = bool(full_train)
        self.ema_weights = None
        self.best = -1.0
        self.last = float("nan")

    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params["epochs"]
        last_epoch = epoch + 1 == total_epochs
        if self.ema_decay > 0 and epoch + 1 >= max(2, total_epochs // 2):
            cur = self.detector.get_weights()
            if self.ema_weights is None:
                self.ema_weights = [w.copy() for w in cur]
            else:
                d = self.ema_decay
                self.ema_weights = [d * e + (1.0 - d) * w for e, w in zip(self.ema_weights, cur)]

        if (epoch + 1) % self.every == 0 or last_epoch:
            m, per, _ = evaluate(keras_predict_fn(self.detector), self.items, self.hw)
            used_ema = False
            if self.ema_weights is not None and (last_epoch or (epoch + 1) >= int(total_epochs * 0.75)):
                live = self.detector.get_weights()
                self.detector.set_weights(self.ema_weights)
                m_ema, per_ema, _ = evaluate(keras_predict_fn(self.detector), self.items, self.hw)
                if m_ema >= m or (self.full_train and last_epoch):
                    m, per, used_ema = m_ema, per_ema, True
                if not (last_epoch and used_ema):
                    self.detector.set_weights(live)

            self.last = m
            tag = " (EMA)" if used_ema else ""
            txt = " ".join(f"{CLASS_NAMES[k]}={v:.3f}" for k, v in per.items())
            print(f"\n  val mAP50{tag} = {m:.4f}   [{txt}]")
            if m > self.best or (self.full_train and last_epoch):
                self.best = max(self.best, m)
                if used_ema:
                    live = self.detector.get_weights()
                    self.detector.set_weights(self.ema_weights)
                    self.detector.save(self.out / "best.keras")
                    if not last_epoch:
                        self.detector.set_weights(live)
                else:
                    self.detector.save(self.out / "best.keras")
                print(f"  -> best.keras (mAP50 {m:.4f}{tag})")
        logs["val_map50"] = self.last                 # ghi vào log.csv (nan ở epoch không đo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="chỉ dùng N ảnh train (chạy thử)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, epochs=a.epochs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "input_hw": list(cfg["input_hw"])}, indent=2), encoding="utf-8")
    print("GPU:", tf.config.list_physical_devices("GPU") or "KHÔNG CÓ (train trên CPU sẽ rất chậm)")

    all_items = load_items(a.data, "train")
    train_items, val_items = split_train_val(all_items, cfg["val_frac"], cfg["seed"])
    if cfg.get("full_train"):
        train_items = all_items
    if a.limit:
        train_items, val_items = train_items[:a.limit], val_items[:max(16, a.limit // 10)]
    print(f"train {len(train_items)} ảnh, val {len(val_items)} ảnh, input {cfg['input_hw']}"
          + (" (FULL TRAIN 100%)" if cfg.get("full_train") else ""))
    hw, bs = cfg["input_hw"], min(cfg["batch"], len(train_items))
    ds_train = make_dataset(train_items, hw, bs, training=True, aug=cfg["aug"],
                            neighbor_weight=cfg.get("neighbor_weight", 0.25))
    steps = len(train_items) // bs

    # Train tiếp nếu đã có last.keras (Colab hay ngắt phiên giữa chừng)
    initial_epoch, log_path = 0, out / "log.csv"
    if (out / "last.keras").exists() and log_path.exists():
        model = tf.keras.models.load_model(out / "last.keras", custom_objects=CUSTOM_OBJECTS)
        with open(log_path, encoding="utf-8") as f:
            initial_epoch = max(0, sum(1 for l in f if l.strip()) - 1)   # bỏ dòng trống (CSVLogger trên Windows)
        print(f"Train tiếp từ epoch {initial_epoch}")
    else:
        model = build_and_compile(cfg, steps)
        print(f"Params: {model.count_params():,}  (giới hạn 200,000)")
    assert model.count_params() <= 200_000
    ds_train = add_dummy_targets(ds_train, model)

    vm = ValMap(val_items, hw, out, cfg["eval_every"], detector_of(model),
                ema_decay=cfg.get("ema_decay", 0.9), full_train=cfg.get("full_train", False))
    if initial_epoch and (out / "best.json").exists():
        vm.best = json.loads((out / "best.json").read_text())["val_map50"]

    class SaveBest(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (out / "best.json").write_text(json.dumps({"val_map50": vm.best}))

    model.fit(ds_train, epochs=cfg["epochs"], initial_epoch=initial_epoch, verbose=2,
              callbacks=[vm, SaveBest(),
                         tf.keras.callbacks.ModelCheckpoint(out / "last.keras"),
                         tf.keras.callbacks.CSVLogger(log_path, append=True)])
    print(f"Xong. mAP50 val tốt nhất = {vm.best:.4f}  -> {out / 'best.keras'}")
    return vm.best


if __name__ == "__main__":
    main()
