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
from edgedet.data.kfold import multilabel_kfold_split
from edgedet.eval.evaluate import evaluate, keras_predict_fn
from edgedet.losses.center_loss import ActRangePenalty, KDBoxLoss, KDHmLoss, box_l1_loss, heatmap_focal_loss
from edgedet.models.edgedet import build_edgedet

DEFAULTS = {
    "backbone": "custom",       # "custom" (H1-C) | "mnv3s" (H1-P, <=200K) | "mnv3l" (Teacher ~3.3M)
    "pretrained": False,
    "mnv3_relu6": False,        # chỉ với backbone mnv3s: chặn ReLU ở 6 (giảm mất mát INT8, xem experiments.md #3)
    "decoupled_head": False,    # tách riêng nhánh conv cho heatmap và box trong head_center
    "input_hw": [144, 192],
    "neck_c": 80, "head_c": 80,
    "epochs": 80, "batch": 64,
    "lr": 2e-3, "warmup_epochs": 3, "weight_decay": 0.01,
    "box_weight": 1.0,          # loss tổng = focal(heatmap) + box_weight * (L1 + GIoU)(box)
    "neighbor_weight": 0.25,    # trọng số giám sát box ở 4 ô sát tâm (0 = chỉ giám sát 1 ô tâm)
    "ema_decay": 0.85,          # trung bình trượt trọng số theo epoch ở 25% epoch cuối (0 = tắt)
    "close_mosaic": 15,         # tắt Mosaic ở N epoch cuối để tinh chỉnh trên ảnh gốc nguyên vẹn
    "full_train": False,        # True = train trên 100% dữ liệu (6000 ảnh) để nộp bài cuối
    "aug": {},                  # ghi đè DEFAULT_AUG trong data/dataset.py
    "val_frac": 0.1, "seed": 0,
    "eval_every": 2,            # tính mAP val mỗi N epoch (tốn ~vài giây trên GPU)
    # Phạt activation vượt ngưỡng ở backbone MobileNetV3 (giảm range -> INT8 bớt mất điểm), vd.
    #   {"threshold": 16, "weight": 0.01}. null = tắt. Xem losses/center_loss.py::ActRangePenalty.
    "act_penalty": None,
    # Chưng cất tri thức (Teacher-Student KD): {"teacher_backbone": "mnv3l", "teacher_epochs": 55, "hm_weight": 1.5, "box_weight": 0.5}
    "kd": None,
    "init_from": None,          # tên thí nghiệm cùng kiến trúc để warm-start Student (vd. "h1p_v2_240x320")
}
CUSTOM_OBJECTS = {"heatmap_focal_loss": heatmap_focal_loss, "box_l1_loss": box_l1_loss,
                  "ActRangePenalty": ActRangePenalty, "KDHmLoss": KDHmLoss, "KDBoxLoss": KDBoxLoss}
_OUR_PREFIXES = ("x8", "x16", "neck", "head", "image")


def penalty_layers(model):
    """Các tensor bị lượng tử hóa trong phần MobileNetV3: output ReLU, Add và BN 'project' (tuyến tính).
    (BN có ReLU phía sau được gộp vào conv khi convert -> tensor INT8 là output của ReLU.)"""
    L = tf.keras.layers
    return [l.name for l in model.layers
            if not l.name.startswith(_OUR_PREFIXES)
            and (isinstance(l, (L.ReLU, L.Add)) or (isinstance(l, L.BatchNormalization) and "project" in l.name))]


def load_config(path, **override):
    p = Path(path)
    if not p.exists():
        candidates = [
            Path(__file__).resolve().parents[2] / path,
            Path(__file__).resolve().parents[2] / "configs" / Path(path).name,
            Path("configs") / Path(path).name,
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    cfg = {**DEFAULTS, **json.loads(p.read_text(encoding="utf-8"))}
    cfg.update({k: v for k, v in override.items() if v is not None})
    cfg["input_hw"] = tuple(cfg["input_hw"])
    return cfg


def build_and_compile(cfg, steps_per_epoch, teacher=None, out_dir=None):
    model = build_edgedet(cfg["backbone"], "center", cfg["input_hw"], neck_c=cfg["neck_c"], head_c=cfg["head_c"],
                          pretrained=cfg["pretrained"], mnv3_relu6=cfg["mnv3_relu6"],
                          decoupled_head=cfg.get("decoupled_head", False))
    init_from = cfg.get("init_from")
    if init_from and out_dir is not None:
        for name in ([init_from] if isinstance(init_from, str) else init_from):
            cand = Path(out_dir).resolve().parent / name / "best.keras"
            if cand.exists():
                print(f"Params Student: warm-start từ {cand}")
                model.set_weights(tf.keras.models.load_model(cand, compile=False).get_weights())
                break
    total = max(1, cfg["epochs"] * steps_per_epoch)
    warm = min(cfg["warmup_epochs"], cfg["epochs"] // 4) * steps_per_epoch     # chạy thử ít epoch -> warmup ngắn lại
    # Learning rate: tăng dần trong vài epoch đầu (warmup – tránh "sốc" khi trọng số còn ngẫu nhiên),
    # sau đó giảm theo hình cos về ~0 ở epoch cuối (bước nhỏ dần để "tinh chỉnh").
    lr = tf.keras.optimizers.schedules.CosineDecay(cfg["lr"] * 0.01, max(1, total - warm), alpha=0.01,
                                                   warmup_target=cfg["lr"], warmup_steps=warm)
    opt = tf.keras.optimizers.AdamW(lr, weight_decay=cfg["weight_decay"])
    opt.exclude_from_weight_decay(var_names=["bn", "bias"])    # không "co" BatchNorm và bias về 0
    loss = {"head_hm": heatmap_focal_loss, "head_box": box_l1_loss}
    weights = {"head_hm": 1.0, "head_box": cfg["box_weight"]}
    extra_outs = []
    pen = cfg["act_penalty"]
    if pen:
        # Model train = model detect + thêm output là các activation cần phạt (dùng chung trọng số).
        # Khi lưu best.keras / xuất INT8 chỉ lấy 2 output đầu (xem detector_of) -> file nộp không đổi.
        assert cfg["backbone"] == "mnv3s", "act_penalty chỉ dành cho backbone mnv3s"
        names = penalty_layers(model)
        extra_outs += [model.get_layer(n).output for n in names]
        for n in names:
            loss[n] = ActRangePenalty(pen["threshold"])
            weights[n] = pen["weight"]
        print(f"act_penalty: {len(names)} lớp, threshold={pen['threshold']}, weight={pen['weight']}")
    kd = cfg.get("kd")
    if kd and teacher is not None:
        L = tf.keras.layers
        teacher._name = "teacher_detector"
        teacher.trainable = False
        hm_t, box_t = teacher(model.input, training=False)
        kd_hm = L.Concatenate(axis=-1, name="kd_hm")([model.outputs[0], hm_t])
        kd_box = L.Concatenate(axis=-1, name="kd_box")([model.outputs[1], box_t, hm_t])
        extra_outs += [kd_hm, kd_box]
        loss["kd_hm"] = KDHmLoss()
        loss["kd_box"] = KDBoxLoss()
        weights["kd_hm"] = float(kd.get("hm_weight", 1.5))
        weights["kd_box"] = float(kd.get("box_weight", 0.5))
        print(f"Params KD Teacher ({kd.get('teacher_backbone', 'mnv3l')}): {teacher.count_params():,} "
              f"(hm_weight={weights['kd_hm']}, box_weight={weights['kd_box']})")
    if extra_outs:
        model = tf.keras.Model(model.input, list(model.outputs) + extra_outs, name=model.name)
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
        ema_start = max(2, int(total_epochs * 0.75))
        if self.ema_decay > 0 and epoch + 1 >= ema_start:
            cur = self.detector.get_weights()
            if self.ema_weights is None:
                self.ema_weights = [w.copy() for w in cur]
            else:
                d = self.ema_decay
                self.ema_weights = [d * e + (1.0 - d) * w for e, w in zip(self.ema_weights, cur)]

        if (epoch + 1) % self.every == 0 or last_epoch:
            m, per, _ = evaluate(keras_predict_fn(self.detector), self.items, self.hw)
            used_ema = False
            if self.ema_weights is not None and (last_epoch or (epoch + 1) >= ema_start + 4):
                live = self.detector.get_weights()
                self.detector.set_weights(self.ema_weights)
                m_ema, per_ema, _ = evaluate(keras_predict_fn(self.detector), self.items, self.hw)
                if (m_ema >= m - 0.0020) if self.full_train else (m_ema >= m):
                    m, per, used_ema = m_ema, per_ema, True
                if not (last_epoch and used_ema):
                    self.detector.set_weights(live)

            self.last = m
            tag = " (EMA)" if used_ema else ""
            txt = " ".join(f"{CLASS_NAMES[k]}={v:.3f}" for k, v in per.items())
            print(f"\n  val mAP50{tag} = {m:.4f}   [{txt}]")
            if m > self.best:
                self.best = m
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


def ensure_teacher(cfg, train_items, val_items, out_dir, epochs_override=None):
    """Đảm bảo đã có checkpoint Teacher FP32 (best.keras) trong <drive>/teachers/ trước khi chưng cất sang Student."""
    kd = cfg.get("kd")
    if not kd:
        return None
    t_bb = kd.get("teacher_backbone", "mnv3l")
    h, w = cfg["input_hw"]
    is_full = bool(cfg.get("full_train", False))
    t_root = Path(out_dir).resolve().parents[1] / "teachers"
    t_dir = t_root / f"teacher_{t_bb}_{h}x{w}_{'full' if is_full else 'val'}"
    t_best = t_dir / "best.keras"
    t_done = t_dir / "done.json"
    if t_done.exists() and t_best.exists():
        print(f"Params Teacher ({t_bb}): nạp sẵn từ {t_best}")
        return tf.keras.models.load_model(t_best, compile=False)

    t_dir.mkdir(parents=True, exist_ok=True)
    t_epochs = epochs_override or int(kd.get("teacher_epochs", 25 if is_full else 55))
    t_cfg = {
        **DEFAULTS,
        "backbone": t_bb,
        "pretrained": bool(cfg.get("pretrained", True)),
        "decoupled_head": True,
        "input_hw": (h, w),
        "neck_c": int(kd.get("teacher_neck_c", 160)),
        "head_c": int(kd.get("teacher_head_c", 160)),
        "epochs": t_epochs,
        "batch": cfg["batch"],
        "lr": float(kd.get("teacher_lr", 8e-4 if is_full else 2e-3)),
        "box_weight": cfg["box_weight"],
        "neighbor_weight": cfg.get("neighbor_weight", 0.25),
        "ema_decay": 0.85,
        "close_mosaic": min(10, max(1, t_epochs // 4)),
        "full_train": is_full,
        "aug": cfg.get("aug", {}),
        "act_penalty": None,
        "kd": None,
    }
    bs = min(t_cfg["batch"], len(train_items))
    steps = max(1, len(train_items) // bs)
    t_init_ep, t_log = 0, t_dir / "log.csv"
    if (t_dir / "last.keras").exists() and t_log.exists():
        t_model = tf.keras.models.load_model(t_dir / "last.keras", custom_objects=CUSTOM_OBJECTS)
        with open(t_log, encoding="utf-8") as f:
            t_init_ep = max(0, sum(1 for l in f if l.strip()) - 1)
        print(f"Train tiếp Teacher ({t_bb}) từ epoch {t_init_ep}/{t_epochs}")
    else:
        t_model = build_and_compile(t_cfg, steps)
        val_teacher_best = t_root / f"teacher_{t_bb}_{h}x{w}_val" / "best.keras"
        if is_full and val_teacher_best.exists():
            print(f"Params Teacher ({t_bb}): warm-start từ {val_teacher_best} ({t_epochs} epoch full data)")
            t_model.set_weights(tf.keras.models.load_model(val_teacher_best, compile=False).get_weights())
        else:
            print(f"Params Teacher ({t_bb}): {t_model.count_params():,} -> bắt đầu train Teacher ({t_epochs} epoch)")

    ds_t = make_dataset(train_items, (h, w), bs, training=True, aug=t_cfg["aug"],
                        neighbor_weight=t_cfg["neighbor_weight"])
    vm_t = ValMap(val_items, (h, w), t_dir, cfg["eval_every"], detector_of(t_model),
                  ema_decay=t_cfg["ema_decay"], full_train=is_full)
    if t_init_ep and (t_dir / "best.json").exists():
        vm_t.best = json.loads((t_dir / "best.json").read_text())["val_map50"]

    class SaveTBest(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (t_dir / "best.json").write_text(json.dumps({"val_map50": vm_t.best}))

    t_cbs = [vm_t, SaveTBest(), tf.keras.callbacks.ModelCheckpoint(t_dir / "last.keras"),
             tf.keras.callbacks.CSVLogger(t_log, append=True)]
    close_m = int(t_cfg["close_mosaic"])
    split_ep = t_epochs - close_m
    if t_cfg.get("aug", {}).get("mosaic", 0.0) > 0 and 0 < close_m < t_epochs:
        if t_init_ep < split_ep:
            t_model.fit(ds_t, epochs=split_ep, initial_epoch=t_init_ep, verbose=2, callbacks=t_cbs)
            t_init_ep = split_ep
        clean_aug = {**t_cfg.get("aug", {}), "mosaic": 0.0}
        ds_clean = make_dataset(train_items, (h, w), bs, training=True, aug=clean_aug,
                                neighbor_weight=t_cfg["neighbor_weight"])
        t_model.fit(ds_clean, epochs=t_epochs, initial_epoch=t_init_ep, verbose=2, callbacks=t_cbs)
    else:
        t_model.fit(ds_t, epochs=t_epochs, initial_epoch=t_init_ep, verbose=2, callbacks=t_cbs)
    t_done.write_text(json.dumps({"val_map50": vm_t.best, "epochs": t_epochs}), encoding="utf-8")
    print(f"Xong Teacher. mAP50 val tốt nhất = {vm_t.best:.4f} -> {t_best}")
    return tf.keras.models.load_model(t_best, compile=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int, help="ghi đè batch size (vd. 16, 32)")
    ap.add_argument("--limit", type=int, help="chỉ dùng N ảnh train (chạy thử)")
    ap.add_argument("--fold", type=int, default=None, help="chỉ số fold khi train K-Fold (vd. 0, 1, 2)")
    ap.add_argument("--n_splits", type=int, default=3, help="tổng số fold (mặc định 3)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, epochs=a.epochs)
    if a.batch:
        cfg["batch"] = a.batch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "input_hw": list(cfg["input_hw"])}, indent=2), encoding="utf-8")
    print("GPU:", tf.config.list_physical_devices("GPU") or "KHÔNG CÓ (train trên CPU sẽ rất chậm)")

    all_items = load_items(a.data, "train")
    if a.fold is not None:
        folds = multilabel_kfold_split(all_items, n_splits=a.n_splits, seed=cfg.get("seed", 2026))
        train_items, val_items = folds[a.fold]
        print(f"K-Fold Stratified: Chạy Fold {a.fold + 1}/{a.n_splits}")
    else:
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
    steps = max(1, len(train_items) // bs)
    teacher = ensure_teacher(cfg, train_items, val_items, out, epochs_override=a.epochs)

    # Train tiếp nếu đã có last.keras (Colab hay ngắt phiên giữa chừng)
    initial_epoch, log_path = 0, out / "log.csv"
    if (out / "last.keras").exists() and log_path.exists():
        try:
            model = tf.keras.models.load_model(out / "last.keras", custom_objects=CUSTOM_OBJECTS)
        except Exception:
            model = build_and_compile(cfg, steps, teacher=teacher, out_dir=out)
            model.load_weights(out / "last.keras")
        with open(log_path, encoding="utf-8") as f:
            initial_epoch = max(0, sum(1 for l in f if l.strip()) - 1)   # bỏ dòng trống (CSVLogger trên Windows)
        print(f"Train tiếp từ epoch {initial_epoch}")
    else:
        model = build_and_compile(cfg, steps, teacher=teacher, out_dir=out)
        print(f"Params: {detector_of(model).count_params():,}  (giới hạn 200,000)")
    assert detector_of(model).count_params() <= 200_000
    ds_train = add_dummy_targets(ds_train, model)

    vm = ValMap(val_items, hw, out, cfg["eval_every"], detector_of(model),
                ema_decay=cfg.get("ema_decay", 0.85), full_train=cfg.get("full_train", False))
    if initial_epoch and (out / "best.json").exists():
        vm.best = json.loads((out / "best.json").read_text())["val_map50"]

    class SaveBest(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (out / "best.json").write_text(json.dumps({"val_map50": vm.best}))

    cbs = [vm, SaveBest(), tf.keras.callbacks.ModelCheckpoint(out / "last.keras"),
           tf.keras.callbacks.CSVLogger(log_path, append=True)]
    close_m = int(cfg.get("close_mosaic", 0))
    split_ep = cfg["epochs"] - close_m
    if cfg.get("aug", {}).get("mosaic", 0.0) > 0 and 0 < close_m < cfg["epochs"]:
        if initial_epoch < split_ep:
            model.fit(ds_train, epochs=split_ep, initial_epoch=initial_epoch, verbose=2, callbacks=cbs)
            initial_epoch = split_ep
        print(f"\n--- Tắt Mosaic ở {close_m} epoch cuối ({split_ep}->{cfg['epochs']}) để tinh chỉnh trên ảnh gốc ---")
        clean_aug = {**cfg.get("aug", {}), "mosaic": 0.0}
        ds_clean = add_dummy_targets(make_dataset(train_items, hw, bs, training=True, aug=clean_aug,
                                                  neighbor_weight=cfg.get("neighbor_weight", 0.25)), model)
        model.fit(ds_clean, epochs=cfg["epochs"], initial_epoch=initial_epoch, verbose=2, callbacks=cbs)
    else:
        model.fit(ds_train, epochs=cfg["epochs"], initial_epoch=initial_epoch, verbose=2, callbacks=cbs)
    print(f"Xong. mAP50 val tốt nhất = {vm.best:.4f}  -> {out / 'best.keras'}")
    return vm.best


if __name__ == "__main__":
    main()
