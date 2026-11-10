"""So sánh 2 đường convert cho CÙNG một kiến trúc nhỏ:
  A) Keras  -> TFLiteConverter (INT8 PTQ)
  B) PyTorch -> ONNX -> onnx2tf (INT8 PTQ)
Kiến trúc: stem conv s2 -> 3 khối inverted residual (có ADD) -> nhánh s16 upsample nearest + ADD -> head 1x1 (11 kênh).
Đo: op-list, kích thước .tflite, sai lệch output INT8 so với FP32 trên cùng input.

Chạy: .venv\\Scripts\\python scripts/analysis/framework_probe.py
Output: experiments/phase1_analysis/framework/
"""
import json, os, subprocess, sys
import numpy as np

OUT = "experiments/phase1_analysis/framework"
# onnx2tf 2.6.9: backend mặc định "flatbuffer_direct" sinh per-channel quant sai cho depthwise
# (quantized_dimension=3) -> dùng "tf_converter". Đổi bằng biến môi trường ONNX2TF_BACKEND.
BACKEND = os.environ.get("ONNX2TF_BACKEND", "tf_converter")
os.makedirs(OUT, exist_ok=True)
H = W = 128
rng = np.random.default_rng(0)
CALIB = rng.random((32, H, W, 3), dtype=np.float32)
TEST = rng.random((8, H, W, 3), dtype=np.float32)


def tflite_ops_and_run(path, x_nhwc):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=path, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = sorted({o["op_name"] for o in it._get_ops_details()})
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    ys = []
    for x in x_nhwc:
        x = x[None]
        if i["shape"][1] == 3:  # NCHW input
            x = x.transpose(0, 3, 1, 2)
        if i["dtype"] in (np.int8, np.uint8):
            s, z = i["quantization"]
            x = np.clip(np.round(x / s + z), -128 if i["dtype"] == np.int8 else 0, 127 if i["dtype"] == np.int8 else 255).astype(i["dtype"])
        it.set_tensor(i["index"], x)
        it.invoke()
        y = it.get_tensor(o["index"]).astype(np.float32)
        if o["dtype"] in (np.int8, np.uint8):
            s, z = o["quantization"]
            y = (y - z) * s
        if y.shape[1] == 11 and y.shape[-1] != 11:
            y = y.transpose(0, 2, 3, 1)
        ys.append(y)
    return ops, np.concatenate(ys), str(i["dtype"].__name__), str(o["dtype"].__name__)


def keras_path():
    import tensorflow as tf
    L = tf.keras.layers

    def cbr(x, c, k=1, s=1, act=True):
        x = L.Conv2D(c, k, s, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        return L.ReLU(6.0)(x) if act else x

    def ir(x, c, s, t=4):
        y = cbr(x, x.shape[-1] * t)
        y = L.DepthwiseConv2D(3, s, padding="same", use_bias=False)(y)
        y = L.ReLU(6.0)(L.BatchNormalization()(y))
        y = cbr(y, c, act=False)
        return L.Add()([x, y]) if s == 1 and x.shape[-1] == c else y

    inp = L.Input((H, W, 3))
    x = cbr(inp, 16, 3, 2)
    x = ir(x, 16, 2)
    x8 = ir(ir(x, 24, 2), 24, 1)
    x16 = ir(x8, 48, 2)
    up = L.Resizing(H // 8, W // 8, interpolation="nearest")(cbr(x16, 32))
    f = L.Add()([cbr(x8, 32), up])
    out = L.Conv2D(11, 1)(f)
    m = tf.keras.Model(inp, out)
    y_fp = m.predict(TEST, verbose=0)
    conv = tf.lite.TFLiteConverter.from_keras_model(m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([c[None]] for c in CALIB)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    p = f"{OUT}/keras_int8.tflite"
    open(p, "wb").write(conv.convert())
    return m.count_params(), y_fp, p


def torch_path():
    import torch, torch.nn as nn, torch.nn.functional as F

    def cbr(ci, co, k=1, s=1, act=True):
        mods = [nn.Conv2d(ci, co, k, s, k // 2, bias=False), nn.BatchNorm2d(co)]
        return nn.Sequential(*(mods + [nn.ReLU6()] if act else mods))

    class IR(nn.Module):
        def __init__(self, ci, co, s, t=4):
            super().__init__()
            self.res = s == 1 and ci == co
            self.b = nn.Sequential(cbr(ci, ci * t), nn.Conv2d(ci * t, ci * t, 3, s, 1, groups=ci * t, bias=False),
                                   nn.BatchNorm2d(ci * t), nn.ReLU6(), cbr(ci * t, co, act=False))

        def forward(self, x):
            return x + self.b(x) if self.res else self.b(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem, self.s4 = cbr(3, 16, 3, 2), IR(16, 16, 2)
            self.s8 = nn.Sequential(IR(16, 24, 2), IR(24, 24, 1))
            self.s16 = IR(24, 48, 2)
            self.l8, self.l16, self.head = cbr(24, 32), cbr(48, 32), nn.Conv2d(32, 11, 1)

        def forward(self, x):
            x8 = self.s8(self.s4(self.stem(x)))
            up = F.interpolate(self.l16(self.s16(x8)), scale_factor=2, mode="nearest")
            return self.head(self.l8(x8) + up)

    net = Net().eval()
    with torch.no_grad():
        y_fp = net(torch.from_numpy(TEST.transpose(0, 3, 1, 2))).numpy().transpose(0, 2, 3, 1)
    onnx_p = f"{OUT}/torch.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, H, W), onnx_p, opset_version=18, input_names=["input"], output_names=["out"])
    np.save(f"{OUT}/calib.npy", CALIB)  # NHWC float32 cho onnx2tf -cind
    cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_p, "-o", f"{OUT}/onnx2tf", "-oiqt", "-qt", "per-channel", "-tb", BACKEND,
           "-cind", "input", f"{OUT}/calib.npy", "[[[[0,0,0]]]]", "[[[[1,1,1]]]]", "-iqd", "int8", "-oqd", "int8"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    open(f"{OUT}/onnx2tf.log", "w", encoding="utf-8").write(r.stdout + r.stderr)
    tfl = [f for f in os.listdir(f"{OUT}/onnx2tf") if f.endswith("full_integer_quant.tflite") and "with_int16" not in f] if os.path.isdir(f"{OUT}/onnx2tf") else []
    n_params = sum(p.numel() for p in net.parameters())
    return n_params, y_fp, (f"{OUT}/onnx2tf/{tfl[0]}" if tfl else None), r.returncode


def err(a, b):
    return {"mae": float(np.abs(a - b).mean()), "max_abs": float(np.abs(a - b).max()),
            "rel_mae": float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-9))}


res = {}
kp, ky, kpath = keras_path()
ops, yq, di, do = tflite_ops_and_run(kpath, TEST)
res["keras"] = {"params": int(kp), "tflite_bytes": os.path.getsize(kpath), "ops": ops, "io": f"{di}/{do}", "int8_vs_fp32": err(yq, ky)}
tp, ty, tpath, rc = torch_path()
if tpath:
    ops, yq, di, do = tflite_ops_and_run(tpath, TEST)
    res["torch_onnx2tf"] = {"params": int(tp), "tflite_bytes": os.path.getsize(tpath), "ops": ops, "io": f"{di}/{do}",
                            "int8_vs_fp32": err(yq, ty), "file": tpath}
else:
    res["torch_onnx2tf"] = {"params": int(tp), "error": f"onnx2tf exit {rc}, xem {OUT}/onnx2tf.log"}
json.dump(res, open(f"{OUT}/framework_probe.json", "w"), indent=2)
print(json.dumps(res, indent=2))
