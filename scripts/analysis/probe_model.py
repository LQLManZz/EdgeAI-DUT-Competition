"""Probe model: params, MACs, convert INT8, op-list, ước tính peak activation RAM.

Không train, trọng số ngẫu nhiên. Mọi số in ra là [TĐ] (tự đo).
Output: experiments/phase1_analysis/{probe_results.csv, json/, tflite/}

Chạy:  .venv\\Scripts\\python scripts/analysis/probe_model.py --ref     # kiểm chứng script (MobileNetV2 1.0/224)
       .venv\\Scripts\\python scripts/analysis/probe_model.py --all     # toàn bộ CONFIGS trong zoo.py
"""
import argparse, csv, json, os, sys, tempfile
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import zoo  # noqa: E402

OUT = "experiments/phase1_analysis"


def count_macs(model):
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            _, h, w, _ = layer.output.shape
            kh, kw = layer.kernel_size
            total += h * w * kh * kw * layer.input.shape[-1] * layer.depth_multiplier
        elif isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, cout = layer.output.shape
            kh, kw = layer.kernel_size
            total += h * w * kh * kw * (layer.input.shape[-1] // layer.groups) * cout
        elif isinstance(layer, tf.keras.layers.Dense):
            total += layer.input.shape[-1] * layer.units
    return int(total)


def to_int8(model):
    shape = model.input_shape[1:]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([np.random.rand(1, *shape).astype(np.float32)] for _ in range(20))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def analyze_tflite(tflite_bytes):
    """Op-list + peak activation theo liveness (CẬN DƯỚI: chưa tính scratch buffer của kernel)."""
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(tflite_bytes)
    interp = tf.lite.Interpreter(model_path=f.name)
    os.unlink(f.name)
    ops = interp._get_ops_details()
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    produced = {o: i for i, op in enumerate(ops) for o in op["outputs"]}
    last_use = {}
    for i, op in enumerate(ops):
        for t in op["inputs"]:
            last_use[t] = i
    for t in interp.get_output_details():
        last_use[t["index"]] = len(ops)
    inputs = {t["index"] for t in interp.get_input_details()}
    acts = set(produced) | inputs

    def nbytes(i):
        return int(np.prod(tensors[i]["shape"])) * np.dtype(tensors[i]["dtype"]).itemsize

    peak, peak_op = 0, None
    for i, op in enumerate(ops):
        s = sum(nbytes(t) for t in acts if produced.get(t, -1) <= i <= last_use.get(t, -1))
        if s > peak:
            peak, peak_op = s, f"#{i} {op['op_name']}"
    op_counts = {}
    for op in ops:
        op_counts[op["op_name"]] = op_counts.get(op["op_name"], 0) + 1
    biggest = max(acts, key=nbytes)
    out = interp.get_output_details()[0]
    return {
        "op_counts": op_counts,
        "peak_activation_bytes": peak,
        "peak_at_op": peak_op,
        "largest_tensor_bytes": nbytes(biggest),
        "largest_tensor_shape": tensors[biggest]["shape"].tolist(),
        "output_shape": out["shape"].tolist(),
        "io_dtype": f'{interp.get_input_details()[0]["dtype"].__name__}/{out["dtype"].__name__}',
    }


def probe(model, save=True):
    tfl = to_int8(model)
    r = {"name": model.name, "params": int(model.count_params()), "macs": count_macs(model),
         "tflite_int8_bytes": len(tfl), **analyze_tflite(tfl)}
    if save:
        os.makedirs(f"{OUT}/tflite", exist_ok=True)
        os.makedirs(f"{OUT}/json", exist_ok=True)
        open(f"{OUT}/tflite/{model.name}.tflite", "wb").write(tfl)
        json.dump(r, open(f"{OUT}/json/{model.name}.json", "w"), indent=2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="store_true", help="kiểm chứng: MobileNetV2 1.0/224 (paper: 3.4M params, 300M MAdds)")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.ref:
        m = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, weights=None)
        print({"params": m.count_params(), "macs": count_macs(m)})
        return
    rows = []
    for cfg in zoo.CONFIGS:
        tf.keras.backend.clear_session()
        m = zoo.build(**cfg)
        r = probe(m)
        rows.append({k: v for k, v in r.items() if k != "op_counts"} | {"ops": " ".join(sorted(r["op_counts"]))})
        print(f'{r["name"]:38s} params={r["params"]:>8,} MACs={r["macs"]/1e6:7.1f}M '
              f'tflite={r["tflite_int8_bytes"]/1024:6.1f}KiB peak={r["peak_activation_bytes"]/1024:7.1f}KiB')
    with open(f"{OUT}/probe_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
