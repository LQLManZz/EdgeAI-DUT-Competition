"""Giai đoạn 2: xuất các ứng viên EdgeDet ra INT8 (trọng số ngẫu nhiên) + bảng params theo layer.

Chạy:  .venv\\Scripts\\python scripts/analysis/phase2_export.py
Output:
  experiments/phase2_arch/{probe_results.csv, json/, tflite/}       (để benchmark trên kit)
  docs/02_architecture/param_table_<model>.md                       (D4 – bảng phân bổ params)
"""
import csv, math, os, sys
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from edgedet.models.edgedet import build_edgedet, folded_params  # noqa: E402
import probe_model  # noqa: E402

probe_model.OUT = "experiments/phase2_arch"
L = tf.keras.layers
SIZES = [(96, 128), (144, 192), (192, 256), (240, 320)]   # 4:3 như ảnh nguồn 320x240, chia hết cho 16
CANDIDATES = [
    *[dict(backbone="custom", head="center", input_hw=s) for s in SIZES],   # H1-C
    *[dict(backbone="mnv3s", head="center", input_hw=s) for s in SIZES],    # H1-P
    dict(backbone="custom", head="yolo", input_hw=(144, 192)),               # H2
    dict(backbone="mnv3s", head="yolo", input_hw=(144, 192)),
]


def stage_of(name):
    for pre, st in [("stem", "Backbone · stem"), ("neck", "Neck · PAN-lite"), ("head", "Head"),
                    ("x8", "Backbone · khối thêm stride 8"), ("x16", "Backbone · khối thêm stride 16")]:
        if name.startswith(pre):
            return st
    if name.startswith("b") and name[1:2].isdigit():
        return f"Backbone · {name.split('_')[0]}"
    return "Backbone · MobileNetV3-Small (pretrained)"


def param_table(m):
    """Bảng: mỗi layer có tham số -> loại, output shape, params Keras, params sau gộp BN, MACs."""
    rows = []
    for l in m.layers:
        p = l.count_params()
        if not p:
            continue
        shp = tuple(l.output.shape[1:])
        if isinstance(l, L.DepthwiseConv2D):
            k = l.kernel.shape
            typ, fold, macs = f"DW {k[0]}x{k[1]}/{l.strides[0]}", math.prod(k) + k[2] * k[3], shp[0] * shp[1] * k[0] * k[1] * k[2]
        elif isinstance(l, L.Conv2D):
            k = l.kernel.shape
            typ, fold, macs = f"Conv {k[0]}x{k[1]}/{l.strides[0]}", math.prod(k) + l.filters, shp[0] * shp[1] * math.prod(k)
        elif isinstance(l, L.BatchNormalization):
            typ, fold, macs = "BN (gộp vào conv)", 0, 0
        else:
            typ, fold, macs = type(l).__name__, p, 0
        rows.append((stage_of(l.name), l.name, typ, "×".join(map(str, shp)), p, fold, macs))
    return rows


def write_md(m, rows, path):
    tot_p, tot_f, tot_m = sum(r[4] for r in rows), sum(r[5] for r in rows), sum(r[6] for r in rows)
    lines = [f"# Bảng phân bổ params – `{m.name}`", "",
             "Sinh tự động bởi `scripts/analysis/phase2_export.py` – không sửa tay. [TĐ]", "",
             f"- **Tổng params (Keras, gồm cả BN): {tot_p:,}** (giới hạn 200.000 → còn {200000 - tot_p:,})",
             f"- Trainable: {sum(int(tf.size(v)) for v in m.trainable_weights):,} · Sau khi gộp BN (hằng số trong .tflite): {tot_f:,}",
             f"- MACs: {tot_m / 1e6:.2f}M · Input {m.input_shape[1]}×{m.input_shape[2]}×3 · Output " + ", ".join("×".join(map(str, o.shape[1:])) for o in m.outputs),
             "", "## Tổng theo khối", "", "| Khối | Params (Keras) | % | Params (gộp BN) | MACs |", "|---|---|---|---|---|"]
    agg = {}
    for r in rows:
        a = agg.setdefault(r[0], [0, 0, 0])
        a[0] += r[4]; a[1] += r[5]; a[2] += r[6]
    for st, (p, f, mc) in agg.items():
        lines.append(f"| {st} | {p:,} | {100 * p / tot_p:.1f}% | {f:,} | {mc / 1e6:.2f}M |")
    lines.append(f"| **Tổng** | **{tot_p:,}** | 100% | **{tot_f:,}** | **{tot_m / 1e6:.2f}M** |")
    lines += ["", "## Chi tiết từng layer", "", "| Khối | Layer | Loại | Output (H×W×C) | Params | Gộp BN | MACs |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]:,} | {r[5]:,} | {r[6]:,} |" for r in rows]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    os.makedirs("docs/02_architecture", exist_ok=True)
    out = []
    for cfg in CANDIDATES:
        tf.keras.backend.clear_session()
        m = build_edgedet(**cfg)
        rows = param_table(m)
        assert sum(r[4] for r in rows) == m.count_params()
        assert m.count_params() <= 200_000, m.name
        if cfg["input_hw"] == (144, 192):
            write_md(m, rows, f"docs/02_architecture/param_table_{m.name}.md")
        r = probe_model.probe(m)
        r["folded_params"] = folded_params(m)
        out.append({k: v for k, v in r.items() if k != "op_counts"} | {"ops": " ".join(sorted(r["op_counts"]))})
        print(f'{r["name"]:32s} params={r["params"]:>8,} folded={r["folded_params"]:>8,} MACs={r["macs"]/1e6:6.1f}M '
              f'tflite={r["tflite_int8_bytes"]/1024:6.1f}KiB peak={r["peak_activation_bytes"]/1024:6.1f}KiB ops={out[-1]["ops"]}')
    with open("experiments/phase2_arch/probe_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
