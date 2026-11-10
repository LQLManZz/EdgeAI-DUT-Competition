"""Chạy hàng đợi thí nghiệm trên Colab: train -> xuất INT8 -> submission -> chẩn đoán INT8 từng lớp.

  python scripts/colab_run.py --drive /content/drive/MyDrive/EdgeAI           # chạy hàng đợi
  python scripts/colab_run.py --drive /content/drive/MyDrive/EdgeAI --summary # chỉ in bảng kết quả

Hàng đợi nằm trong configs/queue.json (Claude sửa file này rồi đóng gói lại edgeai_code.zip),
nên các ô Colab KHÔNG BAO GIỜ phải sửa. Mỗi tên trong hàng đợi ứng với configs/<tên>.json,
kết quả ghi vào <drive>/experiments/<tên>/.

Mỗi bước chỉ chạy nếu chưa có file kết quả của nó -> chạy lại bao nhiêu lần cũng được:
  train      bỏ qua nếu đã có int8_report.json; nếu bị ngắt giữa chừng, train.py tự train tiếp từ last.keras
  export     -> int8_report.json      (mAP FP32 / INT8 trên val)
  submission -> submission.csv        (tự ghép <drive>/intent_submission.csv của nhóm TextCNN nếu có)
  debug      -> int8_layer_debug.csv  (range từng lớp, xem experiments.md)
  errors     -> error_analysis.txt + errors.png (phần mAP còn thiếu: box lệch / nhầm lớp / bỏ sót)
Log đầy đủ của từng bước: <drive>/experiments/<tên>/<bước>.log
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
# Dòng log của train được in ra màn hình (log đầy đủ vẫn ghi vào file)
TRAIN_SHOW = re.compile(r"GPU|Params|act_penalty|Train tiếp|val mAP|best\.keras|Xong|Error|Traceback|^\s*File ").search


def from_line(pattern):
    """Bộ lọc: ẩn log TF lặt vặt, in mọi dòng kể từ dòng đầu tiên khớp `pattern` (phần kết quả)."""
    seen = [False]
    def show(line):
        seen[0] = seen[0] or bool(re.search(pattern, line))
        return seen[0]
    return show


def run(step, args, log_path, show=None):
    """Chạy `python -m ...` ở tiến trình con (giải phóng bộ nhớ GPU sau mỗi bước), ghi log, dừng nếu lỗi."""
    print(f"\n--- {step}: {' '.join(args)}", flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        p = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        tail = []
        for line in p.stdout:
            log.write(line)
            tail = (tail + [line])[-30:]
            if show is None or show(line):
                print(line, end="", flush=True)
        if p.wait() != 0:
            print("".join(tail))
            sys.exit(f"!! Bước {step} lỗi (mã {p.returncode}). Log đầy đủ: {log_path}")


def run_experiment(name, drive, train_extra=()):
    cfg = ROOT / "configs" / f"{name}.json"
    if not cfg.exists():
        sys.exit(f"!! Không có {cfg.relative_to(ROOT)} trong edgeai_code.zip")
    out = Path(drive) / "experiments" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n========== {name}   ({cfg.read_text(encoding='utf-8').strip()})", flush=True)

    if not (out / "int8_report.json").exists():
        run("train", ["-m", "edgedet.train.train", "--config", str(cfg), "--data", str(DATA), "--out", str(out), *train_extra],
            out / "train.log", TRAIN_SHOW)
        run("export", ["-m", "edgedet.quant.export_int8", "--model", str(out / "best.keras"), "--data", str(DATA)],
            out / "export.log", from_line(r"^\{"))          # chỉ in phần JSON kết quả
    else:
        print("train + export: đã xong trước đó (có int8_report.json) -> bỏ qua")

    if not (out / "submission.csv").exists():
        args = ["-m", "edgedet.eval.make_submission", "--tflite", str(out / "model_int8.tflite"),
                "--data", str(DATA), "--out", str(out / "submission.csv")]
        intent = Path(drive) / "intent_submission.csv"
        if intent.exists():
            args += ["--intent", str(intent)]
        run("submission", args, out / "submission.log", from_line(r"^Dự đoán"))

    if not (out / "int8_layer_debug.csv").exists():
        run("debug", ["-m", "edgedet.quant.debug_int8", "--model", str(out / "best.keras"), "--data", str(DATA)],
            out / "debug.log", from_line(r"lớp\. rmse/scale"))

    if not (out / "error_analysis.txt").exists():
        run("errors", ["-m", "edgedet.eval.error_analysis", "--tflite", str(out / "model_int8.tflite"), "--data", str(DATA)],
            out / "errors.log", from_line(r"^mAP50"))


def summary(drive):
    rows = []
    for d in sorted((Path(drive) / "experiments").glob("*/")):
        r = d / "int8_report.json"
        if r.exists():
            j = json.loads(r.read_text())
            rows.append((d.name, j.get("val_map50_fp32"), j.get("val_map50_int8"), j.get("int8_drop"),
                         j["file_bytes"] // 1024, "có" if (d / "submission.csv").exists() else "chưa"))
        elif (d / "best.json").exists():
            rows.append((d.name, json.loads((d / "best.json").read_text())["val_map50"], None, None, None, "đang train"))
    f = lambda v: "–" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    print(f"\n{'thí nghiệm':<22} {'FP32':>7} {'INT8':>7} {'mất':>7} {'KB':>5}  submission")
    for r in sorted(rows, key=lambda r: -(r[2] or 0)):
        print(f"{r[0]:<22} {f(r[1]):>7} {f(r[2]):>7} {f(r[3]):>7} {f(r[4]):>5}  {r[5]}")
    print("(xếp theo INT8 – điểm thật, vì submission được dự đoán bằng file INT8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="thư mục Drive chứa zip + experiments/")
    ap.add_argument("--summary", action="store_true", help="chỉ in bảng kết quả")
    ap.add_argument("--smoke", action="store_true", help="chạy thử trên máy: 2 epoch, 128 ảnh")
    a = ap.parse_args()
    if not a.summary:
        queue = json.loads((ROOT / "configs" / "queue.json").read_text(encoding="utf-8"))["queue"]
        print("Hàng đợi:", queue)
        for name in queue:
            run_experiment(name, a.drive, ["--epochs", "2", "--limit", "128"] if a.smoke else [])
    summary(a.drive)


if __name__ == "__main__":
    main()
