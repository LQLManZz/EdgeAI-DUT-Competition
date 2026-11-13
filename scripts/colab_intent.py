"""Chạy hàng đợi TextCNN intent trên Colab, chọn model, ghép vào submission detection.

  python scripts/colab_intent.py --drive /content/drive/MyDrive/EdgeAI            # chạy hàng đợi
  python scripts/colab_intent.py --drive /content/drive/MyDrive/EdgeAI --summary  # chỉ in bảng + ghép lại

Hàng đợi: configs/textcnn_queue.json (Claude sửa rồi đóng gói lại textcnn_code.zip -> các ô Colab không phải sửa).
Mỗi tên ứng với configs/<tên>.json; kết quả ghi vào <drive>/experiments_intent/<tên>/ (xem src/textcnn/train.py).
Thí nghiệm đã xong (có report.json) tự bỏ qua; bị ngắt giữa chừng thì chạy lại, bước nào xong rồi sẽ bỏ qua.

Sau hàng đợi:
  1. In bảng mọi thí nghiệm, xếp theo stress INT8 (thước đo phân biệt; CV gần như luôn ~100%).
  2. Chép intent_submission.csv của model được chọn -> <drive>/intent_submission.csv
     (chọn: "use" trong textcnn_queue.json nếu có, không thì model đạt mọi kiểm tra có stress INT8 cao nhất,
      hòa thì CV INT8 cao hơn, hòa nữa thì ít tham số hơn). scripts/colab_run.py của detection tự dùng file này.
  3. Với mỗi <drive>/experiments/<detection>/submission.csv đã có: ghi submission_with_intent.csv bên cạnh
     (giữ nguyên dòng DETECTION, thay dòng INTENT) -> nộp file này lên Kaggle.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
SHOW = re.compile(r"^(GPU|config|== |  CV|  stress|   epoch|CV acc|!!|Traceback|\S*Error)|đã có").search


def run(name, drive, smoke):
    cfg = ROOT / "configs" / f"{name}.json"
    if not cfg.exists():
        sys.exit(f"!! Không có {cfg.relative_to(ROOT)} trong textcnn_code.zip")
    out = Path(drive) / "experiments_intent" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n========== {name}", flush=True)
    args = [sys.executable, "-u", "-m", "textcnn.train", "--config", str(cfg), "--data", str(DATA), "--out", str(out)]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"}
    with open(out / "train.log", "a", encoding="utf-8") as log:
        p = subprocess.Popen(args + (["--smoke"] if smoke else []), cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        tail = []
        for line in p.stdout:
            log.write(line)
            tail = (tail + [line])[-30:]
            if SHOW(line):
                print(line, end="", flush=True)
        if p.wait() != 0:
            print("".join(tail))
            sys.exit(f"!! {name} lỗi (mã {p.returncode}). Log đầy đủ: {out / 'train.log'}")


def summary(drive, use):
    rows = []
    for d in sorted((Path(drive) / "experiments_intent").glob("*/")):
        if (d / "report.json").exists():
            r = json.loads((d / "report.json").read_text(encoding="utf-8"))
            rows.append({"name": d.name, "params": r["params"], "KB": r["file_bytes"] // 1024,
                         "cv_fp32": r["cv"]["acc_fp32"], "cv_int8": r["cv"]["acc_int8"],
                         "stress_fp32": r["stress"]["pooled_acc_fp32"], "stress_int8": r["stress"]["pooled_acc_int8"],
                         "checks_ok": all(r["checks"].values())})
    if not rows:
        print("Chưa có thí nghiệm nào xong.")
        return
    rows.sort(key=lambda r: (r["checks_ok"], r["stress_int8"], r["cv_int8"], -r["params"]), reverse=True)
    print(f"\n{'thí nghiệm':<18} {'params':>7} {'KB':>4} {'CV FP32':>8} {'CV INT8':>8} {'str FP32':>9} {'str INT8':>9}  luật")
    for r in rows:
        print(f"{r['name']:<18} {r['params']:>7} {r['KB']:>4} {r['cv_fp32']:>8.4f} {r['cv_int8']:>8.4f} "
              f"{r['stress_fp32']:>9.4f} {r['stress_int8']:>9.4f}  {'đạt' if r['checks_ok'] else 'KHÔNG ĐẠT'}")
    print("(CV = chia theo khuôn câu; str = stress test, bỏ hẳn 1 cụm từ khỏi train. INT8 = file .tflite sẽ nộp)")

    best = use or rows[0]["name"]
    src = Path(drive) / "experiments_intent" / best / "intent_submission.csv"
    if not src.exists():
        sys.exit(f"!! Không có {src}")
    shutil.copy(src, Path(drive) / "intent_submission.csv")
    print(f"\nModel intent dùng để nộp: {best} ({'chọn tay trong textcnn_queue.json' if use else 'tự chọn theo bảng'})"
          f" -> {Path(drive) / 'intent_submission.csv'}")

    intent = pd.read_csv(src, dtype=str).set_index("sample_id")["prediction"]
    # So sánh dự đoán trên 4.000 câu test giữa các model đã xong với `best`
    other_subs = {}
    for r in rows:
        p_sub = Path(drive) / "experiments_intent" / r["name"] / "intent_submission.csv"
        if r["name"] != best and p_sub.exists():
            other_subs[r["name"]] = pd.read_csv(p_sub, dtype=str).set_index("sample_id")["prediction"]
    if other_subs:
        print(f"\nĐối chiếu dự đoán trên {len(intent)} câu test so với {best}:")
        diff_mask = pd.Series(False, index=intent.index)
        for name, s in other_subs.items():
            d = (s.reindex(intent.index) != intent)
            diff_mask |= d
            print(f"  {name:<18}: khác {int(d.sum()):>3} câu ({100.0 * (1.0 - d.mean()):.2f}% trùng)")
        if diff_mask.any() and (DATA / "intent_test.csv").exists():
            q_map = pd.read_csv(DATA / "intent_test.csv", dtype=str).set_index("question_id")["question"]
            print(f"  Mẫu câu bất đồng (tối đa 10/{int(diff_mask.sum())}):")
            for qid in intent.index[diff_mask][:10]:
                picks = ", ".join([f"{best}={intent[qid]}"] + [f"{n}={s.get(qid, '?')}" for n, s in other_subs.items()])
                print(f"    [{qid}] {q_map.get(qid, '')}  -> {picks}")

    for det in sorted((Path(drive) / "experiments").glob("*/submission.csv")):
        sub = pd.read_csv(det, dtype=str, keep_default_na=False)
        m = sub.task_type == "INTENT"
        if m.sum() != len(intent) or not sub.sample_id[m].isin(intent.index).all():
            print(f"!! {det}: ID INTENT không khớp -> bỏ qua")
            continue
        sub.loc[m, "prediction"] = sub.sample_id[m].map(intent)
        sub.to_csv(det.with_name("submission_with_intent.csv"), index=False)
        print(f"   ghép -> {det.with_name('submission_with_intent.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="thư mục Drive chứa textcnn_code.zip")
    ap.add_argument("--summary", action="store_true", help="chỉ in bảng + chọn + ghép")
    ap.add_argument("--smoke", action="store_true", help="chạy thử: 2 epoch, 1 fold, 2 cue")
    a = ap.parse_args()
    q = json.loads((ROOT / "configs" / "textcnn_queue.json").read_text(encoding="utf-8"))
    if not a.summary:
        print("Hàng đợi:", q["queue"])
        for name in q["queue"]:
            run(name, a.drive, a.smoke)
    summary(a.drive, q.get("use"))


if __name__ == "__main__":
    main()
