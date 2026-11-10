"""Gộp kết quả probe (PC) + benchmark ESP32-S3 thành 1 CSV.

Chạy: .venv/Scripts/python scripts/analysis/parse_bench.py
Input : experiments/phase1_analysis/probe_results.csv, esp32s3_bench_dcache*.txt
Output: experiments/phase1_analysis/phase1_summary.csv
"""
import csv, glob, os, re

D = "experiments/phase1_analysis"
probe = {r["name"]: r for r in csv.DictReader(open(f"{D}/probe_results.csv", encoding="utf-8"))}
bench, ops, fails = {}, {}, {}
for path in sorted(glob.glob(f"{D}/esp32s3_bench_dcache*.txt")):
    tag = re.search(r"dcache(\w+)\.txt", path).group(1)
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line.strip())
        p = line.split(",")
        if p[0] == "BENCH":
            bench[(p[1], tag, p[2])] = (int(p[3]), float(p[4]))
        elif p[0] == "FAIL":
            fails[p[1]] = p[2]
        elif p[0] == "OPSBEGIN" and tag == "32k":
            cur = p[1]; ops[cur] = {}
        elif p[0] == "OPSEND":
            cur = None
        elif cur and len(p) == 2 and not line.startswith('"'):
            ops[cur][p[0]] = int(p[1])

rows = []
for name, r in probe.items():
    o = ops.get(name, {})
    tot = sum(o.values()) or 1
    top = sorted(o.items(), key=lambda kv: -kv[1])
    rows.append({
        "model": name, "params": r["params"], "MACs_M": round(int(r["macs"]) / 1e6, 2),
        "tflite_KiB": round(int(r["tflite_int8_bytes"]) / 1024, 1),
        "peak_liveness_KiB": round(int(r["peak_activation_bytes"]) / 1024, 1),
        "arena_KiB": round(bench.get((name, "32k", "PSRAM"), (0, 0))[0] / 1024, 1) or "",
        "ms_psram_dc32k": bench.get((name, "32k", "PSRAM"), (0, ""))[1],
        "ms_sram_dc32k": bench.get((name, "32k", "SRAM"), (0, ""))[1],
        "ms_psram_dc64k": bench.get((name, "64k", "PSRAM"), (0, ""))[1],
        "ms_sram_dc64k": bench.get((name, "64k", "SRAM"), (0, ""))[1],
        "MMAC_per_s_psram32k": round(int(r["macs"]) / 1e3 / bench[(name, "32k", "PSRAM")][1], 1) if (name, "32k", "PSRAM") in bench else "",
        "op_time_share": " ".join(f"{k}:{100*v/tot:.0f}%" for k, v in top[:4]),
        "tflm_status": f"FAIL {fails[name]}" if name in fails else ("OK" if (name, "32k", "PSRAM") in bench else "n/a"),
        "ops": r["ops"],
    })
with open(f"{D}/phase1_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f'{r["model"]:38s} {r["params"]:>7} {r["MACs_M"]:>6} {r["arena_KiB"]:>7} {r["ms_psram_dc32k"]!s:>8} {r["ms_sram_dc32k"]!s:>7} {r["ms_psram_dc64k"]!s:>8} {r["tflm_status"]}')
