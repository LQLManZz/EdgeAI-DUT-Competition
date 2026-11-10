"""Copy các .tflite probe vào firmware/bench_probe/main/models và sinh models_gen.h + models.cmake.

Chạy sau probe_model.py --all:
    .venv\\Scripts\\python scripts/analysis/gen_bench_models.py
"""
import glob, os, re, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase1_analysis/tflite"
DST = "firmware/bench_probe/main/models"

os.makedirs(DST, exist_ok=True)
for f in glob.glob(f"{DST}/*.tflite"):
    os.remove(f)
names = []
for f in sorted(glob.glob(f"{SRC}/*.tflite")):
    shutil.copy(f, DST)
    names.append(os.path.basename(f))

sym = lambda b: re.sub(r"[^A-Za-z0-9]", "_", b)  # = string(MAKE_C_IDENTIFIER) của ESP-IDF
lines = ["// Sinh tự động bởi scripts/analysis/gen_bench_models.py – không sửa tay",
         "#pragma once", "#include <stddef.h>", "#include <stdint.h>", ""]
for b in names:
    lines += [f"extern const uint8_t _binary_{sym(b)}_start[];", f"extern const uint8_t _binary_{sym(b)}_end[];"]
lines += ["", "struct BenchModel { const char* name; const uint8_t* data; size_t size; };",
          "static const BenchModel kModels[] = {"]
for b in names:
    s = sym(b)
    lines.append(f'  {{"{b[:-7]}", _binary_{s}_start, (size_t)(_binary_{s}_end - _binary_{s}_start)}},')
lines += ["};", "static const int kNumModels = sizeof(kModels) / sizeof(kModels[0]);", ""]
open("firmware/bench_probe/main/models_gen.h", "w", encoding="utf-8").write("\n".join(lines))
open("firmware/bench_probe/main/models.cmake", "w", encoding="utf-8").write(
    "set(BENCH_MODELS\n" + "".join(f"    models/{b}\n" for b in names) + ")\n")
print(f"{len(names)} models -> {DST}")
