# EdgeAI – Tiny Object Detection cho ESP32-S3

Dự án dự thi **Edge AI Challenge 2026 lần 2** (CLB PIONEER – Khoa Điện tử & Trí tuệ nhân tạo, ĐHBK Đà Nẵng).
Phạm vi chính: nhánh **Object Detection** (7 lớp) chạy INT8 bằng TensorFlow Lite Micro trên ESP32-S3 N16R8 + OV7670.

Ràng buộc chi tiết: [docs/00_contest_constraints.md](docs/00_contest_constraints.md).

## Ràng buộc cốt lõi

| Tiêu chí              | Giới hạn                                  |
| --------------------- | ----------------------------------------- |
| Parameters (detector) | ≤ 200.000                                 |
| Định dạng             | TensorFlow Lite INT8                      |
| Dung lượng file model | ≤ 1,5 MB                                  |
| Input                 | ≤ 320 × 240 (khuyến nghị 96², 128², 160²) |
| Ensemble / TTA        | Không                                     |
| Phản hồi mỗi lượt hỏi | ≤ 10 s                                    |
| Pretrained            | Được, phải khai báo                       |

## Cấu trúc thư mục

```
references/          Thể lệ + slide Training 1 của BTC (gốc, không sửa)
docs/
  00_contest_constraints.md
  01_survey/         D1 báo cáo khảo sát, D2 bảng so sánh
  02_architecture/   D3 đề xuất kiến trúc, D4 sơ đồ + bảng phân bổ params
  03_training/       D5 training from scratch + knowledge distillation
  04_quantization/   D6 pipeline INT8 (PTQ/QAT)
  05_deployment/     D7 pipeline ESP32-S3 (TFLM, ESP-NN, ESP-IDF)
  06_benchmark/      D8 benchmark trên phần cứng thật
data/                raw/ (dataset Kaggle), processed/, external/ (data ngoài – phải khai báo)
configs/             Cấu hình thí nghiệm
src/edgedet/         Mã nguồn detector (models, data, losses, train, quant, eval, utils)
scripts/analysis/    Script phân tích (đếm params/MACs, op-list, ước tính RAM)
notebooks/           Notebook Kaggle
experiments/         Kết quả từng lần chạy (mỗi thí nghiệm một thư mục)
models/              checkpoints/ (FP32), tflite/ (INT8)
firmware/            Project ESP-IDF (starter kit BTC + tích hợp)
```

## Trạng thái

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1 | Khảo sát kiến trúc hiện có (D1, D2) | **Xong** – chờ duyệt shortlist ([survey_report](docs/01_survey/survey_report.md), [comparison_table](docs/01_survey/comparison_table.md)) |
| 2 | Thiết kế kiến trúc riêng (D3, D4) | **Xong phần thiết kế** – [D3](docs/02_architecture/D3_architecture_proposal.md), [D4](docs/02_architecture/D4_architecture_diagram.md); chọn biến thể cuối bằng mAP ở GĐ3 |
| 3 | Training + Knowledge Distillation (D5) | **Đang làm**. Mục tiêu: bài nộp Kaggle trước 18/11. Người mới bắt đầu đọc [tổng quan](docs/03_training/00_tong_quan_cho_nguoi_moi.md) rồi [learning guide](docs/03_training/learning_guide.md). Pipeline đầy đủ: dữ liệu → train → INT8 → submission; notebook Colab ở `notebooks/edgedet_colab.ipynb` |
| 4 | Quantization INT8 (D6) | Chưa bắt đầu |
| 5 | Deployment ESP32-S3 (D7) | Chưa bắt đầu (đã có firmware benchmark `firmware/bench_probe`) |
| – | Benchmark phần cứng thật (D8) | Microbenchmark GĐ1 xong ([phase1_esp32s3_microbench](docs/06_benchmark/phase1_esp32s3_microbench.md)) |

## Môi trường

| Thành phần                | Phiên bản                                   | Ghi chú                   |
| ------------------------- | ------------------------------------------- | ------------------------- |
| Python (venv `.venv`)     | 3.12                                        | `py -3.12 -m venv .venv`  |
| TensorFlow                | 2.18.1                                      | probe, convert INT8       |
| PyTorch (CPU) + onnx2tf   | 2.6.9 (onnx2tf)                             | chỉ để so sánh framework  |
| ESP-IDF                   | 5.2.6 (`C:\Users\long5\esp\esp-idf-v5.2.6`) | cùng bản với repo mẫu BTC |
| esp-tflite-micro / esp-nn | 1.4.2 / 1.4.2                               | component manager         |

## Tái tạo số liệu Giai đoạn 1

```bash
.venv/Scripts/python scripts/analysis/probe_model.py --ref      # kiểm chứng script (MobileNetV2 1.0/224)
.venv/Scripts/python scripts/analysis/probe_model.py --all      # 29 model probe -> experiments/phase1_analysis/
.venv/Scripts/python scripts/analysis/gen_bench_models.py       # copy .tflite vào firmware/bench_probe
.venv/Scripts/python scripts/analysis/framework_probe.py        # Keras vs PyTorch->onnx2tf
```
Benchmark trên kit (PowerShell, đã `. export.ps1` của ESP-IDF):
```
cd firmware/bench_probe; idf.py build; idf.py -p COM8 flash; cd ../..
python scripts/analysis/read_serial.py COM8 experiments/phase1_analysis/esp32s3_bench_dcache32k.txt
.venv/Scripts/python scripts/analysis/parse_bench.py            # -> phase1_summary.csv
```

## Kiểm tra pipeline training (GĐ3)

```bash
.venv/Scripts/python -m pytest tests -q              # IoU, AP, encode→decode (tính tay)
.venv/Scripts/python scripts/sanity_overfit.py       # học thuộc 8 ảnh giả, mAP50 phải ≈ 1
```

Train → INT8 → submission (máy local cần `PYTHONPATH=src`; trên Colab dùng notebook):
```bash
python -m edgedet.train.train --config configs/h1c_144x192.json --data data/raw --out experiments/h1c_144x192
python -m edgedet.quant.export_int8 --model experiments/h1c_144x192/best.keras --data data/raw
python -m edgedet.eval.make_submission --tflite experiments/h1c_144x192/model_int8.tflite --intent <csv nhóm TextCNN> --out submission.csv
```

## Quy ước số liệu

Mọi số liệu trong docs gắn một nhãn:
- **[CB]** – số công bố, có link nguồn trong `docs/01_survey/sources.md`.
- **[TĐ]** – tự đo bằng script trong repo, có đường dẫn tới output trong `experiments/`.
- **[ƯT]** – ước tính tay, ghi công thức.

Không có nguồn thì để trống, không điền số.
