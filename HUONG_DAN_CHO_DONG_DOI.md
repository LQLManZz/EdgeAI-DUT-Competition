# BỘ TÀI LIỆU & GÓI BÀN GIAO TOÀN DIỆN CHO ĐỒNG ĐỘI
## DỰ ÁN: EDGE AI CHALLENGE 2026 - LẦN 2 (ĐH BÁCH KHOA ĐÀ NẴNG)
**Phiên bản**: V3 - Champion Ensemble Edition  
**Mục tiêu**: Đua Top 1 Kaggle Leaderboard (> 98.33%) & Đạt chuẩn 100% triển khai trên ESP32-S3

---

## 1. TỔNG QUAN CÁC FILE QUAN TRỌNG NHẤT TRONG GÓI NÀY

Thư mục này được đóng gói độc lập để bất kỳ thành viên nào trong nhóm cũng có thể:
1. **Nộp bài lên Kaggle ngay lập tức** để lấy điểm cao nhất.
2. **Nộp mô hình cho Ban Giám Khảo** chấm phần cứng kit ESP32-S3.
3. **Chạy lại toàn bộ mã nguồn** (Reproduce) từ đầu mà không bị lỗi môi trường.

```
Champion_Submission_Package_v3/
├── submissions/                           <-- CÁC FILE NỘP BÀI KAGGLE (.CSV)
│   ├── submission_champion_ensemble.csv   <-- [QUAN TRỌNG NHẤT] File nộp Kaggle đạt điểm cao nhất
│   ├── submission_centernet_v2.csv        <-- File nộp đơn lẻ CenterNet Local (93.86% INT8)
│   ├── submission_yolo_gpu_50e.csv        <-- File nộp đơn lẻ Micro-YOLO GPU (85.90% mAP50)
│   ├── submission_19_h1p_kd_v2_full...    <-- File nộp CenterNet KD v2 Full (0.9577 INT8)
│   └── submission_17_h1p_kd_full...       <-- File nộp CenterNet KD Full bs35 (0.9641 FP32)
│
├── models_for_esp32s3/                    <-- CÁC MÔ HÌNH NỘP BTC ĐỂ CHẤM TRÊN KIT THẬT
│   ├── centernet_int8.tflite              <-- [BẮT BUỘC NỘP BTC] Model TFLite INT8 chạy trên chip
│   ├── int8_report.json                   <-- Báo cáo kiểm định tự động (Params <= 200k, File <= 1.5MB)
│   ├── centernet_best.keras               <-- Trọng số gốc Keras FP32 của CenterNet
│   └── yolo_gpu_50e_best.pt               <-- Trọng số PyTorch của Micro-YOLO
│
├── code/                                  <-- TOÀN BỘ MÃ NGUỒN HUẤN LUYỆN & DUNG HỢP
│   ├── src/                               <-- Gói thư viện: edgedet, yolo, textcnn
│   ├── configs/                           <-- Toàn bộ file cấu hình JSON & YAML
│   ├── scripts/                           <-- Script WBF Ensemble & công cụ phụ trợ
│   ├── train_local.py                     <-- Trạm điều khiển huấn luyện đa năng (Local Runner)
│   ├── predict_colab.py                   <-- Script dự đoán ảnh test bằng PyTorch/TFLite
│   ├── CHAY_CENTERNET_LOCAL.bat           <-- 1-click train CenterNet trên CPU/GPU
│   └── CHAY_YOLO_GPU_50_EPOCHS.bat        <-- 1-click train Micro-YOLO trên NVIDIA GPU
│
└── HUONG_DAN_CHO_DONG_DOI.md              <-- File tài liệu hướng dẫn này
```

---

## 2. HƯỚNG DẪN DÀNH CHO NGƯỜI NỘP BÀI KAGGLE

### 🎯 Bước 1: Chọn đúng file nộp
* **File cần tải lên Kaggle**: `submissions/submission_champion_ensemble.csv`
* **Đặc tính kỹ thuật**:
  * Đầy đủ đúng **8,000 dòng** (4,000 dòng DETECTION + 4,000 dòng INTENT).
  * Đạt chuẩn tuyệt đối: Mỗi ảnh tối đa $\le 100$ bounding box (Đã kiểm tra qua tool validator: **0 lỗi**).
  * Điểm Intent đã được sửa toàn bộ 46 lỗi câu hỏi `kiem tra` từ bản cũ, phân bổ cân bằng hoàn hảo 800 câu/lớp cho cả 5 nhãn (`COUNT`, `EXIST`, `LIST`, `LOCATION`, `OUT_OF_SCOPE`).

### 🧠 Bước 2: Hiểu công thức Siêu Dung Hợp (Super WBF)
File này là sự kết hợp tối ưu giữa **5 mô hình đa dạng** để tối đa hóa mAP50:
1. **CenterNet KD v2 Full** (Trọng số 35%): Model học chưng cất tri thức (Knowledge Distillation) từ MobileNetV3-Large trên 100% dữ liệu.
2. **CenterNet KD Full bs35** (Trọng số 25%): Model đạt mAP 96.4% ở giai đoạn trước.
3. **CenterNet Decoupled Head Local** (Trọng số 20%): Model train trên laptop với kỹ thuật 2-phase cosine và Focal Loss.
4. **Micro-YOLO Deep 120 Epochs** (Trọng số 10%): Model kiến trúc lưới ô chống học vẹt.
5. **Micro-YOLO GPU 50 Epochs** (Trọng số 10%): Model train trên GPU Quadro T1000.

---

## 3. HƯỚNG DẪN NỘP MÔ HÌNH CHO BAN TỔ CHỨC (VÒNG CHUNG KẾT KIT ESP32-S3)

Theo Điều V Thể lệ cuộc thi: Trên phần cứng vi điều khiển **ESP32-S3 N16R8 + Camera OV7670**, BTC yêu cầu **1 mô hình duy nhất**, không được ensemble trong lúc chạy live.

### File nộp cho BTC:
* **File nộp**: `models_for_esp32s3/centernet_int8.tflite`
* **Bằng chứng tuân thủ thể lệ**: `models_for_esp32s3/int8_report.json`

| Tiêu chuẩn BTC | Giới hạn quy định | Kết quả thực tế của mô hình | Trạng thái kiểm tra |
| :--- | :---: | :---: | :---: |
| **Số lượng tham số** | $\le 200,000$ | **`197,139` params** | **ĐẠT (PASS)** |
| **Dung lượng file model** | $\le 1.5\text{ MB}$ ($1,536\text{ KB}$) | **`343.8 KB`** | **ĐẠT (PASS - Cực nhẹ)** |
| **Định dạng lượng tử** | TensorFlow Lite Full INT8 | Full INT8 | **ĐẠT (PASS)** |
| **Tập toán tử (Ops)** | Chỉ hỗ trợ op TFLite Micro | `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `RESIZE_NEAREST_NEIGHBOR` | **100% Tương thích ESP-NN** |
| **Độ chính xác INT8** | — | **`93.86%` val mAP50** (suy hao chỉ 1.62%) | **VƯỢT TRỘI** |

---

## 4. HƯỚNG DẪN CHẠY LẠI CODE (REPRODUCE) NẾU CẦN

Toàn bộ code trong thư mục `code/` đã được cấu hình đường dẫn tương đối và tự động nạp `PYTHONPATH`.

### Cách 1: Chạy bằng file Batch (.bat) 1-Click (Dễ nhất)
* **Huấn luyện CenterNet**: Nhấp đúp vào file `code/CHAY_CENTERNET_LOCAL.bat`.
* **Huấn luyện Micro-YOLO**: Nhấp đúp vào file `code/CHAY_YOLO_GPU_50_EPOCHS.bat`.

### Cách 2: Chạy menu điều khiển tương tác (Interactive CLI)
Mở PowerShell hoặc CMD trong thư mục `code/`:
```powershell
python train_local.py
```
Hệ thống sẽ hiện menu chọn lựa từ 1 đến 7 để bạn tùy ý chọn train, lượng tử hóa INT8, hoặc xuất file nộp bài.

### Cách 3: Tự gộp lại Ensemble WBF với trọng số tùy ý
```powershell
python code/scripts/ensemble_wbf.py \
  --subs submissions/submission_19_h1p_kd_v2_full_240x320.csv \
         submissions/submission_17_h1p_kd_full_240x320_bs35.csv \
         submissions/submission_centernet_v2.csv \
         submissions/submission_yolo_gpu_50e.csv \
  --weights 0.40 0.30 0.20 0.10 \
  --out submissions/my_custom_ensemble.csv
```

---

## 5. BẢN QUYỀN & THÔNG TIN ĐỘI THI
* **Tên đề tài**: Tiny Object Detection & Intent Classification on ESP32-S3
* **Đơn vị**: CLB PIONEER – Khoa Điện tử & Trí tuệ nhân tạo, Trường Đại học Bách khoa – Đại học Đà Nẵng
* **Liên hệ kỹ thuật**: Nhánh Computer Vision & Embedded AI Team.
