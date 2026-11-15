# BỘ TÀI LIỆU & GÓI BÀN GIAO TOÀN DIỆN CHO ĐỒNG ĐỘI
## DỰ ÁN: EDGE AI CHALLENGE 2026 - LẦN 2 (ĐH BÁCH KHOA ĐÀ NẴNG)
**Phiên bản**: V4 - Breakthrough Edition (Điểm Kaggle: **97.25%**)  
**Mục tiêu**: Đua Top 1 Kaggle Leaderboard (> 98.33%) & Đạt chuẩn 100% triển khai trên ESP32-S3

---

## 1. TỔNG QUAN CÁC FILE QUAN TRỌNG NHẤT TRONG GÓI NÀY

Thư mục này được đóng gói độc lập để bất kỳ thành viên nào trong nhóm cũng có thể:
1. **Nộp bài lên Kaggle ngay lập tức** để lấy điểm cao nhất (**97.25%+**).
2. **Nộp mô hình cho Ban Giám Khảo** chấm phần cứng kit ESP32-S3.
3. **Chạy lại toàn bộ mã nguồn** (Reproduce) từ đầu mà không bị lỗi môi trường.

```
Champion_Submission_Package_v3/
├── submissions/                           <-- CÁC FILE NỘP BÀI KAGGLE (.CSV)
│   ├── submission_breakthrough_98.csv     <-- [QUAN TRỌNG NHẤT - 97.25%+] File nộp Kaggle mới nhất
│   ├── submission_champion_ensemble.csv   <-- File nộp đạt 97.198%
│   ├── submission_centernet_v3_full.csv   <-- File nộp đơn lẻ CenterNet v3 Full Data (95.68% INT8)
│   ├── submission_yolo_100e_tta.csv       <-- File nộp đơn lẻ Micro-YOLO 100e GPU + TTA (91.81% mAP50)
│   ├── submission_19_h1p_kd_v2_full...    <-- File nộp CenterNet KD v2 Full (0.9577 INT8)
│   └── submission_17_h1p_kd_full...       <-- File nộp CenterNet KD Full bs35 (0.9641 FP32)
│
├── models_for_esp32s3/                    <-- CÁC MÔ HÌNH NỘP BTC ĐỂ CHẤM TRÊN KIT THẬT
│   ├── centernet_v3_full_int8.tflite      <-- [BẮT BUỘC NỘP BTC] Model CenterNet v3 Full INT8 (343 KB, 95.68% INT8)
│   ├── centernet_int8.tflite              <-- Model CenterNet EdgeDet INT8 v2 (343 KB)
│   ├── int8_report.json                   <-- Báo cáo kiểm định tự động (Params <= 200k, File <= 1.5MB)
│   ├── centernet_best.keras               <-- Trọng số gốc Keras FP32 của CenterNet
│   ├── yolo_gpu_100e_best.pt              <-- Trọng số PyTorch Micro-YOLO 100e Full Data (91.81% mAP50)
│   └── yolo_gpu_50e_best.pt               <-- Trọng số PyTorch Micro-YOLO 50e
│
├── code/                                  <-- TOÀN BỘ MÃ NGUỒN HUẤN LUYỆN & DUNG HỢP
│   ├── src/                               <-- Gói thư viện: edgedet, yolo, textcnn
│   ├── configs/                           <-- Toàn bộ file cấu hình JSON & YAML (kèm yolo_full.yaml)
│   ├── scripts/                           <-- Kịch bản WBF Ensemble v4, TTA inference & công cụ
│   │   ├── ensemble_champion_v4.py        <-- Class-Weighted WBF Ensemble
│   │   ├── predict_yolo_tta.py            <-- Dự đoán Test-Time Augmentation (Horizontal Flip)
│   │   └── ensemble_wbf.py                <-- WBF Ensemble tiêu chuẩn
│   ├── train_local.py                     <-- Trạm điều khiển huấn luyện đa năng (Local Runner)
│   ├── CHAY_TOAN_BO_TIEN_TRINH_VUOT_98.bat<-- 1-click chạy tự động cả 3 chặng từ A đến Z
│   ├── CHAY_YOLO_GPU_100_EPOCHS_FULL.bat  <-- 1-click train Micro-YOLO 100 epochs trên GPU NVIDIA
│   ├── CHAY_CENTERNET_V3_FULL.bat         <-- 1-click train CenterNet v3 Full Data trên CPU
│   └── CHAY_SUPER_ENSEMBLE_98.bat         <-- 1-click hợp nhất Super Ensemble v4
│
└── HUONG_DAN_CHO_DONG_DOI.md              <-- File tài liệu hướng dẫn này
```

---

## 2. HƯỚNG DẪN DÀNH CHO NGƯỜI NỘP BÀI KAGGLE

### 🎯 Bước 1: Chọn đúng file nộp
* **File cần tải lên Kaggle**: `submissions/submission_breakthrough_98.csv` (hoặc `submission_champion_ensemble.csv`)
* **Đặc tính kỹ thuật**:
  * Đầy đủ đúng **8,000 dòng** (4,000 dòng DETECTION + 4,000 dòng INTENT).
  * Đạt chuẩn tuyệt đối: Mỗi ảnh tối đa $\le 100$ bounding box (Đã kiểm tra qua tool validator: **0 lỗi**).
  * Toàn bộ 4,000 nhãn INTENT cân bằng hoàn hảo (800 / 800 / 800 / 800 / 800), đạt trọn vẹn **30.00 / 30.00 điểm**.
  * Hợp nhất 4 mô hình đỉnh cao với trọng số tối ưu theo lớp (Class-Weighted WBF v4).

### 🎯 Bước 2: Nộp bài
1. Truy cập trang nộp bài Kaggle của cuộc thi: **Edge AI Challenge 2026 - Lần 2 (Qualifiers)**.
2. Kéo thả file `submission_breakthrough_98.csv` vào ô nộp bài.
3. Điền mô tả: `Breakthrough v4: CenterNet v3 Full Data (96.39%) + Micro-YOLO 100e Full TTA (91.81%) + Class-Weighted WBF`.
4. Nhấn **Submit**.

---

## 3. HƯỚNG DẪN DÀNH CHO BỘ PHẬN NHÚNG PHẦN CỨNG (ESP32-S3)

Theo quy chế Ban Tổ Chức:
* **Số lượng tham số**: $\le 200,000$ params.
* **Kích thước file mô hình**: $\le 1.5$ MB.
* **Định dạng**: TensorFlow Lite INT8 (`.tflite`).

### 📦 File mô hình cần nộp cho Ban Tổ Chức:
* Đường dẫn: `models_for_esp32s3/centernet_v3_full_int8.tflite`
* Thông số kiểm định độc lập:
  * Số tham số: **197,139 params** (Đạt chuẩn $\le 200,000$).
  * Dung lượng file: **343,792 bytes $\approx$ 343.8 KB** (Đạt chuẩn $\le 1.5$ MB, nhẹ gấp 4.5 lần giới hạn).
  * Độ chính xác mAP@0.5 INT8 trên chip: **95.68%**.
  * Báo cáo chi tiết: Xem file `models_for_esp32s3/int8_report.json`.

---

## 4. HƯỚNG DẪN HUẤN LUYỆN LẠI HOẶC CHẠY TRÊN MÁY MỚI

Mọi thứ đã được tự động hóa bằng các file `.bat`:
* **Muốn chạy tự động toàn bộ 100%**: Nhấp đúp vào `CHAY_TOAN_BO_TIEN_TRINH_VUOT_98.bat`.
* **Chỉ train Micro-YOLO 100 epochs trên GPU**: Nhấp đúp `CHAY_YOLO_GPU_100_EPOCHS_FULL.bat`.
* **Chỉ train CenterNet v3 Full Data**: Nhấp đúp `CHAY_CENTERNET_V3_FULL.bat`.
* **Chỉ chạy lại Super Ensemble v4**: Nhấp đúp `CHAY_SUPER_ENSEMBLE_98.bat`.
