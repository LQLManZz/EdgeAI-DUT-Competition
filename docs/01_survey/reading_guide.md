# Hướng dẫn đọc – Giai đoạn 1

> **Trạng thái 09/11/2026: đã thực hiện xong** – kết quả trong [survey_report.md](survey_report.md) và [comparison_table.md](comparison_table.md). Giữ file này làm hướng dẫn đọc lại nguồn gốc.

Mục tiêu: trong **1–2 ngày**, mỗi kiến trúc trả lời được các câu hỏi chung ở mục 0 và điền một dòng vào [comparison_table.md](comparison_table.md). Không cần đọc hết paper — đọc đúng phần được chỉ.

Thứ tự đề xuất (quan trọng nhất trước): **Nhóm A → Nhóm B → Nhóm C**.

---

## 0. Câu hỏi chung cho mọi kiến trúc

| # | Câu hỏi | Ghi vào |
|---|---|---|
| Q1 | Params, FLOPs/MACs, input size của **phiên bản nhỏ nhất được công bố**? Đo trên dataset nào (COCO, VOC...)? | Bảng, nhãn [CB] |
| Q2 | Backbone, neck, head gồm những khối gì? Stride đầu ra bao nhiêu? | Report |
| Q3 | Head có anchor không? Decode + NMS nằm trong model hay ngoài? | Report |
| Q4 | Dùng những op nào? Có op "lạ" với MCU không (hard-swish, SE, resize/upsample, shuffle, transpose, gather, softmax lớn)? | Bảng + [tflm_checklist.md](tflm_checklist.md) |
| Q5 | Muốn xuống **≤ 200K params** phải cắt gì (width multiplier, bỏ tầng, bỏ FPN)? Cắt rồi còn ý nghĩa không? | Report, nhãn [ƯT] hoặc [TĐ] |
| Q6 | Ở input 96/128/160, tensor activation lớn nhất là bao nhiêu byte INT8? | Bảng, nhãn [ƯT]/[TĐ] |
| Q7 | Vật nhỏ, vật che khuất, nhiều vật gần nhau: kiến trúc xử lý thế nào? | Report |
| Q8 | Có ai từng chạy nó trên MCU (ESP32, Cortex-M) chưa? Số liệu latency/RAM có nguồn không? | Bảng [CB] hoặc để trống |

**Bẫy chung**
- mAP trên COCO (80 lớp, ảnh thật) **không so sánh trực tiếp** được với dataset cuộc thi (7 lớp, synthesis). Chỉ dùng để so tương đối giữa các model **cùng điều kiện đo**.
- "FLOPs" trong nhiều paper thực ra là **MACs** (hoặc ngược lại). Ghi rõ paper dùng định nghĩa nào.
- Params của cả detector ≠ params của backbone. Thể lệ giới hạn **toàn bộ detector**.
- File `.tflite` nhỏ **không có nghĩa** RAM nhỏ.

---

## Nhóm A – Ứng viên sát bài toán MCU (đọc kỹ)

### A1. FOMO (Edge Impulse)
- Nguồn: docs.edgeimpulse.com → tìm "FOMO: Object detection for constrained devices"; mã tham khảo trong các repo `edgeimpulse` trên GitHub.
- Đọc: phần mô tả kiến trúc (backbone cắt cụt, output heatmap ở độ phân giải thấp), phần giới hạn.
- Câu hỏi riêng: FOMO trả về **tâm** chứ không trả bbox → ảnh hưởng gì tới **mAP50 ở Kaggle**? Còn với chung kết (đếm + vị trí trái/giữa/phải) thì sao?
- Bẫy: số liệu "chạy trên MCU X" của Edge Impulse là của họ, ghi nguồn cụ thể.

### A2. CenterNet – "Objects as Points"
- Nguồn: arXiv **1904.07850**; repo `xingyizhou/CenterNet`.
- Đọc: Mục 3 (heatmap + offset + size), phần decode bằng max-pool 3×3 thay NMS.
- Câu hỏi riêng: Head chỉ gồm vài conv 1×1/3×3 → bao nhiêu params với 7 lớp? Decode bằng max-pool có làm được trên MCU không (trong model hay trong C)?
- Bẫy: CenterNet gốc dùng backbone rất lớn (Hourglass, DLA) — chỉ lấy **ý tưởng head**, không lấy số params gốc.

### A3. YOLO-Fastest / YOLO-FastestV2
- Nguồn: GitHub `dog-qiuqiu/Yolo-Fastest`, `dog-qiuqiu/Yolo-FastestV2` (README có bảng params/FLOPs/mAP).
- Đọc: README, file cấu hình model (backbone, số anchor, số đầu ra).
- Câu hỏi riêng: Backbone kiểu ShuffleNetV2 dùng **channel shuffle** (RESHAPE + TRANSPOSE) và **split** → TFLM có hỗ trợ / ESP-NN có tăng tốc không? Anchor-based: cần tính lại anchor cho 7 lớp?
- Bẫy: không có paper peer-review → ghi rõ là số liệu từ README.

### A4. TinyissimoYOLO
- Nguồn: tìm trên arXiv theo tên *"TinyissimoYOLO: A Quantized, Low-Memory Footprint, TinyML Object Detection Network for Low Power Microcontrollers"* và bài tiếp theo của cùng nhóm (ETH Zürich).
- Đọc: bảng kiến trúc, bảng params/memory, phần triển khai trên MCU.
- Câu hỏi riêng: Params có vượt 200K không? Họ chạy trên MCU nào, framework gì (có phải TFLM không)?
- Bẫy: phần cứng của họ có thể có bộ tăng tốc CNN riêng → latency **không chuyển** sang ESP32-S3.

### A5. MCUNet / MCUNetV2 (bài học về RAM)
- Nguồn: arXiv **2007.10319** (MCUNet), **2110.15352** (MCUNetV2); repo `mit-han-lab/mcunet`.
- Đọc: phần phân tích **peak SRAM** theo từng tầng (vì sao tầng đầu ăn RAM nhất), patch-based inference ở V2.
- Câu hỏi riêng: Bài học nào áp dụng được ngay khi chọn input size và số kênh tầng đầu?
- Bẫy: TinyEngine của họ ≠ TFLM; con số RAM của TinyEngine thấp hơn TFLM.

---

## Nhóm B – Kiến trúc bắt buộc theo đề (đọc vừa đủ)

### B1. MobileNetV2 (+ SSDLite)
- Nguồn: arXiv **1801.04381**.
- Đọc: Mục 3 (inverted residual, linear bottleneck), Bảng 2 (kiến trúc), mục SSDLite, bảng kết quả detection.
- Câu hỏi riêng: Với width multiplier 0.35 và input nhỏ, backbone còn bao nhiêu params? Expansion ratio 6 làm activation phình ra bao nhiêu lần ở tầng đầu?
- Gợi ý: đây là model dễ kiểm chứng script nhất (params α=1.0, 224 đã công bố).

### B2. MobileNetV3
- Nguồn: arXiv **1905.02244**.
- Đọc: Mục 5 (h-swish, SE), Bảng 1–2 (Large/Small), mục detection (SSDLite).
- Câu hỏi riêng: h-swish và SE (global pool → FC → hard-sigmoid → MUL) có trong op resolver TFLM của starter kit không? Có được ESP-NN tăng tốc không? Nếu thay bằng ReLU thì mất gì?

### B3. GhostNet
- Nguồn: arXiv **1911.11907**; repo `huawei-noah/Efficient-AI-Backbones`.
- Đọc: Mục 3.1 (Ghost module: conv sinh một phần kênh + depthwise "cheap operation" + CONCAT).
- Câu hỏi riêng: Ghost module tiết kiệm params nhưng thêm CONCAT và tensor trung gian → ảnh hưởng RAM thế nào? Ở quy mô < 200K params, tỉ lệ ghost còn có lợi không?

### B4. NanoDet / NanoDet-Plus
- Nguồn: GitHub `RangiLyu/nanodet` (không có paper; README có bảng model zoo).
- Đọc: README (bảng params/FLOPs/mAP), cấu hình model (ShuffleNetV2 + Ghost-PAN + head kiểu GFL), phần export.
- Câu hỏi riêng: Head GFL dự đoán **phân phối** cho mỗi cạnh box (softmax trên nhiều bin) → output lớn, decode phức tạp hơn. Có đáng với bài toán 7 lớp không? Phần nào có thể giữ (ví dụ chỉ ý tưởng label assignment)?

### B5. PicoDet
- Nguồn: arXiv **2111.00902**; PaddleDetection `configs/picodet`.
- Đọc: Mục 3 (ESNet backbone, CSP-PAN, head, label assignment SimOTA), bảng kết quả.
- Câu hỏi riêng: PicoDet-XS/S có bao nhiêu params? Thành phần nào (label assignment, loss) dùng lại được **mà không tốn params**?
- Bẫy: PaddlePaddle → TFLite phải qua nhiều bước chuyển đổi; ghi lại nếu định dùng pretrained.

### B6. EfficientDet / EfficientDet-Lite
- Nguồn: arXiv **1911.09070**; repo `google/automl` (thư mục efficientdet); EfficientDet-Lite trong TensorFlow Lite Model Maker.
- Đọc: Mục BiFPN, bảng D0; trang model EfficientDet-Lite0.
- Câu hỏi riêng: Lite0 nhỏ nhất bao nhiêu params? BiFPN cần bao nhiêu RESIZE/ADD có trọng số? Có thu nhỏ được về 200K mà vẫn giữ BiFPN không?
- Gợi ý: nhiều khả năng chỉ đóng vai trò **tham chiếu** hoặc **teacher**, không phải ứng viên — nhưng phải có số liệu để chứng minh.

### B7. SSD / SSD-MobileNet
- Nguồn: arXiv **1512.02325**; TF Object Detection API model zoo (`tensorflow/models`, thư mục `research/object_detection`).
- Đọc: Mục 2 (default boxes, multi-scale feature maps), cấu hình `ssd_mobilenet_v2` / `ssdlite`.
- Câu hỏi riêng: Số anchor × số lớp ở mỗi feature map → params của head là bao nhiêu? Op `TFLite_Detection_PostProcess` có trong TFLM không, hay phải decode bằng C?

---

## Nhóm C – Tham chiếu / Teacher (đọc lướt)

### C1. YOLOv5n / YOLOv8n / YOLOX-Nano / YOLO11n
- Nguồn: repo Ultralytics, `Megvii-BaseDetection/YOLOX` (README có bảng params/FLOPs).
- Mục đích: (1) chứng minh bằng số liệu vì sao **không dùng trực tiếp** (params so với 200K), (2) chọn **teacher** cho knowledge distillation ở Giai đoạn 3.

### C2. ESP-DL (Espressif)
- Nguồn: GitHub `espressif/esp-dl` (README + thư mục models).
- Câu hỏi riêng: Espressif đã chạy detector nào trên ESP32-S3, latency bao nhiêu? Kỹ thuật tối ưu nào dùng lại được?
- Bẫy: **ESP-DL không phải TFLite Micro**. Thể lệ yêu cầu TFLM → số latency của ESP-DL **không áp dụng trực tiếp**, chỉ là mốc tham khảo.

---

## Kết thúc Giai đoạn 1 – bạn cần có
1. `comparison_table.md` điền đủ, mỗi ô có nhãn.
2. `survey_report.md` có phần "Bài học thiết kế" và **shortlist 2–3 hướng** kèm lý do bằng số liệu.
3. `framework_comparison` (trong report): Keras vs PyTorch→onnx2tf, có op-list thực tế khi convert thử.
4. Gửi lại để review trước khi sang Giai đoạn 2.
