# D3 – Đề xuất kiến trúc Tiny Object Detector: **EdgeDet**

> Giai đoạn 2 · 09/11/2026 · Trạng thái: **kiến trúc đã cố định về cấu trúc; chọn biến thể cuối (H1-C / H1-P / H2) và độ phân giải bằng mAP ở GĐ3.**
> Cơ sở: shortlist đã duyệt trong [../01_survey/survey_report.md](../01_survey/survey_report.md) §7. Sơ đồ và bảng params: [D4_architecture_diagram.md](D4_architecture_diagram.md).
> Mã nguồn: `src/edgedet/models/edgedet.py` (model), `src/edgedet/utils/decode.py` (decode tham chiếu).

## 1. Tóm tắt đề xuất

**EdgeDet** gồm ba phần:
- Backbone **inverted-residual không SE/h-swish**: kênh bội 8/16, ReLU6, downsample sớm.
- Neck **PAN-lite**: kéo stride 16 về stride 8 bằng `RESIZE_NEAREST_NEIGHBOR` + ADD.
- Head **anchor-free kiểu CenterNet** ở stride 8, có hai output INT8 tách riêng: heatmap 7 lớp và box 4 giá trị.

| Thuộc tính | Giá trị (H1-C, input 144×192) | Giới hạn | Nguồn |
|---|---|---|---|
| Params (Keras, gồm BN) | **185.307** | ≤ 200.000 ✅ | [TĐ] `phase2_export.py` |
| Params sau gộp BN | 174.507 | – | [TĐ] |
| File `.tflite` INT8 | **277 KB** | ≤ 1,5 MB ✅ | [TĐ] |
| Input | 144×192 (mặc định; quét 96×128 → 240×320) | ≤ 320×240 ✅ | – |
| Op TFLM | CONV_2D, DEPTHWISE_CONV_2D, ADD, RESIZE_NEAREST_NEIGHBOR | chỉ op TFLM hỗ trợ ✅ | [TĐ] trên kit |
| Ensemble / TTA | Không | ✅ | – |
| Latency Invoke trên ESP32-S3 | xem §6 | ≤ 10 s / lượt hỏi ✅ | [TĐ] |

## 2. Lý do từng lựa chọn (truy về bằng chứng GĐ1)

| Lựa chọn | Lý do | Bằng chứng |
|---|---|---|
| Inverted residual, **không SE/h-swish** | Nhanh nhất trên ESP-NN (95–103 MMAC/s); SE + h-swish làm tăng latency +28% | survey §3.2, L2 |
| **Số kênh bội 16** (16/32/48/64/80; expand 64/128/192/256) | ESP-NN depthwise chạy bản C khi `ch % 8 != 0` | L1, [CB:S22] |
| Stem Conv3×3/2 → 16 kênh, rồi b1 DW/2 | Downsample ×4 ngay đầu để giảm peak RAM | L3, MCUNet [CB:S15] |
| Dồn 72,5% params vào stride 16 | Feature ngữ nghĩa ở độ phân giải thấp thì rẻ về MACs/RAM | D4 |
| Thêm `b3b` ở stride 8 | Tăng năng lực cho vật nhỏ (chuột, điện thoại nằm xa) | yêu cầu §5 của đề |
| **PAN-lite** 16→8 bằng `Resizing(nearest)` | +2–5% latency, +4K params; tránh TILE | L4, L9 |
| Head **CenterNet** | Ra tâm trực tiếp (khớp chấm LOCATION/COUNT); decode = max-pool 3×3 + ngưỡng, không NMS; head chỉ chiếm 4,7% params | L5, survey §3.11 |
| **Tách output** `hm` (7) và `box` (4) | Mỗi output có scale/zero-point INT8 riêng. Gộp chung thì logits heatmap (~[-10, 10]) và kích thước box phải dùng chung 1 scale → mất độ chính xác | thiết kế; kiểm tra `.tflite`: 2 output, scale khác nhau [TĐ] |
| Mã hóa box: `log(w/8), log(h/8)` và offset ∈ [0,1] | Giữ dải giá trị nhỏ (≈ [-1, 4]) → lượng tử hóa tốt; w,h luôn dương | thiết kế |
| Bias heatmap = −4,595 (prior p = 0,01) | Ổn định focal loss lúc đầu (cách làm của RetinaNet/CenterNet) | thực hành chuẩn |
| Input **4:3** (96×128, 144×192, 192×256, 240×320) | Không méo vật so với ảnh nguồn 320×240; chia hết cho 16 | L7 |
| ReLU6 | Có kernel ESP-NN tối ưu; ổn định dải activation khi lượng tử hóa | [CB:S20] |

## 3. Ba biến thể để quyết định bằng thực nghiệm (GĐ3)

| Mã | Mô tả | Params | Câu hỏi cần trả lời |
|---|---|---|---|
| **H1-C** | Backbone custom (bảng D4), train từ đầu | 185.307 | Có đủ tốt mà không cần pretrained? |
| **H1-P** | MobileNetV3-Small minimalistic α=1.0 pretrained ImageNet (đến stride 16) + khối thêm `x8a`, `x16a`, `x16b` (`decoupled_head=true`: tách nhánh conv `hm` và `box`) | 189.379 / **197.139** | Pretrained có thắng H1-C trên dataset synthesis và trên ảnh thật OV7670 không? |
| **H2** (C/P) | Backbone như trên + head YOLO anchor (3 anchor × 12) + NMS | 187.332 / 191.404 | Anchor-based có mAP50 cao hơn CenterNet không? (anchor tính bằng k-means trên dataset) |

Cả ba dùng chung neck, chung độ phân giải, chung pipeline dữ liệu, nên so sánh công bằng. Quét độ phân giải (H3) chạy sau trên biến thể thắng.

**Khai báo pretrained (nếu chọn H1-P):** trọng số ImageNet của `tf.keras.applications.MobileNetV3Small(alpha=1.0, minimalistic=True)`, do Keras/Google phát hành (file ~2,1 MB tải từ storage.googleapis.com qua Keras). Phải ghi vào hồ sơ dự thi.

## 4. Hợp đồng input / output

**Input** (cả PC và ESP32-S3 phải giống hệt nhau):
1. Ảnh nguồn 320×240 RGB (OV7670: RGB565 → RGB888).
2. Resize về `H×W` của model (4:3, không méo; dùng cùng phép nội suy ở PC và trên chip, đề xuất bilinear hoặc area, chốt ở GĐ3).
3. Chuẩn hóa `x/127.5 − 1` → [-1, 1]. Đây là quy ước của MobileNetV3 pretrained; H1-C dùng chung để so sánh công bằng.
4. Lượng tử hóa INT8 theo `scale`/`zero_point` của tensor input trong `.tflite`.

**Output** (2 tensor int8, stride 8). **Không giả định thứ tự output**: tra theo shape hoặc tên. Bản convert thử cho ra `box` ở vị trí 0 và `hm` ở vị trí 1.

| Tensor | Shape (144×192) | Ý nghĩa |
|---|---|---|
| `hm` | 18×24×7 | logits heatmap theo lớp (`classes.txt`: 0 bottle … 6 toy_car) |
| `box` | 18×24×4 | `log(w/8)`, `log(h/8)`, `off_x`, `off_y` (w,h theo pixel input; offset trong ô) |

**Decode** (`decode_center`, đã kiểm thử bằng ca thủ công):
1. `p = sigmoid(hm)`; giữ điểm là cực đại 3×3 trong cùng lớp và `p ≥ ngưỡng`. Khi lượng tử hóa INT8 làm 2 ô cạnh nhau có cùng giá trị đỉnh (`p == nb`), phá vỡ đồng hạng (`tie_break`) bằng trung bình lân cận 3×3 để chỉ chọn đúng 1 ô tâm đồi Gauss.
2. Trộn nhẹ điểm xếp hạng `(1 - α)·p + α·mean_3x3` (`α = 0,1`) để phá đồng hạng giữa các box có cùng mức INT8 (1/256) khi tính AP toàn cục; lọc trùng cùng lớp bằng NMS (`iou_thr = 0,65`); lấy top-100.
3. Tâm = `((x + off_x)·8, (y + off_y)·8)`; kích thước = `exp(·)·8`; nhân tỉ lệ về ảnh gốc 320×240.
3. Kaggle: ghi `{"class_id", "score", "bbox":[x_min, y_min, x_max, y_max]}`.
4. Chung kết:
   - COUNT/EXIST/LIST = số peak có `p ≥ ngưỡng đếm`, theo từng lớp. Ngưỡng đếm tối ưu riêng trên tập val, khác ngưỡng mAP.
   - LOCATION = vùng của `cx` trên ảnh gốc: trái nếu < 1/3 chiều rộng, giữa, phải nếu ≥ 2/3. Quy tắc chính xác theo BTC.

H2 dùng `decode_yolo` (sigmoid + anchor + NMS theo lớp) với cùng định dạng kết quả.

## 5. Rủi ro và cách giảm

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| 185K params không đủ năng lực cho 7 lớp + che khuất | Trung bình | KD từ YOLO teacher (GĐ3); tăng độ phân giải (không tốn params) |
| Domain gap synthesis → ảnh thật OV7670 | **Cao** | Augmentation màu/nhiễu/blur mô phỏng OV7670; thu ảnh thật khi có kit BTC; pretrained (H1-P) có thể giúp |
| Vật sát nhau trùng tâm trong 1 ô 8 px | Thấp–TB | Input lớn hơn (192×256: ô = 10 px ảnh gốc); ngưỡng peak 3×3 |
| Sai số lượng tử hóa ở heatmap | TB | Output tách riêng; QAT nếu PTQ tụt > 1–2 điểm mAP50 (GĐ4) |
| Phiên bản TFLM của BTC khác 1.4.2 | TB | Chỉ dùng 4 op cơ bản nhất; đo lại khi nhận starter kit |
| Thứ tự output `.tflite` thay đổi giữa các lần convert | Thấp | Tra theo shape (7 kênh so với 4 kênh) trong firmware |

## 6. Đo trên ESP32-S3 thật [TĐ]

Thiết lập giống GĐ1: ESP32-S3 N16R8, 240 MHz, ESP-IDF 5.2.6, esp-tflite-micro/esp-nn 1.4.2, D-cache 32 KB, trọng số ngẫu nhiên (latency và arena không phụ thuộc giá trị trọng số), trung bình 5 lần `Invoke()`.
Log: `experiments/phase2_arch/esp32s3_bench_dcache32k.txt`.

| Biến thể | Input | MACs | Peak liveness | **Arena thực** | **Invoke (arena PSRAM)** | Invoke (arena SRAM) | MMAC/s |
|---|---|---|---|---|---|---|---|
| H1-C | 96×128 | 16,7M | 84 KB | 141 KB | 140,9 ms | **110,2 ms** | 119 |
| **H1-C** | **144×192** | 37,7M | 189 KB | **225 KB** | **322,2 ms** | – (vừa SRAM nhưng sát biên) | 117 |
| H1-C | 192×256 | 67,0M | 336 KB | 372 KB | 571,9 ms | – | 117 |
| H1-C | 240×320 | 104,6M | 525 KB | 561 KB | 897,3 ms | – | 117 |
| H1-P | 96×128 | 15,6M | 84 KB | 193 KB | 134,1 ms | 106,2 ms | 116 |
| H1-P | 144×192 | 35,1M | 189 KB | **375 KB** | 326,2 ms | – | 108 |
| H1-P | 192×256 | 62,3M | 336 KB | 629 KB | 570,9 ms | – | 109 |
| H1-P | 240×320 | 97,4M | 525 KB | 954 KB | 895,2 ms | – | 109 |
| H2-C (yolo) | 144×192 | 38,5M | 189 KB | 225 KB | 324,0 ms | – | 119 |
| H2-P (yolo) | 144×192 | 35,9M | 189 KB | 375 KB | 328,4 ms | – | 109 |

Profile H1-C @144×192: CONV_2D 61%, DEPTHWISE 34%, ADD 5%, RESIZE 0,3%. Arena gồm head 190 KB, tail 35 KB, persistent buffer 31 KB.

**Nhận xét**
1. **Mọi biến thể và mọi độ phân giải đều ≤ 0,9 s/Invoke**, chưa tới 10% ngân sách 10 s. Ngay cả 240×320 cũng còn dư rất nhiều cho chụp ảnh, tiền xử lý, TextCNN và HTTP.
2. EdgeDet đạt **~117 MMAC/s**, cao hơn mọi probe ở GĐ1 (tốt nhất 103). Điều này xác nhận quy tắc kênh bội 16 và không dùng SE/h-swish.
3. **H1-C tiết kiệm RAM hơn H1-P rõ rệt**: arena chỉ bằng 0,59–0,73 lần (ví dụ 225 KB so với 375 KB ở 144×192) với latency gần như bằng nhau. Peak liveness của hai biến thể **bằng nhau** (cùng 189 KB @144×192), nên phần chênh lệch không đến từ activation mà từ scratch buffer của kernel và cách planner xếp tensor (kênh 24/40/72/88 của MobileNetV3 so với kênh bội 16 của H1-C). Chưa phân tích từng tensor; sẽ làm ở GĐ5. H1-C @144×192 thậm chí vừa SRAM nội (225 KB < 236 KB) nếu không bật Wi-Fi; thực tế vẫn đặt trong PSRAM.
4. Head YOLO (H2) và CenterNet (H1) chênh **< 2 ms**, nên chọn head hoàn toàn theo mAP.
5. Ngân sách RAM khi tích hợp, ở trường hợp lớn nhất H1-P @240×320: arena 954 KB + framebuffer 150 KB + input 225 KB ≈ 1,3 MB PSRAM trên 8 MB → **không có vấn đề**.

## 7. Việc tiếp theo (GĐ3)
1. Dataset Kaggle → `data/raw/`; pipeline augmentation; chia train/val.
2. Loss: focal loss (heatmap, Gaussian theo kích thước vật + phạt nhẹ logit ngoài $[-8, 8]$ để giữ dải lượng tử INT8 hẹp) + L1 kết hợp **GIoU** trên `log(w/8)`, `log(h/8)` và offset, giám sát cả 4 ô sát tâm (`neighbor_weight = 0,25`) để chống lệch đỉnh INT8.
3. Train H1-C, H1-P, H2 ở 144×192 cùng lịch, kết hợp **EMA trọng số** nửa sau quá trình train → so mAP50 / P / R / lỗi đếm / độ lệch tâm.
4. Khi chốt cấu hình cuối, train trên **100% dữ liệu (`full_train: true`, 6.000 ảnh)**; chỉ dùng thêm teacher/pretrained nếu các bước trên chưa đạt trần mục tiêu.
