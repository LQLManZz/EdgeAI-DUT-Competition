# D1 – Báo cáo khảo sát Tiny Object Detector cho ESP32-S3

> Giai đoạn 1 · Ngày 09/11/2026 · Trạng thái: **hoàn thành, chờ đội duyệt shortlist**
> Bảng số liệu: [comparison_table.md](comparison_table.md) · Nguồn: [sources.md](sources.md) · Checklist triển khai: [tflm_checklist.md](tflm_checklist.md)
> Benchmark phần cứng: [../06_benchmark/phase1_esp32s3_microbench.md](../06_benchmark/phase1_esp32s3_microbench.md)

## Tóm tắt (đọc phần này trước)

1. **Không model công bố nào lọt ≤ 200K params ở dạng gốc.** Nhỏ nhất có số liệu: YOLO-FastestV2 0.25M [CB:S11], ESPDet-Pico 0.36M [CB:S18], TinyissimoYOLO 0.42M [CB:S12], PicoDet-XS 0.70M [CB:S06], SSDLite-MobileNetV2-0.35 0.93M [CB:S02]. → Bắt buộc **tự thiết kế / thu nhỏ**; chỉ mượn ý tưởng và (tùy chọn) trọng số backbone pretrained.
2. **Latency không phải nút thắt.** Đo trên kit thật: detector ~200K params ở **240×320 (độ phân giải gốc) mất 1,44 s/Invoke**, ở 128×128 mất 0,13–0,29 s [TĐ]. Ngân sách 10 s/lượt còn rất rộng. Do số params không phụ thuộc độ phân giải, **có thể tăng input để giúp vật nhỏ mà không tốn params**.
3. **Nút thắt thật là: (a) 200K params cho năng lực model, (b) RAM.** Arena đo được lớn hơn ước tính liveness 1,1–3,1×. Không model ~200K params nào ở 128² vừa SRAM nội (~236 KB khối liền) → arena đặt trong **PSRAM** (8 MB, chậm hơn SRAM 10–25%; bật D-cache 64 KB giúp giảm 6–16% latency khi arena ở PSRAM) [TĐ].
4. **Tốc độ trên ESP32-S3 phụ thuộc mạnh vào số kênh và loại op, không chỉ MACs.** Cùng ngân sách params:
   - Backbone kiểu MobileNetV3-Small *minimalistic* (không SE, không h-swish) đạt **95–103 MMAC/s**, kiểu MobileNetV2 α=0.5 đạt **85–90 MMAC/s**.
   - MobileNetV1 α=0.3 chỉ đạt **47 MMAC/s**, GhostNet **40**, ShuffleNetV2 **54** [TĐ].
   - Nguyên nhân đã xác minh trong mã nguồn ESP-NN: depthwise có số kênh không chia hết cho 8 rơi về bản C [CB:S22]. Ghost và Shuffle còn tốn thêm CONCAT/TRANSPOSE/SLICE và tensor trung gian lớn.
5. **Bẫy convert:** `UpSampling2D` của Keras 3 sinh op **TILE**, không có trong TFLM, nên `AllocateTensors` lỗi trên kit [TĐ]. Dùng `Resizing(nearest)` → `RESIZE_NEAREST_NEIGHBOR`.
6. **Framework:** khuyến nghị **TensorFlow/Keras** cho student. Đường PyTorch→ONNX→onnx2tf chạy được nhưng sinh thêm `PAD`, và backend mặc định của onnx2tf 2.6.9 làm hỏng quant per-channel [TĐ]. Teacher (YOLO/Ultralytics, PyTorch) vẫn dùng được cho distillation bằng cách xuất soft-label ra file.
7. **Shortlist sang Giai đoạn 2** (mục 7): (H1) backbone inverted-residual kiểu MobileNetV3-Small-minimalistic / MobileNetV2, kênh bội 8–16, PAN-lite về stride 8, head anchor-free kiểu CenterNet; (H2) như H1 nhưng head anchor-based kiểu YOLO-Fastest (so sánh head bằng mAP); (H3) quét độ phân giải input 4:3 (96×128 → 240×320) trên kiến trúc thắng.

---

## 1. Mục tiêu và ràng buộc

Ràng buộc chi tiết: [../00_contest_constraints.md](../00_contest_constraints.md).

- Detector ≤ 200.000 params, TFLite INT8 ≤ 1,5 MB, input ≤ 320×240, 1 lần suy luận, không ensemble/TTA.
- Chạy bằng **TensorFlow Lite Micro** trên ESP32-S3 N16R8 (SRAM 512 KB, PSRAM 8 MB, Flash 16 MB) + OV7670.
- 7 lớp. Vòng loại chấm **mAP50**. Chung kết chấm **đếm / tồn tại / liệt kê / vị trí trái-giữa-phải theo tâm bbox**, ≤ 10 s mỗi lượt.
- Hệ quả: cần **recall + precision tốt cho đếm**, **tâm box chính xác**; độ khít bbox (mAP50-95) là thứ yếu.

## 2. Phương pháp khảo sát

| Bước | Cách làm | Nhãn số liệu |
|---|---|---|
| Thu thập số công bố | Paper gốc (arXiv), README/model zoo của repo chính thức, tài liệu Espressif/Ultralytics | [CB:Sxx] |
| Thu nhỏ về ≤ 200K | Dựng model *probe* đại diện cho mỗi họ trong Keras (`scripts/analysis/zoo.py`). Chỉnh width để các họ có **cùng ngân sách 130–200K params**. Trọng số ngẫu nhiên. | [TĐ] |
| Params / MACs / size / op | `scripts/analysis/probe_model.py`: đếm params, MACs (conv/depthwise/dense), convert INT8 PTQ, liệt kê op, ước tính peak RAM theo liveness | [TĐ] |
| Chạy trên phần cứng thật | `firmware/bench_probe` (ESP-IDF 5.2.6, esp-tflite-micro 1.4.2, esp-nn 1.4.2): arena thực, latency `Invoke`, profile theo op, phân rã arena, SRAM vs PSRAM, D-cache 32 vs 64 KB | [TĐ] |
| So sánh framework | `scripts/analysis/framework_probe.py`: cùng kiến trúc, Keras→TFLite vs PyTorch→ONNX→onnx2tf | [TĐ] |
| Ước tính tay | Ghi công thức | [ƯT] |

**Kiểm chứng công cụ.** Script đo MobileNetV2 1.0/224 được 3.538.984 params và 300,8M MACs; paper công bố 3.4M params và 300M MAdds [CB:S01]. Lệch nhỏ ở params là do Keras đếm cả tham số BatchNorm.

**Giới hạn của probe:** trọng số ngẫu nhiên nên **không có số accuracy** từ probe. Latency, arena và op không phụ thuộc giá trị trọng số, nên các số đó dùng được. Accuracy sẽ đo ở Giai đoạn 2–3 trên dataset cuộc thi.

## 3. Khảo sát từng kiến trúc

### 3.1 MobileNetV2 (+ SSDLite)
- **Kiến trúc:** inverted residual (1×1 expand ×6 → depthwise 3×3 → 1×1 linear), skip ADD. SSDLite thay conv thường ở head SSD bằng depthwise separable [CB:S01].
- **Công bố:** ImageNet 3.4M params / 300M MAdds; SSDLite-MNv2 COCO 22.1 mAP, 4.3M params, 0.8B MAdds [CB:S01]. SSDLite-MNv2-0.35: 13.7 mAP, **0.93M**, 0.16B [CB:S02].
- **Thu về ≤ 200K [TĐ]:** α=0.5, cắt ở stride 16 → 193K params (head CenterNet), 24,2M MACs @128². α=0.35 cắt ở stride 8 → chỉ 24K params.
- **Trên kit [TĐ]:** 283–290 ms @128², arena 322 KB (PSRAM), ~85–90 MMAC/s. Chỉ dùng CONV/DEPTHWISE/ADD (+RESIZE nếu có PAN).
- **Phù hợp:** expand ×6 làm phình activation ở tầng đầu (peak liveness 240 KB @128²) → tốn RAM. Có **trọng số ImageNet** cho α=0.35/0.5 trong Keras.
- **Kết luận: TỐT** (ứng viên backbone).

### 3.2 MobileNetV3 (Small, Large, minimalistic)
- **Kiến trúc:** inverted residual + SE (global pool → FC → hard-sigmoid → MUL) + h-swish, kernel 3×3/5×5. Bản *minimalistic* bỏ SE, h-swish và 5×5 [CB:S02].
- **Công bố:** SSDLite-V3-Small 16.0 mAP / 2.49M / 0.21B; bản giảm kênh 16.1 / 1.77M / 0.16B [CB:S02].
- **Thu về ≤ 200K [TĐ]:** Small-minimalistic α=1.0 cắt ở stride 16 → 128K params, **12,6M MACs** @128². Bản đầy đủ α=0.75 → 162K, 11,4M MACs.
- **Trên kit [TĐ]:**
  - Bản minimalistic: **133 ms**, arena **230 KB** (nhỏ nhất trong các họ), **95–103 MMAC/s** (cao nhất).
  - Bản đầy đủ: 170 ms dù ít MACs hơn. MUL của SE chiếm 15% thời gian, HARD_SWISH chiếm 5%; SE/h-swish không được tính vào MACs nên MACs đánh giá thấp chi phí thật.
- **Phù hợp:** stem stride 2 + downsample sớm → peak activation thấp (112 KB liveness). Keras có trọng số ImageNet cho bản minimalistic.
- **Kết luận: TỐT** (ứng viên backbone số 1; dùng bản minimalistic).

### 3.3 GhostNet
- **Kiến trúc:** Ghost module = conv 1×1 sinh n/s kênh "intrinsic" + depthwise "cheap op" sinh phần còn lại + CONCAT [CB:S03].
- **Công bố:** 0.5×: 2.6M params, 42M FLOPs, 66.2% top-1 ImageNet [CB:S03] (phần lớn params nằm ở classifier).
- **Thu về ≤ 200K [TĐ]:** width 1.75 → 172–177K params, nhưng **43,7–45,2M MACs** @128² (gấp ~1,8× MobileNetV2 cùng params).
- **Trên kit [TĐ]:** **1,10–1,12 s**, arena **1,44 MB** @128² (lớn nhất), chỉ ~40 MMAC/s.
  - Nguyên nhân 1: số kênh sau khi chia đôi (n/2) thường không chia hết cho 8 → depthwise chạy bản C [CB:S22].
  - Nguyên nhân 2: CONCAT và tensor expand ở độ phân giải cao.
- **Kết luận: LOẠI** ở dạng chuẩn. Ý tưởng "ít params nhờ cheap op" bị triệt tiêu bởi chi phí RAM/latency trên MCU này.

### 3.4 NanoDet / NanoDet-Plus
- **Kiến trúc:** ShuffleNetV2 + Ghost-PAN + head kiểu FCOS/GFL. Head dự đoán **phân phối** khoảng cách 4 cạnh (softmax trên `reg_max+1` bin). Label assignment động (AGM + DSLA) [CB:S04].
- **Công bố:** NanoDet-m 320: 20.6 mAP(0.5:0.95), 0.72 GFLOPs, **0.95M** params, 980 KB INT8; Plus-m 320: 27.0, 0.9G, **1.17M** [CB:S04].
- **Thu về ≤ 200K:** phải bỏ Ghost-PAN và thu backbone, vì các thành phần này đã chậm khi đo riêng (3.3, 3.5). Head GFL đo được [TĐ]: output 16×16×39 so với 16×16×11 của head CenterNet, latency gần như không đổi (291 ms so với 290 ms). Cái giá nằm ở **decode** (softmax + tích phân trên MCU) và độ phức tạp khi huấn luyện.
- **Kết luận: LOẠI kiến trúc**, **giữ ý tưởng** label assignment động / GFL loss làm phương án thử ở GĐ3 (không tốn params).

### 3.5 PicoDet (PP-PicoDet)
- **Kiến trúc:** ESNet (ShuffleNet cải tiến + SE) + CSP-PAN + head chia sẻ, SimOTA [CB:S05].
- **Công bố:** XS 320: 23.5 mAP / **36.1 mAP50**, **0.70M**, 0.67 GFLOPs; S 320: 29.1 / 43.4, 1.18M, 0.97G [CB:S06].
- **Thu về ≤ 200K:** cần cắt ~3,5× params. ESNet mang cả nhược điểm của ShuffleNet (TRANSPOSE/slice) lẫn SE. Pretrained ở PaddlePaddle, chuyển sang TFLite qua nhiều bước.
- **Kết luận: LOẠI kiến trúc**, **giữ ý tưởng** SimOTA/label assignment và VFL/GIoU loss cho GĐ3.

### 3.6 EfficientDet / EfficientDet-Lite
- **Kiến trúc:** EfficientNet + BiFPN (nhiều lần resize + weighted ADD) + head chia sẻ [CB:S07].
- **Công bố:** D0: 3.9M params, 2.54B FLOPs, 34.3 AP; Lite0: **3.2M** params, 26.41 mAP float / 26.10 int8 [CB:S08].
- **Thu về ≤ 200K:** phải cắt 16×; BiFPN nhiều nhánh làm tăng tensor sống đồng thời (RAM). Không hợp lý.
- **Kết luận: LOẠI**. Chỉ cân nhắc làm teacher (nhưng YOLO hiện đại mạnh hơn ở cùng cỡ).

### 3.7 SSD / SSD-MobileNet
- **Kiến trúc:** multi-scale feature maps + default boxes (anchor), head conv cho mỗi scale [CB:S09].
- **Công bố:** SSD512 (VGG) 26.8 mAP, 36.1M; SSDLite-MNv2 4.3M; bản nhỏ nhất có số liệu là SSDLite-MNv2-0.35: 0.93M, 13.7 mAP [CB:S01, S02].
- **Thu về ≤ 200K [ƯT]:** head SSD tốn `A×(4+C+1)` kênh cho mỗi scale. Với 6 anchor, 8 lớp tính cả nền, 3 scale, mỗi scale 64 kênh, conv 1×1: 3×64×72 ≈ 14K params, nên head không đắt nếu dùng 1×1. Phần đắt là nhiều scale với nhiều tensor sống, cộng anchor/NMS trên MCU. Op `DETECTION_POSTPROCESS` có trong TFLM dưới dạng custom op [CB:S21], nhưng decode bằng C linh hoạt hơn.
- **Kết luận: ĐƯỢC NẾU SỬA**. Ý tưởng anchor-based nằm trong H2 (head kiểu YOLO 1–2 scale), không dùng SSD đầy đủ.

### 3.8 YOLO-Fastest / YOLO-FastestV2
- **Kiến trúc:** V1.1 dùng backbone depthwise kiểu EfficientNet-lite; V2 dùng ShuffleNetV2 + head tách obj/cls/reg, anchor-based (3 anchor), 2 scale [CB:S10, S11].
- **Công bố:** V1.1 320: COCO mAP50 24.4%, 0.252 BFLOPs, **0.35M**, VOC 61.0% [CB:S10]; V2 352: mAP50 24.1%, 0.212 GFLOPs, **0.25M** [CB:S11].
- **Thu về ≤ 200K:** chỉ cần cắt ~20–40%, gần nhất với ngân sách trong các model công bố. Họ ShuffleNet đo trên kit [TĐ]: 470 ms @128² (163K params), ~54 MMAC/s, TRANSPOSE + RESHAPE + STRIDED_SLICE chiếm ~8% thời gian.
- **Kết luận: ĐƯỢC NẾU SỬA**. Giữ head anchor-based 1–2 scale (H2), thay backbone Shuffle bằng inverted-residual kênh bội 8.

### 3.9 TinyissimoYOLO
- **Công bố:** 422K params, weights < 0.5 MB, ≤ 3 lớp, 8-bit; 180 fps trên MAX78000 nhờ **bộ tăng tốc CNN phần cứng** [CB:S12].
- **Nhận xét:** vượt 200K (2,1×), chỉ hỗ trợ tới 3 lớp trong khi đề có 7, và latency không chuyển sang ESP32-S3 vì S3 không có NPU.
- **Kết luận: LOẠI**. Bài học: model YOLO-style cỡ vài trăm K params là khả thi trên MCU.

### 3.10 FOMO (Edge Impulse)
- **Kiến trúc:** MobileNetV2 α=0.35 cắt ở stride 8 + conv 1×1 → heatmap (C+1 lớp) trên lưới 1/8. Chỉ trả **centroid**, không có kích thước box [CB:S13, vendor].
- **Thu về ≤ 200K [TĐ]:** mặc nhiên đã nhỏ: 21K params, 5,9M MACs @96². Trên kit: 103 ms (PSRAM) / **78 ms (SRAM)**, arena 186 KB, vừa SRAM nội.
- **Phù hợp:** tâm vật đủ cho COUNT/EXIST/LIST/LOCATION ở chung kết. Nhưng **không có bbox → mAP50 ở Kaggle (70% điểm vòng loại) rất thấp**, nên không dùng được nguyên bản.
- **Kết luận: ĐƯỢC NẾU SỬA**: thêm 2 kênh w,h (+ 2 kênh offset) chính là CenterNet → H1.

### 3.11 CenterNet ("Objects as Points")
- **Kiến trúc:** heatmap theo lớp (peak = tâm vật) + hồi quy w,h + offset. Decode bằng max-pool 3×3, **không cần NMS** [CB:S14].
- **Công bố:** 28.1 AP @142 FPS, 37.4 AP @52 FPS trên COCO với backbone lớn [CB:S14]. Không dùng được nguyên bản; chỉ lấy ý tưởng head.
- **Head trên probe [TĐ]:** dwsep + conv 1×1 → 11 kênh (7 + 2 + 2); vài nghìn params; latency head không đáng kể.
- **Phù hợp:** ra trực tiếp **tâm** (khớp cách chấm LOCATION) và đếm bằng số peak. Decode chỉ gồm max-pool 3×3 (ESP-NN tối ưu) + so ngưỡng. Vật gần nhau vẫn tách được nếu tâm cách nhau ≥ 1 ô lưới, nên stride 8 tốt hơn 16.
- **Kết luận: TỐT** (ứng viên head số 1).

### 3.12 MCUNet / MCUNetV2 (bài học về RAM)
- **Công bố:** patch-based inference giảm peak memory 4–8×; detection VOC trên MCU tăng 16.9 mAP so với SOTA trước đó [CB:S15]. Engine TinyEngine ≠ TFLM.
- **Áp dụng:** tầng đầu (độ phân giải cao) quyết định peak RAM. Đo trên kit xác nhận: peak luôn nằm trong ~10 op đầu (`peak_at_op` trong `probe_results.csv`) [TĐ]. Patch-based inference không có sẵn trong TFLM.
- **Kết luận:** dùng làm **nguyên tắc thiết kế** (downsample sớm, ít kênh ở độ phân giải cao).

### 3.13 Họ YOLO hiện đại (tham chiếu / teacher)
- **Công bố:** YOLO11n 640: 2.6M params, 6.5B FLOPs, 39.5 mAP50-95 [CB:S16]; YOLOX-Nano 0.91M, YOLOv5n 1.9M [CB:S04].
- **ESP32-S3 (ESP-DL, không phải TFLM):** YOLO11n 320: 961,5 ms (model), mAP50-95 0.276; 640: 3906 ms [CB:S17]. ESPDet-Pico 224: 0.36M params, 126 ms [CB:S18].
- **Kết luận:** vượt 200K từ 4,5× đến 13× → **không dùng làm student**. YOLO11n/YOLOv8n dùng làm **teacher** cho knowledge distillation (GĐ3). ESPDet-Pico cho thấy Espressif tự chọn hướng "YOLO thu nhỏ ~0.36M params": cùng hướng với shortlist nhưng phải nhỏ hơn ~2×.

## 4. So sánh tổng hợp
Xem [comparison_table.md](comparison_table.md). Năm nhận xét chính:

1. **MACs không dự đoán được latency trên S3.** Throughput dao động 40–103 MMAC/s giữa các họ ở cùng ngân sách params. Phải đo trên kit, không suy từ FLOPs.
2. **Ở cùng ~170–200K params, latency @128² đi từ 133 ms đến 1123 ms** (chênh 8,4×), arena từ 230 KB đến 1,44 MB (6,3×) [TĐ].
3. **Head gần như không ảnh hưởng latency.** CenterNet, YOLO-anchor, GFL và FOMO trên cùng backbone: 290–292 ms [TĐ]. Chọn head theo **accuracy + độ phức tạp decode**, không theo tốc độ.
4. **Stride 8 (PAN-lite) gần như miễn phí:** +2–5% latency, +4K params so với chỉ dùng stride 16 [TĐ]. RESIZE_NEAREST_NEIGHBOR < 0,1% thời gian.
5. **Độ phân giải input tăng latency gần tuyến tính theo số pixel nhưng không tốn params.** Họ MNv2-0.5: 96² 169 ms → 128² 290 ms → 160² 463 ms → 240×320 1439 ms [TĐ].

## 5. So sánh framework (`scripts/analysis/framework_probe.py`, kết quả: `experiments/phase1_analysis/framework/framework_probe.json`)

Cùng kiến trúc (stem + 4 khối inverted residual + PAN-lite + head 11 kênh, input 128²):

| Tiêu chí | Keras → TFLiteConverter | PyTorch → ONNX → onnx2tf 2.6.9 |
|---|---|---|
| Op-list INT8 [TĐ] | ADD, CONV_2D, DEPTHWISE_CONV_2D, RESIZE_NEAREST_NEIGHBOR | như trái **+ PAD** (padding đối xứng của PyTorch ≠ SAME) |
| Kích thước `.tflite` [TĐ] | 53.448 B | 53.440 B |
| Convert INT8 | 1 bước, ổn định | Backend mặc định `flatbuffer_direct` **lỗi** (per-channel quant của depthwise sai `quantized_dimension`); phải dùng `-tb tf_converter` |
| Input/Output | int8/int8, NHWC | int8/int8, NHWC (onnx2tf tự chuyển layout) |
| QAT | `tensorflow-model-optimization` (cần `tf_keras` / Keras 2 – rủi ro tương thích với Keras 3, kiểm chứng ở GĐ4) | QAT ở PyTorch rồi convert lại → thêm rủi ro |
| Teacher YOLO cho KD | Teacher chạy riêng (Ultralytics), lưu soft-label/feature ra file | Teacher và student cùng framework, KD online dễ hơn |

Sai số INT8/FP32 đo được **không** dùng để so sánh vì trọng số ngẫu nhiên. Phép so sánh có nghĩa sẽ làm ở GĐ4 với model đã train.

**Kết luận framework: TensorFlow/Keras** cho student (ít bước, op sạch nhất, khớp TFLM của BTC). KD làm theo kiểu **offline**: chạy teacher PyTorch trên tập train, lưu heatmap/box mềm ra file, rồi student Keras học từ đó. Cách này không cần hai framework chạy cùng lúc.

## 6. Bài học thiết kế (dùng cho Giai đoạn 2)

| # | Bài học | Bằng chứng |
|---|---|---|
| L1 | Mọi conv/depthwise dùng **số kênh bội 8, ưu tiên 16**. | ESP-NN fallback C khi `ch % 8 != 0` [CB:S22]; MNv1 α=0.3: 47 so với MNv1 α=0.25: 70–75 MMAC/s [TĐ] |
| L2 | Dùng **inverted residual không SE/h-swish** (kiểu MNv3-minimalistic). Tránh Ghost/Shuffle/Concat nhiều nhánh. | 95–103 MMAC/s so với 40–54 [TĐ]; SE+h-swish +28% latency [TĐ] |
| L3 | **Downsample sớm, ít kênh ở độ phân giải cao.** Tầng đầu quyết định peak RAM. | peak trong ~10 op đầu [TĐ]; MCUNet [CB:S15] |
| L4 | Một nhánh **PAN-lite**: stride 16 → upsample nearest (`Resizing`) + ADD vào stride 8. Head đơn scale ở stride 8. | +2–5% latency [TĐ]; giúp vật nhỏ và vật gần nhau |
| L5 | Head **anchor-free kiểu CenterNet** (7 heatmap + 2 wh + 2 offset). Decode bằng max-pool 3×3 + ngưỡng, không cần NMS. | Head ~ vài K params, không ảnh hưởng latency [TĐ]; khớp chấm LOCATION/COUNT |
| L6 | **Dồn params cho backbone/neck**, vì head rẻ. | head ≤ 5% params |
| L7 | **Độ phân giải là "miễn phí" về params.** Quét 96×128 → 240×320 (4:3, khớp ảnh nguồn) bằng mAP. | 240×320: 1,44 s ≪ 10 s [TĐ] |
| L8 | Arena đặt trong **PSRAM**. Ngân sách: arena ≤ ~1,5 MB + framebuffer 150 KB. Tránh op động (SHAPE/PACK/STRIDED_SLICE) và **TILE**. | [TĐ] |
| L9 | Keras: dùng `Resizing(nearest)`, không dùng `UpSampling2D`. Tránh `Lambda` cắt kênh. | TILE → `AllocateTensors` lỗi [TĐ] |
| L10 | Pretrained khả dụng: Keras MobileNetV2 (α 0.35/0.5…) và MobileNetV3-Small-minimalistic (α 1.0) ImageNet. Phải khai báo nếu dùng. | thể lệ III.2 |

## 7. Shortlist cho Giai đoạn 2

| Hướng | Mô tả | Ưu | Nhược / rủi ro | Bằng chứng |
|---|---|---|---|---|
| **H1** (ưu tiên) | Backbone inverted-residual **kiểu MNv3-Small-minimalistic** (hoặc MNv2), kênh bội 8/16, cắt ở stride 16, **PAN-lite → stride 8**, head **CenterNet** 11 kênh. Có thể khởi tạo từ trọng số ImageNet. | Nhanh nhất (133–139 ms @128²), arena nhỏ nhất (230 KB), op đơn giản, ra tâm trực tiếp | Năng lực ~130K params của backbone có thể thiếu → còn ~60K params để mở rộng neck/head | Bảng 2 [TĐ] |
| **H2** | Như H1 nhưng head **anchor-based kiểu YOLO-Fastest** (3 anchor × (5+7), 1–2 scale) + NMS bằng C | Bbox regression quen thuộc, nhiều tài liệu; dễ KD từ YOLO teacher | Cần tính anchor theo dataset; NMS trên MCU (rẻ với ≤ 100 box) | Head không ảnh hưởng latency [TĐ]; [CB:S10,S11] |
| **H3** | Quét **độ phân giải** 96×128, 128×160, 192×256, 240×320 cho kiến trúc thắng H1/H2 | Có thể tăng mAP vật nhỏ mà không tốn params | Arena và latency tăng (240×320: 1,24 MB, 1,44 s – vẫn trong ngân sách) | [TĐ] |

**Loại:** GhostNet, ShuffleNetV2/NanoDet/PicoDet (backbone), EfficientDet(-Lite), SSD đầy đủ, TinyissimoYOLO, YOLO-n (làm student), FOMO nguyên bản. Lý do bằng số liệu ở mục 3.

**Đề xuất thực nghiệm GĐ2:** train H1 và H2 với cùng backbone, cùng input 128×128 (rồi thêm 96×128) trên dataset Kaggle; so sánh mAP50, precision/recall, lỗi đếm và độ lệch tâm. Sau đó chạy H3 trên hướng thắng. Mọi biến thể phải qua `firmware/bench_probe` trước khi chốt.

## 8. Hạn chế của khảo sát
- **Không có accuracy cho model thu nhỏ**: probe dùng trọng số ngẫu nhiên. mAP công bố đo trên COCO/VOC, không so trực tiếp được với dataset cuộc thi (7 lớp, ảnh synthesis).
- **Bản TFLM của BTC chưa biết**: đo bằng esp-tflite-micro 1.4.2 / esp-nn 1.4.2. Nếu starter kit dùng bản cũ hơn, latency và danh sách op có thể khác → đo lại khi có kit BTC (23/11 hoặc 20/11).
- Benchmark chưa có Wi-Fi, camera, HTTP chạy song song; SRAM khả dụng thực tế sẽ ít hơn.
- Arena lớn hơn liveness 1,1–3,1×: mới có giả thuyết (scratch buffer của ESP-NN, planner), chưa phân tích từng tensor.
- Số liệu Edge Impulse (FOMO) và ESP-DL là của nhà cung cấp, đo trên engine khác TFLM.
