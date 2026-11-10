# Checklist triển khai TFLM / ESP32-S3

Mục đích: không kết luận "chạy được trên ESP32-S3" chỉ vì model nhỏ hoặc convert được `.tflite`.
Mọi mục **[TĐ]** dưới đây đã được kiểm chứng trên kit thật (ESP32-S3 N16R8 rev v0.2, ESP-IDF 5.2.6, esp-tflite-micro 1.4.2, esp-nn 1.4.2) bằng `firmware/bench_probe`.

## 1. Bốn đại lượng khác nhau – đừng lẫn

| Đại lượng | Là gì | Giới hạn / ngân sách | Đo bằng |
|---|---|---|---|
| **Params** | Số tham số của cả detector | ≤ 200.000 (thể lệ) | `model.count_params()` – `scripts/analysis/probe_model.py` |
| **File size** | Dung lượng `.tflite` INT8 | ≤ 1,5 MB (thể lệ) | kích thước file |
| **RAM (arena)** | Activation + scratch + persistent buffer trong tensor arena | SRAM nội: ~236 KB khối liền lớn nhất còn trống [TĐ]; PSRAM 8 MB | `interpreter.arena_used_bytes()` trên thiết bị |
| **Latency** | Thời gian của cả lượt hỏi theo cách đo của cuộc thi | ≤ 10 s | `esp_timer_get_time()` trên thiết bị |

## 2. Op – kết quả đã kiểm chứng

| Op | Có trong TFLM? [CB:S21] | ESP-NN tối ưu? [CB:S20] | Ghi chú [TĐ] |
|---|---|---|---|
| CONV_2D, DEPTHWISE_CONV_2D | Có | Có | Chiếm 85–100% thời gian mọi model đo được |
| ADD, MUL | Có | Có | |
| MAX_POOL_2D, AVERAGE_POOL_2D | Có | Có | |
| HARD_SWISH, MEAN, SOFTMAX | Có | Có (có trong danh sách kernel) | MobileNetV3 đầy đủ: HARD_SWISH 5%, MUL (SE) 15% thời gian |
| CONCATENATION | Có | Không | Ghost: 5% thời gian |
| RESIZE_NEAREST_NEIGHBOR | Có | Không | < 0,1% thời gian (PAN-lite 1 lần upsample) |
| TRANSPOSE, RESHAPE, STRIDED_SLICE, PACK, SHAPE | Có | Không | ShuffleNet (channel shuffle + split): ~8% thời gian |
| **TILE** | **Không** | – | `Failed to get registration from op code TILE` → `AllocateTensors` lỗi |
| DETECTION_POSTPROCESS | Có (custom op) | Không | Không cần nếu decode bằng C |

### Bẫy convert đã gặp
- **Keras 3 `UpSampling2D` → TILE** (+ EXPAND_DIMS/RESHAPE/SHAPE/STRIDED_SLICE). Dùng `keras.layers.Resizing(h, w, interpolation="nearest")` → `RESIZE_NEAREST_NEIGHBOR`. [TĐ]
- **PyTorch → ONNX → onnx2tf** sinh thêm `PAD` trước conv stride 2 (padding đối xứng của PyTorch ≠ SAME của TF). Backend mặc định `flatbuffer_direct` của onnx2tf 2.6.9 sinh per-channel quant lỗi cho depthwise (`quantized_dimension=3`) → phải dùng `-tb tf_converter`. [TĐ]
- `Lambda` cắt kênh (split) → `STRIDED_SLICE` + `SHAPE` + `PACK` (op động, chậm, không tăng tốc).

## 3. Quy tắc số kênh cho ESP-NN (quan trọng)
- Depthwise: `channels % 8 != 0` (với ch_mult = 1) → **rơi về bản C thường** [CB:S22]. Depthwise nhanh nhất khi `channels % 16 == 0`.
- Conv 1×1 stride 1 có nhánh tối ưu riêng khi `channels % 8 == 0` [CB:S22].
- Hệ quả đo được [TĐ]: MobileNetV1 α=0.3 (kênh 9/19/38…) chỉ đạt ~47 MMAC/s, còn MobileNetV2 α=0.5 (kênh bội 8) đạt ~85–90 MMAC/s, cả hai cùng arena PSRAM.
- → **Mọi kiến trúc đề xuất phải dùng số kênh là bội của 8 (ưu tiên 16).**

## 4. RAM – điều đo được trên kit
- Sau khi boot, SRAM nội còn ~347 KB, khối liền lớn nhất ~236 KB (chưa bật Wi-Fi, camera, HTTP). [TĐ]
- Arena thực tế lớn hơn ước tính liveness **1,1–3,1×**. Phần "head" (activation + scratch buffer của kernel) chiếm >90%. Ví dụ: MobileNetV1 α=0.3 128² ước tính 112 KB, arena thực 352 KB (head 329 KB). [TĐ] (Giả thuyết: scratch của ESP-NN và cách planner xếp tensor; cần đo kỹ hơn ở GĐ5.)
- Arena trong SRAM nội nhanh hơn PSRAM **~10–25%** với model nhỏ. [TĐ]
- Ở cấu hình hiện tại, mọi model 130–200K params / 128² đều cần arena ≥ 230 KB → **phải đặt arena trong PSRAM**, giống mẫu của BTC.
- Còn phải cộng thêm: framebuffer camera (RGB565 320×240 = 150 KB), input tensor, stack, Wi-Fi/HTTP.

## 5. Bẫy firmware đã gặp
- `EMBED_FILES` của ESP-IDF **không đảm bảo căn lề 16 byte** → phải kiểm tra `(uintptr_t)data % 16` hoặc dùng mảng C `alignas(16)`.
- `tflite::MicroProfiler` (~48 KB) đặt trên stack → **double exception**. Đặt trong PSRAM/static.
- Task chạy TFLM cần stack ≥ 32 KB (main task mặc định không đủ).
- Destructor `MicroInterpreter` sau khi `AllocateTensors()` lỗi → `LoadProhibited`. Không gọi `delete` trong nhánh lỗi.
- Windows + cổng USB-Serial/JTAG: reset bằng RTS khiến chip vào **download mode**. Dùng `esptool --after watchdog_reset` (xem `scripts/analysis/read_serial.py`). Console phải đặt `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y`.

## 6. Checklist cho mỗi ứng viên (GĐ2 trở đi)
- [ ] Tất cả op có trong TFLM của starter kit BTC (kiểm tra lại khi có kit: phiên bản có thể khác 1.4.2)?
- [ ] Mọi conv/depthwise có số kênh bội 8 (ưu tiên 16)?
- [ ] Không có TILE, GATHER, op động (SHAPE/PACK) trong đồ thị?
- [ ] `arena_used_bytes` đo trên kit ≤ ngân sách (PSRAM) và đã cộng framebuffer + Wi-Fi?
- [ ] Latency Invoke + tiền xử lý + decode/NMS ≪ 10 s?
- [ ] Output INT8 trên kit khớp `tf.lite.Interpreter` trên PC (cùng input)?

## 7. Những điều KHÔNG được coi là bằng chứng
Convert `.tflite` thành công, xuất ONNX thành công, hay chạy được bằng `tf.lite.Interpreter` trên PC đều không đủ. Bằng chứng là `AllocateTensors()` + `Invoke()` chạy trên ESP32-S3 thật và output khớp với PC.
