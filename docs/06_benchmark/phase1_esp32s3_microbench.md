# D8 (phần GĐ1) – Microbenchmark kiến trúc trên ESP32-S3 thật

> Ngày đo: 09/11/2026 · Mục đích: đo **chi phí triển khai** (arena, latency, op) của từng họ kiến trúc *trước khi* train, để chọn hướng thiết kế bằng số liệu.
> Đây **không phải** benchmark model cuối cùng: model probe có trọng số ngẫu nhiên, nên không có số accuracy. D8 đầy đủ (mAP, latency end-to-end, RAM) sẽ làm ở GĐ5.

## 1. Thiết lập

| Mục | Giá trị |
|---|---|
| Board | ESP32-S3 N16R8 (QFN56, rev v0.2), PSRAM octal 8 MB (AP, 3V), Flash 16 MB QIO 80 MHz, COM8 = USB-Serial/JTAG |
| CPU | 240 MHz; benchmark chạy trên core 1, task riêng (stack 32 KB) |
| SDK | ESP-IDF **v5.2.6** (cùng bản với repo mẫu BTC), `-O2` (`CONFIG_COMPILER_OPTIMIZATION_PERF`) |
| Inference | `espressif/esp-tflite-micro` **1.4.2** + `espressif/esp-nn` **1.4.2** (`CONFIG_NN_OPTIMIZED=y`) |
| Model | 29 file `.tflite` INT8 (I/O int8) từ `scripts/analysis/zoo.py`, nhúng vào flash bằng `EMBED_FILES` |
| Tensor arena | PSRAM (3 MB cấp sẵn), và SRAM nội nếu `arena_used + 17 KB < khối liền lớn nhất` |
| Đo | 1 lần warm-up + 5 lần `Invoke()` bằng `esp_timer_get_time()`. Độ lệch min–max < 0,1 ms với đa số model |
| Profile | `MicroProfiler` (tổng thời gian theo loại op) + `RecordingMicroInterpreter::PrintAllocations()` |
| Cấu hình cache | A: D-cache 32 KB, line 32 B (mặc định IDF) · B: D-cache 64 KB, line 64 B |
| Không bật | Wi-Fi, camera, HTTP server (sẽ làm giảm SRAM khả dụng khi tích hợp) |

Mã nguồn: `firmware/bench_probe/`. Log thô: `experiments/phase1_analysis/esp32s3_bench_dcache32k.txt`, `…_dcache64k.txt`. Bảng gộp: `experiments/phase1_analysis/phase1_summary.csv`.

Bộ nhớ sau boot (1 KB = 1024 B): SRAM nội trống 347 KB, khối liền lớn nhất 236 KB (cấu hình A). Cấu hình B còn 316 KB / 236 KB, tức cache 64 KB lấy thêm ~31 KB SRAM. PSRAM trống 8,1 MB.

## 2. Kết quả chính (cấu hình A, arena PSRAM, input 128×128)

| Họ (probe) | Params | MACs | Arena | Latency | MMAC/s |
|---|---|---|---|---|---|
| MNv3-Small-minimalistic s16 | 128K | 12,6M | 230 KB | **133 ms** | 95 |
| MNv3-Small-minimalistic + PAN s8 | 133K | 14,4M | 230 KB | **139 ms** | 103 |
| MNv3-Small đầy đủ (SE + h-swish) s16 | 162K | 11,4M | 242 KB | 170 ms | 67 |
| MNv2-0.5 s16 | 193K | 24,2M | 322 KB | 283 ms | 85 |
| MNv2-0.5 + PAN s8 | 197K | 26,0M | 322 KB | 290 ms | 90 |
| MNv1-0.3 s16 (kênh lẻ) | 168K | 16,6M | 352 KB | 351 ms | 47 |
| ShuffleNetV2-2.25 s16 | 163K | 25,3M | 294 KB | 470 ms | 54 |
| GhostNet-1.75 s16 | 172K | 43,7M | 1444 KB | 1103 ms | 40 |
| MNv2-0.5 + PAN dùng `UpSampling2D` | 197K | 26,0M | – | **lỗi: `Failed to get registration from op code TILE`** | – |

Bảng đầy đủ (head, độ phân giải, SRAM/PSRAM): [../01_survey/comparison_table.md](../01_survey/comparison_table.md), Bảng 2–2d.

## 3. Ảnh hưởng của cấu hình cache (B so với A, arena PSRAM)

| Probe | A: D$ 32 KB | B: D$ 64 KB | Δ |
|---|---|---|---|
| MNv3-S-min + PAN 128² | 139,2 ms | 125,0 ms | −10% |
| MNv2-0.5 + PAN 128² | 289,9 ms | 254,0 ms | −12% |
| MNv2-0.5 + PAN 96² | 169,0 ms | 141,8 ms | −16% |
| MNv2-0.5 + PAN 240×320 | 1438,7 ms | 1234,8 ms | −14% |
| GhostNet-1.75 + PAN 128² | 1123,1 ms | 990,3 ms | −12% |
| ShuffleV2-2.25 + PAN 128² | 485,0 ms | 448,3 ms | −8% |
| MNv1-0.25 s8 128² | 93,3 ms | 78,4 ms | −16% |

- Arena trong **PSRAM**: D-cache 64 KB / line 64 B giảm latency **6–16%**.
- Arena trong **SRAM nội**: không đổi (ví dụ MNv1-0.25 96²: 39,36 ms ở cả hai cấu hình).
- Đánh đổi: cache 64 KB lấy thêm 32 KB SRAM nội.

## 4. Arena: thực tế so với ước tính

| Probe | Peak liveness (PC) | Arena thực (kit) | Tỉ lệ | "Head" (activation + scratch) | Persistent |
|---|---|---|---|---|---|
| MNv1-0.3 s16 128² | 112 KB | 352 KB | 3,1× | 328,7 KB | 20,9 KB buffer + ~2 KB struct |
| GhostNet-1.75 s16 128² | 784 KB | 1444 KB | 1,8× | 1410,6 KB | 28,0 KB buffer + ~4 KB struct |
| MNv3-S-min s16 128² | 112 KB | 230 KB | 2,1× | – | – |
| MNv2-0.5 + PAN 128² | 240 KB | 322 KB | 1,3× | – | – |
| ShuffleV2-2.25 s16 128² | 260 KB | 294 KB | 1,1× | – | – |

→ **Không dùng ước tính liveness làm ngân sách RAM.** Phải đo `arena_used_bytes()` trên kit. Giả thuyết cho phần chênh lệch là scratch buffer của kernel ESP-NN (conv/depthwise xin scratch trong vùng head) cộng cách planner xếp tensor. Sẽ kiểm chứng ở GĐ5.

## 5. Kết luận cho thiết kế
1. Latency không phải nút thắt (≤ 1,5 s kể cả 240×320); **params và chất lượng mới là nút thắt.**
2. Ưu tiên backbone inverted-residual **không SE/h-swish, kênh bội 8/16** (MNv3-S-minimalistic / MNv2).
3. Arena đặt trong PSRAM. Đề xuất bật D-cache 64 KB nếu tích hợp Wi-Fi + camera vẫn còn đủ SRAM; cần kiểm chứng ở GĐ5.
4. Không dùng `UpSampling2D` (Keras 3), Ghost, channel shuffle.

## 6. Bẫy firmware gặp trong lúc đo
Xem [../01_survey/tflm_checklist.md](../01_survey/tflm_checklist.md) mục 5: EMBED_FILES không căn lề, profiler trên stack gây double exception, destructor sau `AllocateTensors` lỗi, reset USB-Serial/JTAG trên Windows, console USB-Serial/JTAG.
