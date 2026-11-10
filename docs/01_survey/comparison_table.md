# D2 – Bảng so sánh kiến trúc

Nhãn: **[CB:Sxx]** công bố (xem [sources.md](sources.md)) · **[TĐ]** tự đo (`experiments/phase1_analysis/`) · **[ƯT]** ước tính tay. Ô trống = không có nguồn.
FLOPs/MACs ghi đúng cách gọi của nguồn (nhiều repo ghi "FLOPs" nhưng thực chất là MACs).

## Bảng 1 – Phiên bản gốc (số công bố)

| Model                     | Phiên bản nhỏ nhất | Input     | Params | FLOPs / MACs | mAP (dataset)                   | Backbone                      | Neck        | Head                     | Nguồn           |
| ------------------------- | ------------------ | --------- | ------ | ------------ | ------------------------------- | ----------------------------- | ----------- | ------------------------ | --------------- |
| MobileNetV2 (classifier)  | 1.0                | 224²      | 3.4M   | 300M MAdds   | 72.0 top-1 (ImageNet)           | Inverted residual             | –           | FC                       | [CB:S01]        |
| MobileNetV2-SSDLite       | 1.0                | 320²      | 4.3M   | 0.8B MAdds   | 22.1 mAP (COCO)                 | MNv2                          | –           | SSDLite (anchor)         | [CB:S01]        |
| MobileNetV2-0.35-SSDLite  | 0.35               | 320²      | 0.93M  | 0.16B MAdds  | 13.7 mAP (COCO)                 | MNv2 0.35                     | –           | SSDLite                  | [CB:S02]        |
| MobileNetV3-Small-SSDLite | Small (giảm kênh)  | 320²      | 1.77M  | 0.16B MAdds  | 16.1 mAP (COCO)                 | MNv3-S (SE, h-swish)          | –           | SSDLite                  | [CB:S02]        |
| GhostNet (classifier)     | 0.5×               | 224²      | 2.6M   | 42M FLOPs    | 66.2 top-1 (ImageNet)           | Ghost bottleneck              | –           | FC                       | [CB:S03]        |
| NanoDet-m                 | m                  | 320²      | 0.95M  | 0.72G FLOPs  | 20.6 mAP (COCO)                 | ShuffleNetV2 1.0×             | PAN         | GFL, anchor-free         | [CB:S04]        |
| NanoDet-Plus-m            | m                  | 320²      | 1.17M  | 0.9G FLOPs   | 27.0 mAP (COCO)                 | ShuffleNetV2 1.0×             | Ghost-PAN   | GFL + AGM/DSLA           | [CB:S04]        |
| PicoDet-XS                | XS                 | 320²      | 0.70M  | 0.67G FLOPs  | 23.5 mAP / 36.1 mAP50 (COCO)    | ESNet                         | CSP-PAN     | GFL-style, SimOTA        | [CB:S06]        |
| EfficientDet-Lite0        | Lite0              | 320²      | 3.2M   | –            | 26.41 float / 26.10 int8 (COCO) | EfficientNet-Lite0            | BiFPN       | shared, anchor           | [CB:S08]        |
| SSD512 (VGG)              | –                  | 512²      | 36.1M  | 99.5B MAdds  | 26.8 mAP (COCO)                 | VGG-16                        | multi-scale | SSD (anchor)             | [CB:S01, S09]   |
| YOLO-Fastest 1.1          | 1.1                | 320²      | 0.35M  | 0.252 BFLOPs | 24.4 mAP50 (COCO); 61.0 (VOC)   | depthwise (EffNet-lite style) | FPN 2 scale | YOLO (anchor)            | [CB:S10]        |
| YOLO-FastestV2            | V2                 | 352²      | 0.25M  | 0.212 GFLOPs | 24.1 mAP50 (COCO)               | ShuffleNetV2                  | FPN 2 scale | tách obj/cls/reg, anchor | [CB:S11]        |
| TinyissimoYOLO            | –                  | –         | 422K   | –            | – (≤ 3 lớp)                     | –                             | –           | YOLO                     | [CB:S12]        |
| ESPDet-Pico               | pico               | 224²      | 0.36M  | 0.17 GFLOPs  | 88.4 mAP50 (COCO, chỉ lớp cat)  | (YOLO11-based)                | –           | –                        | [CB:S18]        |
| FOMO                      | MNv2 0.35          | 96² (tùy) | –      | –            | – (không có bbox)               | MNv2 0.35 cắt stride 8        | –           | heatmap centroid         | [CB:S13] vendor |
| CenterNet                 | ResNet-18          | 512²      | –      | –            | 28.1 AP @142 FPS (COCO)         | ResNet/DLA/Hourglass          | deconv      | heatmap + wh + offset    | [CB:S14]        |
| YOLO11n (teacher)         | n                  | 640²      | 2.6M   | 6.5B FLOPs   | 39.5 mAP50-95 (COCO)            | C3k2                          | PAN         | decoupled, anchor-free   | [CB:S16]        |
| YOLOX-Nano                | Nano               | 416²      | 0.91M  | 1.08G FLOPs  | 25.8 mAP (COCO)                 | CSPDarknet-dw                 | PAN         | decoupled                | [CB:S04]        |


**Nhận xét:** không model công bố nào ≤ 200K params. Gần nhất là YOLO-FastestV2 (0.25M).

## Bảng 2 – Thu về ≤ 200K params và đo trên ESP32-S3 thật

Probe Keras, trọng số ngẫu nhiên, head CenterNet 11 kênh, INT8 PTQ, input 128×128.
Kit: ESP32-S3 N16R8 rev v0.2, 240 MHz, ESP-IDF 5.2.6, esp-tflite-micro 1.4.2, esp-nn 1.4.2, D-cache 32 KB, arena trong PSRAM, trung bình 5 lần Invoke.
Nguồn: `experiments/phase1_analysis/phase1_summary.csv`. **Tất cả [TĐ].** Với D-cache 64 KB (line 64 B), latency PSRAM giảm thêm 6–16% (cột `ms_psram_dc64k` trong CSV).

| Họ (probe)                                 | Thu về ≤200K bằng cách   | Params | MACs  | .tflite | Peak liveness | **Arena thực**          | **Latency** | MMAC/s | Op chiếm thời gian                           | Op không ESP-NN                        | Đánh giá             |
| ------------------------------------------ | ------------------------ | ------ | ----- | ------- | ------------- | ----------------------- | ----------- | ------ | -------------------------------------------- | -------------------------------------- | -------------------- |
| MNv3-Small-minimal, s16                    | α=1.0, cắt stride 16     | 128K   | 12,6M | 207 KB  | 112 KB        | **230 KB**              | **133 ms**  | 95     | CONV 64%, DW 35%                             | –                                      | **Tốt**              |
| MNv3-Small-minimal + PAN s8                | + nhánh stride 8         | 133K   | 14,4M | 210 KB  | 112 KB        | 230 KB                  | 139 ms      | 103    | CONV 65%, DW 32%                             | RESIZE (0%)                            | **Tốt**              |
| MNv3-Small đầy đủ (SE, h-swish), s16       | α=0.75                   | 162K   | 11,4M | 269 KB  | 128 KB        | 242 KB                  | 170 ms      | 67     | CONV 47%, DW 30%, **MUL 15%**, **HSWISH 5%** | –                                      | Được (bỏ SE/h-swish) |
| MNv2, s16                                  | α=0.5, cắt stride 16     | 193K   | 24,2M | 313 KB  | 240 KB        | 322 KB                  | 283 ms      | 85     | CONV 59%, DW 39%                             | –                                      | **Tốt**              |
| MNv2 + PAN s8                              | α=0.5                    | 197K   | 26,0M | 316 KB  | 240 KB        | 322 KB                  | 290 ms      | 90     | CONV 60%, DW 37%                             | RESIZE (0%)                            | **Tốt**              |
| MNv1 (DW-separable), s16                   | α=0.3 (kênh lẻ 9/19/38…) | 168K   | 16,6M | 230 KB  | 112 KB        | 352 KB                  | 351 ms      | **47** | CONV 66%, DW 34%                             | (DW fallback C)                        | Được nếu kênh bội 8  |
| GhostNet, s16                              | width 1.75               | 172K   | 43,7M | 266 KB  | 784 KB        | **1444 KB**             | **1103 ms** | 40     | CONV 47%, DW 44%, CONCAT 5%                  | CONCAT                                 | **Loại**             |
| ShuffleNetV2 (YOLO-FastestV2/NanoDet), s16 | width 2.25               | 163K   | 25,3M | 251 KB  | 260 KB        | 294 KB                  | 470 ms      | 54     | CONV 73%, DW 16%, TRANSPOSE 4%, RESHAPE 2%   | TRANSPOSE, RESHAPE, SLICE, PACK, SHAPE | Loại (backbone)      |
| Bẫy: MNv2 + PAN dùng `UpSampling2D`        | –                        | 197K   | 26,0M | 320 KB  | –             | **AllocateTensors lỗi** | –           | –      | –                                            | **TILE (không có trong TFLM)**         | Loại                 |

### Bảng 2b – Head trên cùng backbone (MNv2 α=0.5 + PAN s8, 128²) [TĐ]

| Head | Kênh output | Params | Latency | Decode trên MCU |
|---|---|---|---|---|
| CenterNet (7 heatmap + 2 wh + 2 offset) | 16×16×11 | 197.115 | 290 ms | max-pool 3×3 + ngưỡng, không NMS |
| YOLO anchor (3 × (5+7)) | 16×16×36 | 198.340 | 291 ms | sigmoid + anchor + NMS |
| GFL (7 + 4×8 bin) – NanoDet/PicoDet | 16×16×39 | 198.487 | 291 ms | softmax 8 bin × 4 cạnh + NMS |
| FOMO (7 + nền) | 16×16×8 | 195.384 | 292 ms | argmax/ngưỡng, **không có box** |

→ Head không ảnh hưởng latency. Chọn head theo accuracy và độ phức tạp decode.

### Bảng 2c – Độ phân giải input [TĐ]

| Probe | 96×96 | 96×128 | 128×128 | 160×160 | 240×320 |
|---|---|---|---|---|---|
| MNv2-0.5 + PAN: MACs | 14,6M | 19,5M | 26,0M | 40,6M | 121,9M |
| … arena | 228 KB | 274 KB | 322 KB | 458 KB | 1245 KB |
| … latency (PSRAM) | 169 ms | 216 ms | 290 ms | 463 ms | **1439 ms** |
| GhostNet-1.75 + PAN: latency | 639 ms | 836 ms | 1123 ms | 1773 ms | – |
| ShuffleV2-2.25 + PAN: latency | 279 ms | 361 ms | 485 ms | 783 ms | – |

Params **không đổi** theo độ phân giải (197K). Ngay cả 240×320 vẫn chỉ dùng ~14% ngân sách 10 s.

### Bảng 2d – Model rất nhỏ (kiểu FOMO): SRAM nội vs PSRAM [TĐ]

| Probe | Params | MACs | Arena | PSRAM | **SRAM nội** | Lợi |
|---|---|---|---|---|---|---|
| MNv1-0.25 s8 96² | 15K | 3,7M | 98 KB | 49,3 ms | 39,4 ms | −20% |
| MNv1-0.25 s8 128² | 15K | 6,6M | 153 KB | 93,3 ms | 69,8 ms | −25% |
| MNv2-0.35 s8 96² (CenterNet) | 24K | 6,3M | 187 KB | 106,4 ms | 80,8 ms | −24% |
| MNv2-0.35 s8 96² (FOMO) | 21K | 5,9M | 186 KB | 102,7 ms | 78,0 ms | −24% |
| ShuffleV2-2.25 + PAN 96² | 169K | 15,4M | 181 KB | 279,3 ms | 253,0 ms | −9% |

Tham chiếu [CB:S19]: person detection (MobileNetV1 0.25, 96×96 grayscale) trên ESP32-S3 với ESP-NN là 54 ms. Cùng cỡ với MNv1-0.25 96² đo được ở trên.

## Bảng 3 – Phù hợp với bài toán cuộc thi

| Model / ý tưởng            | Vật nhỏ                         | Che khuất / vật gần nhau                       | Ra bbox (mAP50 Kaggle) | Tâm box (LOCATION/COUNT) | Decode/NMS trên MCU   | Kết luận                |
| -------------------------- | ------------------------------- | ---------------------------------------------- | ---------------------- | ------------------------ | --------------------- | ----------------------- |
| CenterNet head @stride 8   | Tốt nếu stride 8 + input đủ lớn | Tách được nếu tâm cách nhau ≥ 1 ô (8 px input) | Có (wh)                | **Trực tiếp**            | Rất rẻ (max-pool 3×3) | **H1**                  |
| YOLO anchor head (Fastest) | Phụ thuộc anchor                | NMS có thể gộp nhầm vật sát nhau               | Có                     | Gián tiếp (tâm box)      | NMS O(n²), n nhỏ – rẻ | **H2**                  |
| GFL head (NanoDet/PicoDet) | Tốt (multi-scale gốc)           | Tốt                                            | Có, khít               | Gián tiếp                | softmax + NMS – vừa   | Giữ ý tưởng loss/assign |
| FOMO                       | Tốt (stride 8)                  | Tốt                                            | **Không**              | Trực tiếp                | Rẻ                    | Loại nguyên bản         |
| SSD multi-anchor           | Trung bình                      | Trung bình                                     | Có                     | Gián tiếp                | Nhiều anchor + NMS    | Loại                    |
| EfficientDet BiFPN         | Tốt                             | Tốt                                            | Có                     | Gián tiếp                | NMS                   | Loại (params)           |

## Công thức ước tính
- Params Conv2D: `k·k·C_in·C_out (+C_out bias)`; DepthwiseConv2D: `k·k·C (+C)`; BatchNorm được gộp vào conv khi convert.
- MACs Conv2D: `H_out·W_out·k·k·C_in·C_out`; DepthwiseConv2D: `H_out·W_out·k·k·C`.
- Activation INT8 (byte): `H·W·C`.
- Peak liveness: tổng byte các tensor còn sống tại mỗi op, theo thứ tự thực thi (cận dưới: chưa tính scratch của kernel).
