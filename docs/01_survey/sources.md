# Nguồn tham khảo – Giai đoạn 1

Mỗi nguồn có một mã `Sxx`; bảng so sánh trích dẫn dạng `[CB:S01]`. Ngày truy cập: **09/11/2026**.
Số liệu tự đo **[TĐ]** không nằm ở đây mà dẫn tới file trong `experiments/phase1_analysis/`.

## Paper / repo kiến trúc

| Mã | Loại | Tên | Link | Vị trí số liệu |
|---|---|---|---|---|
| S01 | paper | MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018) | https://arxiv.org/abs/1801.04381 | Bảng 4 (ImageNet: 3.4M params, 300M MAdds), Bảng 6 (SSDLite COCO: 22.1 mAP, 4.3M, 0.8B) |
| S02 | paper | Searching for MobileNetV3 (Howard et al., 2019) | https://arxiv.org/abs/1905.02244 | Bảng 6 (SSDLite: V3-Small 16.0 mAP / 2.49M / 0.21B; V3-Small giảm kênh 16.1 / 1.77M / 0.16B; V2 0.35 13.7 / 0.93M / 0.16B), Bảng 3 (V3-Small 1.0: 67.4 top-1, 56M MAdds, 2.5M) |
| S03 | paper | GhostNet: More Features from Cheap Operations (Han et al., 2020) | https://arxiv.org/abs/1911.11907 | Bảng ImageNet (0.5×: 2.6M, 42M FLOPs, 66.2; 1.0×: 5.2M, 141M, 73.9), mục 3.1 (Ghost module) |
| S04 | repo | NanoDet / NanoDet-Plus (RangiLyu) | https://github.com/RangiLyu/nanodet | README "Benchmarks" (NanoDet-m 320: 20.6 mAP, 0.72G, 0.95M; Plus-m 320: 27.0, 0.9G, 1.17M; YOLOX-Nano 416: 25.8, 1.08G, 0.91M; YOLOv5-n 640: 28.4, 4.5G, 1.9M) |
| S05 | paper | PP-PicoDet (Yu et al., 2021) | https://arxiv.org/abs/2111.00902 | Abstract (PicoDet-S 0.99M, 30.6 mAP) |
| S06 | repo | PaddleDetection – configs/picodet | https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet | Model zoo (XS 320: 23.5/36.1 mAP, 0.70M, 0.67G; S 320: 29.1/43.4, 1.18M, 0.97G) |
| S07 | paper | EfficientDet (Tan et al., 2020) | https://arxiv.org/abs/1911.09070 | Abstract, BiFPN |
| S08 | repo | google/automl – efficientdet README | https://github.com/google/automl/blob/master/efficientdet/README.md | Bảng D0 (3.9M, 2.54B, 34.3 AP); bảng Lite (lite0: 26.41 float / 26.10 int8, 3.2M) |
| S09 | paper | SSD: Single Shot MultiBox Detector (Liu et al., 2016) | https://arxiv.org/abs/1512.02325 | Default boxes; số SSD512 lấy qua Bảng 6 của S01 (26.8 mAP, 36.1M, 99.5B) |
| S10 | repo | Yolo-Fastest (dog-qiuqiu) | https://github.com/dog-qiuqiu/Yolo-Fastest | README (1.1, 320: COCO mAP50 24.40%, 0.252 BFLOPs, 0.35M, 1.4 MB; VOC 61.02%) |
| S11 | repo | Yolo-FastestV2 (dog-qiuqiu) | https://github.com/dog-qiuqiu/Yolo-FastestV2 | README (352: COCO mAP50 24.10%, 0.212 GFLOPs, 0.25M; ShuffleNetV2 backbone, anchor-based) |
| S12 | paper | TinyissimoYOLO (Moosmann et al., 2023) | https://arxiv.org/abs/2306.00001 | Abstract (422K params, < 0.5 MB weights, ≤ 3 lớp, MAX78000 có bộ tăng tốc CNN) |
| S13 | docs (vendor) | Edge Impulse – FOMO | https://docs.edgeimpulse.com/ (mục FOMO); slide Edge Impulse 2022: https://www.edge-ai-vision.com/wp-content/uploads/2022/05/E3W04_Jongboom_Edge_Impulse_2022.pdf | MobileNetV2 0.35 cắt cụt, heatmap stride 8, chỉ ra centroid; "< 100 KB" là tuyên bố của nhà cung cấp |
| S14 | paper | Objects as Points – CenterNet (Zhou et al., 2019) | https://arxiv.org/abs/1904.07850 | Abstract (28.1 AP @142 FPS; 37.4 AP @52 FPS), mục 3 |
| S15 | paper | MCUNetV2 (Lin et al., 2021) | https://arxiv.org/abs/2110.15352 | Abstract (giảm peak memory 4–8×, +16.9 mAP VOC trên MCU) |
| S16 | docs | Ultralytics YOLO11 | https://docs.ultralytics.com/models/yolo11/ | Bảng COCO (YOLO11n 640: 39.5 mAP50-95, 2.6M, 6.5B) |

## Phần cứng / triển khai ESP32-S3

| Mã | Loại | Tên | Link | Vị trí số liệu |
|---|---|---|---|---|
| S17 | component | espressif/coco_detect (ESP-DL) | https://components.espressif.com/components/espressif/coco_detect | Bảng model (yolo11n 320 trên S3: model 961.5 ms, mAP50-95 0.276; 640: 3906.3 ms, 0.370) |
| S18 | repo | espressif/esp-detection (ESPDet-Pico) | https://github.com/espressif/esp-detection | Bảng (224×224: 0.36M, 0.17 GFLOPs, S3 126.2 ms, mAP50 88.4 trên tập con "cat" của COCO) |
| S19 | repo | espressif/esp-tflite-micro | https://github.com/espressif/esp-tflite-micro | README (person detection trên S3: 2300 ms không ESP-NN, 54 ms có ESP-NN); bản dùng thử: component **1.4.2** |
| S20 | repo | espressif/esp-nn | https://github.com/espressif/esp-nn ; `docs/performance.md` | Danh sách kernel tối ưu (conv, depthwise, FC, add, mul, max/avg pool, ReLU6, hard-swish, softmax, mean) + bảng speedup S3; bản dùng thử: **1.4.2** |
| S21 | source | `micro_mutable_op_resolver.h` (esp-tflite-micro) | https://github.com/espressif/esp-tflite-micro/blob/master/tensorflow/lite/micro/micro_mutable_op_resolver.h | Danh sách `AddXxx()`: có HARD_SWISH, RESIZE_NEAREST_NEIGHBOR, TRANSPOSE, SPLIT, GATHER, MEAN, DETECTION_POSTPROCESS (custom)…; **không có TILE** |
| S22 | source | esp-nn `src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c`, `esp_nn_conv_esp32s3.c` (v1.4.2) | trong `firmware/bench_probe/managed_components/espressif__esp-nn/` (tải qua component manager) | Chú thích: "ch_mult == 1, channels % 8 != 0: Fallback to C implementation"; conv 1×1 có nhánh `channels % 8 == 0` |
| S23 | tài liệu BTC | Thể lệ + slide Training 1 | `references/` | Xem `docs/00_contest_constraints.md` |

## Ưu tiên nguồn
1. Paper gốc / repo chính thức của tác giả.
2. Tài liệu chính thức (TensorFlow, Espressif, Ultralytics).
3. Tài liệu nhà cung cấp (Edge Impulse) – đánh dấu "vendor".
