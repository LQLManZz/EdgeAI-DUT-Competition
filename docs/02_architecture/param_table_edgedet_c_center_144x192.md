# Bảng phân bổ params – `edgedet_c_center_144x192`

Sinh tự động bởi `scripts/analysis/phase2_export.py` – không sửa tay. [TĐ]

- **Tổng params (Keras, gồm cả BN): 185,307** (giới hạn 200.000 → còn 14,693)
- Trainable: 178,107 · Sau khi gộp BN (hằng số trong .tflite): 174,507
- MACs: 37.66M · Input 144×192×3 · Output 18×24×7, 18×24×4

## Tổng theo khối

| Khối | Params (Keras) | % | Params (gộp BN) | MACs |
|---|---|---|---|---|
| Backbone · stem | 496 | 0.3% | 448 | 2.99M |
| Backbone · b1 | 528 | 0.3% | 432 | 0.69M |
| Backbone · b2 | 4,288 | 2.3% | 3,808 | 2.90M |
| Backbone · b3 | 10,496 | 5.7% | 9,632 | 4.04M |
| Backbone · b3b | 10,496 | 5.7% | 9,632 | 4.04M |
| Backbone · b4 | 12,608 | 6.8% | 11,696 | 2.56M |
| Backbone · b5 | 21,888 | 11.8% | 20,592 | 2.18M |
| Backbone · b6 | 25,024 | 13.5% | 23,680 | 2.51M |
| Backbone · b7 | 37,376 | 20.2% | 35,648 | 3.79M |
| Backbone · b8 | 37,376 | 20.2% | 35,648 | 3.79M |
| Neck · PAN-lite | 16,080 | 8.7% | 15,120 | 4.73M |
| Head | 8,651 | 4.7% | 8,171 | 3.46M |
| **Tổng** | **185,307** | 100% | **174,507** | **37.66M** |

## Chi tiết từng layer

| Khối | Layer | Loại | Output (H×W×C) | Params | Gộp BN | MACs |
|---|---|---|---|---|---|---|
| Backbone · stem | `stem_conv` | Conv 3x3/2 | 72×96×16 | 432 | 448 | 2,985,984 |
| Backbone · stem | `stem_bn` | BN (gộp vào conv) | 72×96×16 | 64 | 0 | 0 |
| Backbone · b1 | `b1_dw` | DW 3x3/2 | 36×48×16 | 144 | 160 | 248,832 |
| Backbone · b1 | `b1_dwbn` | BN (gộp vào conv) | 36×48×16 | 64 | 0 | 0 |
| Backbone · b1 | `b1_proj_conv` | Conv 1x1/1 | 36×48×16 | 256 | 272 | 442,368 |
| Backbone · b1 | `b1_proj_bn` | BN (gộp vào conv) | 36×48×16 | 64 | 0 | 0 |
| Backbone · b2 | `b2_exp_conv` | Conv 1x1/1 | 36×48×64 | 1,024 | 1,088 | 1,769,472 |
| Backbone · b2 | `b2_exp_bn` | BN (gộp vào conv) | 36×48×64 | 256 | 0 | 0 |
| Backbone · b2 | `b2_dw` | DW 3x3/2 | 18×24×64 | 576 | 640 | 248,832 |
| Backbone · b2 | `b2_dwbn` | BN (gộp vào conv) | 18×24×64 | 256 | 0 | 0 |
| Backbone · b2 | `b2_proj_conv` | Conv 1x1/1 | 18×24×32 | 2,048 | 2,080 | 884,736 |
| Backbone · b2 | `b2_proj_bn` | BN (gộp vào conv) | 18×24×32 | 128 | 0 | 0 |
| Backbone · b3 | `b3_exp_conv` | Conv 1x1/1 | 18×24×128 | 4,096 | 4,224 | 1,769,472 |
| Backbone · b3 | `b3_exp_bn` | BN (gộp vào conv) | 18×24×128 | 512 | 0 | 0 |
| Backbone · b3 | `b3_dw` | DW 3x3/1 | 18×24×128 | 1,152 | 1,280 | 497,664 |
| Backbone · b3 | `b3_dwbn` | BN (gộp vào conv) | 18×24×128 | 512 | 0 | 0 |
| Backbone · b3 | `b3_proj_conv` | Conv 1x1/1 | 18×24×32 | 4,096 | 4,128 | 1,769,472 |
| Backbone · b3 | `b3_proj_bn` | BN (gộp vào conv) | 18×24×32 | 128 | 0 | 0 |
| Backbone · b3b | `b3b_exp_conv` | Conv 1x1/1 | 18×24×128 | 4,096 | 4,224 | 1,769,472 |
| Backbone · b3b | `b3b_exp_bn` | BN (gộp vào conv) | 18×24×128 | 512 | 0 | 0 |
| Backbone · b3b | `b3b_dw` | DW 3x3/1 | 18×24×128 | 1,152 | 1,280 | 497,664 |
| Backbone · b3b | `b3b_dwbn` | BN (gộp vào conv) | 18×24×128 | 512 | 0 | 0 |
| Backbone · b3b | `b3b_proj_conv` | Conv 1x1/1 | 18×24×32 | 4,096 | 4,128 | 1,769,472 |
| Backbone · b3b | `b3b_proj_bn` | BN (gộp vào conv) | 18×24×32 | 128 | 0 | 0 |
| Backbone · b4 | `b4_exp_conv` | Conv 1x1/1 | 18×24×128 | 4,096 | 4,224 | 1,769,472 |
| Backbone · b4 | `b4_exp_bn` | BN (gộp vào conv) | 18×24×128 | 512 | 0 | 0 |
| Backbone · b4 | `b4_dw` | DW 3x3/2 | 9×12×128 | 1,152 | 1,280 | 124,416 |
| Backbone · b4 | `b4_dwbn` | BN (gộp vào conv) | 9×12×128 | 512 | 0 | 0 |
| Backbone · b4 | `b4_proj_conv` | Conv 1x1/1 | 9×12×48 | 6,144 | 6,192 | 663,552 |
| Backbone · b4 | `b4_proj_bn` | BN (gộp vào conv) | 9×12×48 | 192 | 0 | 0 |
| Backbone · b5 | `b5_exp_conv` | Conv 1x1/1 | 9×12×192 | 9,216 | 9,408 | 995,328 |
| Backbone · b5 | `b5_exp_bn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · b5 | `b5_dw` | DW 3x3/1 | 9×12×192 | 1,728 | 1,920 | 186,624 |
| Backbone · b5 | `b5_dwbn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · b5 | `b5_proj_conv` | Conv 1x1/1 | 9×12×48 | 9,216 | 9,264 | 995,328 |
| Backbone · b5 | `b5_proj_bn` | BN (gộp vào conv) | 9×12×48 | 192 | 0 | 0 |
| Backbone · b6 | `b6_exp_conv` | Conv 1x1/1 | 9×12×192 | 9,216 | 9,408 | 995,328 |
| Backbone · b6 | `b6_exp_bn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · b6 | `b6_dw` | DW 3x3/1 | 9×12×192 | 1,728 | 1,920 | 186,624 |
| Backbone · b6 | `b6_dwbn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · b6 | `b6_proj_conv` | Conv 1x1/1 | 9×12×64 | 12,288 | 12,352 | 1,327,104 |
| Backbone · b6 | `b6_proj_bn` | BN (gộp vào conv) | 9×12×64 | 256 | 0 | 0 |
| Backbone · b7 | `b7_exp_conv` | Conv 1x1/1 | 9×12×256 | 16,384 | 16,640 | 1,769,472 |
| Backbone · b7 | `b7_exp_bn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · b7 | `b7_dw` | DW 3x3/1 | 9×12×256 | 2,304 | 2,560 | 248,832 |
| Backbone · b7 | `b7_dwbn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · b7 | `b7_proj_conv` | Conv 1x1/1 | 9×12×64 | 16,384 | 16,448 | 1,769,472 |
| Backbone · b7 | `b7_proj_bn` | BN (gộp vào conv) | 9×12×64 | 256 | 0 | 0 |
| Backbone · b8 | `b8_exp_conv` | Conv 1x1/1 | 9×12×256 | 16,384 | 16,640 | 1,769,472 |
| Backbone · b8 | `b8_exp_bn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · b8 | `b8_dw` | DW 3x3/1 | 9×12×256 | 2,304 | 2,560 | 248,832 |
| Backbone · b8 | `b8_dwbn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · b8 | `b8_proj_conv` | Conv 1x1/1 | 9×12×64 | 16,384 | 16,448 | 1,769,472 |
| Backbone · b8 | `b8_proj_bn` | BN (gộp vào conv) | 9×12×64 | 256 | 0 | 0 |
| Neck · PAN-lite | `neck_l16_conv` | Conv 1x1/1 | 9×12×80 | 5,120 | 5,200 | 552,960 |
| Neck · PAN-lite | `neck_l16_bn` | BN (gộp vào conv) | 9×12×80 | 320 | 0 | 0 |
| Neck · PAN-lite | `neck_l8_conv` | Conv 1x1/1 | 18×24×80 | 2,560 | 2,640 | 1,105,920 |
| Neck · PAN-lite | `neck_l8_bn` | BN (gộp vào conv) | 18×24×80 | 320 | 0 | 0 |
| Neck · PAN-lite | `neck_fuse_dw` | DW 3x3/1 | 18×24×80 | 720 | 800 | 311,040 |
| Neck · PAN-lite | `neck_fuse_dwbn` | BN (gộp vào conv) | 18×24×80 | 320 | 0 | 0 |
| Neck · PAN-lite | `neck_fuse_pw_conv` | Conv 1x1/1 | 18×24×80 | 6,400 | 6,480 | 2,764,800 |
| Neck · PAN-lite | `neck_fuse_pw_bn` | BN (gộp vào conv) | 18×24×80 | 320 | 0 | 0 |
| Head | `head_dw` | DW 3x3/1 | 18×24×80 | 720 | 800 | 311,040 |
| Head | `head_dwbn` | BN (gộp vào conv) | 18×24×80 | 320 | 0 | 0 |
| Head | `head_pw_conv` | Conv 1x1/1 | 18×24×80 | 6,400 | 6,480 | 2,764,800 |
| Head | `head_pw_bn` | BN (gộp vào conv) | 18×24×80 | 320 | 0 | 0 |
| Head | `head_hm` | Conv 1x1/1 | 18×24×7 | 567 | 567 | 241,920 |
| Head | `head_box` | Conv 1x1/1 | 18×24×4 | 324 | 324 | 138,240 |
