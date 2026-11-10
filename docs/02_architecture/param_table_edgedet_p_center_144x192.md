# Bảng phân bổ params – `edgedet_p_center_144x192`

Sinh tự động bởi `scripts/analysis/phase2_export.py` – không sửa tay. [TĐ]

- **Tổng params (Keras, gồm cả BN): 189,379** (giới hạn 200.000 → còn 10,621)
- Trainable: 181,299 · Sau khi gộp BN (hằng số trong .tflite): 177,259
- MACs: 35.05M · Input 144×192×3 · Output 18×24×7, 18×24×4

## Tổng theo khối

| Khối | Params (Keras) | % | Params (gộp BN) | MACs |
|---|---|---|---|---|
| Backbone · MobileNetV3-Small (pretrained) | 95,112 | 50.2% | 88,176 | 17.87M |
| Backbone · khối thêm stride 16 | 62,400 | 32.9% | 59,328 | 6.30M |
| Backbone · khối thêm stride 8 | 7,136 | 3.8% | 6,464 | 2.70M |
| Neck · PAN-lite | 16,080 | 8.5% | 15,120 | 4.73M |
| Head | 8,651 | 4.6% | 8,171 | 3.46M |
| **Tổng** | **189,379** | 100% | **177,259** | **35.05M** |

## Chi tiết từng layer

| Khối | Layer | Loại | Output (H×W×C) | Params | Gộp BN | MACs |
|---|---|---|---|---|---|---|
| Backbone · MobileNetV3-Small (pretrained) | `conv` | Conv 3x3/2 | 72×96×16 | 432 | 448 | 2,985,984 |
| Backbone · MobileNetV3-Small (pretrained) | `conv_bn` | BN (gộp vào conv) | 72×96×16 | 64 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_depthwise` | DW 3x3/2 | 36×48×16 | 144 | 160 | 248,832 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_depthwise_bn` | BN (gộp vào conv) | 36×48×16 | 64 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_project` | Conv 1x1/1 | 36×48×16 | 256 | 272 | 442,368 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_project_bn` | BN (gộp vào conv) | 36×48×16 | 64 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_expand` | Conv 1x1/1 | 36×48×72 | 1,152 | 1,224 | 1,990,656 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_expand_bn` | BN (gộp vào conv) | 36×48×72 | 288 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_depthwise` | DW 3x3/2 | 18×24×72 | 648 | 720 | 279,936 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_depthwise_bn` | BN (gộp vào conv) | 18×24×72 | 288 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_project` | Conv 1x1/1 | 18×24×24 | 1,728 | 1,752 | 746,496 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_1_project_bn` | BN (gộp vào conv) | 18×24×24 | 96 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_expand` | Conv 1x1/1 | 18×24×88 | 2,112 | 2,200 | 912,384 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_expand_bn` | BN (gộp vào conv) | 18×24×88 | 352 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_depthwise` | DW 3x3/1 | 18×24×88 | 792 | 880 | 342,144 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_depthwise_bn` | BN (gộp vào conv) | 18×24×88 | 352 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_project` | Conv 1x1/1 | 18×24×24 | 2,112 | 2,136 | 912,384 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_2_project_bn` | BN (gộp vào conv) | 18×24×24 | 96 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_expand` | Conv 1x1/1 | 18×24×96 | 2,304 | 2,400 | 995,328 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_expand_bn` | BN (gộp vào conv) | 18×24×96 | 384 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_depthwise` | DW 3x3/2 | 9×12×96 | 864 | 960 | 93,312 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_depthwise_bn` | BN (gộp vào conv) | 9×12×96 | 384 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_project` | Conv 1x1/1 | 9×12×40 | 3,840 | 3,880 | 414,720 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_3_project_bn` | BN (gộp vào conv) | 9×12×40 | 160 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_expand` | Conv 1x1/1 | 9×12×240 | 9,600 | 9,840 | 1,036,800 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_expand_bn` | BN (gộp vào conv) | 9×12×240 | 960 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_depthwise` | DW 3x3/1 | 9×12×240 | 2,160 | 2,400 | 233,280 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_depthwise_bn` | BN (gộp vào conv) | 9×12×240 | 960 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_project` | Conv 1x1/1 | 9×12×40 | 9,600 | 9,640 | 1,036,800 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_4_project_bn` | BN (gộp vào conv) | 9×12×40 | 160 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_expand` | Conv 1x1/1 | 9×12×240 | 9,600 | 9,840 | 1,036,800 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_expand_bn` | BN (gộp vào conv) | 9×12×240 | 960 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_depthwise` | DW 3x3/1 | 9×12×240 | 2,160 | 2,400 | 233,280 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_depthwise_bn` | BN (gộp vào conv) | 9×12×240 | 960 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_project` | Conv 1x1/1 | 9×12×40 | 9,600 | 9,640 | 1,036,800 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_5_project_bn` | BN (gộp vào conv) | 9×12×40 | 160 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_expand` | Conv 1x1/1 | 9×12×120 | 4,800 | 4,920 | 518,400 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_expand_bn` | BN (gộp vào conv) | 9×12×120 | 480 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_depthwise` | DW 3x3/1 | 9×12×120 | 1,080 | 1,200 | 116,640 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_depthwise_bn` | BN (gộp vào conv) | 9×12×120 | 480 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_project` | Conv 1x1/1 | 9×12×48 | 5,760 | 5,808 | 622,080 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_6_project_bn` | BN (gộp vào conv) | 9×12×48 | 192 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_expand` | Conv 1x1/1 | 9×12×144 | 6,912 | 7,056 | 746,496 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_expand_bn` | BN (gộp vào conv) | 9×12×144 | 576 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_depthwise` | DW 3x3/1 | 9×12×144 | 1,296 | 1,440 | 139,968 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_depthwise_bn` | BN (gộp vào conv) | 9×12×144 | 576 | 0 | 0 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_project` | Conv 1x1/1 | 9×12×48 | 6,912 | 6,960 | 746,496 |
| Backbone · MobileNetV3-Small (pretrained) | `expanded_conv_7_project_bn` | BN (gộp vào conv) | 9×12×48 | 192 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16a_exp_conv` | Conv 1x1/1 | 9×12×192 | 9,216 | 9,408 | 995,328 |
| Backbone · khối thêm stride 16 | `x16a_exp_bn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16a_dw` | DW 3x3/1 | 9×12×192 | 1,728 | 1,920 | 186,624 |
| Backbone · khối thêm stride 16 | `x16a_dwbn` | BN (gộp vào conv) | 9×12×192 | 768 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16a_proj_conv` | Conv 1x1/1 | 9×12×64 | 12,288 | 12,352 | 1,327,104 |
| Backbone · khối thêm stride 16 | `x16a_proj_bn` | BN (gộp vào conv) | 9×12×64 | 256 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16b_exp_conv` | Conv 1x1/1 | 9×12×256 | 16,384 | 16,640 | 1,769,472 |
| Backbone · khối thêm stride 16 | `x16b_exp_bn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · khối thêm stride 8 | `x8a_exp_conv` | Conv 1x1/1 | 18×24×96 | 2,304 | 2,400 | 995,328 |
| Backbone · khối thêm stride 16 | `x16b_dw` | DW 3x3/1 | 9×12×256 | 2,304 | 2,560 | 248,832 |
| Backbone · khối thêm stride 8 | `x8a_exp_bn` | BN (gộp vào conv) | 18×24×96 | 384 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16b_dwbn` | BN (gộp vào conv) | 9×12×256 | 1,024 | 0 | 0 |
| Backbone · khối thêm stride 8 | `x8a_dw` | DW 3x3/1 | 18×24×96 | 864 | 960 | 373,248 |
| Backbone · khối thêm stride 16 | `x16b_proj_conv` | Conv 1x1/1 | 9×12×64 | 16,384 | 16,448 | 1,769,472 |
| Backbone · khối thêm stride 8 | `x8a_dwbn` | BN (gộp vào conv) | 18×24×96 | 384 | 0 | 0 |
| Backbone · khối thêm stride 16 | `x16b_proj_bn` | BN (gộp vào conv) | 9×12×64 | 256 | 0 | 0 |
| Backbone · khối thêm stride 8 | `x8a_proj_conv` | Conv 1x1/1 | 18×24×32 | 3,072 | 3,104 | 1,327,104 |
| Neck · PAN-lite | `neck_l16_conv` | Conv 1x1/1 | 9×12×80 | 5,120 | 5,200 | 552,960 |
| Backbone · khối thêm stride 8 | `x8a_proj_bn` | BN (gộp vào conv) | 18×24×32 | 128 | 0 | 0 |
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
