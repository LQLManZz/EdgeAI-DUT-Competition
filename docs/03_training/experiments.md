# Bảng thí nghiệm GĐ3

**Quy tắc:**
- Mỗi dòng chỉ đổi **một** thứ so với dòng tham chiếu.
- Tập val cố định: 600 ảnh, `seed=0`, `val_frac=0.1`.
- mAP tính theo `src/edgedet/eval/map50.py` (kiểu "all"). Mọi số đều là **[TĐ]**.

| #   | Tên (thư mục experiments/)     | Thay đổi so với tham chiếu                                                                                    | Epoch | val mAP50 FP32              | val mAP50 INT8              | Kaggle   | Ghi chú                                                                                                                                                                                                                                                                                                                                           |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- | ----- | --------------------------- | --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | `local_h1c_144x192_e12`        | Baseline H1-C 144×192, config mặc định, **chỉ 12 epoch**, CPU local                                           | 12    | 0,2602                      | 0,2542                      | –        | Chạy để kiểm chứng pipeline. Loss vẫn đang giảm. Khoảng 72 s/epoch trên CPU.                                                                                                                                                                                                                                                                      |
| 1   | `h1c_144x192` (Drive)          | **Baseline chính thức**: H1-C 144×192, config mặc định, Colab T4                                              | 80    | **0,8154**                  | **0,8130**                  | chưa nộp | Khoảng 8–11 s/epoch (T4), tổng khoảng 14 phút. AP INT8: bottle 0,82 · cup 0,90 · phone 0,75 · book 0,89 · **mouse 0,71** · box 0,93 · **toy_car 0,69**. Ở ngưỡng 0,05 trung bình 41 box/ảnh. Convert bằng TF 2.21 (Colab).                                                                                                                        |
| 2   | `h1p_144x192` (Drive)          | So với #1 chỉ đổi backbone sang **MobileNetV3-Small minimalistic pretrained ImageNet** (H1-P, 189.379 params) | 80    | **0,8708** (+5,5)           | **0,8309** (+1,8)           | chưa nộp | FP32 cao hơn ở cả 7 lớp (mouse 0,71→0,80; toy_car 0,69→0,79). Nhưng **INT8 mất 4,0 điểm** (#1 chỉ mất 0,24). AP INT8: bottle 0,81 · cup 0,89 · phone 0,82 · book 0,88 · mouse 0,75 · box 0,90 · toy_car 0,76. File 330 KB, cùng 4 op. Khoảng 12 s/epoch. Phải khai báo trọng số pretrained với BTC.                                               |
| 3   | `h1p_relu6_144x192` (Drive)    | So với #2 chỉ đổi: chặn mọi ReLU của MobileNetV3 pretrained ở 6 (`mnv3_relu6: true`)                          | 80    | 0,6732 (−19,8 so với #2)    | không đo                    | –        | **Thất bại.** FP32 kém cả H1-C train từ đầu. Bỏ hướng này.                                                                                                                                                                                                                                                                                        |
| 4   | `h1p_192x256` (Drive)          | So với #2 chỉ đổi input lên **192×256**                                                                       | 80    | **0,9285** (+5,8 so với #2) | 0,8307 (bằng #2)            | chưa nộp | **INT8 mất 9,8 điểm.** Train khoảng 20 phút. AP INT8: bottle 0,91 · cup 0,90 · phone 0,84 · **book 0,73** · mouse 0,80 · **box 0,83** · toy_car 0,79. So với #2: vật nhỏ tăng (bottle +0,10, mouse +0,05), nhưng vật lớn giảm (book −0,15, box −0,07). Cùng 189.379 params, 330 KB, 4 op. 54,5 box/ảnh ở ngưỡng 0,05.                             |
| 5   | `h1p_pen_192x256` (Drive)      | So với #4 chỉ thêm **phạt activation** (threshold 16, weight 0,01, 28 tensor MobileNetV3)                     | 80    | 0,9210 (−0,75 so với #4)    | **0,9050** (+7,4 so với #4) | chưa nộp | **INT8 chỉ mất 1,6 điểm** (#4 mất 9,8). AP INT8: bottle 0,92 · cup 0,93 · phone 0,89 · book 0,90 · mouse 0,88 · box 0,97 · **toy_car 0,84**. Cùng 189.379 params, 330 KB, 4 op. Nhược điểm: phạt làm chậm giai đoạn đầu. mAP val dao động 0,14–0,31 từ epoch 6 đến 36, trong khi #4 đã đạt 0,86 ở epoch 35. Đường cong vẫn còn tăng ở epoch cuối. |
| 6   | `h1p_pen_240x320` (Drive)      | So với #5 chỉ đổi input lên **240×320** (độ phân giải gốc)                                                    | 80    | **0,9460** (+2,5)           | **0,9217** (+1,7)           | chưa nộp | **Tốt nhất hiện tại.** INT8 mất 2,4 điểm. AP INT8: bottle 0,94 · cup 0,96 · phone 0,91 · book 0,90 · mouse 0,88 · box 0,96 · toy_car 0,90. File 330 KB, cùng 4 op. Khoảng 35 phút train. Range lớn nhất là `re_lu` (stem) 179, max_abs_error 1,9. Phải đo thời gian chạy trên kit (ước tính khoảng 0,9 s/ảnh).                                    |
| 7   | `h1p_pen_192x256_e120` (Drive) | So với #5 chỉ đổi **120 epoch**                                                                               | 120   | 0,9398 (+1,9)               | 0,9184 (+1,3)               | chưa nộp | INT8 mất 2,1 điểm. Giai đoạn đầu còn dao động mạnh hơn #5: mAP rơi xuống 0,08 ở khoảng epoch 14, vì learning rate cao kéo dài hơn. Đường cong đi ngang ở 0,939–0,940 trong 10 epoch cuối.                                                                                                                                                         |
| 8   | `h1p_pen_240x320_e120` (Drive) | So với #6 chỉ đổi **120 epoch**                                                                               | 120   | **0,9542** (+0,8, cao nhất) | 0,9181 (−0,4)               | chưa nộp | **INT8 mất 3,6 điểm**, nên điểm thật thấp hơn #6. AP INT8: bottle 0,96 · cup 0,95 · phone 0,92 · **book 0,88** · mouse 0,89 · **box 0,93** · toy_car 0,89. Nhầm book → box tăng lên 26 lần (#6: 14). mAP val kẹt ở 0,22–0,33 từ epoch 7 đến khoảng 37. Range lớn nhất lại nhỏ hơn #6 (stem 127 so với 179).                              |
| 9   | `h1p_pen_240x320_bw2` (Drive)  | So với #6 chỉ đổi **`box_weight` 1 → 2**                                                                      | 80    | 0,9415 (−0,45)              | **0,9257** (+0,4)           | chưa nộp | **INT8 tốt nhất hiện tại**, chỉ mất 1,6 điểm. AP INT8: bottle 0,95 · cup 0,95 · phone 0,92 · book 0,91 · mouse 0,89 · box 0,96 · toy_car 0,90. "Box lệch" giảm từ +2,8 xuống +2,5; recall vật lớn 0,96 → 0,98. Chênh 0,4 điểm so với #6 có thể chỉ là nhiễu, vì chưa đo độ dao động giữa các lần train.                                       |
| 10  | `h1p_pen_240x320_bw2_r2` (Drive) | **Y hệt #9**, train lại để đo nhiễu                                                                       | 80    | 0,9360 (−0,55 so với #9)    | 0,9025 (**−2,3** so với #9) | chưa nộp | **INT8 mất 3,35 điểm, trong khi #9 cùng config chỉ mất 1,6.** AP INT8: bottle 0,92 · cup 0,94 · phone 0,89 · book 0,88 · mouse 0,87 · box 0,96 · toy_car 0,86. Range lớn nhất tương tự #9 (180 so với 192). Submission đã ghép intent thật.                                                                                                |
| 11  | `h1p_pen_240x320_bw3` (Drive)  | So với #9 chỉ đổi `box_weight` 2 → 3                                                                          | 80    | 0,9313                      | 0,9200                      | chưa nộp | INT8 chỉ mất 1,1 điểm (thấp nhất trong các bản H1-P). "Box lệch" +2,7, "nhầm lớp" +2,5. Không phân biệt được với #9/#10 vì nằm trong mức nhiễu. Submission đã ghép intent thật.                                                                                                                                                              |

## Nhận xét từ baseline #0

- **INT8 chỉ mất 0,6 điểm** (0,260 → 0,254), nên lượng tử hóa PTQ đang hoạt động tốt.
- AP theo lớp (INT8):
  - book 0,45; cup 0,34; box 0,32; phone 0,32;
  - **bottle 0,11; mouse 0,13; toy_car 0,11**.

  Ba lớp yếu là các vật nhỏ, nên đây là lý do thử input lớn hơn.
- Ở ngưỡng score 0,05, model trả khoảng 72 box mỗi ảnh. Nhiều box trùng vị trí nhưng khác lớp, tức model còn lẫn lớp. Hiện tượng này sẽ giảm khi train lâu hơn.
- mAP val bằng 0 trong khoảng 6 epoch đầu, dù loss đã giảm. Lý do: heatmap chưa vượt ngưỡng hoặc chưa đúng chỗ. Không cần lo nếu mAP bắt đầu tăng sau đó.

## Nhận xét #2 so với #1

- Pretrained giúp rõ: +5,5 điểm FP32, đều trên mọi lớp. Thể lệ cho dùng pretrained, miễn là khai báo.
- Hiện **điểm INT8 mới là điểm thật**, vì file nộp được dự đoán bằng model INT8. Theo điểm này, H1-P chỉ hơn H1-C 1,8 điểm, vì lượng tử hóa làm mất 4 điểm.
- **Giả thuyết, chưa kiểm chứng:** MobileNetV3 của Keras dùng `ReLU()` không chặn trên (`keras/src/applications/mobilenet_v3.py:543`), còn khối tự thiết kế dùng ReLU6. Activation không chặn có khoảng giá trị rộng, nên INT8 bị chia thô hơn.
- Hướng thử: đổi ReLU của backbone thành ReLU6 rồi fine-tune (khoảng 15 phút train), hoặc dùng QAT (train có mô phỏng INT8).

## Nhận xét #3

- Đã đo trước khi train: trên ảnh thật, activation của MobileNetV3 pretrained (chưa fine-tune) lên tới **157,8**, và 16/28 lớp ReLU không bị chặn.
- Chặn đột ngột ở 6 làm FP32 giảm từ 0,87 xuống 0,67.
- Giải thích có thể, chưa kiểm chứng: nhiều neuron bị kẹt ở 6. Đạo hàm của ReLU6 ở vùng đó bằng 0, nên gradient không kéo chúng về được, và đặc trưng pretrained bị phá.
- Bài học: không đổi hàm kích hoạt của mạng pretrained một cách đột ngột. Muốn giảm khoảng giá trị activation thì phải **co giãn tương đương** (chia BN cho s, nhân conv kế tiếp với s) hoặc dùng QAT.
- Bước tiếp theo: đo sai số INT8 **từng lớp** của #2 (`src/edgedet/quant/debug_int8.py`) để biết chính xác 4 điểm bị mất nằm ở đâu, rồi mới chọn cách sửa.

## Chẩn đoán INT8 từng lớp của #2 [TĐ]

Lệnh `debug_int8.py` chạy với 200 ảnh train để chọn scale và 32 ảnh val để đo sai số.

| | H1-C (12 epoch, làm mốc) | H1-P #2 |
|---|---|---|
| rmse/scale trung vị / lớn nhất | 0,25 / 0,42 | 0,27 / 0,47 |
| **range lớn nhất** (255 × scale) | khoảng 35 | **309** |
| max_abs_error lớn nhất | khoảng 0,2 | **2,3** |

- Phép làm tròn ở từng lớp vẫn đúng kỹ thuật, vì rmse/scale gần 0,29. Vấn đề là **range rộng**: mỗi mức INT8 thô hơn 3–9 lần, nên sai số tuyệt đối lớn gấp khoảng 10 lần.
- Các lớp range lớn nhất nằm ở đầu MobileNetV3 pretrained:

  | Lớp | Range |
  |---|---|
  | `expanded_conv_project_bn` | 309 |
  | `expanded_conv_2_project_bn` | 257 |
  | `re_lu` (sau stem) | 249 |
  | `re_lu_2` | 180 |
  | `expanded_conv_1_project_bn` | 168 |

- **`expanded_conv_2_add` (range 244) chính là f8**, đầu vào stride 8 của neck. Đặc trưng chi tiết dùng cho vật nhỏ bị lượng tử hóa thô nhất.
- Lưu ý khi đọc output của `debug_int8.py`: **rmse/scale không phát hiện được vấn đề range**, vì chỉ số này đã chia cho scale. Phải xem `range` và `max_abs_error`.
- Hướng sửa, chưa thử:

  | Cách | Nội dung | Đánh giá |
  |---|---|---|
  | (a) Phạt activation | Phạt phần activation vượt ngưỡng (ví dụ \|x\| > 20) ở các lớp trên khi fine-tune | Mềm, có gradient, khác với cắt cứng ở #3 |
  | (b) Co giãn tương đương theo kênh | Cross-layer equalization (Nagel 2019) | Không cần train, nhưng phức tạp với các nhánh residual |
  | (c) QAT | Train có mô phỏng INT8 | tfmot chưa hỗ trợ Keras 3, phải tự viết |
  | Loại bỏ | Activation int16 | Có thể vi phạm điều kiện "INT8" của thể lệ, và ESP-NN không tăng tốc int16 |

## Nhận xét #4

- Input lớn giúp FP32 rất rõ (0,87 → 0,93), đúng như dự đoán từ #0 là vật nhỏ cần nhiều pixel hơn.
- Nhưng INT8 mất gần 10 điểm, nên điểm thật không tăng. **Lượng tử hóa bây giờ là nút thắt duy nhất**: nếu giữ được mức mất của H1-C (khoảng 0,3 điểm) thì #4 sẽ đạt khoảng 0,92.
- Lớp mất nhiều nhất khi lên INT8 là vật lớn (book, box). Giả thuyết, chưa kiểm chứng: vật lớn dựa nhiều vào đặc trưng sâu (stride 16) và giá trị box lớn, nơi range rộng làm sai số tuyệt đối lớn. Kiểm tra bằng `debug_int8.py` cho #4 nếu cần.
- Quyết định: chạy phạt activation **trực tiếp ở 192×256** (`h1p_pen_192x256`), so với #4. Vẫn chỉ đổi một thứ, và đây là cấu hình có trần FP32 cao nhất. `h1p_pen_144x192` để dự phòng.

## Nhận xét #5

- Phạt activation giải quyết được vấn đề INT8: INT8 tăng từ 0,8307 lên 0,9050, mức mất giảm từ 9,8 còn 1,6 điểm. Điều này xác nhận chẩn đoán "range rộng" ở phần trên.
- Cái giá phải trả là FP32 giảm 0,75 điểm, và việc học bị chậm trong khoảng 35 epoch đầu. Lúc đầu activation của mạng pretrained rất lớn, nên phần phạt lấn át loss detection khi learning rate còn cao.
- So đường cong: #4 (không phạt) đi ngang quanh 0,925–0,928 trong 15 epoch cuối, tức là gần trần ở cấu hình này. #5 vẫn còn tăng (0,9183 → 0,9210).

## Phân tích lỗi (`error_analysis.py`, INT8, val 600 ảnh) [TĐ]

| | #5 192×256 | #6 240×320 | #7 192×256 e120 | #8 240×320 e120 | #9 240×320 bw2 |
|---|---|---|---|---|---|
| mAP50 INT8 | 0,9050 | 0,9217 | 0,9184 | 0,9181 | **0,9257** |
| + nới IoU xuống 0,1 (**box lệch**) | +3,8 | +2,8 | +3,3 | +2,7 | +2,5 |
| + bỏ qua lớp (**nhầm lớp**) | +2,8 | +2,7 | +2,4 | +2,7 | +2,3 |
| Vật không có box đúng lớp nào gần (**bỏ sót**) | 0,6% | 0,4% | 0,5% | 0,3% | 0,3% |
| Recall vật nhỏ (<28 px) / vừa / lớn, score ≥ 0,3 | 0,83 / 0,95 / 0,97 | 0,86 / 0,95 / 0,96 | 0,83 / 0,96 / 0,97 | 0,87 / 0,96 / 0,96 | 0,85 / 0,96 / 0,98 |
| Số lần nhầm book → box | – | 14 | – | 26 | 11 |

- Model hầu như không bỏ sót vật (dưới 1%). Phần còn thiếu chia đều cho **box chưa khít** (khoảng 3 điểm) và **nhầm lớp** (khoảng 2,5 điểm).
- **Vật nhỏ là điểm yếu rõ nhất**: recall khoảng 0,85, so với khoảng 0,95 ở vật vừa và lớn. Input 240×320 giúp vật nhỏ nhiều nhất (+0,03).
- Các cặp hay nhầm, ổn định qua cả 3 thí nghiệm:
  - **book → box** (13–14 lần, là cặp nhiều nhất). Hai lớp cùng là khối chữ nhật.
  - mouse → toy_car / phone.
  - phone → mouse / toy_car.
- #6 và #7 đều thắng #5, nên bước tiếp theo là ghép lại: 240×320 với 120 epoch.

## Nhận xét #8 và #9

- **Train lâu hơn làm INT8 mất nhiều hơn**, lặp lại ở cả hai độ phân giải:

  | | 80 epoch | 120 epoch |
  |---|---|---|
  | 192×256 | #5 mất 1,6 | #7 mất 2,1 |
  | 240×320 | #6 mất 2,4 | #8 mất 3,6 |

  FP32 của #8 cao nhất từ trước tới giờ (0,954), nhưng phần tăng thêm bị lượng tử hóa lấy hết.
- Nguyên nhân **chưa rõ**. `debug_int8.py` không cho thấy range lớn hơn (stem của #8 là 127, của #6 là 179), nên lần này không phải vấn đề "range rộng" như ở #4. Giả thuyết chưa kiểm chứng: train lâu làm model quyết định sát ranh giới giữa book và box hơn, nên sai số INT8 nhỏ cũng đủ lật lớp. Dấu hiệu ủng hộ: nhầm book → box tăng từ 14 lên 26 lần, và book/box là hai lớp tụt nhiều nhất.
- `box_weight` 2 (#9) có tác dụng đúng hướng: phần "box lệch" giảm từ +2,8 xuống +2,5, và recall vật lớn tăng 0,96 → 0,98. Đổi lại, recall vật nhỏ giảm nhẹ (0,86 → 0,85).
- **Cẩn thận với chênh lệch nhỏ.** Chưa có lần train lặp lại nào để biết mAP dao động bao nhiêu giữa hai lần train cùng config. Lưu ý: đổi `seed` trong config sẽ đổi cả cách chia val, nên không dùng `seed` để đo được.
- **Ở cả 3 lần, model tốt nhất là epoch cuối cùng.** Vậy khi train lại trên train+val (không còn val để chọn checkpoint), chỉ cần lấy epoch cuối.
- Giai đoạn đầu vẫn dao động mạnh: #8 kẹt ở 0,22–0,33 trong khoảng 30 epoch, #9 kẹt ở 0,18–0,30 trong khoảng 20 epoch.

## Nhận xét #10 và #11: đo được nhiễu, và nó lớn

| | #9 | #10 (y hệt #9) | Chênh |
|---|---|---|---|
| FP32 | 0,9415 | 0,9360 | 0,55 |
| INT8 | 0,9257 | 0,9025 | **2,3** |
| INT8 mất | 1,6 | 3,35 | 1,75 |

- **Mức INT8 bị mất phụ thuộc vào "may rủi" của từng lần train**: cùng một config mà lúc mất 1,6, lúc mất 3,35. FP32 thì ổn định hơn nhiều (lệch khoảng 0,5).
- Calibration không phải nguồn dao động giữa hai lần train: cả hai dùng đúng 200 ảnh như nhau (`rng(0)` trong `export_int8.py`). Vậy khác biệt nằm ở trọng số.
- **Hệ quả: phải rút lại vài kết luận trước.**
  - #6, #8, #9, #10, #11 (INT8 0,9025–0,9257) **không phân biệt được với nhau**. "box_weight 2 tốt hơn" chưa được chứng minh.
  - Kết luận "train lâu làm INT8 mất nhiều hơn" (Nhận xét #8) **không còn đứng vững**: #8 mất 3,6, chỉ hơi cao hơn mức 3,35 mà #10 mất do may rủi.
  - FP32 của #8 (0,954) cao hơn các bản 80 epoch 0,8–1,4 điểm, lớn hơn nhiễu FP32 (khoảng 0,5), nên 120 epoch **có lẽ thật sự giúp FP32**.
- **Đòn bẩy lớn nhất bây giờ là làm INT8 mất ít và ổn định.** Nếu #8 chỉ mất 1,1 như #11, nó sẽ đạt khoảng 0,94.
- **Đã phát hiện và sửa lỗi Decode trên INT8 (`decode_center`, Giai đoạn 1):**
  - Trên FP32, `p >= nb` (max-pool 3×3) luôn chọn đúng 1 đỉnh vì số thực không bằng nhau. Nhưng trên INT8 (`hm` chỉ có 256 mức), 2 ô cạnh nhau ở đỉnh rất hay lượng tử hóa ra **cùng một số nguyên INT8**, khiến cả 2 đều thỏa `p >= nb` và sinh ra 2 box trùng nhau (trong đó ô lệch tâm đọc phải `box` chưa từng được giám sát lúc train!).
  - Đã bổ sung vào `src/edgedet/utils/decode.py`: (1) `tie_break=True` chọn đúng 1 ô có trung bình 3×3 lớn nhất khi bằng điểm đỉnh; (2) `score_smooth=0.1` phá vỡ đồng hạng 1/256 khi xếp hạng AP toàn cục; (3) `nms_iou=0.65` lọc trùng cùng lớp cho vật lớn.
- Bước tiếp: `int8_probe.py` chạy cho cả 5 bản 240×320 (không train lại), đo trực tiếp mức tăng mAP50 INT8 của `decode_center` mới (mục B) và tự động tạo lại `submission.csv` cho cả 5 bản.

## Phần intent (TextCNN) – 30% điểm

Bảng và phân tích riêng: [textcnn_intent.md](textcnn_intent.md). Tóm tắt [TĐ] ngày 2026-11-11:
- `textcnn_dual`: 62.189 tham số, 81 KB INT8, CV INT8 1,0000, stress INT8 0,9274. **Đã chọn để nộp.**
- Dự đoán nằm ở `MyDrive/EdgeAI/intent_submission.csv`. Mỗi thí nghiệm ở bảng trên có thêm `submission_with_intent.csv` (8000 dòng) trong thư mục Drive của nó.
- Cột "Kaggle" ở bảng trên vẫn là "chưa nộp". Khi nộp, dùng `submission_with_intent.csv`, rồi ghi điểm vào cột đó.

## Ghi chú môi trường Colab (đo 2026-11-10)

- Runtime: T4, TF 2.21, Keras 3.13, 8 nhân CPU.
- **XLA trên GPU không có `ResizeNearestNeighborGrad`**, nên phải train với `jit_compile=False` (đã sửa trong `train.py`).
- File `.tflite` của #1 được convert bằng TF 2.21, không phải 2.18.1 như bản đã đo trên kit. Trước khi chốt model cuối, cần chạy file này qua `firmware/bench_probe` để chắc op version vẫn được esp-tflite-micro 1.4.2 nhận.

## Hàng đợi thí nghiệm (theo thứ tự ưu tiên)

1. ~~`h1c_144x192`: baseline đủ 80 epoch~~. Đã xong ở dòng #1, là tham chiếu cho mọi thí nghiệm sau.
2. ~~`h1p_144x192`: backbone pretrained ImageNet~~. Đã xong ở dòng #2. Còn mở: cách giảm mức INT8 bị mất (ReLU6 hoặc QAT).
3. ~~Độ phân giải 192×256~~. Đã xong ở dòng #4: FP32 0,9285 nhưng INT8 chỉ 0,8307. 240×320 hoãn lại đến khi sửa xong INT8.
3b. ~~#5 `h1p_pen_192x256`~~. Đã xong: INT8 0,9050, tốt nhất hiện tại.
3c. ~~#6 `h1p_pen_240x320` (INT8 0,9217) và #7 `h1p_pen_192x256_e120` (INT8 0,9184)~~. Cả hai đã xong và đều thắng #5.
3d. ~~#8 `h1p_pen_240x320_e120` (INT8 0,9181) và #9 `h1p_pen_240x320_bw2` (INT8 0,9257)~~. Cả hai đã xong; #9 là tốt nhất hiện tại. Không train dài hơn 80 epoch nữa, vì INT8 mất nhiều hơn.
3f. ~~#10 `h1p_pen_240x320_bw2_r2` (INT8 0,9025) và #11 `h1p_pen_240x320_bw3` (INT8 0,9200)~~. Đã xong: nhiễu INT8 giữa hai lần train khoảng 2 điểm.
3g. **Đang xếp hàng** (`configs/queue.json`): chẩn đoán `probe` cho 5 bản 240×320 đã train (#9, #10, #6, #8, #11). Không train lại; các bước đã có kết quả tự bỏ qua. Submission của #9, #6, #8 sẽ được làm lại vì file intent mới hơn.
3e. Ý tưởng sau đó, theo kết quả phân tích lỗi:
   - Tăng dần trọng số phạt (0 → 0,01 trong khoảng 10 epoch) để bớt dao động lúc đầu. Cẩn thận: #8 cho thấy học "hiệu quả hơn" có thể làm INT8 mất nhiều hơn.
   - EMA trọng số.
   - Train lại trên toàn bộ train+val trước khi chốt.
4. Augmentation: tắt bớt hoặc tăng mạnh hơn; thêm làm mờ; ghép 4 ảnh (mosaic).
5. Lịch learning rate và số epoch; `box_weight`.

## Tham chiếu ngoài: model YOLO "mAP87" của nhóm khác (`ANOTHER_WORK/`) [TĐ]

Đo ngày 2026-11-10 bằng `scripts/analysis/eval_other_yolo.py`. Model là kiểu YOLOv8 tự định nghĩa trong Ultralytics: 161K params, input 320×320 letterbox. Phép đo dùng **tập val của họ** (`ANOTHER_WORK/val.txt`, 1000 ảnh, seed 42). Không dùng val của mình vì tập train của họ chứa phần lớn ảnh val của mình. Calibration INT8 dùng 200 ảnh trong `train.txt` của họ. Các file sinh ra nằm trong `ANOTHER_WORK/int8/`.

| Biến thể | mAP50 "all", ngưỡng 0,05, max 100 box | mAP50 "all", ngưỡng 0,001, IoU 0,7 |
|---|---|---|
| FP32 ONNX (= TFLite float32) | **0,8694** | 0,8668 |
| INT8, export nguyên bản (1 output gộp `[1,11,2000]`) | **0,3675** (−50,2) | 0,3092 |
| INT8, tách output box / lớp trước Concat cuối | **0,8083** (−6,1) | 0,7924 |

- Export nguyên bản hỏng vì box (0–318 px) và xác suất lớp (0–1) bị gộp vào **một** tensor INT8 có scale 1,247. Kết quả là điểm lớp chỉ còn 2 mức (0 hoặc 1,25), nên mất hoàn toàn khả năng xếp hạng.
- Backend mặc định `flatbuffer_direct` của onnx2tf sinh ra file INT8 **không chạy được** (LOGISTIC có scale output khác 1/256). Phải dùng `-tb tf_converter`.
- Sau khi tách output vẫn mất 6,1 điểm, nhiều hơn H1-C (−0,24) nhưng ít hơn H1-P 192×256 (−9,8). Lớp mất nhiều nhất là mouse (−0,10) và box (−0,07).
- Bản INT8 tách output có các op: CONV_2D, MUL, LOGISTIC (SiLU), TRANSPOSE, STRIDED_SLICE, CONCATENATION, ADD, RESHAPE, PAD, MAX_POOL_2D, SUB, QUANTIZE, RESIZE_NEAREST_NEIGHBOR, SOFTMAX. Phần lớn nằm **ngoài** tập 4 op đã kiểm chứng trên kit. File nặng 249 KB.
- Input 320×320 vượt giới hạn mỗi chiều ≤ 320×240.
- Kết luận: ở điều kiện thật (INT8), model này đạt khoảng 0,81. Con số này ngang H1-C #1 (0,813) và thấp hơn H1-P #2/#4 (0,831), dù tập val khác nhau nên chỉ so được tương đối.

### Bản cập nhật "run4" (đo lại 2026-11-10 tối)

Họ fine-tune thêm 30 epoch từ run3 (lr0 0,0008, cls 1,2, batch 32), rồi export `micro_yolo_split_240x320.onnx`: input 240×320, đã tách 2 output `boxes [1,4,1500]` và `scores [1,7,1500]`. Trọng số head trong ONNX khớp từng bit với `best_model_mAP87.pt`. Checkpoint ghi mAP50 0,8918 (Ultralytics val, imgsz 320). Thư mục `ANOTHER_WORK_int8/` chứa file sinh ra đã bị xóa; cách tái tạo xem mục v3 bên dưới.

| Biến thể | mAP50 "all", ngưỡng 0,05, max 100 box | mAP50 "all", ngưỡng 0,001, IoU 0,7, max 300 |
|---|---|---|
| FP32 ONNX 240×320 | **0,8754** (COCO-101: 0,8715) | 0,8777 (COCO-101: 0,8738) |
| INT8 (onnxsim + onnx2tf `-tb tf_converter`, 200 ảnh calib) | **0,8375** (−3,8) | 0,8239 |

- Con số 0,892 của họ **không tái hiện được**: đo độc lập chỉ được 0,874–0,878. Box không bị lệch hệ thống (lệch tâm trung bình khoảng 0,1 px, box rộng hơn GT khoảng 3 px).
- ONNX họ xuất ra **không convert thẳng được**. Phần tạo anchor (Shape/Gather/Expand) vẫn nằm trong graph, khiến onnx2tf lỗi ở `/model.15/Expand`. Chạy `onnxsim` trước thì convert được.
- INT8 tốt hơn bản trước: 0,808 lên 0,8375, mức mất giảm từ −6,1 xuống −3,8. Lý do là bỏ được phần đệm letterbox và đã tách output.
- Với INT8, đặt ngưỡng 0,001 lại cho mAP **thấp hơn** ngưỡng 0,05. Bước lượng tử của score là 1/256, nên giữ lại rất nhiều box rác có điểm bằng nhau.
- Danh sách op y hệt bản trước (thêm 1 TRANSPOSE): LOGISTIC/MUL từ SiLU, TRANSPOSE ×22, STRIDED_SLICE ×14, SOFTMAX. File nặng 248 KB.
- So tương đối (khác tập val): INT8 0,8375 của họ thấp hơn #5 h1p_pen_192x256 (INT8 0,9050) của mình.

### Bản v3 "Micro-YOLO Deep" / `best_model_mAP92.pt` (đo 2026-11-10 khuya)

Kiến trúc mới là `micro_yolo_deep.yaml`: cùng độ rộng (×0,125) nhưng C2f lặp 3 lần ở P3/P4 và 2 lần ở neck. Tổng **190K params** và **123M MAC** ở 240×320. Model train từ đầu 120 epoch: AdamW lr0 0,002, cos_lr, cls 1,5, mixup 0,15, copy_paste 0,1, close_mosaic 15, **rect=True**. Checkpoint ghi mAP50 0,9253. Trọng số head trong ONNX khớp từng bit với `.pt`.

| Biến thể | ngưỡng 0,05, IoU 0,5, max 100 | ngưỡng 0,001, IoU 0,7, max 300 |
|---|---|---|
| FP32 ONNX 240×320 | **0,9046** (COCO-101: 0,9015) | 0,9134 (COCO-101: 0,9093) |
| INT8 TFLite (onnxsim, onnx2tf `-tb tf_converter`, 200 ảnh calib) | **0,8969** (−0,8) | 0,8997 (−1,4) |
| `micro_yolo_split_int8.onnx` của họ (ORT QDQ) | không load được | — |

- So với run4: FP32 tăng khoảng +3,0 điểm, INT8 tăng **+5,9 điểm** (0,8375 lên 0,8969). Mức mất do INT8 giảm còn khoảng 1 điểm.
- Số 0,925 họ báo vẫn cao hơn số mình đo (0,909–0,913, cùng cách tính); chênh khoảng 1–1,5 điểm, giống lần trước.
- Bản `micro_yolo_split_int8.onnx` của họ lỗi `Unrecognized attribute: axis for operator DequantizeLinear`, do dùng `per_channel=True` trên graph opset 12. Định dạng ORT QDQ cũng không dùng được cho TFLM.
- `quantize_tflite.py` dùng sai cờ onnx2tf (`-in`, `-ot`; đúng là `-i`, `-o`) và không chạy onnxsim. Representative dataset của nó cho dữ liệu NCHW, trong khi SavedModel do onnx2tf sinh ra là NHWC.
- File INT8 nặng 299 KB, gồm CONV_2D 60, LOGISTIC 56, MUL 57, TRANSPOSE 28, STRIDED_SLICE 14, SOFTMAX 1. Chưa đo latency trên kit.
- So tương đối (khác tập val): INT8 0,897 của họ đã **ngang** #5 h1p_pen_192x256 của mình (INT8 0,905).
- Tái hiện: `eval_other_yolo.py --model X.onnx --dump_calib 200`, rồi `onnxsim`, rồi `onnx2tf -i sim.onnx -o tfs -tb tf_converter -oiqt -cind images calib.npy "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"`, cuối cùng `eval_other_yolo.py --model tfs/*_full_integer_quant.tflite`.
