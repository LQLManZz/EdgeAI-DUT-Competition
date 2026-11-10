# Ràng buộc cuộc thi – tóm tắt từ `references/`

Nguồn:
- **[TL]** `references/1793621605372.pdf` – Thể lệ Edge AI Challenge 2026 lần 2 (Đà Nẵng, 02/11/2026).
- **[SL]** `references/1794141594265.pdf` – Slide Training 1 (Canva, 41 trang).

Khi hai nguồn mâu thuẫn, ghi cả hai và đưa vào mục "Câu hỏi cho BTC".

## 1. Lịch

| Mốc | [TL] | [SL] |
|---|---|---|
| Vòng loại (Kaggle) | 09/11 – 22/11/2026 | 09/11 – 18/11/2026 |
| Công bố vào chung kết + phát kit | 23/11/2026 | 20/11/2026 |
| Chung kết | 05/12/2026 | 29/11/2026 |

→ **Mâu thuẫn.** Tạm lấy mốc sớm hơn (18/11) làm deadline vòng loại cho đến khi BTC xác nhận.

## 2. Bộ nhãn detection (7 lớp, [TL] III.1)

| class_id | Tên | Ghi chú BTC |
|---|---|---|
| 0 | bottle | Nhiều kiểu chai, màu sắc |
| 1 | cup | Nhiều mẫu cốc |
| 2 | phone | Nằm ngang hoặc dựng đứng |
| 3 | book | Nhiều màu bìa, kích thước |
| 4 | mouse | Có dây hoặc không dây |
| 5 | box | Gần hộp chữ nhật |
| 6 | toy_car | Nhiều mẫu, màu, hướng đặt |

(Bảng trong PDF bị lệch cột khi trích text; thứ tự trên theo cột Class ID / Tên class. Phải đối chiếu lại với `classes.txt` khi có dataset.)

## 3. Dữ liệu ([TL] III.2)
- Ảnh nguồn **320 × 240**. Vòng loại: ảnh **tổng hợp (synthesis)**. Chung kết: vật thật do BTC chuẩn bị, chụp bằng OV7670.
- Có: một vật, nhiều vật cùng lớp, nhiều lớp, nhiều vị trí, che khuất một phần, ảnh không có vật.
- Được dùng mọi nguồn dữ liệu ngoài, **phải khai báo** (cả dữ liệu lẫn pretrained).
- Cấm dùng tập test để gán nhãn, pseudo-label, huấn luyện.

## 4. Vòng loại – Kaggle ([TL] IV)
- Một file `submission.csv`: `task_type,sample_id,prediction`.
- DETECTION: JSON `[{"class_id":..,"score":..,"bbox":[x_min,y_min,x_max,y_max]}]`, **tọa độ trên ảnh gốc** (320×240), score ∈ [0,1], tối đa **100 box/ảnh**, ảnh rỗng → `[]`.
- Metric detection: **mAP@0.5**. Điểm: **S = 0,70·(mAP50·100) + 0,30·(Accuracy·100)**.
- Tối đa **4 lần nộp / 24 giờ**; phải chọn bài nộp chính thức; xếp hạng theo private.
- Trước khi vào chung kết, BTC **kiểm tra**: tái tạo kết quả, nguồn dữ liệu, số tham số, kích thước tệp, lượng tử hóa, toán tử, khả năng triển khai.

## 5. Ràng buộc kỹ thuật ([TL] V, [SL] trang 4)

| Hạng mục | Giới hạn |
|---|---|
| Phần cứng | ESP32-S3 **N16R8** + camera **OV7670** |
| Detection | 1 model, 1 lần suy luận/ảnh, không ensemble, không TTA |
| Input detection | Mỗi chiều ≤ 320 × 240; khuyến nghị 96×96, 128×128, 160×160 |
| Model detection | **TensorFlow Lite INT8**, ≤ **200.000 tham số**, file ≤ **1,5 MB** |
| Model intent | **TextCNN, TFLite INT8**, ≤ **200.000 tham số**, file ≤ **1,5 MB**, chạy bằng TFLite Micro trên ESP32-S3 |
| Xử lý câu hỏi | Được dùng từ điển + logic để chuẩn hóa câu, lấy lớp vật, định dạng đáp án; **không được thay hoàn toàn TextCNN bằng luật từ khóa** |
| Toán tử | Chỉ op mà **bản TFLite Micro trong starter kit** hỗ trợ |
| Phản hồi | ≤ **10 s** mỗi lượt hỏi, tính từ khi nhận yêu cầu tới khi trả kết quả |
| Suy luận | Toàn bộ trên ESP32-S3, không offload |

## 6. Phần cứng ([SL] trang 16–19, 36)

| Thành phần | Thông số | Hệ quả |
|---|---|---|
| CPU | 2 nhân Xtensa LX7, ≤ 240 MHz | |
| SRAM nội | 512 KB | Dùng chung cho heap, stack, Wi-Fi, HTTP |
| PSRAM | 8 MB | Chứa framebuffer lớn, tensor arena; chậm hơn SRAM |
| Flash | 16 MB | Firmware + model INT8 |
| Framebuffer 320×240 | Gray 75 KiB · RGB565 150 KiB · RGB888 225 KiB | Tính cùng lúc với input tensor + arena |
| Repo mẫu | Tensor arena mẫu **200 KiB trong PSRAM**; ESP-IDF **5.2.6** | Camera repo mặc định Flash 4 MB → phải chỉnh 16 MB |

Camera xuất **RGB565**; tiền xử lý trên thiết bị (resize, chuẩn hóa, quantize đúng scale/zero-point) phải **khớp chính xác** với lúc training.

## 7. Chung kết ([TL] VI)
Luồng: `POST /ask` → chuẩn hóa câu hỏi → TextCNN → từ điển/logic lấy lớp + vùng → chụp OV7670 → detection → logic → `{"answer": ...}`.

| Intent | Trọng số | Cách chấm | Detector cần gì |
|---|---|---|---|
| COUNT + EXIST | 35% | Đúng/sai | Đếm đúng: ít FP, ít FN, NMS tốt với vật gần nhau |
| LIST | 25% | Đúng sau chuẩn hóa lớp + số lượng | Phân loại đúng + đếm đúng |
| LOCATION | 30% | Trái/giữa/phải theo **tâm bbox** | Tâm box đúng; độ khít box ít quan trọng |
| OUT_OF_SCOPE | 10% | Từ chối đúng | (TextCNN) |

20 lượt hỏi; sai định dạng hoặc > 10 s → 0 điểm lượt đó. Cảnh chung kết **chưa xuất hiện nguyên trạng** trong tập train → cần tổng quát hóa (synthesis → ảnh thật OV7670).

## 8. Nhận xét cho thiết kế (cần kiểm chứng ở các giai đoạn sau)
1. Vòng loại chấm **mAP50**; chung kết chấm **đếm + tâm box**. Cả hai đều không đòi bbox quá khít → mAP50-95 là chỉ số phụ.
2. **Domain gap**: train trên ảnh synthesis, chung kết ảnh thật OV7670 (nhiễu, màu, độ sáng). Nên thu thêm ảnh thật bằng OV7670 sau khi nhận kit.
3. Giới hạn 10 s là cho **cả lượt hỏi** (chụp + tiền xử lý + detection + TextCNN + logic + HTTP), không chỉ inference.
4. Params (≤200K), file size (≤1,5 MB) và RAM (arena) là **ba giới hạn khác nhau**. 200K params INT8 ≈ 200 KB weights, nên giới hạn 1,5 MB gần như không phải nút thắt; nút thắt thực tế là **params** và **RAM activation**.

## 9. Câu hỏi cần hỏi BTC
1. Lịch nào đúng (thể lệ hay slide)?
2. Phiên bản TFLite Micro / esp-tflite-micro trong starter kit, và danh sách op được đăng ký trong `MicroMutableOpResolver`? (Đội đã đo bằng esp-tflite-micro 1.4.2 – nếu BTC dùng bản khác cần đo lại.) Starter kit dùng D-cache 32 hay 64 KB?
3. Decode output + NMS viết bằng C ngoài model có được phép không? (Slide trang 40 có khối "Decode / NMS" → có vẻ được.)
4. Cách đếm 200K tham số: tính trước hay sau khi gộp BatchNorm? Có tính tham số non-trainable không?
5. Input không vuông (vd. 160×120) hay ảnh grayscale có được chấp nhận không?
6. mAP50 tính kiểu nào (COCO 101 điểm hay VOC), trung bình theo lớp?
7. 10 s đo từ lúc nào đến lúc nào, có tính thời gian chụp ảnh không?
8. Teacher model dùng cho knowledge distillation có phải khai báo như pretrained không?
