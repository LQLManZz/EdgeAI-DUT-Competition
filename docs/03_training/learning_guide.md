# Learning guide – học Object Detection qua chính code của dự án

Đọc trước file [00_tong_quan_cho_nguoi_moi.md](00_tong_quan_cho_nguoi_moi.md).

Mỗi bài trong file này gồm 4 phần:
- **khái niệm**;
- **code tương ứng** trong repo;
- **việc cần làm**;
- **câu hỏi tự kiểm tra**.

Bạn tự trả lời các câu hỏi trước khi mở đáp án.

Các bài được viết dần theo tiến độ. Bài 1–5 đã có code chạy được. Bài 6 trở đi viết sau khi có dataset.

---

## Lộ trình

| Ngày | Mục tiêu | Bài học | Mốc kiểm tra |
|---|---|---|---|
| 09–10/11 | Hiểu bài toán, có dữ liệu | 1, 2, 3, 4, 5 | Test + sanity pass; dataset lên Drive, xem được ảnh có vẽ box |
| 11/11 | Train baseline H1-C 144×192 trên Colab | 6 (dữ liệu), 7 (train) | Có số mAP50 trên tập val |
| 12/11 | INT8 + nộp Kaggle lần 1 | 8 (lượng tử hóa), 9 (submission) | Có điểm trên leaderboard |
| 13–15/11 | Thí nghiệm cải tiến: pretrained, độ phân giải, augmentation, lịch learning rate, ngưỡng score | 10 (thí nghiệm có kiểm soát, phân tích lỗi) | Bảng thí nghiệm |
| 16/11 | Train lại cấu hình tốt nhất trên train+val | – | Model cuối |
| 17/11 | Kiểm tra tuân thủ (params, size, op), nháp D5/D6 | – | Checklist |
| 18/11 | Dự phòng, chọn bài nộp chính thức | – | Hạn chót |

**Dataset (đã có trong `data/raw/`):**
- `train/images/00001.png …`: 6000 ảnh PNG 320×240 RGB.
- `train/labels/00001.txt …`: mỗi dòng một vật, theo **định dạng YOLO** `lớp cx cy w h`.
  - Tâm (cx, cy) và kích thước (w, h) được chia cho kích thước ảnh, nên luôn nằm trong 0–1.
  - Ví dụ: dòng `1 0.559 0.296 0.094 0.117` là một cup có tâm tại (179, 71) px, kích thước 30×28 px.
- `test/images/img_0001.png …`: 4000 ảnh không có nhãn. Model đoán trên tập này để nộp.
- `sample_submission.csv`: 4000 dòng DETECTION + 4000 dòng INTENT. Dòng INTENT do nhóm TextCNN điền.

**Bước của bạn (chạy trên Colab):**
1. Đóng gói code và dữ liệu bằng lệnh `.venv/Scripts/python scripts/pack_for_colab.py --data`. Lệnh tạo ra `colab/edgeai_code.zip` (khoảng 30 KB) và `colab/edgeai_data.zip` (khoảng 840 MB).
2. Upload 2 file zip vào Google Drive, đặt trong thư mục `MyDrive/EdgeAI/`.
3. Mở `notebooks/edgedet_colab.ipynb` trên Colab (File → Upload notebook). Bật GPU T4 rồi chọn **Runtime → Run all**.
   - Các ô **không bao giờ phải sửa**. Thí nghiệm nào được chạy do `configs/queue.json` trong zip quyết định; `scripts/colab_run.py` chạy lần lượt train → INT8 → submission → chẩn đoán INT8.
   - Có code mới thì chỉ cần đóng gói lại (bỏ `--data`), upload đè `edgeai_code.zip`, rồi Run all.
   - Bị ngắt giữa chừng thì Run all lại: bước đã xong tự bỏ qua, phần train chạy tiếp từ epoch đang dở.
4. Trong lúc chờ upload, đọc bài 1–5 bên dưới.

Bạn có thể chạy thử trên máy (CPU là đủ):

```bash
.venv/Scripts/python -m pytest tests -q
```

```bash
.venv/Scripts/python scripts/sanity_overfit.py
```

---

## Bài 1 – IoU: đo hai box trùng nhau bao nhiêu

**Khái niệm.**
`IoU = diện tích giao / diện tích hợp`, giá trị trong khoảng 0–1. Kaggle coi box đoán là "trúng" khi IoU ≥ 0,5 và đúng lớp.

**Code.** `src/edgedet/eval/map50.py::iou`. Hàm chỉ có 6 dòng. Phần giao là hình chữ nhật có:
- cạnh trái = max của hai cạnh trái;
- cạnh phải = min của hai cạnh phải.

Nếu cạnh phải ≤ cạnh trái thì hai box không giao nhau.

**Việc cần làm.** Đọc `tests/test_core.py::test_iou_by_hand`. Tự tính trên giấy rồi so với test.

**Tự kiểm tra.**
1. Box thật `[0,0,100,100]`, box đoán `[0,0,100,50]` (đúng vị trí nhưng chỉ bằng nửa chiều cao). IoU bằng bao nhiêu? Box này có được tính là trúng không?
2. Vì sao chung kết (chấm vị trí tâm trái/giữa/phải) dễ tính hơn mAP50?

<details><summary>Đáp án</summary>

1. Giao = 5000, hợp = 10000, nên IoU = 0,5. Vừa đúng ngưỡng: được tính là trúng nếu quy tắc là ≥ 0,5. Chỉ cần box đoán nhỏ hơn thêm chút nữa là trượt.
2. Chung kết chỉ cần tâm box rơi đúng 1 trong 3 vùng. Kích thước box sai nhiều vẫn có thể đúng.
</details>

---

## Bài 2 – mAP50: điểm số Kaggle

**Khái niệm.**

Với mỗi lớp:
1. Gom mọi box đoán của lớp đó trên **toàn bộ** tập ảnh.
2. Xếp theo score từ cao xuống thấp.
3. Đi lần lượt từng box. Mỗi box là TP (khớp được một box thật chưa bị nhận, IoU ≥ 0,5) hoặc FP.
4. Sau mỗi bước, tính precision = TP/(số box đã xét) và recall = TP/(tổng số box thật).
5. **AP** = diện tích dưới đường precision–recall.

**mAP** = trung bình AP của các lớp.

**Code.** `map50.py::average_precision`, các bước 1–4 được đánh số trong code.

**Việc cần làm.** Đọc `test_ap_by_hand`. Ví dụ trong đó có 3 box đoán là TP, FP, TP và AP = 0,833. Tự vẽ đường precision–recall trên giấy.

**Tự kiểm tra.**
1. Thêm 1 box FP có score 0,01 (thấp nhất) thì AP thay đổi thế nào? Nếu box FP đó có score 0,99 thì sao?
2. Vì sao khi nộp Kaggle ta để ngưỡng score thấp (0,05) mà không đặt 0,5?

<details><summary>Đáp án</summary>

1. Với score 0,01, box nằm cuối danh sách, sau khi recall đã đạt tối đa, nên AP **không đổi**. Với score 0,99, box nằm đầu danh sách, precision ở mọi mức recall đều giảm, nên AP giảm mạnh. Bài học: **score phải phản ánh đúng độ tự tin**.
2. Ngưỡng thấp giữ lại các box đúng nhưng model ít tự tin, giúp tăng recall. Box sai có score thấp gần như không làm hại AP, như câu 1 vừa thấy.
</details>

---

## Bài 3 – Mã hóa nhãn (encode): biến box thành "đáp án dạng lưới"

**Khái niệm.**
Mạng nơ-ron xuất ra tensor có kích thước cố định. Nó không xuất ra được một danh sách dài ngắn tùy ý.

Vì vậy ta chia ảnh 144×192 thành lưới 18×24 ô, mỗi ô 8×8 px. Đáp án gồm:
- **heatmap** `(18, 24, 7)`: tại ô chứa tâm một vật lớp k, kênh k = 1. Xung quanh ô đó, giá trị giảm dần theo hình "đồi" Gauss. Các ô xa = 0.
- **box** `(18, 24, 4)`: tại ô chứa tâm, ghi `log(w/8), log(h/8)` và độ lệch của tâm thật so với góc ô (`off_x, off_y` ∈ [0, 1)).
  - Dùng **log** vì vật to và vật nhỏ chênh nhau nhiều lần. Log biến "gấp đôi" thành "cộng thêm 0,69", nên model học dễ hơn.
- **mask**: đánh dấu ô nào có tâm vật. Chỉ những ô này mới bị chấm loss box.

**Code.** `src/edgedet/data/targets.py::encode`. Chiều ngược lại là `utils/decode.py::decode_center`.

**Việc cần làm.** Đọc `test_encode_then_decode_recovers_box`: mã hóa 2 box rồi giải mã lại phải ra đúng 2 box đó (IoU > 0,99).

**Tự kiểm tra.**
1. Vì sao heatmap là "đồi" chứ không chỉ một điểm 1 duy nhất?
2. Hai vật cùng lớp có tâm rơi vào **cùng một ô** thì chuyện gì xảy ra? Điều này nguy hiểm hơn khi input nhỏ hay lớn?

<details><summary>Đáp án</summary>

1. Model đoán lệch 1 ô so với tâm thật thì vẫn "gần đúng", box dựng ra vẫn có IoU cao. Đồi Gauss giảm mức phạt ở các ô lân cận (xem bài 4), nên model học ổn định hơn.
2. Chỉ giữ được một vật, vật kia bị mất (giảm recall, đếm sai). Input càng nhỏ thì ô càng to so với vật, nên càng dễ xảy ra. Đây là một lý do để thử input 192×256 hoặc 240×320.
</details>

---

## Bài 4 – Hàm loss: focal loss và L1

**Khái niệm.**
Loss đo mức độ sai. Gradient descent giảm loss bằng cách chỉnh trọng số.

- **Heatmap: focal loss.** Lưới có khoảng 3000 ô nhưng mỗi vật chỉ có 1 ô dương. Nếu dùng cross-entropy thường, các ô nền dễ đoán áp đảo loss và model học cách "đoán 0 ở mọi nơi". Focal loss xử lý bằng hai cơ chế:
  - Nhân thêm `(1-p)²` hoặc `p²` để **giảm trọng số các ô đã đoán đúng dễ dàng**.
  - Nhân thêm `(1-y)⁴` để **phạt nhẹ** các ô nền nằm gần tâm.
- **Box: L1** = |đáp án − dự đoán|, chỉ tính tại ô có mask = 1.

**Code.** `src/edgedet/losses/center_loss.py`.

**Tự kiểm tra.**
1. Một ô nền có `y = 0`, model đoán `p = 0,01`. Ô này đóng góp nhiều hay ít vào loss?
2. Vì sao chia tổng loss cho số vật `n_pos`?

<details><summary>Đáp án</summary>

1. Rất ít: `p² = 0,0001`, nhân với `−log(0,99) ≈ 0,01`. Ô đã đoán đúng thì gần như không ảnh hưởng.
2. Để loss không phụ thuộc vào việc batch có nhiều hay ít vật. Nhờ vậy learning rate dùng được ổn định cho mọi batch.
</details>

---

## Bài 5 – Kiểm tra "học thuộc" (overfit test) và bẫy BatchNorm

**Khái niệm.**
Trước khi train hàng giờ trên dữ liệu thật, cho model **học thuộc** vài ảnh. Nếu ngay cả việc đó cũng không làm được (mAP trên chính các ảnh đã học thấp), thì chắc chắn có lỗi trong pipeline: encode, loss, decode hoặc tọa độ. Kiểm tra này chỉ mất vài phút và tiết kiệm được hàng giờ.

**Code.** `scripts/sanity_overfit.py`. Script tạo 8 ảnh giả, mỗi lớp một màu, rồi train 400 bước. Kết quả đã đo: loss giảm từ khoảng 10 xuống 0,007, **mAP50 = 1,000**.

**Bẫy đã gặp thật (BatchNorm).** Lần chạy đầu cho mAP = **0,00** dù loss rất thấp. Nguyên nhân:
- BatchNorm chuẩn hóa dữ liệu bằng mean/var.
  - Lúc **train**, nó dùng mean/var của batch hiện tại.
  - Lúc **dự đoán**, nó dùng **trung bình trượt** tích lũy dần.
- Với momentum 0,99, trung bình trượt cần vài trăm bước mới theo kịp. 400 bước là quá ít, nên lúc dự đoán thống kê bị lệch và output sai hẳn.
- Đo được:
  - chế độ train: mAP 1,0;
  - chế độ dự đoán: mAP 0,0;
  - sau khi cập nhật lại thống kê BN: chế độ dự đoán lên 1,0.

Train thật có hàng nghìn bước nên gần như không gặp lỗi này. Nhưng hãy nhớ: **luôn đánh giá ở chế độ dự đoán** (`model.predict`), vì trên chip model chạy ở chế độ đó.

**Tự kiểm tra.**
1. mAP trên tập train = 1,0 có nghĩa là model tốt không?
2. Nếu ở bài toán thật mAP val thấp hơn hẳn mAP train thì đó là hiện tượng gì?

<details><summary>Đáp án</summary>

1. Không. Nó chỉ chứng minh pipeline **có khả năng** học. Model tốt là model đúng trên ảnh **chưa thấy** (tập validation).
2. **Overfitting**: model học thuộc chứ không tổng quát hóa. Cách chữa: thêm augmentation, thêm dữ liệu, dùng pretrained, giảm số epoch.
</details>

---

## Bài 6 – Dữ liệu: đọc nhãn, chia train/val, augmentation

**Khái niệm.**

- **Train / val / test.**
  - Model học trên **train**.
  - Ta chấm điểm trên **val**, gồm 10% ảnh có nhãn được tách riêng và model không bao giờ học trên đó.
  - Kaggle chấm trên **test**, ta không có nhãn của tập này.

  Val là "đề thi thử". Nó phải cố định (seed = 0) thì các thí nghiệm mới so sánh được với nhau. Thể lệ **cấm** dùng tập test để train.

- **Augmentation.** Mỗi epoch, ảnh train được biến đổi ngẫu nhiên: zoom, dịch, lật, đổi màu, thêm nhiễu. Model không thể học thuộc từng pixel, nên phải học "hình dạng của cái cốc". Đây là cách chống overfitting rẻ nhất.
  - **Box phải biến đổi theo ảnh.** Ví dụ khi lật ngang, `x1' = W − x2`.
  - Vật bị cắt mất hơn 60% thì bỏ khỏi nhãn.

- **Thống kê dataset** (đo bằng `stats()`):
  - 6000 ảnh, 38.774 vật, trung bình 6,5 vật mỗi ảnh, nhiều nhất 12.
  - 7 lớp gần cân bằng (5100–5800 vật mỗi lớp). Không có ảnh rỗng.
  - **Vật nhỏ:** chiều rộng trung vị 36 px, 5% vật nhỏ hơn 17 px (trên ảnh 320×240). Ở input 144×192 (tỉ lệ 0,6), vật 17 px chỉ còn khoảng 10 px, tức khoảng 1 ô lưới.

**Code.**
- `src/edgedet/data/dataset.py`:
  - `read_yolo_label` đổi định dạng YOLO `cx cy w h` (0–1) sang `[x1,y1,x2,y2]` theo pixel.
  - `_augment` thực hiện augmentation.
  - `make_dataset` tạo `tf.data`.
- `preprocess()` là phép tiền xử lý **duy nhất** cho val, test và INT8 (resize bilinear rồi `x/127.5 − 1`). Firmware ESP32 sau này phải làm y hệt.

**Việc cần làm.** Chạy mục 3 trong notebook. Nhìn hình "Sau augmentation" và kiểm tra box ôm đúng vật.

**Tự kiểm tra.**
1. Vì sao không được chọn cấu hình tốt nhất dựa trên điểm Kaggle public thay vì dùng val?
2. Dữ liệu vòng loại là ảnh render (synthesis), còn chung kết là ảnh chụp bằng camera OV7670. Augmentation nào giúp model chịu được khác biệt này?

<details><summary>Đáp án</summary>

1. Mỗi ngày chỉ được nộp 4 lần, quá ít để so sánh. Ngoài ra, chọn theo điểm public dễ "overfit vào public leaderboard", trong khi xếp hạng cuối dùng điểm private.
2. Augmentation về màu (sáng, tương phản, bão hòa, sắc độ), nhiễu, và sau này là làm mờ. Camera rẻ cho ảnh nhiễu, lệch màu và mờ.
</details>

---

## Bài 7 – Train: epoch, batch, learning rate, đọc đường cong

**Khái niệm.**
- **Batch** = 64 ảnh được xử lý cùng lúc. Sau mỗi batch có **một** bước gradient descent.
- **Epoch** = một lượt đi hết tập train. Với 5400 ảnh và batch 64, mỗi epoch có 84 bước.
- **Learning rate (lr)** = độ dài mỗi bước gradient descent.
  - Lr quá lớn thì loss nhảy lung tung hoặc thành NaN.
  - Lr quá nhỏ thì học rất chậm.
  - Ta dùng **warmup** (3 epoch đầu tăng dần lr), rồi **cosine** (giảm dần về gần 0).
- **AdamW**: biến thể của gradient descent, tự điều chỉnh bước đi cho từng trọng số. Phần "W" (weight decay) kéo trọng số về gần 0 một chút để chống overfitting.
- **Đọc đường cong:**
  - Loss train giảm và mAP val tăng: đang học tốt.
  - Loss train vẫn giảm nhưng mAP val đứng yên hoặc giảm: **overfitting**.
  - Cả hai cùng đứng yên ở mức kém: model quá nhỏ hoặc lr sai.

**Code.** `src/edgedet/train/train.py`. Config nằm trong `configs/*.json`, các giá trị mặc định ở `DEFAULTS`. Callback `ValMap` đo mAP val mỗi 2 epoch và lưu `best.keras`.

**Việc cần làm.** Chạy mục 4–6 của notebook. Ghi lại:
- mAP val tốt nhất và epoch đạt được;
- thời gian mỗi epoch;
- hình dạng đường cong.

**Tự kiểm tra.**
1. Vì sao lưu `best.keras` (mAP val cao nhất) mà không chỉ lưu model ở epoch cuối?
2. Tăng batch từ 64 lên 128 thì mỗi epoch có bao nhiêu bước? Có nên đổi lr không?

<details><summary>Đáp án</summary>

1. Nếu model bắt đầu overfit ở các epoch cuối, model cuối cùng sẽ kém hơn model ở giữa quá trình.
2. Khoảng 42 bước. Mỗi bước có gradient ít nhiễu hơn, nên thường có thể tăng lr (quy tắc kinh nghiệm: tăng lr theo tỉ lệ với batch). Nhớ chỉ đổi **một** thứ mỗi lần thí nghiệm.
</details>

---

## Bài 8 – Lượng tử hóa INT8

**Khái niệm.**
- Khi train, mọi số là float32. Chip cần int8 (−128…127).
- Mỗi tensor được gán một `scale` và `zero_point`, theo công thức `số thực ≈ (q − zero_point) × scale`.
- Để chọn `scale`, converter cho khoảng 200 **ảnh thật** chạy qua model và đo khoảng giá trị của từng tensor (gọi là representative dataset).
- Hệ quả:
  - Mất một chút độ chính xác do làm tròn, nên ta **luôn đo mAP val của chính file INT8**.
  - Heatmap và box được tách thành 2 output để mỗi output có scale riêng. Nếu gộp chung, kênh heatmap sẽ bị làm tròn thô hơn nhiều.

**Code.** `src/edgedet/quant/export_int8.py`. Script tự kiểm tra:
- ≤ 200K params;
- file ≤ 1,5 MB;
- chỉ dùng 4 op đã chạy thật trên kit.

Kết quả ghi vào `int8_report.json`, gồm `val_map50_fp32`, `val_map50_int8` và `int8_drop`. `evaluate.py::TFLiteDetector` chạy file `.tflite` trên PC theo đúng cách chip sẽ chạy.

**Tự kiểm tra.**
1. Nếu `int8_drop` là 0,10 (mất 10 điểm mAP), bạn nghi ngờ gì trước tiên?
2. Vì sao chọn ảnh representative từ tập train mà không lấy từ tập val?

<details><summary>Đáp án</summary>

1. Nghi ngờ đầu tiên: representative dataset không giống dữ liệu thật (sai tiền xử lý, ví dụ quên chuyển về [−1, 1]). Sau đó mới đến một tensor có khoảng giá trị quá rộng (outlier). Nếu vẫn chưa sửa được thì phải dùng QAT (train có mô phỏng lượng tử hóa).
2. Để tập val hoàn toàn "sạch": không có quyết định nào, kể cả cách chọn scale, dựa trên dữ liệu dùng để chấm điểm.
</details>

---

## Bài 9 – File submission

**Khái niệm.**
- `submission.csv` có 8000 dòng: 4000 dòng `DETECTION` của ta và 4000 dòng `INTENT` của nhóm TextCNN.
- Mỗi dòng DETECTION là một chuỗi JSON chứa danh sách box, **tọa độ trên ảnh gốc 320×240**, tối đa 100 box.
- Ta dự đoán bằng **chính file INT8**, vì đó là model BTC sẽ kiểm tra tái tạo kết quả.

**Code.** `src/edgedet/eval/make_submission.py`:
- Script giữ đúng thứ tự `sample_id` của `sample_submission.csv`.
- Dòng INTENT lấy từ file `--intent` của nhóm TextCNN. Nếu không có file này, dòng INTENT lấy từ file mẫu (hợp lệ nhưng gần như chắc chắn sai).
- Hàm `validate()` kiểm tra định dạng trước khi ghi file.

**Việc cần làm.**
- Hỏi nhóm TextCNN file của họ có những cột nào. Cần `task_type,sample_id,prediction`.
- Sau đó ghép file, kiểm tra script in ra `HỢP LỆ`, rồi nộp.

---

## Bài 10 – Thí nghiệm có kiểm soát (sẽ viết khi có baseline)

Quy tắc: **mỗi lần chỉ đổi một thứ**. Ghi kết quả vào bảng [experiments.md](experiments.md) với các cột:
- tên;
- thay đổi so với baseline;
- mAP val FP32;
- mAP val INT8;
- điểm Kaggle (nếu đã nộp).

Bảng đã có dòng #0: baseline 12 epoch, mAP val 0,26.
