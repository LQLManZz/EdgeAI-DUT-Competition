# TextCNN intent – trạng thái (cập nhật 2026-11-11)

Phần intent chiếm **30%** điểm vòng loại: `S = 0,70·(mAP50·100) + 0,30·(Accuracy·100)`.
Mọi số trong file này đều là **[TĐ]** (tự đo), chạy trên Colab T4 với TF 2.21 ngày 2026-11-11.

**Tóm tắt.** Model **`textcnn_dual`** đã được chọn để nộp: 62.189 tham số, file INT8 81 KB, đạt mọi kiểm tra luật.
- Accuracy khi chia CV theo khuôn câu: INT8 **100%**.
- Stress test: INT8 **92,7%**.
- Dự đoán test đã nằm trong `MyDrive/EdgeAI/intent_submission.csv`.
- Mỗi thí nghiệm detection có thêm file `submission_with_intent.csv` đã ghép sẵn.
- **Chưa nộp Kaggle.**

---

## 1. Nguồn gốc và vì sao viết lại

- Nhóm intent gửi `textcnn_colab.zip` (đặt ở gốc repo).
  - Zip **chỉ có code PyTorch**: `train_textcnn.py` và `stress_test.py`. **Không có trọng số** đã train.
  - 4 file dữ liệu trong zip giống từng byte với `data/raw/`.
- Code gốc **không đạt luật** ở mục V của thể lệ, dòng "Intent model":
  - **Luật yêu cầu:** TFLite INT8, ≤ 200.000 tham số, file ≤ 1,5 MB, chỉ dùng op mà TFLite Micro hỗ trợ.
  - **Code gốc:** dự đoán test bằng **ensemble 3 model PyTorch FP32**.
- Đã viết lại bằng Keras trong `src/textcnn/`. Giữ nguyên ba phần của code gốc:
  - tiền xử lý,
  - augmentation OUT_OF_SCOPE,
  - danh sách cụm từ của stress test.
- Thay đổi so với bản gốc:

| Bản gốc (PyTorch) | Bản mới (Keras) | Lý do |
|---|---|---|
| Ensemble 3 seed, FP32 | 1 model, dự đoán bằng chính file `.tflite` INT8 | Luật: TFLite INT8. BTC sẽ kiểm tra tái tạo kết quả |
| `Conv1d` + max theo thời gian | `Conv2D (k×1)` + `MaxPool2D` | Cùng phép toán, nhưng là op TFLM quen thuộc nhất |
| Che vị trí PAD khi max-pool | Bỏ phần che | Phần che cần op NOT_EQUAL/CAST, dễ vỡ khi INT8; model tự học vector cho `<pad>` |
| `nn.Embedding` | `GatherEmbedding` (gọi thẳng `tf.gather`) | Embedding của Keras 3 sinh thêm LESS/ADD/SELECT_V2 khi đổi sang TFLite |
| Batch tùy ý | Đổi TFLite từ bản sao model **batch = 1** | Reshape cố định, không sinh SHAPE/STRIDED_SLICE/PACK; ESP32 cũng xử lý từng câu |
| Bỏ ký tự ngoài `[a-z0-9 ]` | Bỏ dấu tiếng Việt trước (NFD, đ→d) | Dữ liệu Kaggle không dấu; chung kết có thể gõ có dấu |

## 2. Kiến trúc

```
câu hỏi -> norm() -> token -> id (độ dài cố định, đệm <pad>=0, từ lạ <unk>=1)
  nhánh word: ids(28)  -> Embedding 48 -> Conv2D k=1,2,3 × 64 kênh  -> MaxPool toàn câu
  nhánh char: ids(128) -> Embedding 24 -> Conv2D k=3,4,5 × 96 kênh  -> MaxPool toàn câu
  nối 480 số -> Dropout 0,5 -> Dense 5 = logits (COUNT, EXIST, LIST, LOCATION, OUT_OF_SCOPE)
```

- **Nhánh char** đọc từng ký tự, nên vẫn hiểu được từ chưa gặp lúc train. Ví dụ `kiem` gần với `kiem tra`.
- **Nhánh word** nắm cụm từ chắc hơn.
- Train như sau:
  - 25 epoch, batch 64, AdamW với lr 2e-3, warmup 30% rồi giảm theo cos (tương đương OneCycle của bản gốc).
  - label smoothing 0,05.
  - Token dropout 10%: thay ngẫu nhiên token bằng `<unk>`.
  - Thêm khoảng 50% câu OOS tổng hợp: lấy câu COUNT/EXIST/LOCATION, đổi tên vật trong lớp thành vật ngoài lớp.

## 3. Cách đánh giá (tập test không có nhãn)

| Thước đo | Cách làm | Ý nghĩa |
|---|---|---|
| **CV theo khuôn câu** | `GroupKFold(5)`. Khuôn câu = bỏ lời mở đầu lịch sự + thay tên vật bằng `<O>` | Khuôn câu ở fold kiểm tra chưa từng thấy lúc train. **Mọi config đều đạt khoảng 100%**, nên không phân biệt được |
| **Stress test** | Giấu hẳn 1 cụm từ (vd `vi tri`) khỏi train, kiểm tra trên các câu chứa nó. 22 cụm, 4.034 câu | Mô phỏng cách hỏi mới. **Đây là thước đo dùng để chọn model** |

- Cả hai thước đo đều được đo trên **FP32 (Keras)** và trên **INT8 (`.tflite`)**.
- Điểm thật là cột INT8.

## 4. Kết quả [TĐ] (Colab T4, 2026-11-11)

| Config | Nhánh | Tham số | File INT8 | CV FP32 | CV INT8 | Stress FP32 | Stress INT8 | Luật |
|---|---|---|---|---|---|---|---|---|
| **`textcnn_dual`** | word + char | **62.189** | **81 KB** | 1,0000 | 1,0000 | 0,9281 | **0,9274** | đạt |
| `textcnn_word` | word | 80.581 | 94 KB | 1,0000 | 1,0000 | 0,9197 | 0,9212 | đạt |
| `textcnn_char` | char | 77.605 | 94 KB | 0,9998 | 0,9998 | 0,8513 | 0,8508 | đạt |

- **INT8 gần như không mất gì:** chênh lệch FP32/INT8 ≤ 0,5 điểm, có cụm INT8 còn cao hơn. Trên 4000 câu test, FP32 và INT8 cho cùng kết quả 100% (dual).
- Ba model mạnh ở những chỗ khác nhau:

| Cụm bị giấu | word | char | dual |
|---|---|---|---|
| `vi tri` (227 câu) | **0,29** | 0,96 | 0,79 |
| `phia` (184) | 0,69 | 0,72 | 0,69 |
| `ngang` (159) | 1,00 | **0,63** | 1,00 |
| `dem` (679) | 0,88 | 0,77 | 0,87 |
| `tat ca` (282) | 1,00 | 1,00 | **0,69** |
| `kieu` / `nhom` (238/231) | 1,00 | **0,57 / 0,46** | 1,00 |
| `dung de` (24) | 1,00 | 1,00 | **0,63** |

(Số trong bảng là accuracy INT8.)

- **Kiểm tra luật** (`report.json` → `checks`), cả 3 config đều đạt:
  - ≤ 200K tham số;
  - file ≤ 1,5 MB;
  - INT8 hoàn toàn (không còn tensor float32);
  - chỉ 6 op: `GATHER, RESHAPE, CONV_2D, MAX_POOL_2D, CONCATENATION, FULLY_CONNECTED`.

### Dự đoán trên tập test (không có nhãn → KHÔNG phải accuracy)

Chỉ xem dự đoán để so sánh model. Không dùng test để train hay gán nhãn (thể lệ IV.5).

- **Dual dự đoán đúng 800 câu mỗi lớp.** Nếu test cân bằng như train (1200 câu/lớp) thì đây là dấu hiệu tốt. Nhưng test có cân bằng hay không **chưa được xác nhận**.
- **99 câu test chứa từ chưa từng có trong train:**
  - 53 câu chứa `lech`, dạng "Coc nam **lech** ve ben nao cua hinh anh?". Cả 3 model đều đoán LOCATION, hợp lý.
  - 46 câu chứa `kiem`, dạng "**Kiem** tra xem co hop trong anh khong". **word đoán sai 40 câu thành OUT_OF_SCOPE.** char và dual đoán EXIST, hợp lý. Riêng chỗ này word mất khoảng 1% accuracy.
- Ngoài 99 câu đó, 3 model chỉ bất đồng ở 7 câu. Dual chọn hợp lý ở các câu đó, ví dụ "binh hoa" (vật ngoài 7 lớp) → OUT_OF_SCOPE.

## 5. Chạy trên Colab

1. Trên máy, chạy lệnh dưới. Lệnh tạo `colab/textcnn_code.zip` (code + 4 file dữ liệu intent, khoảng 140 KB) và `colab/textcnn_colab.ipynb`. Upload cả hai vào `MyDrive/EdgeAI/`.
   ```
   .venv/Scripts/python scripts/pack_for_colab.py
   ```
2. Mở notebook, chạy Ô 1 → Ô 4, **không sửa ô nào**:
   - Ô 1: kết nối Drive.
   - Ô 2: giải nén + chạy `tests/test_textcnn.py`.
   - Ô 3: chạy hàng đợi `configs/textcnn_queue.json` + in bảng + chọn model + ghép submission.
   - Ô 4: đường cong, stress theo từng cụm, câu sai.
3. Mỗi config mất khoảng vài phút trên T4. Kết quả nằm ở `MyDrive/EdgeAI/experiments_intent/<tên>/`:
   - `cv.json`, `cv_errors.csv`, `stress.json`, `stress.csv`: đánh giá.
   - `model.keras`, `model_int8.tflite`, `vocab.json`, `log.csv`: model cuối (train trên 100% dữ liệu).
   - `intent_submission.csv`: 4000 dòng INTENT, dự đoán bằng file INT8.
   - `report.json`: tóm tắt + kiểm tra luật.
4. **Chọn model:** lấy `"use"` trong `textcnn_queue.json` nếu có. Nếu không, lấy model đạt mọi kiểm tra có stress INT8 cao nhất; hòa thì xét CV INT8, rồi đến ít tham số hơn.
5. **Ghép submission:** với mỗi `experiments/<detection>/submission.csv` đã có, runner ghi thêm `submission_with_intent.csv` bên cạnh. Dòng DETECTION giữ nguyên, dòng INTENT được thay. Đã kiểm tra: đủ 8000 dòng, thứ tự ID không đổi.

Thêm thí nghiệm mới: tạo `configs/textcnn_<tên>.json`, thêm tên vào hàng đợi, đóng gói lại, upload đè zip.

## 6. Hợp đồng tiền xử lý cho firmware (chung kết)

ESP32 phải làm **y hệt** lúc train, nếu không model nhận đầu vào khác và đoán sai. Mọi thông tin cần thiết nằm trong `vocab.json` cạnh `model_int8.tflite`:

1. **Chuẩn hóa:** chữ thường; `đ` → `d`; bỏ dấu (NFD, xóa ký tự Mn); ký tự ngoài `[a-z0-9 ]` → dấu cách; gộp khoảng trắng.
2. **Nhánh word:** tách theo dấu cách, tra `vocab.word`. Từ không có → 1 (`<unk>`). Cắt hoặc đệm 0 cho đủ **28**.
3. **Nhánh char:** từng ký tự (kể cả dấu cách) tra `vocab.char`. Cắt hoặc đệm cho đủ **128**.
4. **Input / output:**
   - Input là int32, shape `[1, 28]` và `[1, 128]`. Gán theo **tên tensor** (`word_ids`, `char_ids`), không theo thứ tự.
   - Output là int8 logits `[1, 5]`. argmax → `classes` trong `vocab.json`.
5. **Bộ nhớ (ước tính, chưa đo trên kit):** activation lớn nhất là đầu ra mỗi conv char, (126 × 96) int8 ≈ 12 KB. Arena dự kiến vài chục KB.

## 7. Việc còn mở

| Việc | Ưu tiên | Ghi chú |
|---|---|---|
| Nộp Kaggle `submission_with_intent.csv` của detection tốt nhất | **cao** | Hiện là #9 `h1p_pen_240x320_bw2` (mAP50 INT8 0,9257). Ước tính S ≈ 0,7 × 92,6 + 0,3 × ~100 ≈ 94,8 nếu accuracy intent khoảng 99–100% |
| Chạy `model_int8.tflite` trên kit (esp-tflite-micro 1.4.2) | trung bình | 6 op đều là op cơ bản của TFLM nhưng **chưa chạy thử trên kit**; file được convert bằng TF 2.21 |
| Viết tokenizer C cho firmware theo mục 6 | trước chung kết | Đối chiếu id với Python trên vài chục câu |
| Tăng độ bền với cách hỏi mới | thấp cho Kaggle, có ích cho chung kết | Các cụm dual còn yếu: `tat ca` 0,69, `dung de` 0,63, `phia` 0,69, `vi tri` 0,79. Ý tưởng: tăng token dropout 0,1 → 0,2–0,3; thêm dropout riêng cho nhánh word để buộc model dùng nhánh char |
| Hỏi BTC: test intent có cân bằng 5 lớp không | thấp | Nếu có, phân bố 800/lớp của dual là bằng chứng mạnh hơn |

## 8. File liên quan

- Code: `src/textcnn/text.py` (tiền xử lý, augmentation, cụm stress), `src/textcnn/model.py`, `src/textcnn/train.py` (pipeline).
- Runner Colab: `scripts/colab_intent.py`. Config: `configs/textcnn_{word,char,dual}.json`, `configs/textcnn_queue.json`.
- Test: `tests/test_textcnn.py` gồm 5 test: chuẩn hóa, encode, và mỗi config đạt luật khi ở dạng INT8.
- Notebook: `notebooks/textcnn_colab.ipynb`. Code gốc của nhóm intent: `textcnn_colab.zip`.
