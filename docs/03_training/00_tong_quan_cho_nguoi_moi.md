# Tổng quan cho người mới: dự án đã làm gì, đã quyết định gì

Đọc file này trước. Ở đây giả định bạn đã biết: neuron, mạng nhiều lớp, gradient descent.
Những khái niệm mới được giải thích ngay khi xuất hiện. Phần chi tiết kèm code nằm trong [learning_guide.md](learning_guide.md).

---

## 1. Bài toán: Object Detection

**Phân loại ảnh** (classification) trả lời câu hỏi: "ảnh này là gì?". Kết quả là **một** nhãn.

**Phát hiện vật thể** (object detection) trả lời câu hỏi: "trong ảnh **có những vật gì**, **ở đâu**?". Với **mỗi** vật, model phải trả về 3 thứ:

| Thành phần | Ví dụ | Ý nghĩa |
|---|---|---|
| `class_id` | `1` (cup) | Vật thuộc lớp nào trong 7 lớp |
| `bbox` | `[40, 60, 120, 180]` | Hình chữ nhật bao quanh vật, dạng `[x_min, y_min, x_max, y_max]`, đơn vị pixel trên ảnh 320×240 |
| `score` | `0.87` | Độ tự tin của model, nằm trong khoảng 0–1 |

Một ảnh có thể có 0, 1 hoặc nhiều vật. Vì vậy output của detection là một **danh sách** box có độ dài thay đổi.

### Cách Kaggle chấm: mAP50

1. **IoU** (Intersection over Union) đo mức độ trùng nhau của 2 box:
   `IoU = diện tích phần giao / diện tích phần hợp`.
   Hai box trùng khít có IoU = 1. Hai box không chạm nhau có IoU = 0.
2. Một box đoán được tính là **đúng** (True Positive, TP) khi thỏa cả 3 điều kiện:
   - đúng lớp;
   - IoU với một box thật **≥ 0,5**;
   - box thật đó chưa bị box khác "nhận" trước.

   Box đoán không thỏa thì là **sai** (False Positive, FP). Box thật không được box nào nhận là **bỏ sót** (False Negative, FN).
3. **Precision** = TP / (số box đã đoán). Câu hỏi nó trả lời: "trong số box đã đoán, bao nhiêu box đúng?"
4. **Recall** = TP / (số box thật). Câu hỏi nó trả lời: "trong số vật có thật, model tìm được bao nhiêu?"
5. Sắp xếp box đoán theo score từ cao xuống thấp. Đi dần xuống danh sách, ta được một đường cong precision–recall. Diện tích dưới đường cong này là **AP** (Average Precision) của một lớp.
6. **mAP50** là trung bình AP của 7 lớp. Chữ "50" nghĩa là ngưỡng IoU = 0,5.

Hệ quả thực tế:
- Box không cần khít tuyệt đối, chỉ cần IoU ≥ 0,5.
- Nộp thêm box có score thấp **thường không làm hại** mAP. Lý do: các box này nằm cuối danh sách xếp hạng, nên chúng chỉ có thể thêm recall. Vì vậy khi nộp Kaggle, ta để ngưỡng score thấp (khoảng 0,05).

Điểm vòng loại được tính như sau:
`70 × mAP50 + 30 × Accuracy(TextCNN)`.
Phần của chúng ta là 70 điểm mAP50. Phần TextCNN ban đầu do nhóm khác làm. Ngày 2026-11-11, code của họ được viết lại cho đạt luật (TFLite INT8) và train trên Colab. Model `textcnn_dual` đạt 100% CV và 92,7% stress test (INT8), đã có file dự đoán và đã ghép vào submission. Chi tiết: [textcnn_intent.md](textcnn_intent.md).

---

## 2. Vì sao không lấy luôn YOLO?

YOLOv8n là model detection nhỏ phổ biến nhất, nhưng nó có khoảng **3,2 triệu tham số** (params).

Cuộc thi chỉ cho phép **≤ 200.000** params. Ngoài ra model phải chạy được trên vi điều khiển ESP32-S3: RAM nhanh chỉ 512 KB, CPU 240 MHz, không có GPU.

YOLOv8n lớn gấp 16 lần giới hạn. Cắt nhỏ nó đi thì không còn là YOLOv8n nữa. Vì vậy ta phải **tự thiết kế** một model nhỏ.

**Giai đoạn 1 (khảo sát)** làm 3 việc:
- Đọc tài liệu về các họ model nhỏ: MobileNet, ShuffleNet, GhostNet, NanoDet, CenterNet, YOLO-Fastest…
- Thu nhỏ từng họ về khoảng 200K params.
- **Chạy thật trên kit ESP32-S3** để đo tốc độ.

Kết quả đo cho ra mấy quy tắc thiết kế (con số chi tiết ở [comparison_table](../01_survey/comparison_table.md)):

| Quy tắc | Lý do (đã đo) |
|---|---|
| Dùng khối **inverted residual** của MobileNet (conv 1×1 → depthwise 3×3 → conv 1×1) | Nhanh nhất trên chip, khoảng 95–117 triệu phép nhân/giây. GhostNet chậm gấp 2–3 lần. |
| Kích hoạt **ReLU6**, không dùng SE / h-swish | Các phép toán đặc biệt này chậm hoặc không được tăng tốc trên chip. |
| Số kênh là **bội của 8 hoặc 16** | Thư viện tăng tốc ESP-NN chỉ chạy nhánh nhanh khi số kênh chia hết cho 8. |
| Chỉ dùng **4 phép toán**: CONV_2D, DEPTHWISE_CONV_2D, ADD, RESIZE_NEAREST_NEIGHBOR | Chắc chắn TFLite Micro hỗ trợ. Ví dụ op `TILE` làm firmware báo lỗi, ta đã gặp thật. |
| Dùng **Keras/TensorFlow** thay cho PyTorch | Chuyển thẳng sang TFLite INT8, không phải qua bước trung gian dễ lỗi. |

---

## 3. Model của ta: EdgeDet (Giai đoạn 2)

Code nằm ở `src/edgedet/models/edgedet.py`. Model có 3 phần nối tiếp nhau:

```
Ảnh 144×192×3
   │
   ▼
[Backbone]  chồng các lớp conv, mỗi lần thu nhỏ ảnh một nửa
   │          144×192 → 72×96 → 36×48 → 18×24 (f8) → 9×12 (f16)
   │          Lớp càng sâu thì "hiểu" càng trừu tượng (cạnh → hình → bộ phận → vật)
   ▼
[Neck]      phóng to f16 lên 18×24 rồi cộng với f8
   │          → vừa có chi tiết (vị trí chính xác), vừa có ngữ nghĩa (vật là gì)
   ▼
[Head]      tại mỗi ô của lưới 18×24 (mỗi ô ứng với 8×8 pixel ảnh input), model đoán:
              • heatmap: 7 số, số thứ k = "tâm của một vật lớp k nằm ở ô này?"
              • box: 4 số = độ rộng, độ cao, độ lệch tâm x, y bên trong ô
```

### Ý tưởng CenterNet: tìm "tâm" vật

Mỗi vật được biểu diễn bằng **điểm tâm** của nó. Khi train, ở ô chứa tâm vật lớp k, ta muốn heatmap kênh k ≈ 1. Ở các ô xa tâm, ta muốn heatmap ≈ 0. Các ô sát tâm được phép có giá trị trung gian theo dạng "đồi" Gauss.

Khi dự đoán (**decode**):
1. Tìm các ô là **đỉnh** của heatmap, tức lớn hơn 8 ô xung quanh, và vượt ngưỡng.
2. Với mỗi đỉnh, đọc 4 số box để dựng hình chữ nhật.
3. Đổi tọa độ về ảnh gốc 320×240.

Vì sao chọn CenterNet:
- Đơn giản: không cần "anchor" (box mẫu) hay NMS (lọc box trùng) như YOLO.
- Chung kết chấm việc **đếm số vật** và **vị trí tâm** (trái/giữa/phải). CenterNet đoán tâm trực tiếp.
- Đã đo: head YOLO và head CenterNet có tốc độ như nhau. Vì vậy chọn head nào chỉ phụ thuộc vào mAP.

### Đã đo trên kit thật

| Biến thể | Mô tả | Params | Thời gian 1 ảnh @144×192 |
|---|---|---|---|
| H1-C | Backbone tự thiết kế, train từ đầu | 185.307 | 0,32 s |
| H1-P | Backbone MobileNetV3-Small, khởi tạo từ trọng số đã học trên ImageNet (pretrained) | 189.379 | 0,33 s |

Giới hạn là 10 s. Ngay cả input lớn nhất 240×320 cũng chỉ mất 0,9 s.
→ **Phần cứng không phải nút thắt. Việc của ta bây giờ là làm model đoán đúng (mAP cao).**

---

## 4. Đã quyết định / chưa quyết định

| Đã chốt | Chưa chốt – sẽ chọn bằng mAP trên tập validation |
|---|---|
| Keras/TensorFlow → TFLite INT8 | H1-C (train từ đầu) hay H1-P (pretrained) |
| Kiến trúc EdgeDet khoảng 185–190K params | Kích thước input: 144×192, 192×256 hay 240×320 |
| Head CenterNet, 2 output (heatmap, box) | Augmentation nào có ích |
| Decode bằng tìm đỉnh heatmap (`src/edgedet/utils/decode.py`) | Có làm knowledge distillation không (để cuối, nếu còn thời gian) |

**Pretrained** nghĩa là backbone đã được train trước trên 1,2 triệu ảnh ImageNet. Nó đã biết nhận ra cạnh, góc, kết cấu, nên thường học nhanh hơn và tốt hơn khi dataset nhỏ. Nếu dùng, phải khai báo với BTC.

**INT8**: lúc train, trọng số là số thực 32 bit (FP32). Chip cần số nguyên 8 bit, nhanh hơn và nhẹ hơn 4 lần. Việc chuyển FP32 → INT8 gọi là **lượng tử hóa** (quantization). Nó làm mất một chút độ chính xác, vì vậy ta luôn đo mAP cả trước và sau khi chuyển.

---

## 5. Việc tiếp theo

Xem lộ trình theo ngày trong [learning_guide.md](learning_guide.md#lộ-trình). Bước đầu tiên là **xem dataset**: không biết nhãn được lưu thế nào thì không viết được code đọc dữ liệu.
