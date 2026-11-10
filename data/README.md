# Dữ liệu

| Thư mục | Nội dung |
|---|---|
| `raw/` | Dataset gốc từ Kaggle (ảnh 320×240, nhãn, `classes.txt`) – giữ nguyên, không sửa |
| `processed/` | Dữ liệu sau tiền xử lý / chia train-val |
| `external/` | Dữ liệu ngoài tự thu thập hoặc tải về |

## Khai báo dữ liệu ngoài (bắt buộc theo thể lệ)

| Tên | Nguồn (link) | Giấy phép | Số ảnh | Dùng để làm gì |
|---|---|---|---|---|
| | | | | |

## Lưu ý
- **Không dùng tập test** để gán nhãn tay, pseudo-label hay huấn luyện (thể lệ mục IV.5).
- Thứ tự `class_id` phải khớp đúng `classes.txt` của BTC.
