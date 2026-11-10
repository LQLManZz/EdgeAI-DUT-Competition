"""Đóng gói code (+ dataset) thành file zip để upload lên Google Drive cho Colab.

  .venv/Scripts/python scripts/pack_for_colab.py          # -> colab/edgeai_code.zip (vài chục KB)
  .venv/Scripts/python scripts/pack_for_colab.py --data   # thêm colab/edgeai_data.zip (dataset, chỉ cần làm 1 lần)

Upload thư mục colab/ vào Google Drive: MyDrive/EdgeAI/  rồi mở notebooks/edgedet_colab.ipynb trên Colab.
Cùng lệnh này tạo colab/textcnn_code.zip + colab/textcnn_colab.ipynb (TextCNN intent, hàng đợi configs/textcnn_queue.json).
Thí nghiệm nào được chạy: configs/queue.json (xem scripts/colab_run.py) -> các ô Colab không phải sửa.
Mỗi lần Claude sửa code: chạy lại lệnh đầu, upload đè edgeai_code.zip.
Vì sao zip: Colab đọc hàng nghìn file nhỏ trực tiếp từ Drive rất chậm; giải nén 1 file zip vào ổ /content thì nhanh.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"
CODE = ["src", "configs", "tests", "scripts/sanity_overfit.py", "scripts/colab_run.py"]


def add(z, path, skip=None):
    for p in ([path] if path.is_file() else sorted(path.rglob("*"))):
        if p.is_file() and "__pycache__" not in p.parts and not (skip and skip in p.name + "/".join(p.parts)):
            z.write(p, p.relative_to(ROOT).as_posix())


# TextCNN intent: code + 4 file dữ liệu intent (~1 MB) trong 1 zip riêng -> notebooks/textcnn_colab.ipynb
INTENT = ["src/textcnn", "tests/test_textcnn.py", "scripts/colab_intent.py",
          *[f"data/raw/{f}" for f in ("intent_train.csv", "intent_test.csv", "sample_submission.csv", "classes.txt")]]

OUT.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT / "edgeai_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for c in CODE:
        add(z, ROOT / c, skip="textcnn")      # TextCNN nằm trong textcnn_code.zip riêng
print("->", OUT / "edgeai_code.zip", f"{(OUT / 'edgeai_code.zip').stat().st_size / 1024:.0f} KB")

with zipfile.ZipFile(OUT / "textcnn_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for c in INTENT + [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "configs").glob("textcnn_*.json"))]:
        add(z, ROOT / c)
(OUT / "textcnn_colab.ipynb").write_bytes((ROOT / "notebooks" / "textcnn_colab.ipynb").read_bytes())
print("->", OUT / "textcnn_code.zip", f"{(OUT / 'textcnn_code.zip').stat().st_size / 1024:.0f} KB  (+ textcnn_colab.ipynb)")

if "--data" in sys.argv:
    with zipfile.ZipFile(OUT / "edgeai_data.zip", "w", zipfile.ZIP_STORED) as z:   # PNG đã nén sẵn, không nén thêm
        add(z, ROOT / "data" / "raw")
    print("->", OUT / "edgeai_data.zip", f"{(OUT / 'edgeai_data.zip').stat().st_size / 2**20:.0f} MB")
