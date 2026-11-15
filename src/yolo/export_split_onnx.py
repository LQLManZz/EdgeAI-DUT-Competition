"""Xuất mô hình Micro-YOLO sang ONNX với 2 đầu ra tách rời:
- 'boxes': [batch, 4, anchors]
- 'scores': [batch, 7, anchors]
Giúp bộ lượng tử hóa INT8 phân bổ dải tỷ lệ (scale) riêng biệt cho tọa độ và xác suất,
tránh hiện tượng sụp đổ mAP khi chuyển sang TFLite INT8 trên ESP32-S3.
"""
import argparse
from pathlib import Path
import types
import torch
from ultralytics import YOLO


def export_split_onnx(weights_path, output_onnx_path, imgsz=(240, 320)):
    print("=" * 70)
    print(" EXPORT MICRO-YOLO INT8-FRIENDLY (2 DECOUPLED HEADS)")
    print("=" * 70)
    print(f"[+] Trọng số đầu vào: {weights_path}")
    print(f"[+] Kích thước ảnh: {imgsz} (Height={imgsz[0]}, Width={imgsz[1]})")

    model = YOLO(weights_path)
    pytorch_model = model.model
    pytorch_model.eval()

    head = pytorch_model.model[-1]
    head.export = True

    def split_inference(self, x):
        dbox = self._get_decode_boxes(x)
        scores = x["scores"].sigmoid()
        return dbox, scores

    head._inference = types.MethodType(split_inference, head)

    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1], device="cpu")
    pytorch_model.to("cpu")

    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes=None
    )
    print(f"[OK] Đã xuất thành công: {output_onnx_path}")
    print(f"     - Kích thước: {output_onnx_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export Micro-YOLO to decoupled ONNX")
    parser.add_argument("--weights", type=str, required=True, help="File .pt cần export")
    parser.add_argument("--out", type=str, default="models/micro_yolo_split.onnx", help="File ONNX đầu ra")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[240, 320], help="Kích thước [H, W]")
    args = parser.parse_args()

    imgsz = tuple(args.imgsz) if len(args.imgsz) == 2 else (args.imgsz[0], args.imgsz[0])
    export_split_onnx(args.weights, args.out, imgsz=imgsz)


if __name__ == "__main__":
    main()
