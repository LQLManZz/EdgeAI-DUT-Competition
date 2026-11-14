"""Model Soup: Kỹ thuật bình quân trọng số các checkpoint để đột phá mAP mà không tăng chi phí suy luận (Zero-inference cost).
Phù hợp 100% với ràng buộc cuộc thi: "Chỉ suy luận trên 1 mô hình độc lập, không ensemble".

Hỗ trợ:
- Keras/TensorFlow models (.keras) cho EdgeDet
- PyTorch models (.pt) cho Micro-YOLO

Cách dùng:
  python scripts/model_soup.py --models exp1/best.keras exp2/best.keras exp3/best.keras --out soup_best.keras
  python scripts/model_soup.py --models fold0.pt fold1.pt fold2.pt --out soup_yolo.pt
"""
import argparse
import os
from pathlib import Path
import numpy as np


def soup_keras(model_paths, out_path):
    import tensorflow as tf
    print(f"[*] Đang thực hiện Model Soup trên {len(model_paths)} mô hình Keras...")
    models = [tf.keras.models.load_model(p, compile=False) for p in model_paths]
    
    # Lấy danh sách trọng số của từng model
    all_weights = [m.get_weights() for m in models]
    
    # Tính trung bình trọng số từng layer
    soup_weights = []
    n = len(models)
    for layer_idx in range(len(all_weights[0])):
        w_sum = sum(all_weights[m_idx][layer_idx] for m_idx in range(n))
        soup_weights.append(w_sum / float(n))
        
    base_model = models[0]
    base_model.set_weights(soup_weights)
    base_model.save(out_path)
    print(f"[OK] Đã lưu mô hình Keras Model Soup thành công: {out_path}")


def soup_pytorch(model_paths, out_path):
    import torch
    print(f"[*] Đang thực hiện Model Soup trên {len(model_paths)} mô hình PyTorch...")
    state_dicts = []
    ckpt_metas = []
    
    for p in model_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dicts.append(ckpt["model"].float().state_dict())
            ckpt_metas.append(ckpt)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dicts.append(ckpt["state_dict"])
            ckpt_metas.append(ckpt)
        else:
            state_dicts.append(ckpt)
            ckpt_metas.append(None)
            
    # Tính trung bình các tham số float
    n = len(state_dicts)
    soup_dict = {}
    keys = list(state_dicts[0].keys())
    
    for k in keys:
        if state_dicts[0][k].is_floating_point():
            soup_dict[k] = sum(sd[k] for sd in state_dicts) / float(n)
        else:
            soup_dict[k] = state_dicts[0][k]
            
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if ckpt_metas[0] is not None and isinstance(ckpt_metas[0], dict) and "model" in ckpt_metas[0]:
        base_ckpt = ckpt_metas[0]
        base_ckpt["model"].load_state_dict(soup_dict)
        torch.save(base_ckpt, out_path)
    else:
        torch.save(soup_dict, out_path)
        
    print(f"[OK] Đã lưu mô hình PyTorch Model Soup thành công: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Model Soup Weight Averaging")
    ap.add_argument("--models", nargs="+", required=True, help="Danh sách file model checkpoint cần tính trung bình")
    ap.add_argument("--out", required=True, help="File đầu ra (vd. soup_best.keras hoặc soup_best.pt)")
    args = ap.parse_args()

    first_ext = Path(args.models[0]).suffix.lower()
    if first_ext in [".keras", ".h5"]:
        soup_keras(args.models, args.out)
    elif first_ext in [".pt", ".pth", ".bin"]:
        soup_pytorch(args.models, args.out)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {first_ext}. Phải là .keras hoặc .pt")


if __name__ == "__main__":
    main()
