"""K-Fold Stratified Split cho Object Detection đa nhãn.
Dựa trên thuật toán Seirafi's Iterative Multilabel Stratification.
Đảm bảo 7 lớp (bottle, cup, phone, book, mouse, box, toy_car) phân bố cân bằng tuyệt đối trên từng Fold.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np


def multilabel_kfold_split(items, n_splits=3, seed=2026):
    """Chia danh sách items thành n_splits folds cân bằng nhãn.
    Mỗi phần tử items là dict(id=..., path=..., boxes=..., classes=...).
    Trả về: list gồm n_splits tuple: (train_items, val_items).
    """
    rng = np.random.default_rng(seed)

    # 1. Thu thập danh sách lớp của từng item
    image_classes = {}
    for it in items:
        # Lấy unique classes của từng ảnh
        cls_set = set(it["classes"]) if len(it["classes"]) > 0 else set()
        image_classes[it["id"]] = cls_set

    class_indices = defaultdict(list)
    item_by_id = {it["id"]: it for it in items}
    all_ids = sorted(list(image_classes.keys()))
    rng.shuffle(all_ids)

    for img_id in all_ids:
        for c in image_classes[img_id]:
            class_indices[c].append(img_id)

    # 2. Tính số lượng mục tiêu từng lớp trên từng fold
    class_totals = {c: len(ids) for c, ids in class_indices.items()}
    fold_targets = {f: {c: class_totals[c] / n_splits for c in class_totals} for f in range(n_splits)}
    fold_actual = {f: {c: 0 for c in class_totals} for f in range(n_splits)}
    fold_ids = {f: [] for f in range(n_splits)}

    assigned_ids = set()
    classes_sorted = sorted(class_totals.keys(), key=lambda c: class_totals[c])

    # 3. Phân bổ theo deficit
    for c in classes_sorted:
        c_ids = [i for i in class_indices[c] if i not in assigned_ids]
        rng.shuffle(c_ids)

        for img_id in c_ids:
            if img_id in assigned_ids:
                continue
            img_c_list = image_classes[img_id]

            best_fold = None
            best_deficit = -1e9
            for f in range(n_splits):
                deficit = sum(fold_targets[f][cls] - fold_actual[f][cls] for cls in img_c_list)
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_fold = f

            fold_ids[best_fold].append(img_id)
            assigned_ids.add(img_id)
            for cls in img_c_list:
                fold_actual[best_fold][cls] += 1

    # Phân bổ các ảnh rỗng (không có vật thể) đều vào các fold
    remaining = [i for i in all_ids if i not in assigned_ids]
    for idx, img_id in enumerate(remaining):
        fold_ids[idx % n_splits].append(img_id)
        assigned_ids.add(img_id)

    # 4. Tạo kết quả train / val cho từng fold
    folds = []
    for f in range(n_splits):
        val_set_ids = set(fold_ids[f])
        val_items = [item_by_id[i] for i in fold_ids[f]]
        train_items = [it for it in items if it["id"] not in val_set_ids]
        folds.append((train_items, val_items))

    return folds
