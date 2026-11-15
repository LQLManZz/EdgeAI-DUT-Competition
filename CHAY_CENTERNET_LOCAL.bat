@echo off
chcp 65001 > nul
title HUẤN LUYỆN CENTERNET EDGEDET - CHÍNH XÁC NHẤT
cd /d "%~dp0\EdgeAI-DUT-Competition_v3"
set "PYTHONPATH=%~dp0EdgeAI-DUT-Competition_v3\src;%PYTHONPATH%"
set "TF_ENABLE_ONEDNN_OPTS=0"

echo ======================================================================
echo    BẮT ĐẦU HUẤN LUYỆN CENTERNET EDGEDET (MÔ HÌNH CHỦ LỰC)
echo    - Kiến trúc: MobileNetV3-Small + Decoupled Head
echo    - Số tham số: 197,139 (Chuẩn BTC: <= 200,000)
echo    - Kích thước ảnh: 240x320
echo    - Kỹ thuật: Focal loss + L1 box loss + Huber neck penalty + EMA
echo    - Số epoch: 40
echo ======================================================================
echo.

python train_local.py --mode edgedet --config configs/h1p_v2_240x320.json --epochs 40 --batch 16 --name centernet_v2_run

echo.
echo ======================================================================
echo    HUẤN LUYỆN XONG! ĐANG XUẤT LƯỢNG TỬ HÓA INT8 TFLITE...
echo ======================================================================
echo.

python -m edgedet.quant.export_int8 --model experiments\centernet_v2_run\best.keras --data ..\egde-ai-challenge-2026-lan-2-qualifiers

echo.
echo ======================================================================
echo    ĐANG SINH FILE DỰ ĐOÁN NỘP BÀI SUBMISSION.CSV...
echo ======================================================================
echo.

python -m edgedet.eval.make_submission --tflite experiments\centernet_v2_run\model_int8.tflite --data ..\egde-ai-challenge-2026-lan-2-qualifiers --out submissions\submission_centernet_v2.csv

echo.
echo ======================================================================
echo    HOÀN TẤT 100%!
echo    File submission đã được lưu tại:
echo    EdgeAI-DUT-Competition_v3\submissions\submission_centernet_v2.csv
echo ======================================================================
pause
