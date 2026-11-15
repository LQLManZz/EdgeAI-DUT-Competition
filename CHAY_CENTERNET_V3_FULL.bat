@echo off
chcp 65001 > nul
title TRAIN CENTERNET V3 FULL DATA (6,000 ANH) - CHU LUC ESP32-S3

cd /d "%~dp0"
if exist "EdgeAI-DUT-Competition_v3" (
    cd /d "%~dp0EdgeAI-DUT-Competition_v3"
)

set "PYTHONPATH=%cd%\src;%PYTHONPATH%"
set "TF_ENABLE_ONEDNN_OPTS=0"

echo ======================================================================
echo    HUAN LUYEN CENTERNET V3 FULL DATA (MO HINH CHU LUC)
echo    - Du lieu: 100 phan tram du lieu (6,000 anh toan bo)
echo    - Kien truc: MobileNetV3-Small + Decoupled Head
echo    - So tham so: 197,139 (duoi 200,000 chuan BTC)
echo    - Kich thuoc: 240x320
echo    - So epoch: 50
echo    - Ky thuat: Focal loss + GIoU loss + Mosaic + EMA weights
echo ======================================================================
echo.

python train_local.py --mode edgedet --config configs/h1p_v3_full_240x320.json --epochs 50 --batch 16 --name centernet_v3_full

if errorlevel 1 (
    echo [LOI] Huan luyen CenterNet that bai. Vui long kiem tra loi ben tren.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo    HUAN LUYEN XONG! DANG XUAT LUONG TU HOA INT8 TFLITE...
echo ======================================================================
echo.

python -m edgedet.quant.export_int8 --model experiments\centernet_v3_full\best.keras --data ..\egde-ai-challenge-2026-lan-2-qualifiers

echo.
echo ======================================================================
echo    DANG SINH FILE DU DOAN NOP BAI SUBMISSION_CENTERNET_V3_FULL.CSV...
echo ======================================================================
echo.

python -m edgedet.eval.make_submission --tflite experiments\centernet_v3_full\model_int8.tflite --data ..\egde-ai-challenge-2026-lan-2-qualifiers --intent submissions\submission_19_h1p_kd_v2_full_240x320.csv --out submissions\submission_centernet_v3_full.csv

echo.
echo ======================================================================
echo    [HOAN TAT] File du doan da duoc tao thanh cong:
echo    EdgeAI-DUT-Competition_v3\submissions\submission_centernet_v3_full.csv
echo ======================================================================
pause
