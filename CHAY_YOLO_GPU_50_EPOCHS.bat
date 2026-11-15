@echo off
chcp 65001 > nul
title TRAIN MICRO-YOLO TREN GPU NVIDIA QUADRO T1000
cd /d "%~dp0EdgeAI-DUT-Competition_v3"
set "PYTHONPATH=%~dp0EdgeAI-DUT-Competition_v3\src;%PYTHONPATH%"

echo ======================================================================
echo    BAT DAU HUAN LUYEN MICRO-YOLO TREN GPU NVIDIA QUADRO T1000
echo    - Dataset: 5,000 train / 1,000 val
echo    - Input size: 240x320
echo    - Architecture: Micro-YOLO Deep (191k params, duoi 200k chuan BTC)
echo    - Epochs: 50
echo ======================================================================
echo.

python -m yolo.train_yolo --model_cfg configs/micro_yolo_deep.yaml --epochs 50 --batch 16 --name yolo_gpu_50e

echo.
echo ======================================================================
echo    HUAN LUYEN XONG! TAO FILE DU DOAN CHO MICRO-YOLO...
echo ======================================================================
echo.

python ..\predict_colab.py --test_dir ..\egde-ai-challenge-2026-lan-2-qualifiers\test\images

if exist submission.csv (
    copy /y submission.csv submissions\submission_yolo_gpu_50e.csv
    echo.
    echo ======================================================================
    echo    KET HOP ENSEMBLE WBF (CENTERNET 70%% + YOLO 30%%)...
    echo ======================================================================
    echo.
    python scripts\ensemble_wbf.py --subs submissions\submission_centernet_v2.csv submissions\submission_yolo_gpu_50e.csv --weights 0.7 0.3 --out submissions\submission_ensemble_best.csv
)

echo.
echo ======================================================================
echo    HOAN TAT 100%!
echo ======================================================================
pause
