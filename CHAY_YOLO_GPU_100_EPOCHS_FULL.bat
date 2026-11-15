@echo off
chcp 65001 > nul
title TRAIN MICRO-YOLO 100 EPOCHS FULL DATA + TTA ENSEMBLE VUOT 98.33

cd /d "%~dp0"
if exist "EdgeAI-DUT-Competition_v3" (
    cd /d "%~dp0EdgeAI-DUT-Competition_v3"
)

set "PYTHONPATH=%cd%\src;%PYTHONPATH%"

echo ======================================================================
echo    CHIEN DICH BUT PHA VUOT 98.33 KAGGLE LEADERBOARD
echo    HUAN LUYEN MICRO-YOLO 100 EPOCHS TREN 100 PHAN TRAM DATA (6,000 ANH)
echo    - GPU: NVIDIA Quadro T1000 (CUDA)
echo    - Dataset: 6,000 anh train_full (khong bo sot anh nao)
echo    - Model: Micro-YOLO Deep (191k params, duoi 200k chuan ESP32-S3)
echo    - Epochs: 100 (Cosine LR decay, warmup, Mosaic)
echo    - Test-Time Augmentation (TTA): Bat (Lat ngang + WBF)
echo ======================================================================
echo.

python -m yolo.train_yolo --model_cfg configs/micro_yolo_deep.yaml --data_yaml data/yolo_full.yaml --epochs 100 --batch 16 --name yolo_gpu_100e_full

if errorlevel 1 (
    echo [LOI] Huan luyen YOLO that bai. Vui long kiem tra loi ben tren.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo    BAP DAU DU DOAN TREN TAP TEST VOI TTA (TEST-TIME AUGMENTATION)...
echo ======================================================================
echo.

python scripts\predict_yolo_tta.py --weights runs\detect\experiments_yolo\yolo_gpu_100e_full\weights\best.pt --test_dir ..\egde-ai-challenge-2026-lan-2-qualifiers\test\images --base_sub submissions\submission_19_h1p_kd_v2_full_240x320.csv --out submissions\submission_yolo_100e_tta.csv --tta

echo.
echo ======================================================================
echo    THUC HIEN CLASS-WEIGHTED WBF ENSEMBLE V4...
echo ======================================================================
echo.

python scripts\ensemble_champion_v4.py --subs submissions\submission_19_h1p_kd_v2_full_240x320.csv submissions\submission_17_h1p_kd_full_240x320_bs35.csv submissions\submission_centernet_v2.csv submissions\submission_yolo_100e_tta.csv --out ..\submission_breakthrough_98.csv --iou 0.52

echo.
echo ======================================================================
echo    KIEM TRA HOP LE FILE SUBMISSION CUOI CUNG...
echo ======================================================================
echo.

python -c "import pandas as pd, json; df=pd.read_csv('../submission_breakthrough_98.csv'); print('Tong dong:', len(df)); max_b=max(len(json.loads(p)) for p in df[df['task_type']=='DETECTION']['prediction']); print('Max boxes per image:', max_b); assert len(df)==8000 and max_b<=100, 'Loi file!'; print('[HOAN HAO] File san sang nop Kaggle de vuot 98.33%!')"

echo.
echo ======================================================================
echo    DA XUAT FILE: submission_breakthrough_98.csv
echo    HAY NOP LEN KAGGLE DE CHIEM TOP 1 LEADERBOARD!
echo ======================================================================
pause
