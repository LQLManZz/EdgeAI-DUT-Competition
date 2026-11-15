@echo off
chcp 65001 > nul
title TIEN TRINH TU DONG VUOT 98.33 KAGGLE

cd /d "%~dp0EdgeAI-DUT-Competition_v3"
set "PYTHONPATH=%~dp0EdgeAI-DUT-Competition_v3\src;%PYTHONPATH%"
set "TF_ENABLE_ONEDNN_OPTS=0"

echo ======================================================================
echo    TIEN TRINH TU DONG TOAN DIEN VUOT 98.33 KAGGLE
echo    - GPU: NVIDIA Quadro T1000 (CUDA san sang)
echo    - Chang 1: Train Micro-YOLO 100 Epochs Full Data (6,000 anh) + TTA
echo    - Chang 2: Train CenterNet v3 Full Data (6,000 anh) + Xuat INT8
echo    - Chang 3: Chay Class-Weighted WBF Super Ensemble v4
echo    (Du kien: ~1h30 phut. May se tu chay tu dong 100%%)
echo ======================================================================
echo.

echo ----------------------------------------------------------------------
echo [CHANG 1/3] HUAN LUYEN MICRO-YOLO 100 EPOCHS TREN GPU T1000...
echo ----------------------------------------------------------------------
python -m yolo.train_yolo --model_cfg configs/micro_yolo_deep.yaml --data_yaml data/yolo_full.yaml --epochs 100 --batch 16 --name yolo_gpu_100e_full
if errorlevel 1 (
    echo [LOI] Huan luyen YOLO gap loi!
    pause
    exit /b 1
)

echo.
echo [+] Dang chay du doan YOLO voi TTA (Lat ngang + WBF)...
python scripts\predict_yolo_tta.py --weights runs\detect\experiments_yolo\yolo_gpu_100e_full\weights\best.pt --test_dir ..\egde-ai-challenge-2026-lan-2-qualifiers\test\images --base_sub submissions\submission_19_h1p_kd_v2_full_240x320.csv --out submissions\submission_yolo_100e_tta.csv --tta
echo [OK] Hoan tat Chang 1!

echo.
echo ----------------------------------------------------------------------
echo [CHANG 2/3] HUAN LUYEN CENTERNET V3 FULL DATA (6,000 ANH)...
echo ----------------------------------------------------------------------
python train_local.py --mode edgedet --config configs/h1p_v3_full_240x320.json --epochs 50 --batch 16 --name centernet_v3_full
if errorlevel 1 (
    echo [LOI] Huan luyen CenterNet gap loi!
    pause
    exit /b 1
)

echo.
echo [+] Dang xuat luong tu hoa INT8 TFLite cho CenterNet v3...
python -m edgedet.quant.export_int8 --model experiments\centernet_v3_full\best.keras --data ..\egde-ai-challenge-2026-lan-2-qualifiers

echo.
echo [+] Dang tao file submission CenterNet v3 Full...
python -m edgedet.eval.make_submission --tflite experiments\centernet_v3_full\model_int8.tflite --data ..\egde-ai-challenge-2026-lan-2-qualifiers --intent submissions\submission_19_h1p_kd_v2_full_240x320.csv --out submissions\submission_centernet_v3_full.csv
echo [OK] Hoan tat Chang 2!

echo.
echo ----------------------------------------------------------------------
echo [CHANG 3/3] HOP NHAT SUPER ENSEMBLE V4 DE VUOT 98.33...
echo ----------------------------------------------------------------------
python scripts\ensemble_champion_v4.py --subs submissions\submission_19_h1p_kd_v2_full_240x320.csv submissions\submission_17_h1p_kd_full_240x320_bs35.csv submissions\submission_centernet_v3_full.csv submissions\submission_yolo_100e_tta.csv --out ..\submission_breakthrough_98.csv --iou 0.52

echo.
echo [+] Kiem tra hop le file cuoi cung...
python -c "import pandas as pd, json; df=pd.read_csv('../submission_breakthrough_98.csv'); print('Tong so dong:', len(df)); max_b=max(len(json.loads(p)) for p in df[df['task_type']=='DETECTION']['prediction']); print('Max boxes moi anh:', max_b); assert len(df)==8000; print('[HOAN HAO] File submission_breakthrough_98.csv da san sang nop Kaggle de chiem TOP 1!')"

echo.
echo ======================================================================
echo    CHUC MUNG! TAT CA DA HOAN TAT 100%!
echo    File submission nam tai:
echo    D:\edgeAI_phan_biet_mot_so_thu_tu_camera_65k\submission_breakthrough_98.csv
echo ======================================================================
pause
