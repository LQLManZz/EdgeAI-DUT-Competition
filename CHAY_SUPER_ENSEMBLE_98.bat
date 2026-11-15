@echo off
chcp 65001 > nul
title CHAY SUPER ENSEMBLE V4 DE VUOT 98.33

cd /d "%~dp0"
if exist "EdgeAI-DUT-Competition_v3" (
    cd /d "%~dp0EdgeAI-DUT-Competition_v3"
)

set "PYTHONPATH=%cd%\src;%PYTHONPATH%"

echo ======================================================================
echo    BAT DAU CHAY SUPER ENSEMBLE V4 CHO KAGGLE LEADERBOARD
echo    Ket hop cac mo hinh manh nhat:
echo    - CenterNet KD v2 Full (sub19)
echo    - CenterNet KD Full bs35 (sub17)
echo    - CenterNet v3 Full moi (hoac centernet_v2)
echo    - Micro-YOLO 100e TTA moi (hoac yolo_gpu_50e)
echo ======================================================================
echo.

set "CENTER_FILE=submissions\submission_centernet_v3_full.csv"
if not exist "%CENTER_FILE%" (
    set "CENTER_FILE=submissions\submission_centernet_v2.csv"
)

set "YOLO_FILE=submissions\submission_yolo_100e_tta.csv"
if not exist "%YOLO_FILE%" (
    set "YOLO_FILE=submissions\submission_yolo_gpu_50e.csv"
)

echo [+] File CenterNet su dung: %CENTER_FILE%
echo [+] File Micro-YOLO su dung: %YOLO_FILE%
echo.

python scripts\ensemble_champion_v4.py --subs submissions\submission_19_h1p_kd_v2_full_240x320.csv submissions\submission_17_h1p_kd_full_240x320_bs35.csv %CENTER_FILE% %YOLO_FILE% --out ..\submission_breakthrough_98.csv --iou 0.52

echo.
echo ======================================================================
echo    KIEM TRA HOP LE FILE SUBMISSION...
echo ======================================================================
echo.

python -c "import pandas as pd, json; df=pd.read_csv('../submission_breakthrough_98.csv'); print('Tong so dong:', len(df)); max_b=max(len(json.loads(p)) for p in df[df['task_type']=='DETECTION']['prediction']); print('Max boxes moi anh:', max_b); assert len(df)==8000; print('[HOAN HAO] File submission_breakthrough_98.csv san sang nop Kaggle de chiem TOP 1!')"

echo.
echo ======================================================================
echo    File cuoi cung nam tai thu muc goc:
echo    D:\edgeAI_phan_biet_mot_so_thu_tu_camera_65k\submission_breakthrough_98.csv
echo ======================================================================
pause
