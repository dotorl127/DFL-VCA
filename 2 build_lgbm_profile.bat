@echo off
chcp 65001 > nul
setlocal

set PYTHON=.venv\Scripts\python.exe

REM First run build_paired_profile.bat to create this file.
set PAIRED_PROFILE=paired_profile.pkl
set LGBM_PROFILE_OUT=lgbm_profile.pkl

REM 0 means use the value saved in paired_profile.pkl.
set SAMPLE_FPS=0
set INPUT_SIZE=0
set BATCH_SIZE=64

REM Use at most N negative frames per positive frame for each raw video.
set NEGATIVE_RATIO=4

REM Conservative LightGBM defaults for small paired datasets.
set NUM_LEAVES=31
set LEARNING_RATE=0.03
set N_ESTIMATORS=400

if not exist "%PYTHON%" (
    echo [ERROR] Python not found. Run setup_uv_env.bat first.
    pause
    exit /b 1
)

if not exist "%PAIRED_PROFILE%" (
    echo [ERROR] PAIRED_PROFILE not found: %PAIRED_PROFILE%
    echo Run build_paired_profile.bat first.
    pause
    exit /b 1
)

"%PYTHON%" build_lgbm_profile.py ^
  --paired_profile "%PAIRED_PROFILE%" ^
  --out "%LGBM_PROFILE_OUT%" ^
  --sample_fps %SAMPLE_FPS% ^
  --input_size %INPUT_SIZE% ^
  --batch_size %BATCH_SIZE% ^
  --negative_ratio %NEGATIVE_RATIO% ^
  --num_leaves %NUM_LEAVES% ^
  --learning_rate %LEARNING_RATE% ^
  --n_estimators %N_ESTIMATORS%

pause
endlocal
