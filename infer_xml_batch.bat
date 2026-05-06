@echo off
chcp 65001 > nul
setlocal

set PYTHON=.venv\Scripts\python.exe

REM Folder containing raw/original videos to analyze.
set INPUT_DIR=C:\Users\MOON\Desktop\DFL-job\CUT-SET

REM Leave OUT_DIR empty to create a sibling folder named <INPUT_DIR>_xml.
set OUT_DIR=

set PROFILE_PATH=style_profile.pkl

REM Thresholds. Lower values create more KEEP/REVIEW ranges and more split points.
set KEEP_THR=0.74
set REVIEW_THR=0.68

REM 0 means use sample_fps saved in profile. Set 0.5 or 1.0 explicitly if needed.
set SAMPLE_FPS=0
set BATCH_SIZE=64
set SMOOTH_SEC=7
set MIN_KEEP_SEC=3
set MERGE_GAP_SEC=2

if not exist "%PYTHON%" (
    echo [ERROR] Python not found. Run setup_uv_env.bat first.
    pause
    exit /b 1
)

if not exist "%INPUT_DIR%" (
    echo [ERROR] INPUT_DIR not found: %INPUT_DIR%
    echo Edit infer_xml_batch.bat and set INPUT_DIR to your raw videos folder.
    pause
    exit /b 1
)

if not exist "%PROFILE_PATH%" (
    echo [ERROR] PROFILE_PATH not found: %PROFILE_PATH%
    echo Run build_profile.bat first.
    pause
    exit /b 1
)

if "%OUT_DIR%"=="" (
    "%PYTHON%" infer_xml_batch.py ^
      --input_dir "%INPUT_DIR%" ^
      --profile "%PROFILE_PATH%" ^
      --sample_fps %SAMPLE_FPS% ^
      --batch_size %BATCH_SIZE% ^
      --keep_thr %KEEP_THR% ^
      --review_thr %REVIEW_THR% ^
      --smooth_sec %SMOOTH_SEC% ^
      --min_keep_sec %MIN_KEEP_SEC% ^
      --merge_gap_sec %MERGE_GAP_SEC% ^
) else (
    "%PYTHON%" infer_xml_batch.py ^
      --input_dir "%INPUT_DIR%" ^
      --out_dir "%OUT_DIR%" ^
      --profile "%PROFILE_PATH%" ^
      --sample_fps %SAMPLE_FPS% ^
      --batch_size %BATCH_SIZE% ^
      --keep_thr %KEEP_THR% ^
      --review_thr %REVIEW_THR% ^
      --smooth_sec %SMOOTH_SEC% ^
      --min_keep_sec %MIN_KEEP_SEC% ^
      --merge_gap_sec %MERGE_GAP_SEC% ^
)

echo.
echo [DONE] Import generated .xml files into Premiere Pro.
echo        Each XML references its original video and creates a split, marker-labeled sequence.
pause
endlocal
