@echo off
chcp 65001 > nul
setlocal

set PYTHON=.venv\Scripts\python.exe

REM Folder containing already edited videos.
set EDITED_DIR=edited_videos

REM Output style profile.
set PROFILE_OUT=style_profile.pkl

REM 1.0 = one frame per second. Use 0.5 for faster processing.
set SAMPLE_FPS=2.0
set INPUT_SIZE=224
set BATCH_SIZE=64
set CLUSTERS=64

if not exist "%PYTHON%" (
    echo [ERROR] Python not found. Run setup_uv_env.bat first.
    pause
    exit /b 1
)

if not exist "%EDITED_DIR%" (
    echo [ERROR] EDITED_DIR not found: %EDITED_DIR%
    echo Create the folder and put edited videos inside it.
    pause
    exit /b 1
)

"%PYTHON%" build_profile.py ^
  --edited_dir "%EDITED_DIR%" ^
  --out "%PROFILE_OUT%" ^
  --sample_fps %SAMPLE_FPS% ^
  --input_size %INPUT_SIZE% ^
  --batch_size %BATCH_SIZE% ^
  --clusters %CLUSTERS%

pause
endlocal
