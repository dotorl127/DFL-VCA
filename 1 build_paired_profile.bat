@echo off
chcp 65001 > nul
setlocal

set PYTHON=.venv\Scripts\python.exe

REM Folder containing original/raw videos.
set RAW_DIR=raw_videos

REM Folder containing edited videos.
REM Each edited filename should contain the raw filename stem.
REM Example: raw "A001.mp4" -> edited "A001_cut_01.mp4", "A001_highlight.mp4"
set EDITED_DIR=edited_videos

set PROFILE_OUT=paired_profile.pkl

REM 2.0 = two frames per second. Increase for more accurate boundaries, decrease for speed.
set SAMPLE_FPS=2.0
set INPUT_SIZE=224
set BATCH_SIZE=64

REM Raw frame is considered used/KEEP when cosine similarity to paired edited frames is above this.
set MATCH_THR=0.78
set SMOOTH_SEC=2
set MIN_KEEP_SEC=1
set MERGE_GAP_SEC=1

set CLUSTERS=64
set SOURCE_CLUSTERS=32

REM 1 = also pair split raw parts like XXX_1 and XXX_2 to the same edited video.
REM Use this when one edited video can be made by combining multiple split raw videos.
set PAIR_SERIES_PARTS=1

if not exist "%PYTHON%" (
    echo [ERROR] Python not found. Run setup_uv_env.bat first.
    pause
    exit /b 1
)

if not exist "%RAW_DIR%" (
    echo [ERROR] RAW_DIR not found: %RAW_DIR%
    pause
    exit /b 1
)

if not exist "%EDITED_DIR%" (
    echo [ERROR] EDITED_DIR not found: %EDITED_DIR%
    pause
    exit /b 1
)

set PAIR_SERIES_ARG=
if "%PAIR_SERIES_PARTS%"=="1" set PAIR_SERIES_ARG=--pair_series_parts

"%PYTHON%" build_paired_profile.py ^
  --raw_dir "%RAW_DIR%" ^
  --edited_dir "%EDITED_DIR%" ^
  --out "%PROFILE_OUT%" ^
  --sample_fps %SAMPLE_FPS% ^
  --input_size %INPUT_SIZE% ^
  --batch_size %BATCH_SIZE% ^
  --match_thr %MATCH_THR% ^
  --smooth_sec %SMOOTH_SEC% ^
  --min_keep_sec %MIN_KEEP_SEC% ^
  --merge_gap_sec %MERGE_GAP_SEC% ^
  --clusters %CLUSTERS% ^
  --source_clusters %SOURCE_CLUSTERS% ^
  %PAIR_SERIES_ARG%

pause
endlocal
