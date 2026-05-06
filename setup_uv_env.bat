@echo off
chcp 65001 > nul
setlocal

echo ============================================
echo Setup uv environment for video XML project
echo ============================================

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed or not in PATH.
    echo Install uv first:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [INFO] Creating .venv with Python 3.10...
    uv venv .venv --python 3.10
) else (
    echo [INFO] .venv already exists.
)

echo [INFO] Installing PyTorch CUDA 12.1 wheels...
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo [INFO] Installing project dependencies...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

echo.
echo [INFO] Testing torch CUDA...
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo [DONE] Environment is ready.
echo First DINOv2 run will download the model through torch.hub.
pause
endlocal
