@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua co moi truong ao .venv. Hay chay setup.ps1 truoc.
    pause
    exit /b 1
)

if not exist "multi_faceswap_app.py" (
    echo [LOI] Khong tim thay file multi_faceswap_app.py trong thu muc nay.
    pause
    exit /b 1
)

echo Dang chay multi_faceswap_app.py ...
.venv\Scripts\python.exe multi_faceswap_app.py
pause
