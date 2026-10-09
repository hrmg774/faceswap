@echo off
chcp 65001 >nul
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [LOI] Chua co moi truong ao .venv. Hay chay setup.ps1 truoc.
    pause
    exit /b 1
)

if not exist "main.py" (
    echo [LOI] Khong tim thay file main.py trong thu muc nay.
    pause
    exit /b 1
)

echo Dang chay main.py ...
.venv\Scripts\python.exe main.py
pause
