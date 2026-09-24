@echo off
setlocal

cd C:\Users\h1oo7\Desktop\hrmg774\faceswap

if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay moi truong Python .venv.
    echo Hay tao moi truong bang: python -m venv .venv
    pause
    exit /b 1
)

if not exist "multi_faceswap_app.py" (
    echo Khong tim thay file multi_faceswap_app.py.
    pause
    exit /b 1
)

echo Dang khoi dong FaceSwap...
".venv\Scripts\python.exe" "multi_faceswap_app.py"

if errorlevel 1 (
    echo.
    echo Ung dung ket thuc voi loi.
    pause
)

endlocal
