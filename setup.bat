@echo off
echo.
echo ==========================================
echo   CrossFakeNet -- Environment Setup
echo ==========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/4] Installing dependencies (this may take 5-10 minutes)...
pip install -r requirements.txt

if not exist uploads mkdir uploads

echo.
echo ==========================================
echo   Setup complete!
echo.
echo   To run:
echo     venv\Scripts\activate
echo     python app.py
echo.
echo   Then open: http://127.0.0.1:5000
echo ==========================================
echo.
pause
