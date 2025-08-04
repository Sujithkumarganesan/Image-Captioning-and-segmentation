@echo off
echo ============================================
echo  Image Captioning ^& Segmentation Setup
echo ============================================
echo.

REM Test if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python first:
    echo.
    echo METHOD 1 - Microsoft Store (Easiest):
    echo 1. Press Windows+R
    echo 2. Type: ms-windows-store://pdp/?ProductId=9NRWMJP3717K
    echo 3. Press Enter and click Install
    echo.
    echo METHOD 2 - Python.org:
    echo 1. Go to: https://www.python.org/downloads/
    echo 2. Download and install Python
    echo 3. CHECK "Add Python to PATH" during install
    echo.
    echo After installing Python, run this script again.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found!
python --version

REM Check if we're in the right place
if not exist "app.py" (
    echo [ERROR] Cannot find app.py
    echo Make sure you're running this from the project folder:
    echo d:\image project\image-captioning-segmentation\
    echo.
    pause
    exit /b 1
)

echo [OK] Project files found!

REM Create virtual environment
if not exist "venv\" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Try running as administrator
        pause
        exit /b 1
    )
)

echo [OK] Virtual environment ready!

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install packages
echo Installing required packages (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install packages
    echo Check your internet connection and try again
    pause
    exit /b 1
)

echo [OK] All packages installed!

REM Create directories
mkdir static\uploads 2>nul
mkdir static\outputs 2>nul
mkdir data\raw 2>nul
mkdir models\checkpoints 2>nul
mkdir logs 2>nul

echo [OK] Directories created!

echo.
echo ============================================
echo  STARTING APPLICATION...
echo ============================================
echo.
echo Your web browser will open automatically
echo Application URL: http://127.0.0.1:5000
echo.
echo To stop the server: Press Ctrl+C
echo ============================================
echo.

REM Start the application
start http://127.0.0.1:5000
python app.py

pause
