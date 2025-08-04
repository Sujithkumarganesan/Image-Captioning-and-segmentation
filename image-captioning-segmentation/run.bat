@echo off
REM Simple command to run the Image Captioning & Segmentation server
REM Usage: run

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Python not found!
    echo.
    echo Please install Python first:
    echo 1. Go to: https://www.python.org/downloads/
    echo 2. Download and install Python 3.8+
    echo 3. Make sure to check "Add Python to PATH"
    echo.
    echo Then run this command again.
    echo.
    exit /b 1
)

REM Navigate to project directory
cd /d "d:\image project\image-captioning-segmentation"

REM Check if project exists
if not exist "app.py" (
    echo ❌ Project not found at: d:\image project\image-captioning-segmentation
    echo Please make sure the project is in the correct location.
    exit /b 1
)

REM Create virtual environment if needed
if not exist "venv\" (
    echo 📦 Setting up virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist "venv\Lib\site-packages\flask" (
    echo 📦 Installing dependencies...
    pip install -q -r requirements.txt
)

REM Create directories
mkdir static\uploads 2>nul
mkdir static\outputs 2>nul
mkdir models\checkpoints 2>nul

REM Start the application
python app.py
