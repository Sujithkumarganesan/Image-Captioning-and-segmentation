@echo off
REM Image Captioning & Segmentation Application Startup Script
REM This script will start the Flask application and open it in your browser

echo ============================================
echo  Image Captioning ^& Segmentation Server
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================
    echo  PYTHON NOT FOUND
    echo ============================================
    echo.
    echo Python is not installed or not in PATH.
    echo.
    echo QUICK SOLUTIONS:
    echo.
    echo 1. Install from python.org:
    echo    - Go to: https://www.python.org/downloads/
    echo    - Download Python 3.8+
    echo    - IMPORTANT: Check "Add Python to PATH" during install
    echo.
    echo 2. Install from Microsoft Store:
    echo    - Type 'python' in PowerShell
    echo    - Click Install when Store opens
    echo.
    echo 3. Use Anaconda:
    echo    - Download from: https://www.anaconda.com/products/distribution
    echo    - Use "Anaconda Prompt" after installation
    echo.
    echo After installing Python, run this script again.
    echo.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ERROR: app.py not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing/updating requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=True

REM Create necessary directories
if not exist "static\uploads\" mkdir static\uploads
if not exist "static\outputs\" mkdir static\outputs
if not exist "data\raw\" mkdir data\raw
if not exist "models\checkpoints\" mkdir models\checkpoints
if not exist "logs\" mkdir logs

echo.
echo ============================================
echo  Starting the application...
echo ============================================
echo.
echo The application will be available at:
echo http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Start the Flask application
python app.py

REM Deactivate virtual environment when done
deactivate

echo.
echo Application stopped.
pause
