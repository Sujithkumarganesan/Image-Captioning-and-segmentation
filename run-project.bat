@echo off
REM Global command to run Image Captioning & Segmentation project
REM This can be run from any directory

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ PYTHON NOT FOUND
    echo.
    echo Install Python to run this project:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Navigate to project directory
cd /d "d:\image project\image-captioning-segmentation" 2>nul
if errorlevel 1 (
    echo ❌ Project directory not found!
    echo Expected location: d:\image project\image-captioning-segmentation
    pause
    exit /b 1
)

REM Quick setup
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Always install/update dependencies to ensure they're available
echo 📦 Installing/updating dependencies...
echo    This may take a few minutes on first run...

pip install --upgrade pip >nul 2>&1
pip install flask>=2.2.0 >nul 2>&1
pip install torch>=1.12.0 --index-url https://download.pytorch.org/whl/cpu >nul 2>&1
pip install torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu >nul 2>&1  
pip install pillow>=9.0.0 >nul 2>&1
pip install opencv-python>=4.6.0 >nul 2>&1
pip install numpy>=1.21.0 >nul 2>&1
pip install matplotlib>=3.5.0 >nul 2>&1
pip install flask-cors>=3.0.10 >nul 2>&1

echo ✅ Dependencies installed!

REM Create required directories
if not exist "static\uploads" mkdir static\uploads
if not exist "static\outputs" mkdir static\outputs  
if not exist "models\checkpoints" mkdir models\checkpoints

echo.
echo 🚀 Starting Image Captioning ^& Segmentation Server...
echo.

REM Start the server
python app.py
