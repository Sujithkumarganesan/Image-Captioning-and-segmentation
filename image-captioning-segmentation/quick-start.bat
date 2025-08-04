@echo off
echo ======================================
echo     Image Captioning & Segmentation  
echo     Quick Start Launcher
echo ======================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version

:: Install minimal dependencies directly
echo.
echo [INFO] Installing minimal dependencies...
echo This may take a few minutes on first run...

python -m pip install --upgrade pip
python -m pip install flask>=2.2.0 pillow>=9.0.0 numpy>=1.21.0 matplotlib>=3.5.0 flask-cors>=3.0.10 transformers>=4.20.0 requests>=2.25.0 torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Try running as administrator or check your internet connection
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Dependencies installed successfully!

:: Start the application
echo.
echo [INFO] Starting the application...
echo The server will start on: http://localhost:5000
echo.
echo Click the link above or copy-paste it into your browser
echo Press Ctrl+C to stop the server
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo Check the error messages above
    pause
)
