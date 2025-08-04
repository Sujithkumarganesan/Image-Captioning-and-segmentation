#!/bin/bash

# Image Captioning & Segmentation Application Startup Script
# This script will start the Flask application and open it in your browser

echo "============================================"
echo " Image Captioning & Segmentation Server"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing/updating requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=True

# Create necessary directories
mkdir -p static/uploads
mkdir -p static/outputs
mkdir -p data/raw
mkdir -p models/checkpoints
mkdir -p logs

echo
echo "============================================"
echo " Starting the application..."
echo "============================================"
echo
echo "The application will be available at:"
echo "http://127.0.0.1:5000"
echo
echo "Press Ctrl+C to stop the server"
echo "============================================"
echo

# Start the Flask application
python3 app.py

# Deactivate virtual environment when done
deactivate

echo
echo "Application stopped."
