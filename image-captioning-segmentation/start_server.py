#!/usr/bin/env python3
"""
Command line launcher for Image Captioning & Segmentation project
Usage: python start_server.py
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_python():
    """Check if Python is properly installed."""
    try:
        import sys
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            return True
        else:
            print("❌ Python 3.8+ required. Current version:", sys.version)
            return False
    except:
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    packages = [
        "flask>=2.2.0",
        "torch>=1.12.0", 
        "torchvision>=0.13.0",
        "pillow>=9.0.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "flask-cors>=3.0.10"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")
    
    print("✅ Dependencies installed!")

def create_directories():
    """Create necessary directories."""
    dirs = [
        "static/uploads",
        "static/outputs", 
        "models/checkpoints",
        "data/raw",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    print("🚀 IMAGE CAPTIONING & SEGMENTATION SERVER")
    print("=" * 50)
    
    # Check Python
    if not check_python():
        print("\nPlease install Python 3.8+ from: https://python.org/downloads/")
        return
    
    # Change to project directory
    project_dir = Path(__file__).parent / "image-captioning-segmentation"
    if project_dir.exists():
        os.chdir(project_dir)
    else:
        print(f"❌ Project directory not found: {project_dir}")
        return
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found. Make sure you're in the project directory.")
        return
    
    # Install requirements
    try:
        import flask
    except ImportError:
        install_requirements()
    
    # Create directories
    create_directories()
    
    # Start the server
    print("\n🌐 Starting server...")
    print("📍 URL: http://127.0.0.1:5000")
    print("\n💡 Click the link above to open in your browser!")
    print("   Or copy and paste: http://127.0.0.1:5000")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the Flask app
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
