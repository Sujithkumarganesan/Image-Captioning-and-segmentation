# Image Captioning and Segmentation Project

A comprehensive deep learning project that combines image captioning and semantic segmentation capabilities with a web-based interface.

## Features

- **Image Captioning**: Generate descriptive captions for images using transformer-based models
- **Semantic Segmentation**: Perform pixel-level classification and object segmentation
- **Multi-task Learning**: Combined model for both captioning and segmentation
- **Web Interface**: User-friendly web application for easy interaction
- **API Support**: RESTful API endpoints for programmatic access

## Quick Start

### Prerequisites

**Option 1: Install Python (Recommended)**
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation: `python --version`

**Option 2: Install Anaconda/Miniconda**
1. Download from [anaconda.com](https://www.anaconda.com/products/distribution)
2. Follow installation instructions
3. Verify: `conda --version`

### Easy Setup (Windows)

**🚀 One-Click Startup:**
1. Navigate to the project folder
2. Double-click `start.bat`
3. The script will automatically:
   - Install Python dependencies
   - Start the server
   - Open your browser to http://127.0.0.1:5000

### Manual Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-captioning-segmentation
```

2. **Option A: Using Conda**
```bash
conda env create -f environment.yml
conda activate image-caption-seg
```

**Option B: Using Python venv**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

### Running the Application

**Windows:**
```cmd
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Manual start:**
```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5000`

## Project Structure

```
image-captioning-segmentation/
├── config/                 # Configuration files
├── data/                   # Dataset storage
├── src/                    # Source code
├── models/                 # Model implementations
├── utils/                  # Utility functions
├── templates/              # HTML templates
├── static/                 # Static web assets
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks
```

## Models Supported

- **Captioning**: BLIP, CLIP-Cap, Vision Transformer + GPT
- **Segmentation**: DeepLab, U-Net, Mask R-CNN
- **Multi-task**: Custom architecture combining both tasks

## API Endpoints

- `POST /api/v1/caption`: Generate image caption
- `POST /api/v1/segment`: Perform image segmentation
- `POST /api/v1/analyze`: Combined captioning and segmentation
- `GET /api/v1/health`: Check model status

## Troubleshooting

### Python Not Found Error
If you see "Python was not found", you have several options:

1. **Install Python from python.org:**
   - Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
   - ⚠️ **Important:** Check "Add Python to PATH" during installation
   - Restart your command prompt after installation

2. **Use the Microsoft Store:**
   - Type `python` in PowerShell to open Microsoft Store
   - Install Python from there

3. **Use Anaconda (Alternative):**
   - Download from [anaconda.com](https://www.anaconda.com/products/distribution)
   - Use `conda` commands instead of `pip`

### Quick Test
After installing Python, verify it works:
```cmd
python --version
pip --version
```

### Alternative: Use the Startup Scripts
The easiest way is to use our provided startup scripts:
- **Windows:** Double-click `start.bat`
- **Linux/Mac:** Run `./start.sh`

These scripts handle everything automatically!

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
