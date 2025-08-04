# Quick Start Guide

## Running the Application

### Option 1: Using the Startup Script (Recommended)

**Windows:**
1. Double-click `start.bat` or run it from command prompt
2. The script will automatically:
   - Create a virtual environment
   - Install dependencies
   - Start the server
   - Open your browser to http://127.0.0.1:5000

**Linux/Mac:**
1. Make the script executable: `chmod +x start.sh`
2. Run the script: `./start.sh`
3. The application will be available at http://127.0.0.1:5000

### Option 2: Manual Setup

1. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Navigate to http://127.0.0.1:5000

## What You'll See

1. **Homepage:** Overview of features and model status
2. **Upload Page:** Drag & drop or select images for analysis
3. **Results Page:** View generated captions and segmentation maps
4. **API Endpoints:** Available at `/api/v1/` for programmatic access

## Features

- **Image Captioning:** Generate natural language descriptions
- **Image Segmentation:** Identify and color-code object regions  
- **Multi-task Analysis:** Combined captioning and segmentation
- **Web Interface:** User-friendly drag & drop interface
- **REST API:** Programmatic access to AI capabilities

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

Maximum file size: 16MB

## API Usage Examples

### Generate Caption
```python
import requests

files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/v1/caption', files=files)
result = response.json()
print(result['data']['caption'])
```

### Perform Segmentation
```python
import requests

files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/v1/segment', files=files)
result = response.json()
print(result['data']['segmentation_image'])
```

### Combined Analysis
```python
import requests

files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/api/v1/analyze', files=files)
result = response.json()
print(result['data'])
```

## Troubleshooting

### Models Not Loading
- Wait a few moments for models to initialize
- Check the model status on the homepage
- Refresh the page if models seem stuck

### Upload Issues
- Ensure file is a supported image format
- Check file size is under 16MB
- Try a different image if one fails

### Performance
- First run may be slower as models load
- GPU acceleration will be used if available
- Close other applications if running slowly

## Development

To modify the application:

1. **Model Implementation:** Update files in `src/models/`
2. **UI Changes:** Modify templates in `templates/`
3. **API Endpoints:** Edit `app.py`
4. **Configuration:** Update files in `config/`

## Production Deployment

For production deployment, see the Docker configuration:

```bash
docker-compose up -d
```

Or deploy manually with:

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 2 app:app
```

---

**Need Help?** 
- Check the console output for error messages
- Verify all dependencies are installed
- Ensure Python 3.8+ is being used
