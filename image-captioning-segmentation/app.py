#!/usr/bin/env python3
"""
Image Captioning and Segmentation Web Application

A Flask-based web application that provides image captioning and segmentation
capabilities through a user-friendly web interface and REST API.
"""

import os
import sys
import logging
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import torch
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config import DeploymentConfig, BaseConfig
from src.models.model_manager import ModelManager
from src.utils.image_utils import ImageProcessor
from src.utils.response_utils import create_response, handle_error

# Initialize Flask app
app = Flask(__name__)
app.config.update(DeploymentConfig.get_flask_config())

# Enable CORS
CORS(app, origins=DeploymentConfig.CORS_ORIGINS)

# Setup logging
logging.basicConfig(
    level=getattr(logging, DeploymentConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DeploymentConfig.LOG_FILE) if DeploymentConfig.LOG_FILE.parent.exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
model_manager = ModelManager()
image_processor = ImageProcessor()

# Global variables for model loading status
models_loaded = False
loading_status = {"captioning": False, "segmentation": False, "integrated": False}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in DeploymentConfig.ALLOWED_EXTENSIONS


def load_models():
    """Load models in background."""
    global models_loaded, loading_status
    
    try:
        logger.info("Starting model loading...")
        
        # Load captioning model
        if DeploymentConfig.MODEL_SERVING["captioning"]["enabled"]:
            logger.info("Loading captioning model...")
            model_manager.load_captioning_model()
            loading_status["captioning"] = True
            logger.info("Captioning model loaded successfully")
        
        # Load segmentation model  
        if DeploymentConfig.MODEL_SERVING["segmentation"]["enabled"]:
            logger.info("Loading segmentation model...")
            model_manager.load_segmentation_model()
            loading_status["segmentation"] = True
            logger.info("Segmentation model loaded successfully")
            
        # Load integrated model
        if DeploymentConfig.MODEL_SERVING["integrated"]["enabled"]:
            logger.info("Loading integrated model...")
            model_manager.load_integrated_model()
            loading_status["integrated"] = True
            logger.info("Integrated model loaded successfully")
        
        models_loaded = True
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', 
                         models_loaded=models_loaded,
                         loading_status=loading_status)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing."""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = DeploymentConfig.UPLOAD_FOLDER / filename
            file.save(str(filepath))
            
            # Get processing options
            task = request.form.get('task', 'both')
            
            try:
                # Process image
                results = process_image(str(filepath), task)
                
                return render_template('results.html', 
                                     results=results,
                                     filename=filename,
                                     task=task)
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}')
                return redirect(url_for('index'))
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    
    return render_template('upload.html')


def process_image(image_path, task='both'):
    """Process image with specified task."""
    if not models_loaded:
        raise Exception("Models are still loading. Please wait.")
    
    results = {}
    
    # Load image as PIL Image first
    pil_image = image_processor.load_image(image_path)
    
    try:
        if task in ['captioning', 'both']:
            if loading_status["captioning"]:
                # For captioning, pass PIL image directly to avoid conversion issues
                caption = model_manager.generate_caption_from_pil(pil_image)
                results['caption'] = caption
            else:
                results['caption'] = "Captioning model not available"
        
        if task in ['segmentation', 'both']:
            if loading_status["segmentation"]:
                try:
                    # For segmentation, use preprocessed tensor
                    processed_image = image_processor.preprocess_image(pil_image)
                    logger.debug(f"Processed image shape: {processed_image.shape}")
                    
                    segmentation_map = model_manager.segment_image(processed_image)
                    logger.debug(f"Segmentation map shape: {segmentation_map.shape}, dtype: {segmentation_map.dtype}")
                    
                    # Save clean segmentation result
                    seg_filename = f"seg_{Path(image_path).stem}.png"
                    seg_path = DeploymentConfig.OUTPUT_FOLDER / seg_filename
                    image_processor.save_segmentation(segmentation_map, str(seg_path))
                    results['segmentation'] = seg_filename
                    
                    logger.debug(f"Segmentation saved as: {seg_filename}")
                except Exception as seg_error:
                    logger.error(f"Error in segmentation processing: {str(seg_error)}")
                    results['segmentation'] = f"Segmentation failed: {str(seg_error)}"
            else:
                results['segmentation'] = "Segmentation model not available"
        
        # If integrated model is available and both tasks requested
        if task == 'both' and loading_status["integrated"]:
            processed_image = image_processor.preprocess_image(pil_image)
            integrated_results = model_manager.analyze_image(processed_image)
            results.update(integrated_results)
    
    except Exception as e:
        logger.error(f"Error in image processing: {str(e)}")
        raise
    
    return results


# API Routes
@app.route('/api/v1/caption', methods=['POST'])
def api_caption():
    """API endpoint for image captioning."""
    try:
        if not models_loaded or not loading_status["captioning"]:
            return create_response(False, "Captioning model not available", None, 503)
        
        if 'image' not in request.files:
            return create_response(False, "No image provided", None, 400)
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return create_response(False, "Invalid file type", None, 400)
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = DeploymentConfig.UPLOAD_FOLDER / f"temp_{filename}"
        file.save(str(temp_path))
        
        # Process image
        pil_image = image_processor.load_image(str(temp_path))
        caption = model_manager.generate_caption_from_pil(pil_image)
        
        # Clean up
        temp_path.unlink()
        
        return create_response(True, "Caption generated successfully", {"caption": caption})
    
    except Exception as e:
        return handle_error(e, "Error generating caption")


@app.route('/api/v1/segment', methods=['POST'])
def api_segment():
    """API endpoint for image segmentation."""
    try:
        if not models_loaded or not loading_status["segmentation"]:
            return create_response(False, "Segmentation model not available", None, 503)
        
        if 'image' not in request.files:
            return create_response(False, "No image provided", None, 400)
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return create_response(False, "Invalid file type", None, 400)
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = DeploymentConfig.UPLOAD_FOLDER / f"temp_{filename}"
        file.save(str(temp_path))
        
        # Process image
        image = image_processor.load_image(str(temp_path))
        processed_image = image_processor.preprocess_image(image)
        segmentation_map = model_manager.segment_image(processed_image)
        
        # Save result
        result_filename = f"seg_{Path(filename).stem}.png"
        result_path = DeploymentConfig.OUTPUT_FOLDER / result_filename
        image_processor.save_segmentation(segmentation_map, str(result_path))
        
        # Clean up
        temp_path.unlink()
        
        return create_response(True, "Segmentation completed successfully", 
                             {"segmentation_image": result_filename})
    
    except Exception as e:
        return handle_error(e, "Error performing segmentation")


@app.route('/api/v1/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for combined analysis."""
    try:
        if not models_loaded:
            return create_response(False, "Models not available", None, 503)
        
        if 'image' not in request.files:
            return create_response(False, "No image provided", None, 400)
        
        file = request.files['image']
        if not allowed_file(file.filename):
            return create_response(False, "Invalid file type", None, 400)
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = DeploymentConfig.UPLOAD_FOLDER / f"temp_{filename}"
        file.save(str(temp_path))
        
        # Process image
        results = process_image(str(temp_path), 'both')
        
        # Clean up
        temp_path.unlink()
        
        return create_response(True, "Analysis completed successfully", results)
    
    except Exception as e:
        return handle_error(e, "Error performing analysis")


@app.route('/api/v1/health')
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy" if models_loaded else "loading",
        "models": loading_status,
        "device": str(BaseConfig.DEVICE),
        "version": "0.1.0"
    }
    return jsonify(status)


@app.route('/outputs/<filename>')
def uploaded_file(filename):
    """Serve uploaded and output files."""
    return send_from_directory(str(DeploymentConfig.OUTPUT_FOLDER), filename)


@app.route('/uploads/<filename>')
def output_file(filename):
    """Serve uploaded files."""
    return send_from_directory(str(DeploymentConfig.UPLOAD_FOLDER), filename)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error."""
    return create_response(False, "File too large. Maximum size is 16MB.", None, 413)


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('500.html'), 500


def open_browser():
    """Open browser after a delay."""
    url = f"http://{DeploymentConfig.HOST}:{DeploymentConfig.PORT}"
    webbrowser.open(url)


def main():
    """Main entry point."""
    # Setup directories
    DeploymentConfig.setup_directories()
    
    # Start model loading in background
    from threading import Thread
    loading_thread = Thread(target=load_models, daemon=True)
    loading_thread.start()
    
    url = f"http://{DeploymentConfig.HOST}:{DeploymentConfig.PORT}"
    
    # Don't auto-open browser, just show clickable link
    print(f"\n{'='*70}")
    print("🚀 IMAGE CAPTIONING & SEGMENTATION SERVER")
    print(f"{'='*70}")
    print(f"📍 Server Status: STARTING...")
    print(f"🔧 Environment: {DeploymentConfig.FLASK_ENV}")
    print(f"🖥️  Device: {BaseConfig.DEVICE}")
    print(f"{'='*70}")
    print("⏳ Loading AI models in background...")
    print(f"{'='*70}")
    print()
    print("🌐 APPLICATION URL:")
    print(f"   {url}")
    print()
    print("📱 CLICK THE LINK ABOVE TO OPEN IN CHROME")
    print("   Or copy and paste the URL into your browser")
    print()
    print(f"{'='*70}")
    print("📊 Server Logs:")
    print(f"{'='*70}")
    
    # Run the app without auto-opening browser
    app.run(
        host=DeploymentConfig.HOST,
        port=DeploymentConfig.PORT,
        debug=DeploymentConfig.DEBUG,
        threaded=DeploymentConfig.THREADED,
        use_reloader=False  # Prevent double startup in debug mode
    )


if __name__ == '__main__':
    main()
