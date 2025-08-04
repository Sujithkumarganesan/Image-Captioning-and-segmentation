"""
Response Utilities

Handles API response formatting and error handling.
"""

import logging
import traceback
from typing import Any, Dict, Optional
from flask import jsonify

logger = logging.getLogger(__name__)


def create_response(success: bool, 
                   message: str, 
                   data: Optional[Any] = None, 
                   status_code: int = 200) -> tuple:
    """Create standardized API response."""
    response = {
        'success': success,
        'message': message,
        'timestamp': None,  # You can add timestamp if needed
    }
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), status_code


def handle_error(error: Exception, 
                custom_message: str = "An error occurred",
                status_code: int = 500) -> tuple:
    """Handle and format errors for API responses."""
    error_message = str(error)
    
    # Log the full traceback for debugging
    logger.error(f"{custom_message}: {error_message}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Don't expose internal error details in production
    from config import DeploymentConfig
    if DeploymentConfig.is_production():
        error_message = custom_message
    else:
        error_message = f"{custom_message}: {error_message}"
    
    return create_response(
        success=False,
        message=error_message,
        status_code=status_code
    )


def validate_file_upload(file) -> Dict[str, Any]:
    """Validate uploaded file."""
    from config import DeploymentConfig
    
    validation_result = {
        'valid': True,
        'errors': []
    }
    
    # Check if file exists
    if not file or not file.filename:
        validation_result['valid'] = False
        validation_result['errors'].append("No file provided")
        return validation_result
    
    # Check file extension
    allowed_extensions = DeploymentConfig.ALLOWED_EXTENSIONS
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        validation_result['valid'] = False
        validation_result['errors'].append(
            f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Additional file size check (Flask handles this automatically, but we can add custom logic)
    try:
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        max_size = DeploymentConfig.MAX_CONTENT_LENGTH
        if file_size > max_size:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
            )
    except Exception as e:
        logger.warning(f"Could not check file size: {str(e)}")
    
    return validation_result


def format_model_response(results: Dict[str, Any], 
                         include_metadata: bool = True) -> Dict[str, Any]:
    """Format model response with metadata."""
    formatted_response = {
        'results': results
    }
    
    if include_metadata:
        from config import BaseConfig
        formatted_response['metadata'] = {
            'model_device': str(BaseConfig.DEVICE),
            'image_size': BaseConfig.IMAGE_SIZE,
            'version': '0.1.0'
        }
    
    return formatted_response


def create_error_response(error_type: str, 
                         error_message: str,
                         status_code: int = 400) -> tuple:
    """Create error response with error type."""
    return create_response(
        success=False,
        message=f"{error_type}: {error_message}",
        status_code=status_code
    )
