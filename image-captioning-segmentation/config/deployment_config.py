import os
from .base_config import BaseConfig


class DeploymentConfig(BaseConfig):
    """Configuration for deployment and web application."""
    
    # Flask configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # Server configuration
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
    THREADED = True
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = BaseConfig.UPLOADS_DIR
    OUTPUT_FOLDER = BaseConfig.OUTPUTS_DIR
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
    
    # API rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "redis://localhost:6379"
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:5000"]
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE"]
    CORS_HEADERS = ["Content-Type", "Authorization"]
    
    # Session configuration
    SESSION_TYPE = "filesystem"
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_FILE_DIR = BaseConfig.PROJECT_ROOT / "sessions"
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    }
    
    # API endpoints
    API_PREFIX = "/api/v1"
    API_ENDPOINTS = {
        "caption": f"{API_PREFIX}/caption",
        "segment": f"{API_PREFIX}/segment",
        "analyze": f"{API_PREFIX}/analyze",
        "health": f"{API_PREFIX}/health",
        "models": f"{API_PREFIX}/models"
    }
    
    # Model serving configuration
    MODEL_SERVING = {
        "captioning": {
            "enabled": True,
            "model_path": BaseConfig.CHECKPOINT_DIR / "captioning_best.pth",
            "device": BaseConfig.DEVICE,
            "batch_size": 1,
            "max_length": 50
        },
        "segmentation": {
            "enabled": True,
            "model_path": BaseConfig.CHECKPOINT_DIR / "segmentation_best.pth",
            "device": BaseConfig.DEVICE,
            "batch_size": 1,
            "num_classes": 21
        },
        "integrated": {
            "enabled": True,
            "model_path": BaseConfig.CHECKPOINT_DIR / "integrated_best.pth",
            "device": BaseConfig.DEVICE,
            "batch_size": 1
        }
    }
    
    # Caching configuration
    CACHE_TYPE = "simple"  # simple, redis, memcached
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    CACHE_KEY_PREFIX = "img_cap_seg_"
    
    # Redis configuration (if using Redis cache)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Database configuration (if needed)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Logging configuration
    LOG_TO_STDOUT = True
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BaseConfig.PROJECT_ROOT / "logs" / "app.log"
    LOG_MAX_BYTES = 10485760  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Monitoring and metrics
    METRICS_ENABLED = True
    METRICS_PORT = 9090
    PROMETHEUS_MULTIPROC_DIR = BaseConfig.PROJECT_ROOT / "metrics"
    
    # Background task configuration
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    
    # Model inference settings
    INFERENCE_TIMEOUT = 30  # seconds
    MAX_CONCURRENT_REQUESTS = 10
    MODEL_WARMUP = True  # Warm up models on startup
    
    # Image processing limits
    MAX_IMAGE_DIMENSION = 2048
    MIN_IMAGE_DIMENSION = 32
    SUPPORTED_FORMATS = ["JPEG", "PNG", "BMP", "TIFF", "WEBP"]
    
    # Output configuration
    OUTPUT_FORMAT = "json"  # json, xml, html
    INCLUDE_CONFIDENCE_SCORES = True
    SAVE_INTERMEDIATE_RESULTS = False
    
    # Error handling
    FRIENDLY_ERROR_MESSAGES = True
    DETAILED_ERROR_LOGS = True
    ERROR_EMAIL_ENABLED = False
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
    
    # Docker deployment
    DOCKER_ENABLED = os.getenv("DOCKER_ENABLED", "false").lower() == "true"
    CONTAINER_NAME = "image-caption-seg-app"
    
    # Health check configuration
    HEALTH_CHECK_ENABLED = True
    HEALTH_CHECK_ENDPOINT = "/health"
    HEALTH_CHECK_INTERVAL = 30  # seconds
    
    @classmethod
    def get_flask_config(cls):
        """Get Flask-specific configuration."""
        return {
            "SECRET_KEY": cls.SECRET_KEY,
            "DEBUG": cls.DEBUG,
            "MAX_CONTENT_LENGTH": cls.MAX_CONTENT_LENGTH,
            "UPLOAD_FOLDER": str(cls.UPLOAD_FOLDER),
            "SESSION_TYPE": cls.SESSION_TYPE,
            "SESSION_PERMANENT": cls.SESSION_PERMANENT,
            "SESSION_USE_SIGNER": cls.SESSION_USE_SIGNER,
            "SESSION_FILE_DIR": str(cls.SESSION_FILE_DIR)
        }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories for deployment."""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.OUTPUT_FOLDER,
            cls.SESSION_FILE_DIR,
            cls.LOG_FILE.parent,
            cls.PROMETHEUS_MULTIPROC_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_production(cls):
        """Check if running in production mode."""
        return cls.FLASK_ENV == "production"
