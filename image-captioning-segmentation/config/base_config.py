import os
from typing import Dict, Any
from pathlib import Path


class BaseConfig:
    """Base configuration class with common settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    STATIC_DIR = PROJECT_ROOT / "static"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    
    # Model paths
    CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
    PRETRAINED_DIR = MODELS_DIR / "pretrained"
    
    # Data subdirectories
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SPLITS_DIR = DATA_DIR / "splits"
    
    # Static subdirectories
    UPLOADS_DIR = STATIC_DIR / "uploads"
    OUTPUTS_DIR = STATIC_DIR / "outputs"
    CSS_DIR = STATIC_DIR / "css"
    JS_DIR = STATIC_DIR / "js"
    
    # Ensure all directories exist
    for dir_path in [CHECKPOINT_DIR, PRETRAINED_DIR, RAW_DATA_DIR, 
                     PROCESSED_DATA_DIR, SPLITS_DIR, UPLOADS_DIR, 
                     OUTPUTS_DIR, CSS_DIR, JS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Common model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Training parameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    PATIENCE = 10
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Random seeds
    RANDOM_SEED = 42
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        config = {}
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                config[key] = value
        return config
    
    @classmethod
    def update_from_env(cls):
        """Update configuration from environment variables."""
        # Update device based on CUDA availability
        try:
            import torch
            cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            cls.DEVICE = "cpu"
        
        # Update other configs from environment
        if os.getenv("BATCH_SIZE"):
            cls.BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
        
        if os.getenv("LEARNING_RATE"):
            cls.LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
        
        if os.getenv("EPOCHS"):
            cls.EPOCHS = int(os.getenv("EPOCHS"))
