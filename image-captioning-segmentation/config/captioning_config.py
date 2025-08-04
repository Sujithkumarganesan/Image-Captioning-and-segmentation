from .base_config import BaseConfig


class CaptioningConfig(BaseConfig):
    """Configuration for image captioning models."""
    
    # Model architecture
    MODEL_NAME = "blip-image-captioning-base"
    ENCODER_NAME = "resnet50"
    DECODER_NAME = "transformer"
    
    # BLIP specific settings
    BLIP_MODEL_SIZE = "base"  # base, large
    BLIP_PRETRAINED = True
    
    # Vision Transformer settings
    VIT_MODEL_NAME = "vit-base-patch16-224"
    VIT_PRETRAINED = True
    
    # GPT settings for decoder
    GPT_MODEL_NAME = "gpt2"
    GPT_MAX_LENGTH = 50
    GPT_NUM_BEAMS = 5
    GPT_NO_REPEAT_NGRAM_SIZE = 2
    
    # Vocabulary settings
    VOCAB_SIZE = 10000
    MAX_CAPTION_LENGTH = 50
    MIN_CAPTION_LENGTH = 5
    
    # Special tokens
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    # Model dimensions
    ENCODER_DIM = 2048
    DECODER_DIM = 512
    ATTENTION_DIM = 512
    EMBEDDING_DIM = 512
    
    # Training specific
    TEACHER_FORCING_RATIO = 0.5
    CAPTION_LOSS_WEIGHT = 1.0
    
    # Evaluation metrics
    METRICS = ["bleu", "meteor", "rouge", "cider", "spice"]
    
    # Data augmentation for captioning
    AUGMENTATION = {
        "horizontal_flip": 0.5,
        "rotation": 10,
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }
    
    # Inference settings
    BEAM_SIZE = 5
    TEMPERATURE = 1.0
    TOP_K = 50
    TOP_P = 0.9
    
    # Model checkpoints
    PRETRAINED_CAPTIONING_MODEL = BaseConfig.PRETRAINED_DIR / "captioning_pretrained.pth"
    CAPTIONING_CHECKPOINT = BaseConfig.CHECKPOINT_DIR / "captioning_best.pth"
    
    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration."""
        return {
            "model_name": cls.MODEL_NAME,
            "encoder_dim": cls.ENCODER_DIM,
            "decoder_dim": cls.DECODER_DIM,
            "attention_dim": cls.ATTENTION_DIM,
            "embedding_dim": cls.EMBEDDING_DIM,
            "vocab_size": cls.VOCAB_SIZE,
            "max_length": cls.MAX_CAPTION_LENGTH,
            "beam_size": cls.BEAM_SIZE,
            "temperature": cls.TEMPERATURE
        }
