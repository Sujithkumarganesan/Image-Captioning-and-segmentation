from .base_config import BaseConfig
from .captioning_config import CaptioningConfig
from .segmentation_config import SegmentationConfig


class IntegratedConfig(BaseConfig):
    """Configuration for integrated captioning and segmentation model."""
    
    # Model architecture
    MODEL_NAME = "integrated_caption_segment"
    SHARED_BACKBONE = "resnet50"
    
    # Multi-task learning settings
    TASK_WEIGHTS = {
        "captioning": 0.5,
        "segmentation": 0.5
    }
    
    # Shared encoder settings
    SHARED_ENCODER_DIM = 2048
    FEATURE_FUSION_METHOD = "concat"  # concat, add, attention
    
    # Task-specific heads
    CAPTIONING_HEAD_DIM = 512
    SEGMENTATION_HEAD_DIM = 256
    
    # Attention mechanism for multi-task
    USE_CROSS_ATTENTION = True
    ATTENTION_HEADS = 8
    ATTENTION_DIM = 512
    
    # Loss balancing
    LOSS_BALANCING_METHOD = "uncertainty"  # fixed, uncertainty, gradnorm
    UNCERTAINTY_WEIGHT_LEARNING = True
    
    # Gradient balancing
    GRADNORM_ALPHA = 0.16
    
    # Training strategy
    TRAINING_STRATEGY = "joint"  # joint, alternating, curriculum
    
    # Curriculum learning (if used)
    CURRICULUM_EPOCHS = {
        "captioning_only": 20,
        "segmentation_only": 20,
        "joint_training": 60
    }
    
    # Feature sharing levels
    SHARE_ENCODER = True
    SHARE_DECODER = False
    SHARE_ATTENTION = True
    
    # Model ensemble
    USE_ENSEMBLE = False
    ENSEMBLE_WEIGHTS = [0.6, 0.4]  # [captioning_weight, segmentation_weight]
    
    # Data loading for multi-task
    MULTITASK_BATCH_SIZE = BaseConfig.BATCH_SIZE
    TASK_SAMPLING_STRATEGY = "round_robin"  # round_robin, random, weighted
    
    # Evaluation
    EVAL_BOTH_TASKS = True
    EVAL_FREQUENCY = 5  # Every N epochs
    
    # Model checkpoints
    INTEGRATED_CHECKPOINT = BaseConfig.CHECKPOINT_DIR / "integrated_best.pth"
    TASK_SPECIFIC_CHECKPOINTS = {
        "captioning": BaseConfig.CHECKPOINT_DIR / "integrated_captioning.pth",
        "segmentation": BaseConfig.CHECKPOINT_DIR / "integrated_segmentation.pth"
    }
    
    # Inference modes
    INFERENCE_MODE = "both"  # captioning, segmentation, both
    
    # Visualization for integrated model
    SHOW_ATTENTION_MAPS = True
    ATTENTION_MAP_LAYER = -1  # Last layer
    
    @classmethod
    def get_task_config(cls, task: str):
        """Get configuration for specific task."""
        if task == "captioning":
            config = CaptioningConfig.get_model_config()
            config.update({
                "task_weight": cls.TASK_WEIGHTS["captioning"],
                "head_dim": cls.CAPTIONING_HEAD_DIM
            })
            return config
        elif task == "segmentation":
            config = SegmentationConfig.get_model_config()
            config.update({
                "task_weight": cls.TASK_WEIGHTS["segmentation"],
                "head_dim": cls.SEGMENTATION_HEAD_DIM
            })
            return config
        else:
            raise ValueError(f"Unknown task: {task}")
    
    @classmethod
    def get_integrated_config(cls):
        """Get integrated model configuration."""
        return {
            "model_name": cls.MODEL_NAME,
            "shared_backbone": cls.SHARED_BACKBONE,
            "shared_encoder_dim": cls.SHARED_ENCODER_DIM,
            "feature_fusion_method": cls.FEATURE_FUSION_METHOD,
            "use_cross_attention": cls.USE_CROSS_ATTENTION,
            "attention_heads": cls.ATTENTION_HEADS,
            "attention_dim": cls.ATTENTION_DIM,
            "task_weights": cls.TASK_WEIGHTS,
            "loss_balancing_method": cls.LOSS_BALANCING_METHOD,
            "training_strategy": cls.TRAINING_STRATEGY
        }
    
    @classmethod
    def update_task_weights(cls, captioning_weight: float, segmentation_weight: float):
        """Update task weights ensuring they sum to 1."""
        total = captioning_weight + segmentation_weight
        cls.TASK_WEIGHTS = {
            "captioning": captioning_weight / total,
            "segmentation": segmentation_weight / total
        }
