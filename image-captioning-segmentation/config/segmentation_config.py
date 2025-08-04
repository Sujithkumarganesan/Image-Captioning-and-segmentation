from .base_config import BaseConfig


class SegmentationConfig(BaseConfig):
    """Configuration for image segmentation models."""
    
    # Model architecture
    MODEL_NAME = "deeplabv3plus"
    BACKBONE = "resnet50"
    
    # Supported models
    SUPPORTED_MODELS = [
        "deeplabv3plus",
        "unet", 
        "maskrcnn",
        "fcn",
        "pspnet"
    ]
    
    # Backbone options
    SUPPORTED_BACKBONES = [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "resnext50", "resnext101",
        "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
        "mobilenet_v2"
    ]
    
    # Model specific settings
    # DeepLabV3+
    DEEPLABV3_OUTPUT_STRIDE = 16
    DEEPLABV3_PRETRAINED_BACKBONE = True
    
    # U-Net
    UNET_ENCODER_DEPTH = 5
    UNET_DECODER_CHANNELS = [256, 128, 64, 32, 16]
    
    # Mask R-CNN
    MASKRCNN_MIN_SIZE = 800
    MASKRCNN_MAX_SIZE = 1333
    MASKRCNN_ANCHOR_SIZES = [32, 64, 128, 256, 512]
    
    # Dataset specific
    NUM_CLASSES = 21  # PASCAL VOC
    CLASS_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    
    # COCO classes (if using COCO dataset)
    COCO_NUM_CLASSES = 80
    
    # Training parameters
    IGNORE_INDEX = 255
    SEGMENTATION_LOSS_WEIGHT = 1.0
    
    # Loss functions
    LOSS_TYPE = "cross_entropy"  # cross_entropy, focal, dice, combined
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    DICE_SMOOTH = 1e-5
    
    # Data augmentation
    AUGMENTATION = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.1,
        "rotation": 15,
        "scale": (0.8, 1.2),
        "crop_size": (512, 512),
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "gaussian_blur": 0.1,
        "gaussian_noise": 0.1
    }
    
    # Post-processing
    USE_CRF = False  # Conditional Random Field post-processing
    CRF_ITERATIONS = 10
    
    # Evaluation metrics
    METRICS = ["miou", "pixel_accuracy", "mean_accuracy", "dice_score"]
    
    # Visualization
    COLORMAP = "pascal"  # pascal, cityscapes, ade20k
    ALPHA = 0.5  # Overlay transparency
    
    # Model checkpoints
    PRETRAINED_SEGMENTATION_MODEL = BaseConfig.PRETRAINED_DIR / "segmentation_pretrained.pth"
    SEGMENTATION_CHECKPOINT = BaseConfig.CHECKPOINT_DIR / "segmentation_best.pth"
    
    # Inference settings
    TEST_TIME_AUGMENTATION = False
    MULTI_SCALE_INFERENCE = False
    SCALES = [0.75, 1.0, 1.25, 1.5]
    
    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration."""
        return {
            "model_name": cls.MODEL_NAME,
            "backbone": cls.BACKBONE,
            "num_classes": cls.NUM_CLASSES,
            "ignore_index": cls.IGNORE_INDEX,
            "loss_type": cls.LOSS_TYPE,
            "use_crf": cls.USE_CRF
        }
    
    @classmethod
    def get_class_colors(cls):
        """Get enhanced color map for better instance visualization."""
        import numpy as np
        
        # Enhanced color palette with more distinct colors for better visualization
        colors = [
            [0, 0, 0],        # 0: background - black
            [128, 0, 0],      # 1: aeroplane - dark red
            [0, 128, 0],      # 2: bicycle - dark green
            [128, 128, 0],    # 3: bird - olive
            [0, 0, 128],      # 4: boat - dark blue
            [255, 0, 255],    # 5: bottle - magenta
            [0, 255, 255],    # 6: bus - cyan
            [255, 255, 0],    # 7: car - yellow
            [255, 165, 0],    # 8: cat - orange
            [0, 100, 255],    # 9: chair - bright blue
            [128, 0, 128],    # 10: cow - purple
            [255, 192, 203],  # 11: diningtable - pink
            [0, 255, 0],      # 12: dog - bright green
            [255, 20, 147],   # 13: horse - deep pink
            [30, 144, 255],   # 14: motorbike - dodger blue
            [255, 0, 0],      # 15: person - bright red
            [50, 205, 50],    # 16: pottedplant - lime green
            [255, 218, 185],  # 17: sheep - peach
            [75, 0, 130],     # 18: sofa - indigo
            [255, 140, 0],    # 19: train - dark orange
            [220, 20, 60]     # 20: tvmonitor - crimson
        ]
        
        return np.array(colors[:cls.NUM_CLASSES])
    
    @classmethod
    def get_instance_colors(cls, max_instances=10):
        """Get distinct colors for instance segmentation."""
        import numpy as np
        
        # High-contrast colors for instance segmentation (like your reference image)
        instance_colors = [
            [255, 0, 0],      # Bright red
            [0, 255, 0],      # Bright green  
            [0, 0, 255],      # Bright blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
            [255, 192, 203],  # Pink
            [0, 128, 128],    # Teal
        ]
        
        return np.array(instance_colors[:max_instances])
