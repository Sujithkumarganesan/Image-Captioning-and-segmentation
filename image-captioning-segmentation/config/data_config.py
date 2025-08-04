from .base_config import BaseConfig
from pathlib import Path


class DataConfig(BaseConfig):
    """Configuration for datasets and data processing."""
    
    # Dataset types
    SUPPORTED_DATASETS = [
        "coco", "pascal_voc", "flickr30k", "flickr8k", 
        "visual_genome", "ade20k", "cityscapes"
    ]
    
    # COCO Dataset
    COCO_ROOT = BaseConfig.RAW_DATA_DIR / "coco"
    COCO_IMAGES = {
        "train": COCO_ROOT / "images" / "train2017",
        "val": COCO_ROOT / "images" / "val2017",
        "test": COCO_ROOT / "images" / "test2017"
    }
    COCO_ANNOTATIONS = {
        "captions_train": COCO_ROOT / "annotations" / "captions_train2017.json",
        "captions_val": COCO_ROOT / "annotations" / "captions_val2017.json",
        "instances_train": COCO_ROOT / "annotations" / "instances_train2017.json",
        "instances_val": COCO_ROOT / "annotations" / "instances_val2017.json",
        "panoptic_train": COCO_ROOT / "annotations" / "panoptic_train2017.json"
    }
    
    # PASCAL VOC Dataset
    PASCAL_ROOT = BaseConfig.RAW_DATA_DIR / "pascal_voc"
    PASCAL_IMAGES = PASCAL_ROOT / "JPEGImages"
    PASCAL_ANNOTATIONS = PASCAL_ROOT / "Annotations"
    PASCAL_SEGMENTATION = PASCAL_ROOT / "SegmentationClass"
    
    # Flickr30K Dataset
    FLICKR30K_ROOT = BaseConfig.RAW_DATA_DIR / "flickr30k"
    FLICKR30K_IMAGES = FLICKR30K_ROOT / "images"
    FLICKR30K_CAPTIONS = FLICKR30K_ROOT / "results_20130124.token"
    
    # Data splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Image preprocessing
    IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet means
    IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet stds
    IMAGE_SIZE = (224, 224)
    RESIZE_SIZE = (256, 256)
    
    # Data augmentation parameters
    TRAIN_AUGMENTATION = {
        "resize": RESIZE_SIZE,
        "random_crop": IMAGE_SIZE,
        "horizontal_flip": 0.5,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        },
        "random_rotation": 10,
        "gaussian_blur": 0.1,
        "normalize": {
            "mean": IMAGE_MEAN,
            "std": IMAGE_STD
        }
    }
    
    VAL_AUGMENTATION = {
        "resize": RESIZE_SIZE,
        "center_crop": IMAGE_SIZE,
        "normalize": {
            "mean": IMAGE_MEAN,
            "std": IMAGE_STD
        }
    }
    
    # Caption preprocessing
    MAX_CAPTION_LENGTH = 50
    MIN_CAPTION_LENGTH = 5
    VOCAB_THRESHOLD = 5  # Minimum word frequency
    
    # Text processing
    LOWERCASE = True
    REMOVE_PUNCTUATION = True
    REMOVE_NUMBERS = False
    
    # Data loading
    BATCH_SIZE = BaseConfig.BATCH_SIZE
    NUM_WORKERS = BaseConfig.NUM_WORKERS
    PIN_MEMORY = BaseConfig.PIN_MEMORY
    SHUFFLE_TRAIN = True
    DROP_LAST = True
    
    # Segmentation specific
    IGNORE_INDEX = 255
    VOID_CLASSES = [255]  # Classes to ignore during training
    
    # Multi-task data loading
    MULTITASK_SAMPLING = "balanced"  # balanced, caption_heavy, segment_heavy
    SAMPLING_RATIOS = {
        "captioning": 0.5,
        "segmentation": 0.5
    }
    
    # Cache settings
    USE_CACHE = True
    CACHE_DIR = BaseConfig.DATA_DIR / "cache"
    CACHE_SIZE = 1000  # Number of items to cache
    
    # Download URLs (for automatic dataset download)
    DOWNLOAD_URLS = {
        "coco_train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "coco_val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "coco_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "pascal_voc": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    }
    
    @classmethod
    def get_dataset_config(cls, dataset_name: str):
        """Get configuration for specific dataset."""
        if dataset_name == "coco":
            return {
                "root": cls.COCO_ROOT,
                "images": cls.COCO_IMAGES,
                "annotations": cls.COCO_ANNOTATIONS,
                "type": "captioning_segmentation"
            }
        elif dataset_name == "pascal_voc":
            return {
                "root": cls.PASCAL_ROOT,
                "images": cls.PASCAL_IMAGES,
                "annotations": cls.PASCAL_ANNOTATIONS,
                "segmentation": cls.PASCAL_SEGMENTATION,
                "type": "segmentation"
            }
        elif dataset_name == "flickr30k":
            return {
                "root": cls.FLICKR30K_ROOT,
                "images": cls.FLICKR30K_IMAGES,
                "captions": cls.FLICKR30K_CAPTIONS,
                "type": "captioning"
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @classmethod
    def create_data_directories(cls):
        """Create all necessary data directories."""
        directories = [
            cls.COCO_ROOT, cls.PASCAL_ROOT, cls.FLICKR30K_ROOT,
            cls.CACHE_DIR, cls.PROCESSED_DATA_DIR / "images",
            cls.PROCESSED_DATA_DIR / "captions", cls.PROCESSED_DATA_DIR / "masks",
            cls.PROCESSED_DATA_DIR / "features", cls.SPLITS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
