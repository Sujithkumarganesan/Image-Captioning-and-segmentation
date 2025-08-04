"""
Image Processing Utilities

Handles image loading, preprocessing, and postprocessing operations.
"""

import logging
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from pathlib import Path

# Add project root to path for config imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import BaseConfig, DataConfig, SegmentationConfig

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles all image processing operations."""
    
    def __init__(self):
        self.device = BaseConfig.DEVICE
        self.image_size = DataConfig.IMAGE_SIZE
        self.image_mean = DataConfig.IMAGE_MEAN
        self.image_std = DataConfig.IMAGE_STD
        
        # Setup transforms
        self.preprocess_transform = transforms.Compose([
            transforms.Resize(DataConfig.RESIZE_SIZE),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_mean, std=self.image_std)
        ])
        
        self.postprocess_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-m/s for m, s in zip(self.image_mean, self.image_std)],
                std=[1/s for s in self.image_std]
            ),
            transforms.ToPILImage()
        ])
        
        logger.info("ImageProcessor initialized")
    
    def load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """Load image from path or return if already PIL Image."""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image_path, Image.Image):
                image = image_path.convert('RGB')
            else:
                raise ValueError("Input must be file path or PIL Image")
            
            logger.debug(f"Image loaded with size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference."""
        try:
            # Apply preprocessing transforms
            processed_image = self.preprocess_transform(image)
            logger.debug(f"Image preprocessed to shape: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image."""
        try:
            # Remove batch dimension if present
            if len(tensor.shape) == 4:
                tensor = tensor.squeeze(0)
            
            # Apply postprocessing transforms
            image = self.postprocess_transform(tensor.cpu())
            return image
            
        except Exception as e:
            logger.error(f"Error postprocessing image: {str(e)}")
            raise
    
    def save_segmentation(self, segmentation_map: np.ndarray, save_path: str) -> None:
        """Save segmentation map as professional clean colored image."""
        try:
            # Ensure segmentation map is 2D
            if len(segmentation_map.shape) == 3:
                segmentation_map = segmentation_map.squeeze()
            
            # Define professional color palette with high contrast and clean colors
            professional_colors = [
                [0, 0, 0],          # 0: background (black)
                [255, 0, 0],        # 1: bright red
                [0, 255, 0],        # 2: bright green  
                [0, 0, 255],        # 3: bright blue
                [255, 255, 0],      # 4: yellow
                [255, 0, 255],      # 5: magenta
                [0, 255, 255],      # 6: cyan
                [255, 128, 0],      # 7: orange
                [128, 0, 255],      # 8: purple
                [255, 192, 203],    # 9: pink
                [0, 128, 0],        # 10: dark green
                [128, 128, 0],      # 11: olive
                [0, 0, 128],        # 12: navy
                [128, 0, 0],        # 13: maroon
                [0, 128, 128],      # 14: teal
                [255, 165, 0],      # 15: orange (for persons)
                [75, 0, 130],       # 16: indigo
                [255, 20, 147],     # 17: deep pink
                [50, 205, 50],      # 18: lime green
                [220, 20, 60],      # 19: crimson
                [30, 144, 255]      # 20: dodger blue
            ]
            
            # Create colored segmentation image with black background
            height, width = segmentation_map.shape
            colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get unique classes in the segmentation map
            unique_classes = np.unique(segmentation_map)
            logger.debug(f"Unique classes found: {unique_classes}")
            
            # Create clean, solid colored regions for each detected object
            instance_counter = 1
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                
                mask = segmentation_map == class_id
                
                # Use bright, distinct colors for each instance
                color_idx = instance_counter % len(professional_colors)
                if color_idx == 0:  # Skip black (background)
                    color_idx = 1
                    
                colored_segmentation[mask] = professional_colors[color_idx]
                instance_counter += 1
            
            # Apply morphological operations for cleaner edges
            from scipy import ndimage
            
            # Clean up the segmentation with morphological operations
            for class_id in unique_classes:
                if class_id == 0:
                    continue
                    
                # Get mask for this class
                mask = segmentation_map == class_id
                
                # Apply closing to fill holes
                mask_cleaned = ndimage.binary_closing(mask, structure=np.ones((5,5)))
                
                # Apply opening to smooth edges
                mask_cleaned = ndimage.binary_opening(mask_cleaned, structure=np.ones((3,3)))
                
                # Update the colored segmentation
                color_idx = (np.where(unique_classes == class_id)[0][0] % len(professional_colors))
                if color_idx == 0:
                    color_idx = 1
                    
                # Clear the old mask and apply the cleaned one
                old_mask = segmentation_map == class_id
                colored_segmentation[old_mask] = [0, 0, 0]  # Clear old
                colored_segmentation[mask_cleaned] = professional_colors[color_idx]  # Apply cleaned
            
            # Save as PIL Image with high quality
            segmentation_image = Image.fromarray(colored_segmentation, 'RGB')
            
            # Apply a slight gaussian blur for smoother edges
            from PIL import ImageFilter
            segmentation_image = segmentation_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            segmentation_image.save(save_path, 'PNG', quality=95)
            
            logger.debug(f"Professional segmentation saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving segmentation: {str(e)}")
            # Fallback to simple save if advanced processing fails
            try:
                height, width = segmentation_map.shape
                colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
                unique_classes = np.unique(segmentation_map)
                
                simple_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
                
                for i, class_id in enumerate(unique_classes):
                    if class_id == 0:
                        continue
                    mask = segmentation_map == class_id
                    color_idx = i % len(simple_colors)
                    colored_segmentation[mask] = simple_colors[color_idx]
                
                segmentation_image = Image.fromarray(colored_segmentation, 'RGB')
                segmentation_image.save(save_path)
                logger.debug(f"Fallback segmentation saved to: {save_path}")
                
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {str(fallback_error)}")
                raise
    
    def create_overlay(self, original_image: Image.Image, 
                      segmentation_map: np.ndarray,
                      alpha: float = 0.5) -> Image.Image:
        """Create overlay of original image and segmentation."""
        try:
            # Resize original image to match segmentation map
            seg_height, seg_width = segmentation_map.shape
            original_resized = original_image.resize((seg_width, seg_height), Image.LANCZOS)
            
            # Get colored segmentation
            colors = SegmentationConfig.get_class_colors()
            colored_segmentation = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)
            
            for class_id in range(len(colors)):
                mask = segmentation_map == class_id
                colored_segmentation[mask] = colors[class_id]
            
            # Create overlay
            segmentation_image = Image.fromarray(colored_segmentation, 'RGB')
            overlay = Image.blend(original_resized, segmentation_image, alpha)
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            raise
    
    def add_caption_to_image(self, image: Image.Image, 
                           caption: str,
                           font_size: int = 20,
                           position: Tuple[int, int] = (10, 10)) -> Image.Image:
        """Add caption text to image."""
        try:
            # Create a copy of the image
            image_with_caption = image.copy()
            draw = ImageDraw.Draw(image_with_caption)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Add text with background for better visibility
            text_bbox = draw.textbbox(position, caption, font=font)
            padding = 5
            
            # Draw background rectangle
            draw.rectangle([
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding
            ], fill=(0, 0, 0, 128))
            
            # Draw text
            draw.text(position, caption, fill=(255, 255, 255), font=font)
            
            return image_with_caption
            
        except Exception as e:
            logger.error(f"Error adding caption to image: {str(e)}")
            raise
    
    def resize_image(self, image: Image.Image, 
                    target_size: Tuple[int, int],
                    maintain_aspect_ratio: bool = True) -> Image.Image:
        """Resize image to target size."""
        try:
            if maintain_aspect_ratio:
                image.thumbnail(target_size, Image.LANCZOS)
                return image
            else:
                return image.resize(target_size, Image.LANCZOS)
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise
    
    def validate_image(self, image_path: str) -> bool:
        """Validate if file is a valid image."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def get_image_info(self, image: Union[str, Image.Image]) -> dict:
        """Get information about the image."""
        try:
            if isinstance(image, str):
                image = Image.open(image)
            
            info = {
                'size': image.size,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'width': image.width,
                'height': image.height
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            raise
    
    def create_visualization_grid(self, original_image: Image.Image,
                                caption: str,
                                segmentation_map: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None) -> Image.Image:
        """Create a visualization grid showing original, caption, and segmentation."""
        try:
            # Resize original image for consistent display
            display_size = (512, 512)
            original_resized = original_image.resize(display_size, Image.LANCZOS)
            
            # Create image with caption
            captioned_image = self.add_caption_to_image(
                original_resized, caption, font_size=16, position=(10, 10)
            )
            
            if segmentation_map is not None:
                # Create overlay
                overlay_image = self.create_overlay(original_resized, segmentation_map, alpha=0.6)
                
                # Create a grid: original | captioned | overlay
                grid_width = display_size[0] * 3
                grid_height = display_size[1]
                grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
                
                grid_image.paste(original_resized, (0, 0))
                grid_image.paste(captioned_image, (display_size[0], 0))
                grid_image.paste(overlay_image, (display_size[0] * 2, 0))
                
            else:
                # Create a grid: original | captioned
                grid_width = display_size[0] * 2
                grid_height = display_size[1]
                grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
                
                grid_image.paste(original_resized, (0, 0))
                grid_image.paste(captioned_image, (display_size[0], 0))
            
            if save_path:
                grid_image.save(save_path)
                logger.info(f"Visualization grid saved to: {save_path}")
            
            return grid_image
            
        except Exception as e:
            logger.error(f"Error creating visualization grid: {str(e)}")
            raise
