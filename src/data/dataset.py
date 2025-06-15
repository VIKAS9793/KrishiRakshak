"""
Custom PyTorch Dataset for loading plant disease images from CSV files.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional, Any, Callable, Union, List, cast
import logging
from torchvision.transforms import ToTensor
from pathlib import Path
import albumentations as A
from torchvision.transforms.functional import to_tensor
import mmap
import io

logger = logging.getLogger(__name__)


class PlantDiseaseDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Custom Dataset for loading plant disease images from CSV files."""
    
    def __init__(
        self,
        csv_file: str,
        data_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        class_mapping: Optional[Dict[str, int]] = None,
        cache_size: int = 1000,  # Cache size for images
        use_mmap: bool = True,  # Use memory mapping for faster loading
        preload: bool = False  # Preload all images into memory
    ):
        """Initialize dataset.
        
        Args:
            csv_file: Path to CSV file containing image paths and labels
            data_dir: Base directory for images (if paths in CSV are relative)
            transform: Optional transform to be applied on images
            class_mapping: Optional dictionary mapping class names to indices
            cache_size: Number of images to cache in memory
            use_mmap: Whether to use memory mapping for faster loading
            preload: Whether to preload all images into memory
        """
        self.csv_file = csv_file
        self.data_dir = Path(data_dir) if data_dir else Path(csv_file).parent
        self.transform = transform
        self.class_mapping = class_mapping or {}
        self.cache_size = cache_size
        self.use_mmap = use_mmap
        self.preload = preload
        
        # Read CSV file
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['image_path'].values
        self.labels = self.data['label'].values
        
        # Initialize cache
        self.cache = {}
        self.cache_order = []
        
        # Initialize images list
        self.images: List[Union[mmap.mmap, bytes]] = []
        
        # Preload images if requested
        if preload:
            logger.info("Preloading all images into memory...")
            for img_path in self.image_paths:
                full_path = self.data_dir / img_path
                with open(full_path, 'rb') as f:
                    if use_mmap:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        self.images.append(mm)
                    else:
                        self.images.append(f.read())
            logger.info("Preloading complete")
        
        # Validate paths
        self._validate_paths()
        
        logger.info(f"Loaded {len(self)} samples from {csv_file}")
        logger.info(f"Using data directory: {self.data_dir}")
        if self.class_mapping:
            logger.info(f"Using class mapping: {self.class_mapping}")
    
    def _validate_paths(self):
        """Validate that all image paths exist."""
        missing_files = []
        for idx, img_path in enumerate(self.image_paths):
            full_path = self.data_dir / img_path
            if not full_path.exists():
                missing_files.append((idx, str(full_path)))
        
        if missing_files:
            logger.error(f"Found {len(missing_files)} missing files:")
            for idx, path in missing_files[:5]:  # Log first 5 missing files
                logger.error(f"Missing file {idx}: {path}")
            if len(missing_files) > 5:
                logger.error(f"... and {len(missing_files) - 5} more missing files")
            raise FileNotFoundError(f"Found {len(missing_files)} missing image files")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def _load_image(self, idx: int) -> Image.Image:
        """Load image with caching.
        
        Args:
            idx: Index of image to load
            
        Returns:
            Loaded image
        """
        # Check cache first
        if idx in self.cache:
            # Update cache order
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]
        
        # Load image
        if self.preload and self.images:
            # Use preloaded image data
            if self.use_mmap:
                mm = cast(mmap.mmap, self.images[idx])
                mm.seek(0)
                image = Image.open(io.BytesIO(mm.read())).convert('RGB')
            else:
                # For bytes data, use BytesIO directly
                image = Image.open(io.BytesIO(self.images[idx])).convert('RGB')
        else:
            # Load from disk
            img_path = self.data_dir / self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {str(e)}")
                raise
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
        
        # Add to cache
        self.cache[idx] = image
        self.cache_order.append(idx)
        
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index.
        
        Args:
            idx: Index of sample to get
            
        Returns:
            Tuple of (image tensor, label)
        """
        # Load image with caching
        image = self._load_image(idx)
        
        # Get label
        label = self.labels[idx]
        if self.class_mapping:
            label = self.class_mapping[label]
        
        # Apply transforms
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.error(f"Transform failed for image {self.image_paths[idx]}: {str(e)}")
                raise
        else:
            # Convert to tensor if no transform
            image = to_tensor(image)  # Faster than manual conversion
        
        return image, int(label)  # Ensure label is an integer
    
    def clear_cache(self):
        """Clear the image cache."""
        self.cache.clear()
        self.cache_order.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when dataset is deleted."""
        if self.preload and self.use_mmap and self.images:
            for mm in self.images:
                if isinstance(mm, mmap.mmap):
                    mm.close()


# Removed MixupCutmixDataset, get_class_mapping, get_transforms, 
# TransformValidator, DatasetValidator, create_dataset_csvs, 
# calculate_class_weights, and prepare_data_loaders as they are moved to dataset_utils.py
