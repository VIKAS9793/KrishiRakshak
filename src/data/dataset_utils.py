# filename: src/data/dataset_utils.py
"""
Utility functions for data loading, transformation, and dataset validation.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, TypeVar, Protocol, Sized, cast
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from collections import Counter
import logging
import random
import glob
import torchvision.transforms as transforms

from src.data.constants import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataset import PlantDiseaseDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.class_weights import calculate_class_weights, get_class_weights_tensor
from src.utils.training_utils import validate_and_convert_config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Tuple[torch.Tensor, int])
DatasetType = Dataset[T]

class SizedDataset(Protocol, Sized):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]: ...

def get_class_mapping(data_dir: Union[str, Path]) -> Dict[str, int]:
    """
    Get class name to index mapping from the train directory.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Dictionary mapping class names to indices
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(classes)} classes in train directory: {classes}")
    return {cls: idx for idx, cls in enumerate(classes)}


def get_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transforms based on configuration.
    
    Args:
        config: Configuration dictionary containing transform parameters
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    # Get image size from config
    image_size = config.get("image_size", 224)
    
    # Training transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transforms, val_transforms


class TransformValidator:
    """
    Utility class for validating and testing image transformation pipelines.
    """

    @staticmethod
    def create_test_image(size: int = 224) -> np.ndarray:
        """
        Creates a dummy test image for validating transformations.

        Args:
            size: The desired size (width and height) of the square image.

        Returns:
            A NumPy array representing a dummy RGB image with pixel values from 0-255.
        """
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

    @staticmethod
    def validate_transform(transform: A.Compose, num_tests: int = 10) -> bool:
        """
        Validates a given Albumentations transform pipeline by applying it to test images.

        Args:
            transform: The Albumentations Compose object to validate.
            num_tests: The number of test images to run through the transform.

        Returns:
            True if all transformations apply without errors, False otherwise.
        """
        logger.info(f"Validating transform pipeline with {num_tests} tests...")
        try:
            for i in range(num_tests):
                test_image = TransformValidator.create_test_image()
                transformed_image = transform(image=test_image)["image"]
                # Basic check on output shape and type
                assert isinstance(transformed_image, torch.Tensor)
                assert transformed_image.shape[0] == 3 # Channels first
                assert transformed_image.shape[1] == transformed_image.shape[2] # Square image
            logger.info("Transform validation successful.")
            return True
        except Exception as e:
            logger.error(f"Transform validation failed: {e}")
            return False


class DatasetValidator:
    """
    Utility class for validating dataset integrity and structure.
    """

    @staticmethod
    def validate_dataset(dataset: Dataset[Tuple[torch.Tensor, int]], num_samples: int = 10) -> bool:
        """
        Validates a given dataset by attempting to load a number of samples.
        """
        try:
            # Check dataset size
            dataset_size = len(dataset)  # type: ignore
            if dataset_size == 0:
                logger.error("Dataset is empty")
                return False

            # Try to load a few samples
            for i in range(min(num_samples, dataset_size)):
                try:
                    sample, label = dataset[i]
                    if not isinstance(sample, torch.Tensor) or not isinstance(label, int):
                        logger.error(f"Invalid sample types at index {i}")
                        return False
                except Exception as e:
                    logger.error(f"Error loading sample at index {i}: {str(e)}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False


def create_dataset_csvs(data_dir: Union[str, Path]) -> Tuple[str, str, str]:
    """
    Create CSV files for train, validation, and test datasets.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of (train_csv, val_csv, test_csv) paths
    """
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    
    logger.info(f"Creating dataset CSVs in {data_dir}")
    logger.info(f"Train CSV path: {train_csv}")
    logger.info(f"Val CSV path: {val_csv}")
    logger.info(f"Test CSV path: {test_csv}")
    
    # Process each split
    for split_name, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        split_dir = data_dir / split_name
        logger.info(f"Processing {split_name} split in {split_dir}")
        
        if not split_dir.is_dir():
            logger.error(f"Split directory not found: {split_dir}")
            continue

        all_image_paths: List[str] = []
        all_labels: List[str] = []
        class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(class_names)} classes in {split_name}: {class_names}")

        for class_name in tqdm(class_names, desc=f"Scanning {split_name} directories"):
            class_path = split_dir / class_name
            image_files = []
            for ext in ["jpg", "jpeg", "png"]:
                image_files.extend(class_path.glob(f"*.{ext}"))
            logger.info(f"Found {len(image_files)} images in {class_name}")
            
            for img_file in image_files:
                # Store relative path from data_dir
                rel_path = img_file.relative_to(data_dir)
                all_image_paths.append(str(rel_path))
                all_labels.append(class_name)

        if not all_image_paths:
            logger.error(f"No images found in {split_dir}")
            continue

        # Create a DataFrame
        df = pd.DataFrame({"image_path": all_image_paths, "label": all_labels})
        logger.info(f"Created DataFrame with {len(df)} rows")
        logger.info(f"Sample of DataFrame:\n{df.head()}")
        
        # Save CSV file
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Successfully saved {split_name}.csv to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save {split_name}.csv: {str(e)}")
            raise
    
    # Verify CSV files exist
    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            logger.error(f"CSV file not created: {csv_path}")
            raise FileNotFoundError(f"CSV file not created: {csv_path}")
        else:
            logger.info(f"Verified CSV file exists: {csv_path}")
    
    return str(train_csv), str(val_csv), str(test_csv)


def prepare_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int,
    num_workers: int,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Prepare data loaders for training and validation.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, class_mapping)
    """
    # Convert data_dir to Path
    data_dir = Path(data_dir)
    
    # Get class mapping
    class_mapping = get_class_mapping(data_dir)
    
    # Get transforms
    train_transforms, val_transforms = get_transforms(config)
    
    # Get data loading settings
    data_config = config.get("data", {})
    cache_size = data_config.get("cache_size", 2000)
    use_mmap = data_config.get("use_mmap", True)
    preload = data_config.get("preload", False)
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        csv_file=str(data_dir / "train.csv"),
        data_dir=str(data_dir),
        transform=train_transforms,
        class_mapping=class_mapping,
        cache_size=cache_size,
        use_mmap=use_mmap,
        preload=preload
    )
    
    val_dataset = PlantDiseaseDataset(
        csv_file=str(data_dir / "val.csv"),
        data_dir=str(data_dir),
        transform=val_transforms,
        class_mapping=class_mapping,
        cache_size=cache_size,
        use_mmap=use_mmap,
        preload=preload
    )
    
    # Get training settings
    training_config = config.get("training", {})
    pin_memory = training_config.get("pin_memory", True)
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,  # Drop incomplete batches
        timeout=60,  # Timeout for worker processes
        generator=torch.Generator()  # For reproducible shuffling
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=60
    )
    
    return train_loader, val_loader, class_mapping 