"""
Data loading and preprocessing for the PlantVillage dataset.
"""
import os
import shutil
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.config import get_config

class PlantDiseaseDataset(Dataset):
    """Custom dataset for PlantVillage disease classification."""
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        class_to_idx: Dict[str, int],
        transform: Optional[A.Compose] = None,
        is_train: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of corresponding labels
            class_to_idx: Mapping from class name to index
            transform: Albumentations transforms to apply
            is_train: Whether this is the training set
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an image and its label by index."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        class_counts = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts


def get_transforms(img_size: int = 224, is_train: bool = True) -> A.Compose:
    """Get data augmentation and transformation pipeline.
    
    Args:
        img_size: Target image size (height, width)
        is_train: Whether to include training augmentations
        
    Returns:
        Albumentations Compose object with the transformation pipeline
    """
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.CenterCrop(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def prepare_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 4,
    download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Prepare data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        img_size: Target image size (height, width)
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        download: Whether to download the dataset if not found
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Download dataset if needed
    if download:
        # TODO: Implement dataset download from Kaggle
        pass
    
    # Find all images and their labels
    image_paths = []
    labels = []
    class_to_idx = {}
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    # Find all class directories
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Collect all image paths and labels
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
            
        for img_path in cls_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(cls_idx)
    
    # Split into train, validation, and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=random_seed
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_split/(1-test_split),  # Adjust for test split
        stratify=train_val_labels, 
        random_state=random_seed
    )
    
    print(f"Dataset split:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(f"  Test samples: {len(test_paths)}")
    print(f"  Number of classes: {len(class_to_idx)}")
    
    # Create datasets
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)
    
    train_dataset = PlantDiseaseDataset(
        train_paths, train_labels, class_to_idx, transform=train_transform, is_train=True
    )
    
    val_dataset = PlantDiseaseDataset(
        val_paths, val_labels, class_to_idx, transform=val_transform, is_train=False
    )
    
    test_dataset = PlantDiseaseDataset(
        test_paths, test_labels, class_to_idx, transform=val_transform, is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_to_idx


def get_class_weights(data_dir: str) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Tensor of class weights
    """
    class_counts = {}
    data_dir = Path(data_dir)
    
    # Count samples per class
    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir():
            num_samples = len(list(cls_dir.glob('*.*')))
            class_counts[cls_dir.name] = num_samples
    
    # Calculate weights (inverse of class frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = [total_samples / (num_classes * count) for count in class_counts.values()]
    weights = torch.FloatTensor(weights)
    
    return weights
