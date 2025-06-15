"""
Image transformation utilities for training and validation
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import psutil
import gc

def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Get training transforms optimized for performance.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Composition of transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),  # Use antialias for better quality
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),  # Reduced from 15 to 10 degrees
        T.ColorJitter(brightness=0.2, contrast=0.2),  # Simplified color jitter
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Get validation transforms optimized for performance.
    
    Args:
        image_size: Size to resize images to
        
    Returns:
        Composition of transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),  # Use antialias for better quality
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Add progress tracking to training loop
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    process = psutil.Process()
    
    # Add progress bar
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # Log memory usage
        if batch_idx % 10 == 0:
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
        
        # Clear cache periodically
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None 