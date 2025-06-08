"""
Model architectures and training logic for crop disease classification.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models
from torchinfo import summary
import torchmetrics
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

class BaseModel(nn.Module):
    """Base model class with common functionality."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = None
        
    def forward(self, x):
        return self.model(x)
    
    def freeze_base(self):
        """Freeze all layers except the final classifier."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, optimizer_name: str = 'adamw'):
        """Get optimizer for training. Supports 'adam' and 'adamw'."""
        if optimizer_name == 'adam':
            return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def get_scheduler(self, optimizer, patience: int = 5, factor: float = 0.1):
        """Get learning rate scheduler."""
        return ReduceLROnPlateau(optimizer, patience=patience, factor=factor)


class EfficientNetB0(BaseModel):
    """EfficientNet-B0 model for crop disease classification."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes)
        self.model = models.efficientnet_b0(pretrained=pretrained)
        # Replace final classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)


class ResNet18(BaseModel):
    """ResNet-18 model for crop disease classification."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes)
        self.model = models.resnet18(pretrained=pretrained)
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)


class MobileNetV3Small(BaseModel):
    """MobileNetV3-Small model for crop disease classification."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes)
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        # Replace final classifier
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)


def get_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_base: bool = False
) -> nn.Module:
    """Get model by name.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_base: Whether to freeze base layers
        
    Returns:
        Initialized model
    """
    model_map = {
        'efficientnet_b0': EfficientNetB0,
        'resnet18': ResNet18,
        'mobilenet_v3_small': MobileNetV3Small
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(model_map.keys())}")
    
    model = model_map[model_name](num_classes=num_classes, pretrained=pretrained)
    
    if freeze_base:
        model.freeze_base()
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler = None
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=model.num_classes).to(device)
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if scaler is provided
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy(predicted, targets)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'metrics': {
            'train_accuracy': accuracy.compute().item()
        }
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list = None
) -> Dict[str, float]:
    """Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use for validation
        class_names: List of class names for metrics
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=model.num_classes).to(device)
    precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=model.num_classes).to(device)
    recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=model.num_classes).to(device)
    f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=model.num_classes).to(device)
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.append(predicted)
            all_targets.append(targets)
            
            # Update metrics
            accuracy(predicted, targets)
            precision(predicted, targets)
            recall(predicted, targets)
            f1(predicted, targets)
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())
    
    # Calculate per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'val_loss': epoch_loss,
        'val_accuracy': accuracy.compute().item(),
        'val_precision': precision.compute().item(),
        'val_recall': recall.compute().item(),
        'val_f1': f1.compute().item(),
        'class_accuracy': class_acc.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    # Add per-class metrics if class names are provided
    if class_names is not None:
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'accuracy': class_acc[i],
                'samples': int(cm[i].sum())
            }
        metrics['per_class'] = class_metrics
    
    return metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list = None,
    output_dir: str = None
) -> Dict[str, float]:
    """Evaluate model on test set and generate metrics, including IoU, mAP, SSIM, PSNR, and MSE."""
    import torchmetrics
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    all_images = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
            all_images.append(inputs.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    all_images = torch.cat(all_images)
    # Standard metrics
    metrics = validate(model, test_loader, criterion, device, class_names)
    # IoU (for classification, mean IoU over classes)
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=len(class_names)).to(device)
    metrics['IoU'] = iou_metric(all_preds, all_targets).item()
    # mAP (mean average precision)
    map_metric = torchmetrics.AveragePrecision(task='multiclass', num_classes=len(class_names)).to(device)
    metrics['mAP'] = map_metric(all_probs, all_targets).item()
    # MSE (mean squared error between predicted class indices and targets)
    metrics['MSE'] = mean_squared_error(all_targets.numpy(), all_preds.numpy())
    # SSIM and PSNR (average over batch, using first channel if 3-channel)
    ssim_scores = []
    psnr_scores = []
    for i in range(all_images.shape[0]):
        img = all_images[i].numpy().transpose(1,2,0)
        # Reconstruct predicted image as one-hot mask for class (for demo purposes)
        pred_img = np.zeros_like(img)
        pred_img[:,:,0] = (all_preds[i].item() / (len(class_names)-1)) # normalize to [0,1]
        try:
            ssim_score = ssim(img, pred_img, channel_axis=-1, data_range=1.0)
            psnr_score = psnr(img, pred_img, data_range=1.0)
        except Exception:
            ssim_score = float('nan')
            psnr_score = float('nan')
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    metrics['SSIM'] = float(np.nanmean(ssim_scores))
    metrics['PSNR'] = float(np.nanmean(psnr_scores))
    # Save metrics
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        # Plot and save confusion matrix
        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor = None,
    config: dict = None,
    use_wandb: bool = False,
    device: torch.device = None,
    output_dir: str = 'outputs'
) -> Dict[str, List[float]]:
    """Train and validate model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Class weights for imbalanced dataset
        config: Training configuration
        use_wandb: Whether to log to Weights & Biases
        device: Device to use for training
        output_dir: Directory to save model checkpoints and logs
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config is None:
        config = {
            'epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
            'lr_patience': 5,
            'lr_factor': 0.1
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = model.to(device)
    
    # Loss function with class weights if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = model.get_optimizer(
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        optimizer_name=config.get('optimizer_name', 'adamw')
    )
    
    scheduler = model.get_scheduler(
        optimizer=optimizer,
        patience=config.get('lr_patience', 5),
        factor=config.get('lr_factor', 0.1)
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch+1}/{config["epochs"]}')
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        scheduler.step(val_metrics['val_loss'])
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_acc'].append(val_metrics['val_accuracy'])
        
        # Log metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.2f}%")
        
        # Log to wandb
        if use_wandb:
            import wandb
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['val_loss'],
                'val/accuracy': val_metrics['val_accuracy'],
                'val/precision': val_metrics['val_precision'],
                'val/recall': val_metrics['val_recall'],
                'val/f1': val_metrics['val_f1']
            })
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'val_accuracy': val_metrics['val_accuracy']
            }, os.path.join(output_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.get('early_stopping_patience', 10):
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['val_loss'],
        'val_accuracy': val_metrics['val_accuracy']
    }, os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return history
