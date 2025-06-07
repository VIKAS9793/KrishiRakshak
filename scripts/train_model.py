"""
Training script for Plant Disease Classification with optimized settings for 2-day training.
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Train plant disease classification model')
    parser.add_argument('--data_dir', type=str, default='data/plantvillage/raw',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'mobilenet_v3_small', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (width=height)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--cpu_threads', type=int, default=None,
                        help='Number of CPU threads to use')
    
    args = parser.parse_args()
    
    # Set CPU threads if specified
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    
    # Import training modules
    from src.data.dataset import prepare_data_loaders, get_class_weights
    from src.models.model import get_model
    from train import train_model, setup_training_environment
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, class_to_idx = prepare_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=2 if device.type == 'cuda' else 0
    )
    
    # Get model
    print(f"Initializing {args.model_name} model...")
    model = get_model(
        model_name=args.model_name,
        num_classes=len(class_to_idx),
        pretrained=True
    ).to(device)
    
    # Get class weights for imbalanced dataset
    class_weights = get_class_weights(args.data_dir)
    class_weights = class_weights.to(device)
    
    # Training configuration
    config = {
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'lr_patience': 5,
        'lr_factor': 0.1,
        'num_classes': len(class_to_idx),
        'device': str(device)
    }
    
    # Start training
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config,
        use_wandb=args.use_wandb,
        device=device
    )
    
    print("Training completed!")

if __name__ == '__main__':
    main()
