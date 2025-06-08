"""
Training script for Crop Disease Classification using PyTorch.

# HACKATHON DEFAULTS:
# - Pretrained model enabled
# - Fewer epochs (5)
# - Smaller batch size (16)
# These settings are optimized for quick demo and iteration.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Project imports
from src.data.dataset import prepare_data_loaders, get_class_weights
from src.models import get_model, train_model, evaluate_model

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train crop disease classification model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/plantvillage/raw',
                        help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16 for hackathon demo)')
    parser.add_argument('--img-size', type=int, default=128,
                        help='Image size (height=width, default: 128 for hackathon demo)')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Fraction of data to use for testing')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet18', 'mobilenet_v3_small'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights (default: True for hackathon demo)')
    parser.add_argument('--freeze-base', action='store_true',
                        help='Freeze base layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train (default: 5 for hackathon demo)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='Number of epochs to wait before reducing learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='Factor by which to reduce learning rate')
    
    # Logging and saving
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--experiment-name', type=str, default='experiment',
                        help='Name of the experiment')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    
    # New optimizer argument
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help="Optimizer to use for training (default: 'adamw'). Choices: 'adam', 'adamw'.")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return device

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    device = setup_environment(args)
    
    # Prepare data loaders
    print('Preparing data loaders...')
    train_loader, val_loader, test_loader, class_to_idx = prepare_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers
    )
    
    # Save class to index mapping
    with open(os.path.join(args.output_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Get class weights for imbalanced dataset
    class_weights = get_class_weights(args.data_dir)
    
    # Initialize model
    print(f'Initializing {args.model_name} model...')
    model = get_model(
        model_name=args.model_name,
        num_classes=len(class_to_idx),
        pretrained=args.pretrained,
        freeze_base=args.freeze_base
    ).to(device)
    
    # Print model summary
    print(model)
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'lr_patience': args.lr_patience,
        'lr_factor': args.lr_factor,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'model_name': args.model_name,
        'num_classes': len(class_to_idx),
        'optimizer_name': args.optimizer
    }
    
    # Initialize Weights & Biases if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project='crop-disease-classification',
            name=args.experiment_name,
            config=config
        )
    
    try:
        # Train model
        print('Starting training...')
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            config=config,
            use_wandb=args.use_wandb,
            device=device,
            output_dir=args.output_dir
        )
        
        # Evaluate on test set
        print('Evaluating on test set...')
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=list(class_to_idx.keys()),
            output_dir=args.output_dir
        )
        
        # Save test metrics
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print('\nTest Metrics:')
        print(f"Accuracy: {test_metrics['val_accuracy']:.4f}")
        print(f"Precision: {test_metrics['val_precision']:.4f}")
        print(f"Recall: {test_metrics['val_recall']:.4f}")
        print(f"F1-Score: {test_metrics['val_f1']:.4f}")
        
    except KeyboardInterrupt:
        print('Training interrupted. Saving latest model...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
        }, os.path.join(args.output_dir, 'interrupted_model.pth'))
    except Exception as e:
        print(f'Error during training: {e}')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,
        }, os.path.join(args.output_dir, 'interrupted_model.pth'))
        raise
    finally:
        # Cleanup
        if args.use_wandb:
            wandb.finish()
    
    print(f'Training completed. Outputs saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
