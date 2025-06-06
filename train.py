"""
Main training script for Krishi Rakshak - Plant Disease Classification
"""
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import prepare_data_loaders, get_class_weights
from src.models.model import (
    EfficientNetB0, 
    train_model, 
    evaluate_model,
    save_confusion_matrix_plot
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train EfficientNetB0 for Plant Disease Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (height=width)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data to use for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the backbone network')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: model_timestamp)')
    
    # Hardware arguments
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'model_{timestamp}'
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command line arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return device, str(output_dir)


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    device, output_dir = setup_environment(args)
    
    # Prepare data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader, test_loader, class_to_idx = prepare_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    # Save class to index mapping
    with open(Path(output_dir) / 'class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Create model
    print("\nCreating model...")
    model = EfficientNetB0(
        num_classes=len(class_to_idx),
        pretrained=args.pretrained
    )
    
    if args.freeze_backbone:
        model.freeze_backbone()
        print("Froze backbone layers")
    
    model = model.to(device)
    
    # Set up loss function with class weights if needed
    if args.use_class_weights:
        print("Calculating class weights...")
        class_weights = get_class_weights(args.data_dir)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Track validation accuracy
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Train the model
    print(f"\nStarting training for {args.epochs} epochs...")
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        output_dir=output_dir,
        class_names=list(class_to_idx.keys()),
        early_stopping_patience=args.early_stopping_patience,
        save_best_only=True
    )
    
    # Load best model for evaluation
    best_model_path = training_results.get('best_model_path')
    if best_model_path and os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=list(class_to_idx.keys()),
        output_dir=output_dir
    )
    
    # Save test results
    with open(Path(output_dir) / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
