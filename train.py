"""
CPU-Optimized Training Script for Krishi Rakshak
"""
import os
import sys
import argparse
import json
import gc
import psutil
from datetime import datetime
from pathlib import Path

# Core libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from torchinfo import summary

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    INTEL_EXTENSION_AVAILABLE = True
    print("Intel Extension for PyTorch is available - will use optimizations")
except ImportError:
    INTEL_EXTENSION_AVAILABLE = False
    print("Intel Extension for PyTorch not available - using standard PyTorch")

# Memory and system monitoring
from memory_profiler import profile
import py_cpuinfo

# Experiment tracking
import wandb

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import prepare_data_loaders, get_class_weights
from src.models.model import EfficientNetB0, train_model, evaluate_model

def get_system_info():
    """Get detailed system information for logging"""
    cpu_info = py_cpuinfo.get_cpu_info()
    memory_info = psutil.virtual_memory()
    
    system_info = {
        'cpu_brand': cpu_info.get('brand_raw', 'Unknown'),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'memory_total_gb': round(memory_info.total / (1024**3), 2),
        'pytorch_version': torch.__version__,
        'intel_extension_available': INTEL_EXTENSION_AVAILABLE
    }
    
    return system_info

def setup_cpu_optimizations(args):
    """Setup CPU-specific optimizations"""
    
    # Set number of threads
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    else:
        # Use physical cores for training, leave some for system
        num_cores = psutil.cpu_count(logical=False)
        torch.set_num_threads(max(1, num_cores - 1))
    
    print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")
    
    # Disable CUDA optimizations
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    # Enable CPU optimizations
    torch.set_num_interop_threads(1)  # Avoid thread oversubscription
    
    # Set optimal memory allocation
    os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())
    os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())
    
    return torch.device('cpu')

def optimize_model_for_cpu(model):
    """Apply CPU-specific model optimizations"""
    
    if INTEL_EXTENSION_AVAILABLE:
        # Intel Extension optimizations
        model = ipex.optimize(model)
        print("Model optimized with Intel Extension for PyTorch")
    
    # Enable CPU mixed precision if available (PyTorch 2.0+)
    if hasattr(torch, 'cpu') and hasattr(torch.cpu, 'amp'):
        print("CPU mixed precision available")
        return model, True
    
    return model, False

@profile  # Memory profiling decorator
def memory_efficient_training_step(model, data, target, criterion, optimizer, 
                                 scaler=None, accumulation_steps=1):
    """Memory-efficient training step with optional mixed precision"""
    
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass with optional mixed precision
    if scaler is not None:
        with torch.cpu.amp.autocast():
            outputs = model(data)
            loss = criterion(outputs, target) / accumulation_steps
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        if accumulation_steps == 1:
            scaler.step(optimizer)
            scaler.update()
    else:
        # Standard precision
        outputs = model(data)
        loss = criterion(outputs, target) / accumulation_steps
        loss.backward()
        
        if accumulation_steps == 1:
            optimizer.step()
    
    return outputs, loss.item() * accumulation_steps

def parse_args():
    """Enhanced argument parser for CPU training"""
    parser = argparse.ArgumentParser(description='CPU-Optimized Plant Disease Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (recommended: 4-16 for CPU)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (consider 128-192 for faster CPU training)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data to use for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loading workers (0-1 recommended for CPU)')
    
    # CPU-specific arguments
    parser.add_argument('--cpu_threads', type=int, default=None,
                        help='Number of CPU threads (default: auto-detect)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Accumulate gradients for effective larger batch size')
    parser.add_argument('--use_cpu_mixed_precision', action='store_true',
                        help='Use CPU mixed precision (PyTorch 2.0+)')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Enable memory efficient training')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'mobilenet_v3_small', 'resnet18'],
                        help='Model architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate (higher for smaller batches)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Patience for early stopping (higher for CPU)')
    
    # Monitoring arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--project_name', type=str, default='plant-disease-cpu',
                        help='W&B project name')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (batches)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get system information
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Setup CPU optimizations
    device = setup_cpu_optimizations(args)
    
    # Initialize experiment tracking
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args),
            tags=['cpu-training'],
            notes=f"CPU: {system_info['cpu_brand']}"
        )
    
    # Continue with rest of training...
    print(f"Starting CPU-optimized training...")
    
    # TODO: Add data loading, model initialization, and training loop
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
