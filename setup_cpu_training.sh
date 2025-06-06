#!/bin/bash
# setup_cpu_training.sh - Setup script for CPU-optimized plant disease classification

echo "=== Setting up CPU-Optimized Plant Disease Classification Environment ==="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create and activate virtual environment
echo "📦 Creating virtual environment..."
python -m venv plant_disease_env

# Activation command varies by OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source plant_disease_env/Scripts/activate
else
    source plant_disease_env/bin/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install CPU-optimized PyTorch first
echo "🔥 Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Check if Intel CPU is available for Intel Extension
cpu_vendor=$(python -c "import cpuinfo; print(cpuinfo.get_cpu_info()['vendor_id_raw'])" 2>/dev/null || echo "unknown")

if [[ "$cpu_vendor" == "GenuineIntel" ]]; then
    echo "🚀 Intel CPU detected. Installing Intel Extension for PyTorch..."
    pip install intel-extension-for-pytorch
    pip install mkl
else
    echo "ℹ️  Non-Intel CPU detected. Skipping Intel Extension."
fi

# Install main requirements
echo "📚 Installing remaining requirements..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."

# Test PyTorch
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CPU threads available: {torch.get_num_threads()}')
print(f'CPU device: {torch.device(\"cpu\")}')
"

# Test Intel Extension if available
python -c "
try:
    import intel_extension_for_pytorch as ipex
    print('✅ Intel Extension for PyTorch: Available')
except ImportError:
    print('ℹ️  Intel Extension for PyTorch: Not available')
"

# Test other key packages
python -c "
packages = ['numpy', 'pandas', 'matplotlib', 'torchvision', 'albumentations', 'timm']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}: OK')
    except ImportError as e:
        print(f'❌ {pkg}: Failed - {e}')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  plant_disease_env\\Scripts\\activate"
else
    echo "  source plant_disease_env/bin/activate"
fi
echo ""
echo "To start training, run:"
echo "  python train.py --data_dir /path/to/your/dataset --use_wandb"
