# Krishi Rakshak - Plant Disease Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Krishi Rakshak** is an AI-powered crop health monitoring system that helps farmers detect plant diseases using just a smartphone camera. This version is optimized for CPU training and inference, making it accessible on standard hardware.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/VIKAS9793/KrishiRakshak.git
cd krishi-rakshak
```

### 2. Setup Environment

#### Windows:
```bash
# Run the setup script
setup_cpu_training.bat
```

#### Linux/Mac:
```bash
# Make the script executable
chmod +x setup_cpu_training.sh

# Run the setup script
./setup_cpu_training.sh
```

#### Manual Setup:
```bash
# Create and activate virtual environment
python -m venv plant_disease_env

# Activate environment
# Windows:
plant_disease_env\Scripts\activate
# Linux/Mac:
source plant_disease_env/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify CPU support
python -c "import torch; print(f'CPU: {torch.cpu.get_device_name(0)}' if torch.cuda.is_available() else 'Using CPU')"
```

### 4. Prepare Your Dataset
Organize your dataset in the following structure:
```
data/plantvillage/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class2/
│       └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 5. Run Training
```bash
# Quick test run (2 epochs)
python train.py --data_dir data/plantvillage --batch_size 8 --epochs 2 --experiment_name "test_run"

# Full training with default config
python train.py --config configs/train_config.yaml

# Custom training with overrides
python train.py \
    --data_dir data/plantvillage \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 0.01 \
    --use_wandb \
    --experiment_name "efficientnet_b0_full"
```

### 6. Run the Web Interface
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

## 🛠 Configuration

Edit `configs/train_config.yaml` to customize training parameters. Key sections include:

- **Data Configuration**: Set paths, batch sizes, and image augmentations
- **Model Settings**: Choose architecture and training parameters
- **CPU Optimization**: Configure thread usage and memory settings
- **Training Parameters**: Learning rate, schedulers, and early stopping

## 📊 Monitoring

### Weights & Biases (Recommended)
1. Create an account at [wandb.ai](https://wandb.ai/)
2. Run `wandb login` and paste your API key
3. Enable W&B in config or with `--use_wandb`

### TensorBoard
```bash
tensorboard --logdir=experiments
```

## 🚀 Performance Tips

1. **For Faster Training**
   - Reduce image size (e.g., 160x160)
   - Use larger batch sizes that fit in memory
   - Enable mixed precision with `--use_mixed_precision`

2. **For Memory Efficiency**
   - Use gradient accumulation
   - Reduce batch size
   - Enable memory-efficient optimizations

3. **For Intel CPUs**
   - Ensure Intel Extension for PyTorch is installed
   - Enable MKL optimizations

## 📂 Project Structure

```
krishi-rakshak/
├── configs/               # Configuration files
│   └── train_config.yaml  # Main training configuration
├── data/                  # Dataset directory
├── experiments/           # Training outputs and checkpoints
├── logs/                  # Log files
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions
│   └── utils/            # Utility functions
├── app.py                # Gradio web interface
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
└── setup_cpu_training.*  # Environment setup scripts
```

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PlantVillage dataset
- PyTorch and Intel Extension for PyTorch teams
- Open-source community contributors
