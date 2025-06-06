@echo off
:: setup_cpu_training.bat - Windows setup script for CPU-optimized training

echo === Setting up CPU-Optimized Plant Disease Classification Environment ===

:: Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set python_version=%%a
for /f "tokens=1,2 delims=. " %%a in ("%python_version%") do set python_major_minor=%%a.%%b

echo Python version: %python_version%

:: Check Python version >= 3.8
for /f "delims=. tokens=1,2" %%a in ("%python_major_minor%") do (
    if %%a LSS 3 (
        echo ❌ Python 3.8+ is required. Current version: %python_version%
        exit /b 1
    )
    if %%a EQU 3 if %%b LSS 8 (
        echo ❌ Python 3.8+ is required. Current version: %python_version%
        exit /b 1
    )
)

:: Create and activate virtual environment
echo 📦 Creating virtual environment...
python -m venv plant_disease_env

:: Activate the environment
call plant_disease_env\Scripts\activate.bat

:: Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

:: Install CPU-optimized PyTorch
echo 🔥 Installing PyTorch CPU version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Check for Intel CPU
echo Checking CPU type...
python -c "import cpuinfo; print(cpuinfo.get_cpu_info()['vendor_id_raw'])" > cpu_vendor.txt 2>nul
set /p CPU_VENDOR=<cpu_vendor.txt
del cpu_vendor.txt

if "%CPU_VENDOR%"=="GenuineIntel" (
    echo 🚀 Intel CPU detected. Installing Intel Extension for PyTorch...
    pip install intel-extension-for-pytorch
    pip install mkl
) else (
    echo ℹ️  Non-Intel CPU detected. Skipping Intel Extension.
)

:: Install main requirements
echo 📚 Installing remaining requirements...
pip install -r requirements.txt

:: Verify installation
echo ✅ Verifying installation...

echo.
echo PyTorch Information:
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CPU threads available: {torch.get_num_threads()}'); print(f'CPU device: {torch.device(\"cpu\")}')"

echo.
echo Testing Intel Extension for PyTorch:
python -c "try: import intel_extension_for_pytorch as ipex; print('✅ Intel Extension for PyTorch: Available')\nexcept ImportError: print('ℹ️  Intel Extension for PyTorch: Not available')"

echo.
echo Testing key packages:
python -c "packages = ['numpy', 'pandas', 'matplotlib', 'torchvision', 'albumentations', 'timm']; [print(f'✅ {pkg}: OK') if __import__(pkg) else None for pkg in packages]"

echo.
echo 🎉 Setup complete!
echo.
echo To activate the environment in the future, run:
echo    plant_disease_env\Scripts\activate
echo.
echo To start training, run:
echo    python train.py --data_dir data\plantvillage --use_wandb
echo.

pause
