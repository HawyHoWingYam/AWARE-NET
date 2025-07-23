# Environment Setup Instructions

## Updated Dependencies (2025-07-23)

### ⚠️ CRITICAL: Required PyTorch Updates

Before running any training or testing, update PyTorch and torchvision to the latest nightly builds:

```bash
# 1. Update torch (REQUIRED)
pip install --pre --upgrade --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Update torchvision (REQUIRED)  
pip install --pre --upgrade --no-cache-dir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note**: These nightly builds are required for compatibility with the training pipeline. Standard PyTorch versions may cause issues.

### Verification After Update

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
```

## 任務 0.1：環境配置完成

### Method 1: Using Conda (Recommended)

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate aware-net
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Method 2: Using pip (Alternative)

1. **Create a virtual environment:**
   ```bash
   python -m venv aware-net-env
   source aware-net-env/bin/activate  # On Windows: aware-net-env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### GPU Setup Verification

To ensure CUDA is properly configured for PyTorch:

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

### Download Pre-trained Model Weights

Based on the README, you'll need these pre-trained weights:

```bash
# Create weights directory
mkdir -p weights

# Download Res2Net101 weights
wget -O weights/res2net101_26w_4s-02a759a1.pth \
  https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-res2net/res2net101_26w_4s-02a759a1.pth

# Download EfficientNet-B7 weights  
wget -O weights/tf_efficientnet_b7_ns.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth
```

### Directory Structure Setup

Create the required directory structure for data processing:

```bash
mkdir -p data_root/{raw_datasets,processed_data/{train/{real,fake},val/{real,fake},final_test_sets},manifests}
mkdir -p faces/{celebdf,ffpp}
mkdir -p videos/{CelebDF-v2,FF++}
mkdir -p src
```

### Environment Variables (Optional)

You may want to set these environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Add src to Python path
```

### Troubleshooting

1. **CUDA Issues:** If CUDA is not detected, ensure you have compatible NVIDIA drivers and CUDA toolkit installed.

2. **Memory Issues:** For systems with limited RAM, you may need to reduce batch sizes in the configuration.

3. **Face Detection:** If facenet-pytorch has issues, try using mtcnn as an alternative face detector.

### Next Steps

After environment setup, you can proceed to:
1. Task 0.2: Dataset preparation and splitting (preprocess_datasets.py)
2. Begin Stage 1: Training the first stage model (fast filter)