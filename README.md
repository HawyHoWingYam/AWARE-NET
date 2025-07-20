# AWARE-NET: Adaptive Weighted Averaging for Robust Ensemble Network in Deepfake Detection

AWARE-NET is a PyTorch implementation of "Adaptive Weighted Averaging for Robust Ensemble Network in Deepfake Detection". The project implements a two-tier ensemble framework that hierarchically combines multiple instances of three state-of-the-art architectures: Xception, Res2Net101, and EfficientNet-B7.

## Core Architecture

### Framework Design
1. **Tier 1**: Averages predictions within each architecture to reduce model variance
2. **Tier 2**: Learns optimal weights for each architecture's contribution through backpropagation

### Performance Results
- **FF++** (FaceForensics++): AUC 99.22% (no aug.), 99.47% (aug.)
- **CelebDF-v2**: AUC 100% (both configurations)
- **Cross-dataset evaluation**: FF++ → CelebDF-v2, CelebDF-v2 → FF++

## Quick Start Guide

### 1. Environment Setup

#### Install Anaconda/Miniconda
If not already installed:
- **Anaconda** (full version): https://www.anaconda.com/products/distribution
- **Miniconda** (lightweight): https://docs.conda.io/en/latest/miniconda.html

#### Create Conda Environment (RTX 5060Ti/5090 Compatible)
```bash
# Navigate to project directory
cd D:\work\AWARE-NET

# Create environment from environment.yml (without PyTorch)
conda env create -f environment.yml

# Activate environment
conda activate aware-net

# Install PyTorch with CUDA 12.1 support for RTX 5060Ti/5090
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

#### Manual Environment Creation (Alternative Method)
```bash
# Create basic environment
conda create -n aware-net python=3.12

# Activate environment
conda activate aware-net

# Install PyTorch with CUDA 12.1 for RTX 5060Ti/5090
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies via conda (recommended for Windows)
conda install opencv pandas numpy scikit-learn matplotlib seaborn tqdm lightgbm -c conda-forge
conda install timm albumentations tensorboard -c conda-forge
conda install wandb -c conda-forge

# If pip has DLL errors on Windows, try conda alternatives:
conda install -c conda-forge facenet-pytorch mtcnn
# Or use conda-forge alternatives if the above packages are not available
```

#### Troubleshooting Windows pip DLL Errors
If you encounter `ImportError: DLL load failed while importing pyexpat`, try:
```bash
# Option 1: Reinstall pip
conda install pip -c conda-forge --force-reinstall

# Option 2: Use conda instead of pip for all packages
conda install -c conda-forge timm albumentations tensorboard wandb

# Option 3: For packages not available via conda, try:
conda install -c pytorch-nightly facenet-pytorch  # if available
conda install -c conda-forge mtcnn-pytorch  # alternative name

# Option 4: If still having issues, recreate environment
conda deactivate
conda env remove -n aware-net
conda env create -f environment.yml
conda activate aware-net
```

### 2. Pre-trained Model Weights

Download required weights to `weights/` directory:
- **Res2Net101**: `res2net101_26w_4s-02a759a1.pth`
- **EfficientNet-B7**: `tf_efficientnet_b7_ns.pth`

```bash
# Create weights directory
mkdir weights
# Download and place pre-trained weight files in weights/ directory
```

### 3. Dataset Configuration

#### Supported Datasets
- **FF++** (FaceForensics++)
- **CelebDF-v2** 
- **DFDC** (Deepfake Detection Challenge)
- **DF40** (Pre-processed image dataset)

#### Setup Dataset Paths
```bash
# Interactive configuration setup
python scripts/setup_dataset_config.py

# Verify configuration
python scripts/test_config.py
```

### 4. Data Preprocessing

AWARE-NET uses unified **256x256 PNG** format, consistent with DF40 dataset specifications.

#### Preprocessing Commands
```bash
# Activate environment
conda activate aware-net

# View configuration summary
python scripts/preprocess_datasets_v2.py --print-config

# Validate paths (without processing)
python scripts/preprocess_datasets_v2.py --validate-only

# 🚀 GPU-Accelerated Multi-threaded Processing (Recommended)
python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 --video-backend decord --face-detector insightface --workers 4

# Choose specific face detector
python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 --face-detector mediapipe
python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 --face-detector yolov8

# Process all video datasets (automatically skips pre-processed DF40)
python scripts/preprocess_datasets_v2.py

# Process specific datasets with optimized backends
python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 ffpp --video-backend decord --workers 3

# Process only DFDC with single-threaded mode (debugging)
python scripts/preprocess_datasets_v2.py --datasets dfdc --workers 1
```

#### Preprocessing Output
- **Format**: 256x256 PNG images (DF40-compatible)
- **Directory Structure**:
  ```
  processed_data/
  ├── train/
  │   ├── real/
  │   └── fake/
  ├── val/
  │   ├── real/
  │   └── fake/
  ├── final_test_sets/
  │   ├── celebdf_v2/
  │   ├── ffpp/
  │   └── dfdc/
  └── manifests/
      ├── train_manifest.csv
      ├── val_manifest.csv
      └── test_*_manifest.csv
  ```

#### Preprocessing Parameters (configured in config/dataset_paths.json)
- **frame_interval**: 10 (extract every 10th frame)
- **image_size**: [256, 256] (DF40-compatible)
- **max_faces_per_video**: 50
- **bbox_scale**: 1.3 (face bounding box expansion)
- **min_face_size**: 80 (minimum face size)

### 🚀 Multi-threaded GPU Processing (New Feature)

#### Performance Improvements
- **Single-threaded**: ~1.55s/video, 30-40% GPU utilization
- **Multi-threaded (4 workers)**: ~0.4-0.8s/video, **70-85% GPU utilization**
- **Speed improvement**: **2-4x faster processing**

#### Supported Face Detection Backends
1. **InsightFace** (Recommended) - High-performance GPU acceleration
2. **MediaPipe** - Google's optimized face detection
3. **YOLOv8** - General object detection with face capability
4. **OpenCV DNN** - Lightweight CPU fallback
5. **MTCNN** - Backup face detection method

#### Supported Video Backends
1. **Decord** (Recommended for Windows) - GPU-accelerated video processing
2. **TorchVision.io** - PyTorch native video processing
3. **OpenCV** - Universal CPU fallback

#### Multi-threading Configuration
```bash
# Optimal performance (recommended)
--workers 4              # 4 parallel workers (balances GPU utilization)
--video-backend decord    # GPU video processing for Windows
--face-detector insightface  # Fastest GPU face detection

# Custom worker count
--workers 2               # Fewer workers for lower-end GPUs
--workers 1               # Single-threaded for debugging

# Backend selection
--video-backend torchvision  # Alternative video backend
--face-detector mediapipe    # Alternative face detector
```

## Training and Evaluation

### Basic Training
```bash
# Main training pipeline
python main.py

# Individual model training (when src/ directory is populated)
python src/main.py
```

### Testing and Validation
```bash
# Run tests (check for existing test framework first)
python -m pytest  # if pytest is used
python -m unittest discover  # if unittest is used

# Cross-dataset evaluation (when implemented)
python src/cross_evaluation.py
```

## Project Structure

```
AWARE-NET/
├── src/                      # Core implementation files
│   ├── utils/               # Utility modules (dataset_config.py)
│   ├── config.py            # Central configuration
│   ├── model.py             # Model definitions
│   ├── ensemble.py          # Ensemble model implementation
│   ├── dataset.py           # Dataset classes and processing
│   ├── train.py             # Training process manager
│   ├── experiments.py       # Experiment manager
│   └── visualization.py     # Visualization tools
├── scripts/                 # Data processing and setup scripts
│   ├── preprocess_datasets_v2.py
│   ├── setup_dataset_config.py
│   └── test_config.py
├── config/                  # Configuration files
│   └── dataset_paths.json
├── docs/                    # Documentation
├── dataset/                 # Raw video datasets
│   ├── CelebDF-v2/
│   ├── FF++/
│   ├── DFDC/
│   └── DF40/
├── processed_data/          # Processed face images
├── weights/                 # Pre-trained model weights
├── project_instruction/     # Project phase documentation
├── environment.yml          # Conda environment configuration
└── README.md
```

## Common Conda Commands

```bash
# List all environments
conda env list

# Activate AWARE-NET environment
conda activate aware-net

# Deactivate environment
conda deactivate

# Remove environment (if need to recreate)
conda env remove -n aware-net

# Update environment (if environment.yml changes)
conda env update -f environment.yml
```

## Dataset Configuration Management

### Core Configuration Files
- **`src/utils/dataset_config.py`**: Core configuration management class
- **`config/dataset_paths.json`**: Your specific dataset path configuration
- **`scripts/setup_dataset_config.py`**: Interactive configuration setup tool

### Configuration Validation
```bash
# Validate all dataset paths
python scripts/test_config.py

# View current configuration
python scripts/preprocess_datasets_v2.py --print-config
```

## Dataset Statistics (Current Configuration)

After running `python scripts/test_config.py`:
- **CelebDF-v2**: 6,529 videos found
- **FF++**: 9,431 videos found
- **DFDC**: 1,000 videos found (classified)
- **DF40**: 206,662 pre-processed images found

## Troubleshooting

### Common Issues
- **CUDA not available**: Verify NVIDIA drivers (≥537.13 for RTX 5060Ti/5090) and install PyTorch with CUDA 12.1
- **Memory errors**: Reduce batch sizes in configuration (RTX 5060Ti: 16GB, RTX 5090: 32GB VRAM)
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Face detection issues**: Switch between facenet-pytorch and mtcnn
- **RTX 5090/5060Ti compatibility**: Use pytorch-cuda=12.1 or later versions

### Performance Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training when available
- Implement proper data loading with multiple workers

## Environment Variables

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Python path for src imports
```

## Development Guidelines

### Model Training Pipeline
1. Train individual models (Xception, Res2Net101, EfficientNet-B7) with/without augmentation
2. Train ensemble with pre-trained individual models
3. Perform cross-dataset evaluation for generalization testing

### Code Conventions
- Follow existing PyTorch patterns in the codebase
- Use timm library for pre-trained models (MobileNetV4, EfficientNetV2, GenConViT)
- Implement proper error handling for GPU/CPU compatibility
- Use tqdm for progress tracking during long operations

### Face Detection
- Primary: facenet-pytorch for efficient and accurate detection
- Alternative: mtcnn-pytorch as backup option
- Configure detection parameters in main config

## Key Features

### Flexible Dataset Configuration
- JSON-based configuration system supporting multiple dataset formats
- Automatic path validation and dataset discovery
- Support for both video datasets (CelebDF-v2, FF++, DFDC) and image datasets (DF40)

### Unified Preprocessing Pipeline
- Standardized 256x256 PNG output format across all datasets
- MTCNN-based face detection with configurable parameters
- Automatic train/validation/test split generation
- CSV manifest file generation for easy data loading

### DF40 Integration
- Seamless integration with pre-processed DF40 dataset
- Automatic detection and skipping of already processed data
- Consistent image specifications across all datasets

## License

This project is for research purposes only. Please refer to respective dataset licenses for usage restrictions.

## Citation

If you use this code in your research, please cite the original paper:
```bibtex
@article{aware-net,
  title={Adaptive Weighted Averaging for Robust Ensemble Network in Deepfake Detection},
  author={[Author Information]},
  journal={[Journal Information]},
  year={[Year]}
}
```

## Contact

For questions and issues, please refer to the project documentation or create an issue in the repository.