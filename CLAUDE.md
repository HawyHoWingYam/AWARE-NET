# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWARE-NET is a PyTorch implementation of "Adaptive Weighted Averaging for Robust Ensemble Network in Deepfake Detection". The project implements a two-tier ensemble framework that hierarchically combines multiple instances of three state-of-the-art architectures: Xception, Res2Net101, and EfficientNet-B7.

### Core Architecture
1. **Tier 1**: Averages predictions within each architecture to reduce model variance
2. **Tier 2**: Learns optimal weights for each architecture's contribution through backpropagation

## Environment Setup

### Quick Start Commands
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate aware-net

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Alternative: pip installation
pip install -r requirements.txt
```

### Pre-trained Model Weights
Download required weights to `weights/` directory:
- **Res2Net101**: `res2net101_26w_4s-02a759a1.pth`
- **EfficientNet-B7**: `tf_efficientnet_b7_ns.pth`

## Development Commands

### Training and Evaluation
```bash
# Main training pipeline
python main.py

# Individual model training (when src/ directory is populated)
python src/main.py

# Data preprocessing (when implemented)
python preprocessing.py
```

### Data Preprocessing
```bash
# Setup dataset configuration (recommended)
python scripts/setup_dataset_config.py

# Validate dataset paths
python scripts/preprocess_datasets_v2.py --validate-only

# Preprocess datasets with flexible configuration (Task 0.2)
python scripts/preprocess_datasets_v2.py --config config/dataset_paths.json

# Check configuration summary
python scripts/preprocess_datasets_v2.py --print-config
```

### Testing and Validation
```bash
# Run tests (check for existing test framework first)
python -m pytest  # if pytest is used
python -m unittest discover  # if unittest is used

# Cross-dataset evaluation (when implemented)
python src/cross_evaluation.py
```

## Code Architecture

### Key Components
- **config.py**: Central configuration for all project parameters, data paths, model settings, training parameters
- **model.py**: Model definitions including single detectors and ensemble detectors
- **ensemble.py**: Core ensemble model implementation with different combination strategies
- **dataset.py**: DeepfakeDataset class and data processing functions, data splitting functionality
- **train.py**: Training process manager with model training loops, validation, and checkpoint saving
- **experiments.py**: Experiment manager for executing experiments, model evaluation, and result generation
- **visualization.py**: Visualization tools for training curves, ROC curves, confusion matrices
- **cross_evaluation.py**: Cross-dataset evaluation module for testing generalization capability

### Directory Structure
```
├── src/                    # Core implementation files
│   └── utils/             # Utility modules (dataset_config.py)
├── scripts/               # Data processing and setup scripts
│   ├── preprocess_datasets_v2.py
│   └── setup_dataset_config.py
├── config/                # Configuration files
│   └── dataset_paths.json
├── docs/                  # Documentation
│   ├── dataset_configuration_guide.md
│   ├── preprocessing_guide.md
│   └── setup_environment.md
├── dataset/               # Raw video datasets
│   ├── CelebDF-v2/
│   └── FF++/
├── processed_data/        # Processed face images (created by preprocessing)
│   ├── train/
│   ├── val/
│   ├── final_test_sets/
│   └── manifests/
├── weights/               # Pre-trained model weights
└── project_instruction/   # Project phases documentation
```

## Dataset Configuration

### Supported Datasets
- **FF++** (FaceForensics++): AUC 99.22% (no aug.), 99.47% (aug.)
- **CelebDF-v2**: AUC 100% (both configurations)
- **Cross-dataset evaluation**: FF++ → CelebDF-v2, CelebDF-v2 → FF++

### Data Processing
- Default dataset fraction: 50% (configurable)
- Train/Val/Test split: 70/15/15
- Annotation management with caching options

## Configuration Management

### Dataset Configuration (New)
Uses flexible JSON-based configuration system:
- **`src/utils/dataset_config.py`**: Core configuration management class
- **`config/dataset_paths.json`**: Your specific dataset path configuration
- **`scripts/setup_dataset_config.py`**: Interactive configuration setup tool

```bash
# Setup configuration for your dataset structure
python scripts/setup_dataset_config.py

# Validate paths without processing
python scripts/preprocess_datasets_v2.py --validate-only
```

### Model Configuration (Legacy)
All model configuration is centralized in `config.py`:
- Dataset paths and fractions
- Model architectures and parameters  
- Training hyperparameters
- Augmentation settings
- Evaluation metrics

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

## Environment Variables

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Python path for src imports
```

## Troubleshooting

### Common Issues
- **CUDA not available**: Verify NVIDIA drivers and CUDA toolkit installation
- **Memory errors**: Reduce batch sizes in configuration
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Face detection issues**: Switch between facenet-pytorch and mtcnn

### Performance Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training when available
- Implement proper data loading with multiple workers