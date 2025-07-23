# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWARE-NET implements a multi-stage cascade architecture for efficient deepfake detection. The system is designed with a fast filter approach, where Stage 1 serves as a high-speed preliminary filter using MobileNetV4-Hybrid-Medium, followed by more sophisticated analysis stages for complex samples.

### Core Architecture (Current Implementation)
1. **Stage 1 - Fast Filter**: MobileNetV4-Hybrid-Medium with temperature scaling calibration (✅ Complete)
2. **Stage 2-5 - Advanced Analyzers**: Heterogeneous ensemble, cross-attention fusion, temporal analysis (🔄 Planned)

### Current Status
- ✅ **Stage 1 Complete**: Training, calibration, and evaluation pipeline implemented
- 🔄 **Stages 2-5**: Advanced ensemble analyzers in development

## Environment Setup

### Quick Start Commands
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate aware-net

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install PyTorch nightly (required for compatibility)
pip install --pre --upgrade --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre --upgrade --no-cache-dir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Stage 1 Model (Current Implementation)
The current implementation uses MobileNetV4-Hybrid-Medium from the timm library:
- **Model**: `mobilenetv4_hybrid_medium.ix_e550_r256_in1k` (automatically downloaded by timm)
- **Input Size**: 256×256 RGB images
- **Output**: Binary classification (real/fake detection)

## Development Commands

### Stage 1 Training Pipeline
```bash
# Task 1.1: Train MobileNetV4-Hybrid-Medium model
python src/stage1/train_stage1.py --data_dir processed_data --epochs 50 --batch_size 32 --lr 1e-4

# Task 1.2: Calibrate model probabilities using temperature scaling
python src/stage1/calibrate_model.py --model_path output/stage1/best_model.pth --data_dir processed_data

# Task 1.3: Comprehensive performance evaluation 
python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --temp_file output/stage1/calibration_temp.json
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
# Validate dataset paths and configuration
python scripts/preprocess_datasets_v2.py --validate-only

# Test individual Stage 1 components
python src/stage1/train_stage1.py --help  # View training options
python src/stage1/calibrate_model.py --help  # View calibration options  
python src/stage1/evaluate_stage1.py --help  # View evaluation options

# Comprehensive Stage 1 evaluation with metrics
python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --temp_file output/stage1/calibration_temp.json
```

## Code Architecture

### Key Components (Current Implementation)
- **src/stage1/train_stage1.py**: Stage 1 model training with MobileNetV4-Hybrid-Medium
- **src/stage1/calibrate_model.py**: Temperature scaling calibration for probability reliability
- **src/stage1/evaluate_stage1.py**: Comprehensive performance evaluation with reliability diagrams
- **src/stage1/utils.py**: Shared utility functions for Stage 1 pipeline
- **src/utils/dataset_config.py**: Dataset configuration management class
- **scripts/preprocess_datasets_v2.py**: Multi-threaded GPU-accelerated data preprocessing
- **scripts/setup_dataset_config.py**: Interactive dataset configuration setup
- **config/dataset_paths.json**: JSON-based dataset path configuration

### Directory Structure
```
├── src/                    # Core implementation files
│   ├── stage1/            # Stage 1 fast filter implementation
│   │   ├── train_stage1.py       # Model training script
│   │   ├── calibrate_model.py    # Probability calibration
│   │   ├── evaluate_stage1.py    # Performance evaluation
│   │   └── utils.py              # Shared utilities
│   └── utils/             # Utility modules
│       └── dataset_config.py     # Dataset configuration management
├── scripts/               # Data processing and setup scripts
│   ├── preprocess_datasets_v2.py
│   └── setup_dataset_config.py
├── config/                # Configuration files
│   └── dataset_paths.json
├── docs/                  # Documentation
│   └── setup_environment.md
├── output/                # Training outputs
│   └── stage1/           # Stage 1 training results
│       ├── best_model.pth
│       ├── calibration_temp.json
│       └── evaluation_report.json
├── dataset/               # Raw video datasets
│   ├── CelebDF-v2/
│   ├── FF++/
│   ├── DFDC/
│   └── DF40/
├── processed_data/        # Processed face images (created by preprocessing)
│   ├── train/
│   ├── val/
│   ├── final_test_sets/
│   └── manifests/
└── project_instruction/   # Project phases documentation
```

## Dataset Configuration

### Supported Datasets
- **CelebDF-v2**: Celebrity deepfake detection dataset
- **FF++ (FaceForensics++)**: Face manipulation detection dataset  
- **DFDC**: Deepfake Detection Challenge dataset
- **DF40**: Pre-processed face swap dataset (256x256 PNG format)

### Data Processing
- **Output Format**: Unified 256x256 PNG images across all datasets
- **Train/Val/Test Split**: 70/15/15 (configurable)
- **Face Detection**: Multi-backend support (InsightFace, MediaPipe, YOLOv8, MTCNN)
- **GPU Acceleration**: Multi-threaded processing with 70-85% GPU utilization

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

### Stage 1 Model Configuration
Stage 1 training parameters are configured via command-line arguments:
- **Model**: MobileNetV4-Hybrid-Medium from timm library
- **Loss Function**: BCEWithLogitsLoss for binary classification
- **Optimizer**: AdamW with weight decay (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR for smooth learning rate decay
- **Data Augmentation**: RandomHorizontalFlip, ColorJitter, RandomAffine, GaussianBlur

## Development Guidelines

### Stage 1 Training Pipeline (Current Implementation)
1. **Training (Task 1.1)**: Fine-tune MobileNetV4-Hybrid-Medium on combined datasets
2. **Calibration (Task 1.2)**: Apply temperature scaling for probability reliability
3. **Evaluation (Task 1.3)**: Comprehensive performance analysis with reliability diagrams
4. **Cascade Design**: Analyze threshold strategies for multi-stage architecture

### Code Conventions
- Follow existing PyTorch patterns in the codebase
- Use timm library for pre-trained models (MobileNetV4, EfficientNetV2, GenConViT)
- Implement proper error handling for GPU/CPU compatibility
- Use tqdm for progress tracking during long operations

### Face Detection (Multi-Backend Support)
- **Primary**: InsightFace (GPU-accelerated, recommended for performance)
- **Alternatives**: MediaPipe, YOLOv8, OpenCV DNN, MTCNN
- **Configuration**: Configurable via command-line arguments in preprocessing scripts
- **GPU Optimization**: Multi-threaded processing with 70-85% GPU utilization

## Environment Variables

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Python path for src imports
```

## Troubleshooting

### Common Issues
- **CUDA not available**: Verify NVIDIA drivers and install PyTorch nightly builds
- **Memory errors**: Reduce batch sizes in Stage 1 training configuration
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Face detection issues**: Switch between different face detection backends (InsightFace, MediaPipe, YOLOv8)

### Performance Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training when available
- Implement proper data loading with multiple workers
- Monitor GPU utilization during multi-threaded preprocessing

## Personal Memory Notes

- 之後的測試大部分都是我來運行,最好詢問一下我是否我運行.
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.