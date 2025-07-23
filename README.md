# AWARE-NET: Cascade Deepfake Detection System

AWARE-NET implements a multi-stage cascade architecture for efficient deepfake detection. The system is designed with a fast filter approach, where Stage 1 serves as a high-speed preliminary filter using MobileNetV4-Hybrid-Medium, followed by more sophisticated analysis stages for complex samples.

## Current Implementation Status

✅ **Stage 1 Complete**: Fast filter using MobileNetV4-Hybrid-Medium with temperature scaling calibration  
🔄 **Stage 2-5**: Advanced ensemble analyzers (planned)

## Stage 1: Fast Filter Architecture

The current implementation focuses on Stage 1, which serves as the first line of defense in the cascade system:

### Core Components
1. **Model**: MobileNetV4-Hybrid-Medium (efficient mobile-optimized architecture)
2. **Training**: Fine-tuned on combined dataset with comprehensive data augmentation
3. **Calibration**: Temperature scaling for reliable probability outputs
4. **Evaluation**: Comprehensive performance analysis with reliability diagrams

### Stage 1 Performance
- **Validation AUC**: >0.85 (target baseline)
- **Cascade Strategy**: Conservative threshold (>0.98) for high-confidence real samples
- **Processing Speed**: Optimized for real-time inference on mobile devices
- **Calibration**: Significant ECE reduction through temperature scaling

## Quick Start Guide

### 1. Environment Setup

#### Create Conda Environment
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate aware-net

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

#### Update PyTorch (Required)
```bash
# Install latest PyTorch nightly for compatibility
pip install --pre --upgrade --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre --upgrade --no-cache-dir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Dataset Configuration

#### Supported Datasets
- **CelebDF-v2**: Celebrity deepfake detection dataset
- **FF++ (FaceForensics++)**: Face manipulation detection dataset
- **DFDC**: Deepfake Detection Challenge dataset
- **DF40**: Pre-processed face swap dataset

#### Setup Dataset Paths
```bash
# Interactive configuration setup
python scripts/setup_dataset_config.py

# Validate configuration
python scripts/preprocess_datasets_v2.py --validate-only
```

### 3. Data Preprocessing

All datasets are processed to unified **256x256 PNG** format for consistency.

#### GPU-Accelerated Processing
```bash
# Multi-threaded GPU processing (recommended)
python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 --video-backend decord --face-detector insightface --workers 4

# Process all datasets
python scripts/preprocess_datasets_v2.py

# View configuration
python scripts/preprocess_datasets_v2.py --print-config
```

#### Output Structure
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

## Stage 1 Training Pipeline

### Task 1.1: Model Training
```bash
# Train MobileNetV4-Hybrid-Medium model
python src/stage1/train_stage1.py --data_dir processed_data --epochs 50 --batch_size 32 --lr 1e-4
```

**Features**:
- Fine-tuned MobileNetV4-Hybrid-Medium from timm library
- Binary classification with BCEWithLogitsLoss
- AdamW optimizer with CosineAnnealingLR scheduler
- Comprehensive data augmentation (RandomHorizontalFlip, ColorJitter, RandomAffine, GaussianBlur)
- Automatic best model saving based on validation AUC

### Task 1.2: Probability Calibration
```bash
# Calibrate model probabilities using temperature scaling
python src/stage1/calibrate_model.py --model_path output/stage1/best_model.pth --data_dir processed_data
```

**Features**:
- Temperature scaling optimization using scipy.optimize
- Minimizes Negative Log-Likelihood (NLL) loss
- Generates reliability diagrams for calibration visualization
- Saves optimal temperature parameter for inference

### Task 1.3: Performance Evaluation
```bash
# Comprehensive performance evaluation
python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --temp_file output/stage1/calibration_temp.json
```

**Metrics**:
- AUC Score, F1-Score, Accuracy, Confusion Matrix
- Expected Calibration Error (ECE)
- ROC curves and reliability plots
- Cascade threshold analysis (leakage and filtration rates)

## Project Structure

```
AWARE-NET/
├── src/
│   ├── stage1/                    # Stage 1 implementation
│   │   ├── train_stage1.py       # Model training script
│   │   ├── calibrate_model.py    # Probability calibration
│   │   ├── evaluate_stage1.py    # Performance evaluation
│   │   └── utils.py              # Shared utilities
│   └── utils/
│       └── dataset_config.py     # Dataset configuration management
├── scripts/                      # Data processing scripts
│   ├── preprocess_datasets_v2.py
│   └── setup_dataset_config.py
├── config/
│   └── dataset_paths.json       # Dataset path configuration
├── docs/                        # Documentation
├── processed_data/              # Processed face images (created by preprocessing)
├── output/
│   └── stage1/                  # Stage 1 training outputs
│       ├── best_model.pth
│       ├── calibration_temp.json
│       └── evaluation_report.json
├── project_instruction/         # Implementation documentation
└── dataset/                     # Raw video datasets
```

## Technical Implementation Details

### Model Architecture
- **Base Model**: MobileNetV4-Hybrid-Medium (from timm library)
- **Input Size**: 256×256 RGB images
- **Output**: Single node for binary classification (real/fake)
- **Optimization**: Fine-tuned on combined deepfake datasets

### Training Configuration
```python
# Key training parameters
model_name = "mobilenetv4_hybrid_medium.ix_e550_r256_in1k"
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Data Augmentation Strategy
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Calibration Method
- **Technique**: Temperature scaling (simple and effective)
- **Objective**: Minimize Negative Log-Likelihood
- **Formula**: `calibrated_prob = sigmoid(logits / T)`
- **Validation**: ECE and reliability diagrams

## Cascade Strategy Design

### Stage 1 Threshold Analysis
The evaluation script analyzes different confidence thresholds for cascade decisions:
- **High Confidence (>0.98)**: Samples classified as "simple real" - filtered out
- **Low/Medium Confidence**: Samples passed to Stage 2 for detailed analysis
- **Leakage Rate**: Percentage of fake samples incorrectly passed as real
- **Filtration Rate**: Percentage of samples filtered by Stage 1

### Success Metrics
- **Training Convergence**: Validation AUC >0.85, F1-Score >0.80
- **Calibration Effect**: ECE reduction >50%
- **Cascade Efficiency**: At 0.98 threshold, leakage rate <5%, filtration rate >30%

## GPU Processing Features

### Multi-threaded Performance
- **Single-threaded**: ~1.55s/video, 30-40% GPU utilization
- **Multi-threaded (4 workers)**: ~0.4-0.8s/video, **70-85% GPU utilization**
- **Speed improvement**: 2-4x faster processing

### Supported Backends
- **Face Detection**: InsightFace (GPU), MediaPipe, YOLOv8, MTCNN
- **Video Processing**: Decord (GPU), TorchVision.io, OpenCV

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Python path for src imports
```

## Troubleshooting

### Common Issues
- **CUDA not available**: Verify NVIDIA drivers and install PyTorch nightly builds
- **Memory errors**: Reduce batch sizes in training configuration
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Face detection issues**: Switch between different face detection backends

### Performance Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training when available
- Implement proper data loading with multiple workers
- Monitor GPU utilization during processing

## Development Roadmap

### Completed (Stage 1)
- ✅ MobileNetV4-Hybrid-Medium training pipeline
- ✅ Temperature scaling calibration
- ✅ Comprehensive evaluation framework
- ✅ Multi-threaded GPU preprocessing
- ✅ Unified dataset configuration system

### Planned (Stages 2-5)
- 🔄 Stage 2: Heterogeneous ensemble analyzer
- 🔄 Stage 3: Cross-attention fusion
- 🔄 Stage 4: Temporal consistency analysis
- 🔄 Stage 5: Final classification with adaptive thresholding

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

## License

This project is for research purposes only. Please refer to respective dataset licenses for usage restrictions.