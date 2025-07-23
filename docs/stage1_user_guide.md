# Stage 1 User Guide: Fast Filter Implementation

This guide provides comprehensive instructions for using the Stage 1 fast filter implementation in AWARE-NET's cascade deepfake detection system.

## Overview

Stage 1 implements a fast filter using MobileNetV4-Hybrid-Medium to efficiently identify "simple real" samples and pass complex samples to subsequent stages. The implementation consists of three sequential tasks:

1. **Task 1.1**: Model Training - Fine-tune MobileNetV4 on combined datasets
2. **Task 1.2**: Probability Calibration - Apply temperature scaling for reliability
3. **Task 1.3**: Performance Evaluation - Comprehensive analysis with reliability diagrams

## Prerequisites

### Environment Setup
```bash
# Activate the conda environment
conda activate aware-net

# Verify PyTorch installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Required Data
- Preprocessed datasets in `/workspace/AWARE-NET/processed_data/`
- Training and validation manifests: `train_manifest.csv`, `val_manifest.csv`
- Ensure 256×256 PNG format for all face images

## Task 1.1: Model Training

### Basic Usage
```bash
# Navigate to project root
cd /workspace/AWARE-NET

# Run training with default parameters
python src/stage1/train_stage1.py --data_dir processed_data
```

### Advanced Configuration
```bash
# Custom training configuration
python src/stage1/train_stage1.py \
    --data_dir processed_data \
    --model_name mobilenetv4_hybrid_medium.ix_e550_r256_in1k \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --output_dir output/stage1 \
    --save_every 10 \
    --patience 15
```

### Key Parameters
- `--data_dir`: Path to processed dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: AdamW weight decay (default: 1e-5)
- `--output_dir`: Directory to save model weights and logs

### Expected Output
```
output/stage1/
├── best_model.pth          # Best model weights (based on validation AUC)
├── training_log.csv        # Training metrics log
├── training_curves.png     # Loss and AUC curves
└── final_model.pth         # Final epoch model weights
```

### Training Metrics
The training script logs the following metrics per epoch:
- **Train Loss**: BCEWithLogitsLoss on training set
- **Train Accuracy**: Binary classification accuracy
- **Val Loss**: BCEWithLogitsLoss on validation set  
- **Val Accuracy**: Binary classification accuracy
- **Val AUC**: Area Under ROC Curve (primary metric for model selection)
- **Val F1**: F1-Score for balanced evaluation

### Data Augmentation
The training applies comprehensive data augmentation:
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Task 1.2: Probability Calibration

### Basic Usage
```bash
# Run calibration using the best model from training
python src/stage1/calibrate_model.py \
    --model_path output/stage1/best_model.pth \
    --data_dir processed_data
```

### Advanced Configuration
```bash
# Custom calibration with specific validation set
python src/stage1/calibrate_model.py \
    --model_path output/stage1/best_model.pth \
    --data_dir processed_data \
    --val_manifest processed_data/manifests/val_manifest.csv \
    --output_dir output/stage1 \
    --method temperature_scaling \
    --optimize_metric nll
```

### Key Parameters
- `--model_path`: Path to trained model weights
- `--data_dir`: Path to processed dataset directory
- `--val_manifest`: Specific validation manifest file
- `--method`: Calibration method (default: temperature_scaling)
- `--optimize_metric`: Optimization target (nll or ece)

### Calibration Process
1. **Load Model**: Loads the trained MobileNetV4 model
2. **Extract Logits**: Performs inference on validation set
3. **Optimize Temperature**: Finds optimal temperature T using scipy.optimize
4. **Validate Calibration**: Calculates ECE and generates reliability diagrams

### Expected Output
```
output/stage1/
├── calibration_temp.json        # Optimal temperature value
├── reliability_diagram.png      # Before/after calibration comparison
├── calibration_metrics.json     # ECE, NLL, and other calibration metrics
└── calibrated_predictions.csv   # Validation set predictions with calibration
```

### Temperature Scaling Formula
```python
# Original model output
logits = model(images)

# Apply temperature scaling
calibrated_logits = logits / temperature

# Convert to probabilities
calibrated_probs = torch.sigmoid(calibrated_logits)
```

## Task 1.3: Performance Evaluation

### Basic Usage
```bash
# Comprehensive evaluation with calibrated model
python src/stage1/evaluate_stage1.py \
    --model_path output/stage1/best_model.pth \
    --temp_file output/stage1/calibration_temp.json \
    --data_dir processed_data
```

### Advanced Configuration
```bash
# Custom evaluation with specific test set
python src/stage1/evaluate_stage1.py \
    --model_path output/stage1/best_model.pth \
    --temp_file output/stage1/calibration_temp.json \
    --data_dir processed_data \
    --test_manifest processed_data/manifests/val_manifest.csv \
    --output_dir output/stage1 \
    --cascade_thresholds 0.9,0.95,0.98,0.99 \
    --save_predictions
```

### Key Parameters
- `--model_path`: Path to trained model weights
- `--temp_file`: Path to calibration temperature JSON file
- `--test_manifest`: Evaluation dataset manifest
- `--cascade_thresholds`: Comma-separated threshold values for cascade analysis
- `--save_predictions`: Save detailed predictions to CSV

### Evaluation Metrics
The evaluation script calculates comprehensive performance metrics:

#### Core Classification Metrics
- **AUC**: Area Under ROC Curve
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for fake detection
- **Recall**: Sensitivity for fake detection
- **Specificity**: True negative rate for real detection

#### Calibration Metrics
- **ECE**: Expected Calibration Error
- **MCE**: Maximum Calibration Error
- **Brier Score**: Probabilistic scoring rule
- **NLL**: Negative Log-Likelihood

#### Cascade Analysis Metrics
For each threshold (0.9, 0.95, 0.98, 0.99):
- **Filtration Rate**: Percentage of samples classified as "simple real"
- **Leakage Rate**: Percentage of fake samples incorrectly filtered
- **Precision at Threshold**: Accuracy of high-confidence real predictions
- **Coverage**: Percentage of samples passed to next stage

### Expected Output
```
output/stage1/
├── evaluation_report.json       # Complete metrics summary
├── roc_curve.png               # ROC curve visualization
├── confusion_matrix.png        # Confusion matrix heatmap
├── reliability_diagram.png     # Calibration reliability plot
├── cascade_analysis.png        # Threshold analysis visualization
├── predictions.csv             # Detailed predictions (if --save_predictions)
└── threshold_analysis.json     # Cascade strategy metrics
```

## Complete Pipeline Example

### Run All Three Tasks Sequentially
```bash
#!/bin/bash
# Stage 1 Complete Pipeline

echo "Starting Stage 1 Training..."
python src/stage1/train_stage1.py \
    --data_dir processed_data \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4

echo "Starting Probability Calibration..."
python src/stage1/calibrate_model.py \
    --model_path output/stage1/best_model.pth \
    --data_dir processed_data

echo "Starting Performance Evaluation..."
python src/stage1/evaluate_stage1.py \
    --model_path output/stage1/best_model.pth \
    --temp_file output/stage1/calibration_temp.json \
    --data_dir processed_data

echo "Stage 1 Pipeline Complete!"
```

## Monitoring and Logging

### Training Progress
Monitor training progress in real-time:
```bash
# Watch training log
tail -f output/stage1/training_log.csv

# View training curves
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('output/stage1/training_log.csv')
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Val')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(df['epoch'], df['val_auc'])
plt.title('Validation AUC')
plt.show()
"
```

### GPU Utilization
Monitor GPU usage during training:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check GPU memory usage
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB')
"
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python src/stage1/train_stage1.py --batch_size 16

# Enable mixed precision (if supported)
python src/stage1/train_stage1.py --mixed_precision
```

#### 2. Model Not Converging
```bash
# Reduce learning rate
python src/stage1/train_stage1.py --lr 5e-5

# Increase patience for early stopping
python src/stage1/train_stage1.py --patience 20
```

#### 3. Calibration Issues
```bash
# Try different optimization methods
python src/stage1/calibrate_model.py --optimize_method L-BFGS-B

# Use different validation split
python src/stage1/calibrate_model.py --val_split 0.2
```

#### 4. Data Loading Errors
```bash
# Validate data paths
python scripts/preprocess_datasets_v2.py --validate-only

# Check manifest files
head -5 processed_data/manifests/train_manifest.csv
```

### Performance Expectations

#### Training Time (RTX 4090/A100)
- **50 epochs**: ~2-4 hours depending on dataset size
- **Per epoch**: ~3-5 minutes for 100k samples
- **GPU utilization**: 80-95% during training

#### Memory Requirements
- **Training**: ~8-12GB VRAM (batch_size=32)
- **Inference**: ~2-4GB VRAM
- **Calibration**: ~4-6GB VRAM

#### Target Performance Metrics
- **Validation AUC**: >0.85 (minimum acceptable)
- **F1-Score**: >0.80 (balanced performance)
- **ECE (after calibration)**: <0.05 (well-calibrated)
- **Cascade efficiency**: >30% filtration at 0.98 threshold

## Integration with Cascade System

### Understanding Cascade Strategy
Stage 1 serves as a fast filter in the cascade architecture:

1. **High Confidence Real (>0.98)**: Classified as "simple real", filtered out
2. **Medium/Low Confidence**: Passed to Stage 2 for detailed analysis
3. **Fake Samples**: Should mostly pass through (low leakage rate)

### Threshold Selection Guidelines
- **Conservative (0.99)**: Very low leakage, low filtration
- **Balanced (0.98)**: Good balance of filtration and safety
- **Aggressive (0.95)**: High filtration, higher leakage risk

### Next Steps
After completing Stage 1:
1. Analyze cascade metrics to select optimal threshold
2. Prepare data pipeline for Stage 2 (ensemble analyzer)
3. Implement Stage 2 with filtered samples from Stage 1

## Best Practices

### Model Training
1. Always validate data preprocessing before training
2. Monitor validation metrics to prevent overfitting
3. Save checkpoints regularly during long training runs
4. Use early stopping based on validation AUC plateau

### Calibration
1. Use separate validation set for calibration (not training set)
2. Verify calibration quality with reliability diagrams
3. Test different optimization methods if calibration fails

### Evaluation
1. Evaluate on multiple datasets for robustness
2. Analyze failure cases to understand model limitations
3. Document performance metrics for Stage 2 integration

This completes the comprehensive Stage 1 user guide. The implementation provides a solid foundation for the cascade deepfake detection system.