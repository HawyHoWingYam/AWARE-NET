# DeepFake-Ensemble-Detection

## Description
A comprehensive framework for deepfake detection using ensemble learning, combining state-of-the-art CNN architectures (Xception, Res2Net101, and EfficientNet-B7). The framework evaluates performance across multiple deepfake datasets (FF++ and CelebDF) with and without augmentations.

## Key Features
- **Multi-Architecture Ensemble**:
  - Xception: Efficient depthwise separable convolutions
  - Res2Net101_26w_4s: Multi-scale feature extraction
  - EfficientNet-B7-NS: Compound scaling for optimal performance
  - Learnable weights for adaptive ensemble combination

- **Comprehensive Evaluation**:
  - Cross-dataset validation (FF++ and CelebDF)
  - With/without augmentation comparison
  - Extensive metrics tracking (AUC, F1, Precision, Recall)
  - Detailed visualization tools

- **Robust Data Pipeline**:
  - Configurable data fraction usage
  - Automated train/val/test splitting
  - Advanced augmentation techniques
  - Structured annotation management

## Configuration Options

### Data Management
- **Dataset Fraction**: Control how much of the dataset to use (default: 0.5)
- **Train/Val/Test Split**: 70/15/15 split ratio
- **Annotation Management**:
  - `FORCE_NEW_ANNOTATIONS`: Force create new dataset splits (default: False)
  - `ANNOTATION_CACHE`: Cache and reuse dataset splits (default: True)

### Training Pipeline
1. **Data Preparation**:
   - Configure annotation caching in `config.py`
   - Set `FORCE_NEW_ANNOTATIONS=True` for fresh splits
   - Use cached splits for faster repeated runs

2. **Model Training**:
   - Train individual models first
   - Train ensemble model
   - Cross-dataset evaluation

# Recommended training order
1. Train individual models first without augmentation:
   - Xception (good at local features)
   - Res2Net101 (good at multi-scale features)
   - EfficientNet (good at global features)

2. Train individual models with augmentation:
   - Apply geometric + color augmentations
   - 30% augmented data
   - Save best models based on validation loss

3. Train ensemble model:
   - Initialize with pre-trained individual models
   - Fine-tune ensemble weights
   - Use both augmented and non-augmented data

## Usage