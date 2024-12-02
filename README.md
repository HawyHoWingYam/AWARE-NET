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

## Directory Structure