"""
Stage 3: Meta-Model Training and Stacking Ensemble
==================================================

This module implements the third stage of the cascade deepfake detection system,
focusing on meta-model training using K-fold cross-validation and LightGBM ensemble.

Key Components:
- K-fold cross-validation pipeline for unbiased feature generation
- LightGBM meta-model training with hyperparameter optimization
- Stacking ensemble combining EfficientNetV2-B3 and GenConViT features
- Stage 4 cascade integration preparation
"""

__version__ = "1.0.0"
__author__ = "AWARE-NET Research Team"