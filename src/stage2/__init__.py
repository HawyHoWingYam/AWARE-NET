"""
Stage 2: Precision Analyzer Module
=================================

This module implements the second stage of the cascade detection system,
featuring two heterogeneous expert models:

1. EfficientNetV2-B3 - CNN expert specializing in local texture anomaly detection
2. GenConViT - Generative-discriminative hybrid expert for global semantic analysis

Components:
- train_stage2_effnet.py: EfficientNetV2-B3 training script
- train_stage2_genconvit.py: GenConViT training script  
- feature_extractor.py: Embedding extraction for meta-learning
- utils.py: Utility functions for Stage 2
- evaluate_stage2.py: Stage 2 evaluation tools
"""

__version__ = "1.0.0"
__author__ = "AWARE-NET Team"

# Import key components
try:
    from .train_stage2_effnet import main as train_effnet
except ImportError:
    pass

try:
    from .train_stage2_genconvit import main as train_genconvit
except ImportError:
    pass