"""
Stage 1: Fast Filter Module
===========================

This module implements the first stage of the cascade detection system:
a fast filter based on MobileNetV4-Hybrid-Medium for efficient "simple sample" filtering.

Components:
- train_stage1.py: Model training script
- calibrate_model.py: Probability calibration using temperature scaling
- evaluate_stage1.py: Comprehensive evaluation script
- utils.py: Shared utility functions
"""

__version__ = "1.0.0"
__author__ = "AWARE-NET Team"

# Import main components
try:
    from .utils import *
except ImportError:
    pass  # utils.py may not exist initially