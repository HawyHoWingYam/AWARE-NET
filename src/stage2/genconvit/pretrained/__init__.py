"""
Pretrained GenConViT Integration
===============================

Original GenConViT model integration with Hugging Face pretrained weights.
Provides seamless access to official GenConViT implementations while
maintaining AWARE-NET compatibility.

Components:
- OriginalGenConViT: Wrapper for original model architecture
- WeightLoader: Hugging Face model weight loading utilities
- Adapter: AWARE-NET integration adapter layer
"""

from .original_model import OriginalGenConViT, create_original_genconvit
from .weight_loader import (
    HuggingFaceWeightLoader,
    load_pretrained_weights,
    get_available_models,
    download_model_weights
)
from .adapter import (
    AwareNetAdapter,
    get_pretrained_config,
    create_adapter_model
)
from .config import (
    PRETRAINED_CONFIGS,
    get_huggingface_config,
    validate_pretrained_config
)

__all__ = [
    'OriginalGenConViT',
    'create_original_genconvit',
    'HuggingFaceWeightLoader', 
    'load_pretrained_weights',
    'get_available_models',
    'AwareNetAdapter',
    'get_pretrained_config',
    'create_adapter_model'
]