"""
GenConViT Dual-Mode Integration Package
======================================

Unified interface for GenConViT deepfake detection models supporting:
- Hybrid Mode: Custom recreation for perfect AWARE-NET integration
- Pretrained Mode: Original weights with guaranteed performance
- Auto Mode: Intelligent selection based on availability

This package provides seamless switching between implementation approaches
while maintaining a consistent API for the AWARE-NET Stage 2 framework.

Quick Start:
    >>> from src.stage2.genconvit import create_genconvit
    >>> model, manager = create_genconvit(mode="auto", variant="ED")
    >>> output = model(input_tensor)

Advanced Usage:
    >>> from src.stage2.genconvit_manager import GenConViTManager
    >>> manager = GenConViTManager(mode="hybrid", variant="VAE")
    >>> model = manager.create_model()
    >>> 
    >>> # Switch modes seamlessly
    >>> new_model = manager.switch_mode("pretrained")
    >>> 
    >>> # Compare performance
    >>> comparison = manager.compare_modes()
"""

__version__ = "1.0.0"
__author__ = "AWARE-NET Team"

# Import key components for easy access
from .common.base import (
    GenConViTVariant,
    GenConViTOutput,
    GenConViTBase,
    HybridEmbedBase,
    EncoderBase,
    DecoderBase,
    get_device_info
)

from .common.losses import (
    GenConViTLoss,
    ReconstructionLoss,
    ClassificationLoss,
    compute_genconvit_losses
)

# Import manager and convenience functions
try:
    from ..genconvit_manager import (
        GenConViTManager,
        GenConViTMode,
        create_genconvit,
        compare_genconvit_modes
    )
    MANAGER_AVAILABLE = True
except ImportError as e:
    MANAGER_AVAILABLE = False
    _import_error = e

# Import hybrid implementation
try:
    from .hybrid.model import GenConViTED as HybridED, GenConViTVAE as HybridVAE
    from .hybrid.config import get_hybrid_config
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Import pretrained implementation
try:
    from .pretrained.original_model import OriginalGenConViT
    from .pretrained.weight_loader import load_pretrained_weights
    from .pretrained.adapter import get_pretrained_config
    PRETRAINED_AVAILABLE = True
except ImportError:
    PRETRAINED_AVAILABLE = False

# Define what's available for import
__all__ = [
    # Core classes
    'GenConViTVariant',
    'GenConViTOutput', 
    'GenConViTBase',
    'GenConViTLoss',
    
    # Manager (if available)
    'GenConViTManager',
    'create_genconvit',
    'compare_genconvit_modes',
    
    # Hybrid models (if available)
    'HybridED',
    'HybridVAE',
    'get_hybrid_config',
    
    # Pretrained models (if available)
    'OriginalGenConViT',
    'get_pretrained_config',
    
    # Utilities
    'get_device_info',
    'get_availability_info'
]

def get_availability_info():
    """Get information about available GenConViT components"""
    info = {
        'manager_available': MANAGER_AVAILABLE,
        'hybrid_available': HYBRID_AVAILABLE,
        'pretrained_available': PRETRAINED_AVAILABLE,
        'recommended_modes': []
    }
    
    if not MANAGER_AVAILABLE:
        info['manager_error'] = str(_import_error)
    
    # Determine recommended modes
    if PRETRAINED_AVAILABLE:
        info['recommended_modes'].append('pretrained')
    if HYBRID_AVAILABLE:
        info['recommended_modes'].append('hybrid')
    
    if not info['recommended_modes']:
        info['status'] = 'no_modes_available'
        info['message'] = 'No GenConViT modes are available. Check dependencies.'
    elif len(info['recommended_modes']) == 1:
        info['status'] = 'single_mode_available'
        info['message'] = f"Only {info['recommended_modes'][0]} mode available."
    else:
        info['status'] = 'all_modes_available'
        info['message'] = 'All GenConViT modes available. Use auto mode for best selection.'
    
    return info

def _check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        import torchvision
    except ImportError:
        missing_deps.append('torch/torchvision')
    
    try:
        import timm
    except ImportError:
        missing_deps.append('timm')
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        missing_deps.append('huggingface_hub')
    
    return missing_deps

def print_status():
    """Print GenConViT package status"""
    print("GenConViT Dual-Mode Integration Status")
    print("=" * 40)
    
    # Check dependencies
    missing_deps = _check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install torch torchvision timm huggingface_hub")
    else:
        print("✅ All dependencies available")
    
    # Check availability
    info = get_availability_info()
    print(f"\n📊 Component Availability:")
    print(f"   Manager: {'✅' if info['manager_available'] else '❌'}")
    print(f"   Hybrid Mode: {'✅' if info['hybrid_available'] else '❌'}")
    print(f"   Pretrained Mode: {'✅' if info['pretrained_available'] else '❌'}")
    
    print(f"\n🎯 Status: {info['status']}")
    print(f"   {info['message']}")
    
    if info['recommended_modes']:
        print(f"\n🚀 Recommended Usage:")
        if 'auto' in info['recommended_modes'] or len(info['recommended_modes']) > 1:
            print("   from src.stage2.genconvit import create_genconvit")
            print("   model, manager = create_genconvit(mode='auto')")
        else:
            mode = info['recommended_modes'][0]
            print(f"   from src.stage2.genconvit import create_genconvit")
            print(f"   model, manager = create_genconvit(mode='{mode}')")

# Convenience factory function for direct model creation
def create_model(mode: str = "auto", 
                variant: str = "ED", 
                config: dict = None,
                **kwargs):
    """
    Convenience function to create GenConViT model
    
    Args:
        mode: Integration mode ('hybrid', 'pretrained', 'auto')
        variant: Model variant ('ED', 'VAE')
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        GenConViT model instance
        
    Example:
        >>> model = create_model(mode="hybrid", variant="ED")
        >>> output = model(input_tensor)
    """
    if not MANAGER_AVAILABLE:
        raise ImportError(f"GenConViT Manager not available: {_import_error}")
    
    manager = GenConViTManager(mode=mode, variant=variant, **kwargs)
    return manager.create_model(config)

# Initialize and check status on import
if __name__ != "__main__":
    # Only check on import, not when run as script
    availability = get_availability_info()
    if availability['status'] == 'no_modes_available':
        import warnings
        warnings.warn(
            "No GenConViT modes are available. Check your dependencies. "
            "Run `python -m src.stage2.genconvit` for detailed status.",
            ImportWarning
        )

# Script mode - print detailed status
if __name__ == "__main__":
    print_status()
    
    # Demo if components available
    availability = get_availability_info()
    if availability['manager_available'] and availability['recommended_modes']:
        print(f"\n🧪 Running Demo...")
        try:
            # Create manager
            manager = GenConViTManager(mode="auto", variant="ED", verbose=False)
            print(f"✅ Manager created in {manager.mode.value} mode")
            
            # Get model info without creating full model
            print(f"✅ Device: {manager.device}")
            print(f"✅ Cache dir: {manager.cache_dir}")
            
            print(f"\n🎉 GenConViT integration ready!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        print(f"\n⚠️  Demo skipped - components not available")