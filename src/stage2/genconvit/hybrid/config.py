"""
Hybrid GenConViT Configuration
=============================

Configuration management for custom GenConViT recreation.
Based on original paper specifications with AWARE-NET optimizations.
"""

from typing import Dict, Any, Optional
from ..common.base import GenConViTVariant

# Base configuration matching original GenConViT specifications
HYBRID_BASE_CONFIG = {
    # Model Architecture
    'backbone': 'convnext_tiny',
    'embedder': 'swin_tiny_patch4_window7_224', 
    'input_size': 224,  # Original uses 224x224
    'patch_size': 4,
    'embed_dim': 96,
    'num_classes': 1,  # Binary classification
    
    # Encoder-Decoder Architecture (based on GitHub analysis)
    'encoder_channels': [16, 32, 64, 128, 256],  # Original progression
    'decoder_channels': [256, 128, 64, 32, 16],  # Reverse progression
    'encoder_layers': 5,
    'decoder_layers': 5,
    
    # VAE Configuration
    'latent_dim': 4,  # Original uses 4D latent space
    'vae_kl_weight': 0.5,  # Original paper uses 0.5
    
    # Loss Weights (based on analysis)
    'classification_weight': 1.0,  # Classification is primary
    'reconstruction_weight': 1.0,  # Reconstruction equally important
    'kl_weight': 0.5,  # KL divergence for VAE
    
    # Training Configuration
    'batch_size': 16,  # Conservative for complex architecture
    'learning_rate': 1e-4,  # Original paper recommendation
    'weight_decay': 1e-4,
    'optimizer': 'Adam',  # Original uses Adam
    'scheduler': 'StepLR',
    'step_size': 15,  # Original uses step_size=15
    'gamma': 0.1,  # Factor=0.1
    
    # Architecture Details
    'activation': 'relu',  # Original uses ReLU
    'normalization': 'batch',  # BatchNorm
    'dropout': 0.1,
    'use_attention': True,
    'use_residual': True,
    
    # AWARE-NET Specific
    'aware_net_compatible': True,
    'feature_extraction_mode': True,
    'unified_output_format': True,
}

# ED-specific configuration
HYBRID_ED_CONFIG = {
    **HYBRID_BASE_CONFIG,
    'variant': 'ED',
    'use_vae': False,
    'losses': ['classification', 'reconstruction'],
    'model_type': 'encoder_decoder',
    
    # ED-specific weights
    'classification_weight': 0.9,  # Slightly favor classification
    'reconstruction_weight': 0.1,  # Support with reconstruction
}

# VAE-specific configuration  
HYBRID_VAE_CONFIG = {
    **HYBRID_BASE_CONFIG,
    'variant': 'VAE',
    'use_vae': True,
    'losses': ['classification', 'reconstruction', 'kl_divergence'],
    'model_type': 'variational_autoencoder',
    
    # VAE-specific weights
    'classification_weight': 0.7,  # Reduce slightly for VAE complexity
    'reconstruction_weight': 0.2,  # Increase reconstruction importance
    'kl_weight': 0.1,  # Moderate KL constraint
}

# Performance optimization configs
HYBRID_FAST_CONFIG = {
    **HYBRID_ED_CONFIG,
    'backbone': 'convnext_nano',  # Smaller backbone
    'encoder_channels': [16, 32, 64],  # Fewer layers
    'decoder_channels': [64, 32, 16],
    'encoder_layers': 3,
    'decoder_layers': 3,
    'embed_dim': 64,  # Smaller embedding
    'batch_size': 32,  # Larger batch size
}

HYBRID_QUALITY_CONFIG = {
    **HYBRID_VAE_CONFIG,
    'backbone': 'convnext_small',  # Larger backbone
    'encoder_channels': [32, 64, 128, 256, 512],  # More layers
    'decoder_channels': [512, 256, 128, 64, 32],
    'encoder_layers': 5,
    'decoder_layers': 5,
    'embed_dim': 128,  # Larger embedding
    'batch_size': 8,  # Smaller batch for memory
}

def get_hybrid_config(variant: GenConViTVariant, 
                     custom_config: Optional[Dict[str, Any]] = None,
                     optimization: str = 'balanced') -> Dict[str, Any]:
    """
    Get hybrid configuration for GenConViT model
    
    Args:
        variant: Model variant (ED or VAE)
        custom_config: Optional custom configuration overrides
        optimization: Optimization mode ('fast', 'balanced', 'quality')
        
    Returns:
        Complete configuration dictionary
    """
    
    # Select base configuration
    if optimization == 'fast':
        base_config = HYBRID_FAST_CONFIG.copy()
    elif optimization == 'quality':
        base_config = HYBRID_QUALITY_CONFIG.copy()
    else:  # balanced
        if variant == GenConViTVariant.ED:
            base_config = HYBRID_ED_CONFIG.copy()
        else:  # VAE
            base_config = HYBRID_VAE_CONFIG.copy()
    
    # Apply variant-specific settings
    base_config['variant'] = variant.value
    base_config['use_vae'] = (variant == GenConViTVariant.VAE)
    
    # Apply custom overrides
    if custom_config:
        base_config.update(custom_config)
    
    # Validate configuration
    config = _validate_config(base_config)
    
    return config

def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration"""
    
    # Ensure required keys exist
    required_keys = [
        'backbone', 'embedder', 'input_size', 'num_classes',
        'encoder_channels', 'decoder_channels', 'variant'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate dimensions
    if config['input_size'] <= 0:
        raise ValueError("Input size must be positive")
    
    if config['num_classes'] <= 0:
        raise ValueError("Number of classes must be positive")
    
    # Validate encoder/decoder consistency
    if len(config['encoder_channels']) != config['encoder_layers']:
        config['encoder_layers'] = len(config['encoder_channels'])
    
    if len(config['decoder_channels']) != config['decoder_layers']:
        config['decoder_layers'] = len(config['decoder_channels'])
    
    # Validate loss weights
    weights = ['classification_weight', 'reconstruction_weight']
    if config.get('use_vae', False):
        weights.append('kl_weight')
    
    for weight in weights:
        if weight in config and config[weight] < 0:
            config[weight] = abs(config[weight])
    
    # Set derived values
    config['total_encoder_channels'] = sum(config['encoder_channels'])
    config['total_decoder_channels'] = sum(config['decoder_channels'])
    config['is_vae'] = config.get('use_vae', False)
    
    return config

def get_model_size_estimate(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate model size and memory requirements
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary with size estimates
    """
    
    # Rough parameter count estimation
    input_size = config['input_size']
    encoder_params = sum(
        # Conv layers + batch norm + activation
        (3 if i == 0 else config['encoder_channels'][i-1]) * 
        config['encoder_channels'][i] * 9 + config['encoder_channels'][i] * 2
        for i in range(len(config['encoder_channels']))
    )
    
    decoder_params = sum(
        config['decoder_channels'][i] * 
        (config['decoder_channels'][i+1] if i+1 < len(config['decoder_channels']) else 3) * 9 + 
        config['decoder_channels'][i] * 2
        for i in range(len(config['decoder_channels']))
    )
    
    # Hybrid embedding parameters (rough estimate)
    embed_params = config['embed_dim'] * 1000  # Rough backbone estimate
    
    # Classification head
    classifier_params = config['embed_dim'] * config['num_classes']
    
    total_params = encoder_params + decoder_params + embed_params + classifier_params
    
    # Memory estimates (rough)
    model_size_mb = total_params * 4 / 1024 / 1024  # FP32
    batch_memory_mb = (
        config['batch_size'] * 3 * input_size * input_size * 4 / 1024 / 1024 * 3  # Input + reconstruction + intermediate
    )
    
    return {
        'total_parameters': total_params,
        'model_size_mb': model_size_mb,
        'batch_memory_mb': batch_memory_mb,
        'recommended_vram_gb': (model_size_mb + batch_memory_mb) / 1024 * 2,  # 2x safety margin
        'encoder_parameters': encoder_params,
        'decoder_parameters': decoder_params,
        'embedding_parameters': embed_params
    }

def create_config_summary(config: Dict[str, Any]) -> str:
    """Create human-readable configuration summary"""
    
    size_info = get_model_size_estimate(config)
    
    summary = f"""
GenConViT Hybrid Configuration Summary
=====================================

Model Variant: {config['variant']}
Architecture: {config['backbone']} + {config['embedder']}
Input Size: {config['input_size']}x{config['input_size']}

Encoder: {len(config['encoder_channels'])} layers ({config['encoder_channels']})
Decoder: {len(config['decoder_channels'])} layers ({config['decoder_channels']})
Embedding Dimension: {config['embed_dim']}

Loss Configuration:
  Classification Weight: {config['classification_weight']}
  Reconstruction Weight: {config['reconstruction_weight']}
  {"KL Weight: " + str(config['kl_weight']) if config.get('use_vae') else ""}

Training Settings:
  Batch Size: {config['batch_size']}
  Learning Rate: {config['learning_rate']}
  Optimizer: {config['optimizer']}

Model Size Estimates:
  Parameters: {size_info['total_parameters']:,}
  Model Size: {size_info['model_size_mb']:.1f} MB
  Recommended VRAM: {size_info['recommended_vram_gb']:.1f} GB
"""
    
    return summary.strip()

# Preset configurations for common use cases
PRESETS = {
    'development': {
        **HYBRID_FAST_CONFIG,
        'batch_size': 4,
        'encoder_channels': [16, 32],
        'decoder_channels': [32, 16],
    },
    'production_ed': HYBRID_ED_CONFIG,
    'production_vae': HYBRID_VAE_CONFIG,
    'high_performance': HYBRID_QUALITY_CONFIG,
    'memory_efficient': HYBRID_FAST_CONFIG
}

def get_preset_config(preset_name: str, 
                     variant: GenConViTVariant = GenConViTVariant.ED) -> Dict[str, Any]:
    """
    Get a preset configuration
    
    Args:
        preset_name: Name of preset configuration
        variant: Model variant
        
    Returns:
        Preset configuration
    """
    
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    config = PRESETS[preset_name].copy()
    config['variant'] = variant.value
    config['use_vae'] = (variant == GenConViTVariant.VAE)
    
    return _validate_config(config)