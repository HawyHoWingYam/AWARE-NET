"""
Pretrained GenConViT Configuration
=================================

Configuration management for pretrained GenConViT models from Hugging Face.
Maps original model configurations to AWARE-NET compatible format.
"""

from typing import Dict, Any, Optional, List
from ..common.base import GenConViTVariant

# Hugging Face model configurations based on the original repository
PRETRAINED_CONFIGS = {
    'deressa/genconvit': {
        # Model metadata
        'model_name': 'Deressa/GenConViT',
        'repo_id': 'Deressa/GenConViT',
        'model_type': 'genconvit_original',
        'variant': 'ED',  # Default to ED variant
        
        # Original architecture specifications
        'input_size': 224,
        'patch_size': 4,
        'embed_dim': 96,
        'num_classes': 1,
        
        # Original model architecture details
        'backbone': 'convnext_tiny',
        'swin_variant': 'swin_tiny_patch4_window7_224',
        'encoder_layers': 5,
        'decoder_layers': 5,
        'latent_dim': 4,  # For VAE variant
        
        # Training configuration from original paper
        'learning_rate': 1e-4,
        'batch_size': 16,
        'weight_decay': 1e-4,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'step_size': 15,
        'gamma': 0.1,
        
        # Loss weights from paper
        'classification_weight': 1.0,
        'reconstruction_weight': 1.0,
        'kl_weight': 0.5,
        
        # File mappings
        'model_files': {
            'config': 'config.json',
            'weights': 'pytorch_model.bin',
            'tokenizer': 'tokenizer.json'  # If applicable
        },
        
        # AWARE-NET compatibility settings
        'aware_net_compatible': True,
        'feature_extraction_ready': True,
        'unified_output_format': True,
        
        # Version info
        'version': '1.0.0',
        'last_updated': '2024-01-01'
    }
}

# Alternative model sources (if available)
ALTERNATIVE_CONFIGS = {
    'github_original': {
        'source': 'https://github.com/erprogs/GenConViT',
        'model_type': 'github_clone',
        'description': 'Direct clone from original GitHub repository',
        'requires_manual_setup': True
    }
}

def get_huggingface_config(model_name: str = 'deressa/genconvit',
                          variant: GenConViTVariant = GenConViTVariant.ED,
                          custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get Hugging Face model configuration
    
    Args:
        model_name: HuggingFace model identifier
        variant: Model variant (ED or VAE)
        custom_config: Optional custom configuration overrides
        
    Returns:
        Complete model configuration
    """
    
    # Normalize model name
    model_key = model_name.lower().replace('/', '_')
    
    if model_key not in PRETRAINED_CONFIGS:
        available = list(PRETRAINED_CONFIGS.keys())
        raise ValueError(f"Unknown pretrained model '{model_name}'. Available: {available}")
    
    # Get base configuration
    base_config = PRETRAINED_CONFIGS[model_key].copy()
    
    # Apply variant-specific settings
    base_config['variant'] = variant.value
    base_config['use_vae'] = (variant == GenConViTVariant.VAE)
    
    # Adjust loss weights based on variant
    if variant == GenConViTVariant.VAE:
        base_config.update({
            'classification_weight': 0.7,
            'reconstruction_weight': 0.2,
            'kl_weight': 0.1
        })
    else:  # ED variant
        base_config.update({
            'classification_weight': 0.9,
            'reconstruction_weight': 0.1
        })
    
    # Apply custom overrides
    if custom_config:
        base_config.update(custom_config)
    
    # Validate and normalize
    config = validate_pretrained_config(base_config)
    
    return config

def validate_pretrained_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pretrained model configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
    """
    
    # Required keys for pretrained models
    required_keys = [
        'model_name', 'repo_id', 'input_size', 'num_classes',
        'variant', 'embed_dim'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate dimensions
    if config['input_size'] <= 0:
        raise ValueError("Input size must be positive")
    
    if config['embed_dim'] <= 0:
        raise ValueError("Embedding dimension must be positive")
    
    # Validate variant
    if config['variant'] not in ['ED', 'VAE']:
        raise ValueError(f"Invalid variant: {config['variant']}. Must be 'ED' or 'VAE'")
    
    # Set derived values
    config['is_pretrained'] = True
    config['source_type'] = 'huggingface'
    config['integration_mode'] = 'pretrained'
    
    # Feature extraction configuration
    config['feature_extraction'] = {
        'method': 'adapter_extraction',
        'feature_dim': config['embed_dim'],
        'layer_name': 'final_features'
    }
    
    return config

def get_model_download_info(model_name: str) -> Dict[str, Any]:
    """
    Get download information for a pretrained model
    
    Args:
        model_name: Model identifier
        
    Returns:
        Download configuration
    """
    
    model_key = model_name.lower().replace('/', '_')
    
    if model_key not in PRETRAINED_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = PRETRAINED_CONFIGS[model_key]
    
    return {
        'repo_id': config['repo_id'],
        'files_to_download': list(config['model_files'].values()),
        'cache_dir': f"genconvit_cache/{model_key}",
        'revision': config.get('revision', 'main'),
        'requires_auth': config.get('requires_auth', False)
    }

def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available pretrained models
    
    Returns:
        List of model information dictionaries
    """
    
    models = []
    for key, config in PRETRAINED_CONFIGS.items():
        models.append({
            'name': config['model_name'],
            'key': key,
            'repo_id': config['repo_id'],
            'variant': config['variant'],
            'description': f"GenConViT {config['variant']} from {config['repo_id']}",
            'input_size': config['input_size'],
            'parameters': config.get('parameters', 'Unknown'),
            'last_updated': config.get('last_updated', 'Unknown')
        })
    
    return models

def create_pretrained_summary(config: Dict[str, Any]) -> str:
    """
    Create human-readable summary of pretrained model configuration
    
    Args:
        config: Model configuration
        
    Returns:
        Configuration summary string
    """
    
    summary = f"""
GenConViT Pretrained Model Configuration
========================================

Model Information:
- Name: {config['model_name']}
- Repository: {config['repo_id']}
- Variant: {config['variant']}
- Version: {config.get('version', 'Unknown')}

Architecture:
- Input Size: {config['input_size']}x{config['input_size']}
- Embedding Dimension: {config['embed_dim']}
- Number of Classes: {config['num_classes']}
- Backbone: {config.get('backbone', 'Original')}

Training Configuration:
- Batch Size: {config['batch_size']}
- Learning Rate: {config['learning_rate']}
- Optimizer: {config['optimizer']}

Loss Weights:
- Classification: {config['classification_weight']}
- Reconstruction: {config['reconstruction_weight']}
{f"- KL Divergence: {config['kl_weight']}" if config.get('use_vae') else ""}

Integration:
- AWARE-NET Compatible: {config['aware_net_compatible']}
- Feature Extraction Ready: {config['feature_extraction_ready']}
- Source Type: {config['source_type']}
"""
    
    return summary.strip()

# Model capability matrix
MODEL_CAPABILITIES = {
    'deressa/genconvit': {
        'classification': True,
        'reconstruction': True,
        'feature_extraction': True,
        'vae_support': True,
        'aware_net_compatible': True,
        'batch_inference': True,
        'gpu_optimized': True
    }
}

def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """
    Get capability information for a model
    
    Args:
        model_name: Model identifier
        
    Returns:
        Capability dictionary
    """
    
    model_key = model_name.lower().replace('/', '_')
    return MODEL_CAPABILITIES.get(model_key, {})