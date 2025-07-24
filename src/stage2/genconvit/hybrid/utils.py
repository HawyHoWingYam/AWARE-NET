"""
Hybrid GenConViT Utilities
==========================

Utilities for hybrid GenConViT implementation including
model loading, saving, and analysis tools.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .model import GenConViTED, GenConViTVAE, create_genconvit_hybrid
from .config import get_hybrid_config, create_config_summary
from ..common.base import GenConViTVariant, GenConViTOutput

def save_hybrid_model(model: nn.Module, 
                     save_path: str,
                     config: Dict[str, Any],
                     metrics: Optional[Dict[str, float]] = None,
                     epoch: Optional[int] = None) -> None:
    """
    Save hybrid GenConViT model with complete state
    
    Args:
        model: Model to save
        save_path: Path to save the model
        config: Model configuration
        metrics: Optional training metrics
        epoch: Optional epoch number
    """
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'hybrid_genconvit',
        'variant': config['variant'],
        'timestamp': torch.datetime.now().isoformat()
    }
    
    if metrics:
        save_dict['metrics'] = metrics
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(save_dict, save_path)
    
    # Save human-readable config summary
    config_path = Path(save_path).with_suffix('.config.txt')
    with open(config_path, 'w') as f:
        f.write(create_config_summary(config))

def load_hybrid_model(model_path: str, 
                     device: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load hybrid GenConViT model
    
    Args:
        model_path: Path to saved model
        device: Target device
        
    Returns:
        (model, config) tuple
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Validate checkpoint
    required_keys = ['model_state_dict', 'config', 'variant']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Invalid checkpoint: missing key '{key}'")
    
    config = checkpoint['config']
    variant = checkpoint['variant']
    
    # Create model
    model = create_genconvit_hybrid(variant, config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config

def visualize_hybrid_features(model: nn.Module,
                             input_tensor: torch.Tensor,
                             save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Visualize features from hybrid GenConViT model
    
    Args:
        model: Trained hybrid model
        input_tensor: Input tensor [B, 3, H, W]
        save_path: Optional path to save visualizations
        
    Returns:
        Dictionary of feature arrays
    """
    
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Get model output and features
        output = model(input_tensor)
        extracted_features = model.extract_features(input_tensor)
        
        # Convert to numpy for visualization
        features_np = {}
        for key, features in extracted_features.items():
            features_np[key] = features.cpu().numpy()
        
        # Add reconstruction
        features_np['reconstruction'] = output.reconstruction.cpu().numpy()
        
        if save_path:
            _plot_feature_maps(features_np, input_tensor.cpu().numpy(), save_path)
    
    return features_np

def _plot_feature_maps(features: Dict[str, np.ndarray],
                      original_input: np.ndarray,
                      save_path: str):
    """Plot feature maps and reconstructions"""
    
    batch_size = original_input.shape[0]
    sample_idx = 0  # Use first sample in batch
    
    # Prepare plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hybrid GenConViT Feature Visualization', fontsize=16)
    
    # Original image
    original = original_input[sample_idx].transpose(1, 2, 0)
    original = (original - original.min()) / (original.max() - original.min())
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    # Reconstruction
    if 'reconstruction' in features:
        recon = features['reconstruction'][sample_idx].transpose(1, 2, 0)
        recon = np.clip(recon, 0, 1)
        axes[0, 1].imshow(recon)
        axes[0, 1].set_title('Reconstruction')
        axes[0, 1].axis('off')
    
    # Reconstruction error
    if 'reconstruction' in features:
        error = np.abs(original - recon)
        axes[0, 2].imshow(error)
        axes[0, 2].set_title('Reconstruction Error')
        axes[0, 2].axis('off')
    
    # Feature histograms
    feature_keys = ['hybrid_embedding', 'final_features']
    for i, key in enumerate(feature_keys):
        if key in features and i < 2:
            feature_data = features[key][sample_idx]
            axes[1, i].hist(feature_data.flatten(), bins=50, alpha=0.7)
            axes[1, i].set_title(f'{key} Distribution')
            axes[1, i].set_xlabel('Feature Value')
            axes[1, i].set_ylabel('Frequency')
    
    # Feature correlation (if multiple features available)
    if len([k for k in features.keys() if 'features' in k]) >= 2:
        feature_names = [k for k in features.keys() if 'features' in k][:2]
        f1 = features[feature_names[0]][sample_idx]
        f2 = features[feature_names[1]][sample_idx]
        
        # Use subset of features for correlation plot
        subset_size = min(100, len(f1))
        indices = np.random.choice(len(f1), subset_size, replace=False)
        
        axes[1, 2].scatter(f1[indices], f2[indices], alpha=0.6)
        axes[1, 2].set_title(f'{feature_names[0]} vs {feature_names[1]}')
        axes[1, 2].set_xlabel(feature_names[0])
        axes[1, 2].set_ylabel(feature_names[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_model_complexity(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze computational complexity of hybrid model
    
    Args:
        config: Model configuration
        
    Returns:
        Complexity analysis results
    """
    
    from .config import get_model_size_estimate
    
    # Get size estimates
    size_info = get_model_size_estimate(config)
    
    # FLOPs estimation (rough)
    input_size = config['input_size']
    
    # Encoder FLOPs
    encoder_flops = 0
    current_size = input_size
    for i, ch in enumerate(config['encoder_channels']):
        prev_ch = 3 if i == 0 else config['encoder_channels'][i-1]
        # Conv + pooling
        encoder_flops += prev_ch * ch * 9 * current_size * current_size
        current_size //= 2
    
    # Decoder FLOPs (reverse)
    decoder_flops = 0
    for i, ch in enumerate(config['decoder_channels']):
        next_ch = config['decoder_channels'][i+1] if i+1 < len(config['decoder_channels']) else 3
        decoder_flops += ch * next_ch * 4 * current_size * current_size
        current_size *= 2
    
    # Hybrid embedding FLOPs (backbone + embedder)
    backbone_flops = 100_000_000  # Rough estimate for ConvNeXt
    embedder_flops = 200_000_000  # Rough estimate for Swin
    
    total_flops = encoder_flops + decoder_flops + backbone_flops + embedder_flops
    
    analysis = {
        **size_info,
        'encoder_flops': encoder_flops,
        'decoder_flops': decoder_flops,
        'backbone_flops': backbone_flops,
        'embedder_flops': embedder_flops,
        'total_flops': total_flops,
        'flops_per_sample': total_flops,
        'complexity_category': _categorize_complexity(total_flops, size_info['total_parameters'])
    }
    
    return analysis

def _categorize_complexity(flops: int, parameters: int) -> str:
    """Categorize model complexity"""
    
    if flops < 1e9 and parameters < 10e6:
        return 'lightweight'
    elif flops < 5e9 and parameters < 50e6:
        return 'medium'
    elif flops < 20e9 and parameters < 200e6:
        return 'heavy'
    else:
        return 'very_heavy'

def benchmark_hybrid_model(model: nn.Module,
                          input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                          num_runs: int = 100,
                          warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark hybrid model performance
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Benchmark results
    """
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                output = model(dummy_input)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                import time
                start_time = time.time()
                output = model(dummy_input)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            
            times.append(elapsed_time)
    
    # Calculate statistics
    times = np.array(times)
    
    results = {
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
        'fps': 1000.0 / np.mean(times),
        'batch_size': input_shape[0],
        'device': str(device),
        'num_runs': num_runs
    }
    
    return results

def convert_to_aware_net_format(model: nn.Module,
                               config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hybrid model to AWARE-NET compatible format
    
    Args:
        model: Trained hybrid model
        config: Model configuration
        
    Returns:
        AWARE-NET format dictionary
    """
    
    aware_net_format = {
        'model_type': 'genconvit_hybrid',
        'variant': config['variant'],
        'architecture': {
            'backbone': config['backbone'],
            'embedder': config['embedder'],
            'input_size': config['input_size'],
            'embed_dim': config['embed_dim']
        },
        'feature_extractor': {
            'extract_method': 'extract_features',
            'feature_dim': config['embed_dim'] + (
                config.get('latent_dim', 4) if config.get('use_vae') else 
                sum(config['encoder_channels'])
            ),
            'feature_keys': ['final_features']
        },
        'training_config': {
            'loss_weights': {
                'classification': config['classification_weight'],
                'reconstruction': config['reconstruction_weight']
            }
        },
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
        'compatibility': {
            'stage2_ready': True,
            'stage3_ready': True,
            'feature_extraction_ready': True
        }
    }
    
    if config.get('use_vae'):
        aware_net_format['training_config']['loss_weights']['kl_divergence'] = config['kl_weight']
    
    return aware_net_format

def create_hybrid_model_report(model: nn.Module,
                              config: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
    """
    Create comprehensive model report
    
    Args:
        model: Trained model
        config: Model configuration
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    
    # Get analysis
    complexity = analyze_model_complexity(config)
    aware_net_format = convert_to_aware_net_format(model, config)
    
    # Create report
    report = f"""
GenConViT Hybrid Model Report
============================

Model Information:
- Variant: {config['variant']}
- Architecture: {config['backbone']} + {config['embedder']}
- Input Size: {config['input_size']}x{config['input_size']}
- Parameters: {complexity['total_parameters']:,}
- Model Size: {complexity['model_size_mb']:.1f} MB
- Complexity: {complexity['complexity_category']}

Architecture Details:
- Encoder Layers: {len(config['encoder_channels'])}
- Decoder Layers: {len(config['decoder_channels'])}
- Embedding Dimension: {config['embed_dim']}
- VAE Enabled: {config.get('use_vae', False)}

Performance Estimates:
- FLOPs: {complexity['total_flops']:,}
- Recommended VRAM: {complexity['recommended_vram_gb']:.1f} GB
- Memory per Batch: {complexity['batch_memory_mb']:.1f} MB

AWARE-NET Compatibility:
- Stage 2 Ready: ✅
- Stage 3 Ready: ✅
- Feature Extraction: ✅
- Feature Dimension: {aware_net_format['feature_extractor']['feature_dim']}

Loss Configuration:
- Classification Weight: {config['classification_weight']}
- Reconstruction Weight: {config['reconstruction_weight']}
{f"- KL Weight: {config['kl_weight']}" if config.get('use_vae') else ""}

Training Configuration:
- Batch Size: {config['batch_size']}
- Learning Rate: {config['learning_rate']}
- Optimizer: {config['optimizer']}
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report.strip())
    
    return report.strip()