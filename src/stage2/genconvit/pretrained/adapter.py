"""
AWARE-NET Adapter for Pretrained GenConViT
==========================================

Adapter layer that provides seamless integration between pretrained GenConViT
models and the AWARE-NET framework, ensuring consistent interfaces and
feature extraction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..common.base import GenConViTBase, GenConViTOutput, GenConViTVariant
from .original_model import OriginalGenConViT
from .config import get_huggingface_config

class AwareNetAdapter(nn.Module):
    """
    Adapter layer for integrating pretrained GenConViT with AWARE-NET
    
    This adapter provides:
    - Consistent feature extraction interface
    - AWARE-NET compatible output format
    - Stage 3 meta-model integration
    - Runtime mode switching support
    """
    
    def __init__(self, 
                 pretrained_model: OriginalGenConViT,
                 adapt_features: bool = True,
                 feature_dim: Optional[int] = None):
        """
        Args:
            pretrained_model: Pretrained GenConViT model
            adapt_features: Whether to adapt features for AWARE-NET
            feature_dim: Target feature dimension for adaptation
        """
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.adapt_features = adapt_features
        
        # Get model configuration
        self.config = pretrained_model.config if hasattr(pretrained_model, 'config') else {}
        self.variant = pretrained_model.variant
        
        # Feature adaptation layer
        if adapt_features:
            original_feature_dim = self._get_original_feature_dim()
            target_feature_dim = feature_dim or 512  # Default AWARE-NET feature dim
            
            if original_feature_dim != target_feature_dim:
                self.feature_adapter = nn.Sequential(
                    nn.Linear(original_feature_dim, target_feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.LayerNorm(target_feature_dim)
                )
            else:
                self.feature_adapter = nn.Identity()
        else:
            self.feature_adapter = nn.Identity()
        
        # AWARE-NET compatibility flags
        self.aware_net_compatible = True
        self.stage3_ready = True
        self.feature_extraction_ready = True
    
    def _get_original_feature_dim(self) -> int:
        """Get the original model's feature dimension"""
        
        try:
            # Test forward pass to determine feature dimension
            dummy_input = torch.randn(1, 3, self.pretrained_model.input_size, 
                                    self.pretrained_model.input_size)
            
            with torch.no_grad():
                features = self.pretrained_model.extract_features(dummy_input)
                if 'final_features' in features:
                    return features['final_features'].shape[1]
                elif 'combined_features' in features:
                    return features['combined_features'].shape[1]
                else:
                    # Fallback to embed_dim
                    return self.config.get('embed_dim', 96)
                    
        except Exception:
            # Safe fallback
            return self.config.get('embed_dim', 96)
    
    def forward(self, x: torch.Tensor) -> GenConViTOutput:
        """
        Forward pass through adapted model
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            AWARE-NET compatible GenConViTOutput
        """
        
        # Get original model output
        original_output = self.pretrained_model(x)
        
        # Adapt features if needed
        if self.adapt_features and hasattr(original_output, 'features') and original_output.features:
            adapted_features = {}
            
            for key, feature_tensor in original_output.features.items():
                if feature_tensor.dim() == 2:  # Feature vectors
                    adapted_features[key] = self.feature_adapter(feature_tensor)
                else:
                    adapted_features[key] = feature_tensor
            
            # Update the output with adapted features
            original_output.features = adapted_features
        
        return original_output
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features with AWARE-NET compatibility
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of adapted feature tensors
        """
        
        # Get original features
        original_features = self.pretrained_model.extract_features(x)
        
        if not self.adapt_features:
            return original_features
        
        # Adapt features
        adapted_features = {}
        
        for key, feature_tensor in original_features.items():
            if feature_tensor.dim() == 2:  # Feature vectors
                adapted_features[key] = self.feature_adapter(feature_tensor)
            else:
                adapted_features[key] = feature_tensor
        
        # Ensure 'final_features' exists for Stage 3
        if 'final_features' not in adapted_features:
            if 'combined_features' in adapted_features:
                adapted_features['final_features'] = adapted_features['combined_features']
            else:
                # Use the first available feature
                first_key = next(iter(adapted_features))
                adapted_features['final_features'] = adapted_features[first_key]
        
        return adapted_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        
        original_info = (self.pretrained_model.get_model_info() 
                        if hasattr(self.pretrained_model, 'get_model_info') 
                        else {})
        
        adapter_info = {
            'adapter_type': 'aware_net_adapter',
            'adapted_features': self.adapt_features,
            'feature_adapter_type': type(self.feature_adapter).__name__,
            'aware_net_compatible': self.aware_net_compatible,
            'stage3_ready': self.stage3_ready,
            'feature_extraction_ready': self.feature_extraction_ready
        }
        
        # Merge information
        combined_info = {**original_info, **adapter_info}
        combined_info['model_type'] = 'adapted_original_genconvit'
        
        return combined_info
    
    def save_adapter(self, save_path: str):
        """Save adapter configuration and weights"""
        
        save_dict = {
            'adapter_state_dict': self.feature_adapter.state_dict(),
            'config': self.config,
            'adapt_features': self.adapt_features,
            'variant': self.variant.value if hasattr(self.variant, 'value') else str(self.variant),
            'model_info': self.get_model_info()
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)
        
        print(f"✅ Adapter saved to: {save_path}")
    
    def load_adapter(self, load_path: str, device: Optional[str] = None):
        """Load adapter configuration and weights"""
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(load_path, map_location=device)
        
        # Load adapter weights
        if 'adapter_state_dict' in checkpoint:
            self.feature_adapter.load_state_dict(checkpoint['adapter_state_dict'])
        
        # Update configuration
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"✅ Adapter loaded from: {load_path}")
    
    def freeze_pretrained(self):
        """Freeze pretrained model parameters"""
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        print("🔒 Pretrained model parameters frozen")
    
    def unfreeze_pretrained(self):
        """Unfreeze pretrained model parameters"""
        for param in self.pretrained_model.parameters():
            param.requires_grad = True
        print("🔓 Pretrained model parameters unfrozen")
    
    def freeze_adapter(self):
        """Freeze adapter parameters"""
        for param in self.feature_adapter.parameters():
            param.requires_grad = False
        print("🔒 Adapter parameters frozen")
    
    def unfreeze_adapter(self):
        """Unfreeze adapter parameters"""
        for param in self.feature_adapter.parameters():
            param.requires_grad = True
        print("🔓 Adapter parameters unfrozen")

class PretrainedFeatureExtractor(nn.Module):
    """
    Specialized feature extractor for pretrained GenConViT models
    
    Optimized for Stage 3 meta-model integration with consistent
    feature dimensions and formats.
    """
    
    def __init__(self, 
                 adapter_model: AwareNetAdapter,
                 feature_layers: List[str] = None,
                 pooling_method: str = 'adaptive'):
        """
        Args:
            adapter_model: Adapted GenConViT model
            feature_layers: Specific layers to extract features from
            pooling_method: Method for pooling features ('adaptive', 'max', 'avg')
        """
        super().__init__()
        
        self.adapter_model = adapter_model
        self.feature_layers = feature_layers or ['final_features']
        self.pooling_method = pooling_method
        
        # Feature hooks for intermediate layer extraction
        self.feature_hooks = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction"""
        
        def create_hook(name):
            def hook(module, input, output):
                self.feature_hooks[name] = output
            return hook
        
        # Register hooks for specified layers
        for layer_name in self.feature_layers:
            if hasattr(self.adapter_model.pretrained_model, layer_name):
                layer = getattr(self.adapter_model.pretrained_model, layer_name)
                layer.register_forward_hook(create_hook(layer_name))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of extracted features
        """
        
        # Clear previous hooks
        self.feature_hooks.clear()
        
        # Forward pass to collect features
        with torch.no_grad():
            features = self.adapter_model.extract_features(x)
        
        # Add hooked features
        features.update(self.feature_hooks)
        
        # Apply pooling if needed
        pooled_features = {}
        for key, feature_tensor in features.items():
            if feature_tensor.dim() > 2:  # Spatial features
                if self.pooling_method == 'adaptive':
                    pooled = F.adaptive_avg_pool2d(feature_tensor, (1, 1)).flatten(1)
                elif self.pooling_method == 'max':
                    pooled = F.adaptive_max_pool2d(feature_tensor, (1, 1)).flatten(1)
                elif self.pooling_method == 'avg':
                    pooled = F.adaptive_avg_pool2d(feature_tensor, (1, 1)).flatten(1)
                else:
                    pooled = feature_tensor.flatten(1)  # Flatten spatial dimensions
                
                pooled_features[key] = pooled
            else:
                pooled_features[key] = feature_tensor
        
        return pooled_features

def create_adapter_model(model_name: str = 'Deressa/GenConViT',
                        variant: str = 'ED',
                        config: Optional[Dict[str, Any]] = None,
                        adapt_features: bool = True,
                        feature_dim: Optional[int] = None,
                        auto_load_weights: bool = True) -> AwareNetAdapter:
    """
    Factory function to create adapted GenConViT model
    
    Args:
        model_name: HuggingFace model identifier
        variant: Model variant ('ED' or 'VAE')
        config: Optional configuration overrides
        adapt_features: Whether to adapt features for AWARE-NET
        feature_dim: Target feature dimension
        auto_load_weights: Whether to load pretrained weights
        
    Returns:
        AwareNetAdapter instance
    """
    
    from .original_model import create_original_genconvit
    
    # Create original model
    pretrained_model = create_original_genconvit(
        model_name=model_name,
        variant=variant,
        config=config,
        auto_load_weights=auto_load_weights
    )
    
    # Create adapter
    adapter = AwareNetAdapter(
        pretrained_model=pretrained_model,
        adapt_features=adapt_features,
        feature_dim=feature_dim
    )
    
    return adapter

def get_pretrained_config(model_name: str = 'Deressa/GenConViT',
                         variant: str = 'ED',
                         custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration for pretrained GenConViT model
    
    Args:
        model_name: Model identifier
        variant: Model variant
        custom_config: Custom configuration overrides
        
    Returns:
        Complete model configuration
    """
    
    from ..common.base import GenConViTVariant
    variant_enum = GenConViTVariant(variant.upper())
    
    return get_huggingface_config(model_name, variant_enum, custom_config)

def benchmark_adapter_performance(adapter: AwareNetAdapter,
                                 input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                                 num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark adapter model performance
    
    Args:
        adapter: Adapter model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    
    import time
    
    adapter.eval()
    device = next(adapter.parameters()).device
    
    # Warmup
    dummy_input = torch.randn(input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = adapter(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            output = adapter(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_time_ms': sum(times) / len(times),
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min_time_ms': min(times),
        'max_time_ms': max(times),
        'fps': 1000.0 / (sum(times) / len(times)),
        'device': str(device)
    }