"""
GenConViT Common Base Components
===============================

Shared interfaces and base classes for both hybrid and pretrained modes.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum

class GenConViTVariant(Enum):
    """GenConViT model variants"""
    ED = "ED"    # Encoder-Decoder
    VAE = "VAE"  # Variational Autoencoder

class GenConViTOutput:
    """Standardized output format for GenConViT models"""
    
    def __init__(self, 
                 classification: torch.Tensor,
                 reconstruction: torch.Tensor,
                 mu: Optional[torch.Tensor] = None,
                 logvar: Optional[torch.Tensor] = None,
                 features: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            classification: Classification logits [B, num_classes]
            reconstruction: Reconstructed images [B, C, H, W]
            mu: VAE mean (VAE only) [B, latent_dim]
            logvar: VAE log variance (VAE only) [B, latent_dim]
            features: Additional feature maps for analysis
        """
        self.classification = classification
        self.reconstruction = reconstruction
        self.mu = mu
        self.logvar = logvar
        self.features = features or {}
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format"""
        result = {
            'classification': self.classification,
            'reconstruction': self.reconstruction
        }
        
        if self.mu is not None:
            result['mu'] = self.mu
        if self.logvar is not None:
            result['logvar'] = self.logvar
            
        result.update(self.features)
        return result
    
    def __getitem__(self, key: str) -> torch.Tensor:
        """Dictionary-style access"""
        return self.to_dict()[key]
    
    def keys(self):
        """Get available keys"""
        return self.to_dict().keys()

class GenConViTBase(nn.Module, ABC):
    """Abstract base class for GenConViT implementations"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 variant: GenConViTVariant):
        super().__init__()
        self.config = config
        self.variant = variant
        self.input_size = config.get('input_size', 224)
        self.num_classes = config.get('num_classes', 1)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> GenConViTOutput:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            GenConViTOutput with classification and reconstruction
        """
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for ensemble learning
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary of feature tensors
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'variant': self.variant.value,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }

class HybridEmbedBase(nn.Module, ABC):
    """Base class for hybrid embedding implementations"""
    
    def __init__(self, 
                 backbone: str = 'convnext_tiny',
                 embedder: str = 'swin_tiny_patch4_window7_224',
                 input_size: int = 224,
                 embed_dim: int = 96):
        super().__init__()
        self.backbone_name = backbone
        self.embedder_name = embedder
        self.input_size = input_size
        self.embed_dim = embed_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid embedding
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Embedded features [B, embed_dim]
        """
        pass

class EncoderBase(nn.Module, ABC):
    """Base class for encoder implementations"""
    
    def __init__(self, 
                 in_channels: int = 3,
                 channels: list = None):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels or [64, 128, 256, 512, 1024]
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        pass

class DecoderBase(nn.Module, ABC):
    """Base class for decoder implementations"""
    
    def __init__(self, 
                 out_channels: int = 3,
                 channels: list = None):
        super().__init__()
        self.out_channels = out_channels
        self.channels = channels or [1024, 512, 256, 128, 64]
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output"""
        pass

def create_activation(activation: str = 'relu') -> nn.Module:
    """Create activation function"""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
        'silu': nn.SiLU(inplace=True),
        'tanh': nn.Tanh()
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    
    return activations[activation]

def create_normalization(norm_type: str, num_features: int) -> nn.Module:
    """Create normalization layer"""
    normalizations = {
        'batch': nn.BatchNorm2d(num_features),
        'instance': nn.InstanceNorm2d(num_features),
        'layer': nn.GroupNorm(1, num_features),
        'group': nn.GroupNorm(num_features // 4, num_features)
    }
    
    if norm_type not in normalizations:
        raise ValueError(f"Unknown normalization: {norm_type}")
    
    return normalizations[norm_type]

def calculate_conv_output_size(input_size: int, 
                             kernel_size: int, 
                             stride: int = 1, 
                             padding: int = 0, 
                             dilation: int = 1) -> int:
    """Calculate convolution output size"""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def calculate_conv_transpose_output_size(input_size: int, 
                                       kernel_size: int, 
                                       stride: int = 1, 
                                       padding: int = 0, 
                                       output_padding: int = 0) -> int:
    """Calculate transpose convolution output size"""
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_cached': torch.cuda.memory_reserved()
        })
    
    return info