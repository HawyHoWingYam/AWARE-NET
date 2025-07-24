"""
Hybrid GenConViT Model Implementation
====================================

Custom recreation of GenConViT architecture based on paper analysis.
Implements both ED (Encoder-Decoder) and VAE variants with perfect
AWARE-NET integration compatibility.

Architecture Components:
- HybridEmbed: ConvNeXt + Swin Transformer fusion
- Encoder: Progressive downsampling (5 layers)
- Decoder: Progressive upsampling (5 layers) 
- VAE: Latent space with reparameterization trick
- Classification: Binary deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Dict, Any, Optional, Tuple

from ..common.base import (
    GenConViTBase, 
    GenConViTOutput, 
    GenConViTVariant,
    HybridEmbedBase,
    EncoderBase,
    DecoderBase,
    create_activation,
    create_normalization
)

class HybridEmbed(HybridEmbedBase):
    """
    Hybrid embedding combining ConvNeXt backbone with Swin Transformer
    
    Based on original GenConViT HybridEmbed implementation from model_embedder.py
    """
    
    def __init__(self, 
                 backbone: str = 'convnext_tiny',
                 embedder: str = 'swin_tiny_patch4_window7_224',
                 input_size: int = 224,
                 embed_dim: int = 96):
        super().__init__(backbone, embedder, input_size, embed_dim)
        
        # ConvNeXt backbone for local feature extraction
        self.backbone = timm.create_model(
            backbone, 
            pretrained=True, 
            features_only=True,
            out_indices=[0, 1, 2, 3]  # Multi-scale features
        )
        
        # Handle Swin Transformer input size mismatch
        # Original expects 224x224, we need to adapt for different sizes
        if input_size != 224:
            # Use adaptive pooling to resize for Swin
            self.swin_resize = nn.AdaptiveAvgPool2d((224, 224))
        else:
            self.swin_resize = nn.Identity()
        
        # Swin Transformer embedder for global attention
        self.embedder = timm.create_model(
            embedder,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimensions using dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size, input_size)
            backbone_features = self.backbone(dummy_input)
            
            # Resize for Swin if needed
            dummy_swin_input = self.swin_resize(dummy_input)
            embedder_features = self.embedder(dummy_swin_input)
            
            # Calculate combined feature dimensions
            self.backbone_dim = sum(f.shape[1] for f in backbone_features)
            self.embedder_dim = embedder_features.shape[1]
            self.combined_dim = self.backbone_dim + self.embedder_dim
        
        # Fusion layers (matching original implementation style)
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, embed_dim * 2),
            nn.GELU(),  # Original uses GELU in some places
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)  # Add layer norm for stability
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid embedding
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Combined features [B, embed_dim]
        """
        batch_size = x.shape[0]
        
        # ConvNeXt multi-scale features
        backbone_features = self.backbone(x)
        
        # Global average pooling for each scale
        pooled_features = []
        for feature in backbone_features:
            pooled = F.adaptive_avg_pool2d(feature, (1, 1))
            pooled_features.append(pooled.view(batch_size, -1))
        
        # Concatenate multi-scale features
        backbone_combined = torch.cat(pooled_features, dim=1)
        
        # Swin Transformer global features (resize if needed)
        x_resized = self.swin_resize(x)
        embedder_features = self.embedder(x_resized)
        
        # Combine all features
        combined = torch.cat([backbone_combined, embedder_features], dim=1)
        
        # Fusion to final embedding dimension
        fused_features = self.fusion(combined)
        
        return fused_features

class Encoder(EncoderBase):
    """
    Encoder for GenConViT with progressive downsampling
    
    Based on original GenConViT encoder architecture:
    - 5 convolutional layers with increasing channels
    - ReLU activation and MaxPool2d downsampling
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 channels: list = None,
                 activation: str = 'relu',
                 normalization: str = 'batch'):
        
        super().__init__(in_channels, channels)
        
        if channels is None:
            channels = [16, 32, 64, 128, 256]  # Original progression
        
        self.layers = nn.ModuleList()
        prev_ch = in_channels
        
        for i, ch in enumerate(channels):
            layer = nn.Sequential(
                nn.Conv2d(prev_ch, ch, kernel_size=3, stride=1, padding=1),
                create_normalization(normalization, ch),
                create_activation(activation),
                nn.MaxPool2d(kernel_size=2, stride=2)  # Original uses MaxPool2d
            )
            self.layers.append(layer)
            prev_ch = ch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to latent representation
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Encoded features [B, channels[-1], H//32, W//32]
        """
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(DecoderBase):
    """
    Decoder for GenConViT with progressive upsampling
    
    Based on original GenConViT decoder architecture:
    - 5 transposed convolutional layers with decreasing channels
    - ReLU activation and stride=2 upsampling
    """
    
    def __init__(self, 
                 out_channels: int = 3,
                 channels: list = None,
                 activation: str = 'relu',
                 normalization: str = 'batch'):
        
        super().__init__(out_channels, channels)
        
        if channels is None:
            channels = [256, 128, 64, 32, 16]  # Original reverse progression
        
        self.layers = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            if i == len(channels) - 1:
                # Final layer to RGB (no normalization/activation)
                layer = nn.ConvTranspose2d(
                    ch, out_channels, 
                    kernel_size=2, stride=2, padding=0
                )
            else:
                # Intermediate layers
                next_ch = channels[i + 1]
                layer = nn.Sequential(
                    nn.ConvTranspose2d(
                        ch, next_ch,
                        kernel_size=2, stride=2, padding=0
                    ),
                    create_normalization(normalization, next_ch),
                    create_activation(activation)
                )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to reconstructed image
        
        Args:
            x: Encoded features [B, channels[0], H, W]
            
        Returns:
            Reconstructed image [B, out_channels, H_orig, W_orig]
        """
        for layer in self.layers:
            x = layer(x)
        return x

class GenConViTED(GenConViTBase):
    """
    GenConViT Encoder-Decoder variant
    
    Combines ConvNeXt-Swin hybrid features with autoencoder reconstruction
    for deepfake detection through both classification and reconstruction losses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, GenConViTVariant.ED)
        
        # Hybrid embedding layer
        self.hybrid_embed = HybridEmbed(
            backbone=config['backbone'],
            embedder=config['embedder'],
            input_size=config['input_size'],
            embed_dim=config['embed_dim']
        )
        
        # Encoder-Decoder for reconstruction
        self.encoder = Encoder(
            in_channels=3,
            channels=config['encoder_channels'],
            activation=config.get('activation', 'relu'),
            normalization=config.get('normalization', 'batch')
        )
        
        self.decoder = Decoder(
            out_channels=3,
            channels=config['decoder_channels'],
            activation=config.get('activation', 'relu'),
            normalization=config.get('normalization', 'batch')
        )
        
        # Calculate latent dimensions after encoding
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.input_size, self.input_size)
            encoded = self.encoder(dummy)
            self.latent_size = encoded.shape[1] * encoded.shape[2] * encoded.shape[3]
        
        # Classification head combining hybrid and reconstruction features
        classifier_input_dim = config['embed_dim'] + self.latent_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),  # Match original activation choice
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1) / 2),
            nn.Linear(256, self.num_classes)
        )
        
        # Store config for feature extraction
        self.config = config
    
    def forward(self, x: torch.Tensor) -> GenConViTOutput:
        """
        Forward pass through GenConViTED
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            GenConViTOutput with classification and reconstruction
        """
        batch_size = x.shape[0]
        
        # Hybrid embedding features
        hybrid_features = self.hybrid_embed(x)
        
        # Encoder-Decoder path
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        
        # Flatten encoded features for classification
        encoded_flat = encoded.view(batch_size, -1)
        
        # Combine hybrid and reconstruction features
        combined_features = torch.cat([hybrid_features, encoded_flat], dim=1)
        
        # Classification
        classification_logits = self.classifier(combined_features)
        
        # Additional features for analysis
        features = {
            'hybrid_features': hybrid_features,
            'encoded_features': encoded_flat,
            'combined_features': combined_features
        }
        
        return GenConViTOutput(
            classification=classification_logits,
            reconstruction=reconstructed,
            features=features
        )
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for ensemble learning
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of feature tensors
        """
        with torch.no_grad():
            # Get all intermediate features
            hybrid_features = self.hybrid_embed(x)
            encoded = self.encoder(x)
            encoded_flat = encoded.view(x.shape[0], -1)
            combined_features = torch.cat([hybrid_features, encoded_flat], dim=1)
            
            return {
                'hybrid_embedding': hybrid_features,
                'encoder_features': encoded_flat,
                'combined_features': combined_features,
                'final_features': combined_features  # Main feature for Stage 3
            }


class GenConViTVAE(GenConViTBase):
    """
    GenConViT Variational Autoencoder variant
    
    Extends GenConViTED with VAE latent space and KL divergence loss
    for enhanced generative modeling of deepfake artifacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, GenConViTVariant.VAE)
        
        self.latent_dim = config.get('latent_dim', 4)  # Original uses 4D
        
        # Hybrid embedding layer
        self.hybrid_embed = HybridEmbed(
            backbone=config['backbone'],
            embedder=config['embedder'],
            input_size=config['input_size'],
            embed_dim=config['embed_dim']
        )
        
        # VAE Encoder (modified from standard encoder)
        self.encoder = self._create_vae_encoder(
            in_channels=3,
            channels=config['encoder_channels'][:4],  # Use 4 layers for VAE
            activation=config.get('activation', 'relu'),
            normalization=config.get('normalization', 'batch')
        )
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.input_size, self.input_size)
            encoded = self.encoder(dummy)
            self.encoder_output_size = encoded.shape[1] * encoded.shape[2] * encoded.shape[3]
        
        # VAE latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, self.latent_dim)
        
        # VAE Decoder input
        self.fc_decode = nn.Linear(self.latent_dim, self.encoder_output_size)
        
        # VAE Decoder (transposed convolutions)
        self.decoder = self._create_vae_decoder(
            latent_shape=encoded.shape[1:],  # Shape after fc_decode
            channels=config['decoder_channels'][:4],  # Use 4 layers
            activation=config.get('activation', 'relu'),
            normalization=config.get('normalization', 'batch')
        )
        
        # Classification head
        classifier_input_dim = config['embed_dim'] + self.latent_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1) / 2),
            nn.Linear(256, self.num_classes)
        )
        
        self.config = config
    
    def _create_vae_encoder(self, in_channels, channels, activation, normalization):
        """Create VAE-specific encoder"""
        layers = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in channels:
            layer = nn.Sequential(
                nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                create_normalization(normalization, ch),
                create_activation('leaky_relu')  # VAE typically uses LeakyReLU
            )
            layers.append(layer)
            prev_ch = ch
        
        return nn.Sequential(*layers)
    
    def _create_vae_decoder(self, latent_shape, channels, activation, normalization):
        """Create VAE-specific decoder"""
        layers = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            if i == len(channels) - 1:
                # Final layer to RGB
                layer = nn.Sequential(
                    nn.ConvTranspose2d(ch, 3, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid()  # VAE outputs in [0,1] range
                )
            else:
                next_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]
                layer = nn.Sequential(
                    nn.ConvTranspose2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                    create_normalization(normalization, next_ch),
                    create_activation(activation)
                )
            layers.append(layer)
        
        return nn.Sequential(*layers)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        VAE reparameterization trick
        
        Args:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
            
        Returns:
            Sampled latent vector [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> GenConViTOutput:
        """
        Forward pass through GenConViTVAE
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            GenConViTOutput with classification, reconstruction, mu, and logvar
        """
        batch_size = x.shape[0]
        
        # Hybrid embedding features
        hybrid_features = self.hybrid_embed(x)
        
        # VAE Encoder
        encoded = self.encoder(x)
        encoded_flat = encoded.view(batch_size, -1)
        
        # VAE latent space
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        z = self.reparameterize(mu, logvar)
        
        # VAE Decoder
        decoded_flat = self.fc_decode(z)
        decoded = decoded_flat.view(encoded.shape)
        reconstructed = self.decoder(decoded)
        
        # Classification using hybrid + latent features
        combined_features = torch.cat([hybrid_features, z], dim=1)
        classification_logits = self.classifier(combined_features)
        
        # Additional features for analysis
        features = {
            'hybrid_features': hybrid_features,
            'latent_features': z,
            'combined_features': combined_features,
            'encoded_features': encoded_flat
        }
        
        return GenConViTOutput(
            classification=classification_logits,
            reconstruction=reconstructed,
            mu=mu,
            logvar=logvar,
            features=features
        )
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for ensemble learning
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of feature tensors
        """
        with torch.no_grad():
            # Get VAE features
            hybrid_features = self.hybrid_embed(x)
            encoded = self.encoder(x)
            encoded_flat = encoded.view(x.shape[0], -1)
            
            # VAE latent features
            mu = self.fc_mu(encoded_flat)
            logvar = self.fc_logvar(encoded_flat)
            z = self.reparameterize(mu, logvar)
            
            combined_features = torch.cat([hybrid_features, z], dim=1)
            
            return {
                'hybrid_embedding': hybrid_features,
                'latent_mean': mu,
                'latent_logvar': logvar,
                'latent_sample': z,
                'combined_features': combined_features,
                'final_features': combined_features  # Main feature for Stage 3
            }

def create_genconvit_hybrid(variant: str = "ED", 
                           config: Optional[Dict[str, Any]] = None,
                           optimization: str = 'balanced') -> GenConViTBase:
    """
    Factory function to create hybrid GenConViT models
    
    Args:
        variant: Model variant ('ED' or 'VAE')
        config: Optional configuration overrides
        optimization: Optimization mode ('fast', 'balanced', 'quality')
        
    Returns:
        GenConViT model instance
    """
    from .config import get_hybrid_config
    
    variant_enum = GenConViTVariant(variant.upper())
    model_config = get_hybrid_config(variant_enum, config, optimization)
    
    if variant_enum == GenConViTVariant.ED:
        return GenConViTED(model_config)
    else:
        return GenConViTVAE(model_config)