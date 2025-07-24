"""
Original GenConViT Model Wrapper
===============================

Wrapper for the original GenConViT architecture with pretrained weights.
Provides AWARE-NET compatible interface while maintaining original model structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import warnings

from ..common.base import GenConViTBase, GenConViTOutput, GenConViTVariant
from .weight_loader import HuggingFaceWeightLoader
from .config import get_huggingface_config

class OriginalGenConViT(GenConViTBase):
    """
    Wrapper for original GenConViT architecture with pretrained weights
    
    This class provides a compatibility layer between the original GenConViT
    implementation and the AWARE-NET framework, allowing seamless use of
    pretrained weights while maintaining consistent API.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 model_name: str = 'Deressa/GenConViT',
                 auto_load_weights: bool = True):
        """
        Args:
            config: Model configuration
            model_name: HuggingFace model identifier
            auto_load_weights: Whether to automatically load pretrained weights
        """
        
        # Determine variant from config
        variant_str = config.get('variant', 'ED')
        variant = GenConViTVariant(variant_str)
        
        super().__init__(config, variant)
        
        self.model_name = model_name
        self.weight_loader = HuggingFaceWeightLoader()
        
        # Initialize original architecture components
        self._build_original_architecture(config)
        
        # Load pretrained weights if requested
        if auto_load_weights:
            self.load_pretrained_weights()
    
    def _build_original_architecture(self, config: Dict[str, Any]):
        """
        Build original GenConViT architecture based on configuration
        
        Note: This is a simplified recreation based on the paper.
        The actual original implementation may differ in details.
        """
        
        input_size = config['input_size']
        embed_dim = config['embed_dim']
        
        # Original HybridEmbed equivalent (simplified)
        self.feature_extractor = self._create_feature_extractor(config)
        
        # Original encoder-decoder structure
        self.encoder = self._create_encoder(config)
        self.decoder = self._create_decoder(config)
        
        # Classification head
        # Combine features from hybrid embedding and encoder
        classifier_input_dim = embed_dim + self._get_encoder_output_dim(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(256, self.num_classes)
        )
        
        # VAE components (if applicable)
        if self.variant == GenConViTVariant.VAE:
            latent_dim = config.get('latent_dim', 4)
            encoder_output_dim = self._get_encoder_output_dim(config)
            
            self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
            self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, encoder_output_dim)
    
    def _create_feature_extractor(self, config: Dict[str, Any]) -> nn.Module:
        """Create feature extraction module"""
        
        embed_dim = config['embed_dim']
        
        # Simplified hybrid feature extraction
        # In practice, this would use ConvNeXt + Swin combination
        feature_extractor = nn.Sequential(
            # Convolutional feature extraction
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Project to embedding dimension
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        return feature_extractor
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _create_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Create encoder for reconstruction"""
        
        # Simplified encoder (similar to hybrid implementation)
        encoder_layers = []
        in_channels = 3
        channels = [16, 32, 64, 128, 256]
        
        for out_channels in channels:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        
        return nn.Sequential(*encoder_layers)
    
    def _create_decoder(self, config: Dict[str, Any]) -> nn.Module:
        """Create decoder for reconstruction"""
        
        # Simplified decoder (reverse of encoder)
        decoder_layers = []
        channels = [256, 128, 64, 32, 16]
        
        for i, in_channels in enumerate(channels):
            if i == len(channels) - 1:
                # Final layer to RGB
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, 3, kernel_size=2, stride=2, padding=0)
                )
            else:
                out_channels = channels[i + 1]
                decoder_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        return nn.Sequential(*decoder_layers)
    
    def _get_encoder_output_dim(self, config: Dict[str, Any]) -> int:
        """Calculate encoder output dimension"""
        # Simplified calculation
        input_size = config['input_size']
        final_size = input_size // (2 ** 5)  # 5 downsampling layers
        return 256 * final_size * final_size
    
    def forward(self, x: torch.Tensor) -> GenConViTOutput:
        """
        Forward pass through original GenConViT
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            GenConViTOutput compatible with AWARE-NET
        """
        
        batch_size = x.shape[0]
        
        # Feature extraction (hybrid embedding equivalent)
        hybrid_features = self.feature_extractor(x)
        
        # Encoder-decoder path
        encoded = self.encoder(x)
        
        if self.variant == GenConViTVariant.VAE:
            # VAE path
            encoded_flat = encoded.view(batch_size, -1)
            mu = self.fc_mu(encoded_flat)
            logvar = self.fc_logvar(encoded_flat)
            
            # Reparameterization
            z = self._reparameterize(mu, logvar)
            decoded_flat = self.fc_decode(z)
            decoded = decoded_flat.view(encoded.shape)
            reconstructed = self.decoder(decoded)
            
            # Classification with latent features
            combined_features = torch.cat([hybrid_features, z], dim=1)
            
            # Store additional features
            features = {
                'hybrid_features': hybrid_features,
                'latent_mean': mu,
                'latent_logvar': logvar,
                'latent_sample': z,
                'combined_features': combined_features
            }
            
            # Return VAE output
            classification_logits = self.classifier(combined_features)
            
            return GenConViTOutput(
                classification=classification_logits,
                reconstruction=reconstructed,
                mu=mu,
                logvar=logvar,
                features=features
            )
        
        else:
            # Standard ED path
            reconstructed = self.decoder(encoded)
            
            # Classification
            encoded_flat = encoded.view(batch_size, -1)
            combined_features = torch.cat([hybrid_features, encoded_flat], dim=1)
            classification_logits = self.classifier(combined_features)
            
            # Store features
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
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features for ensemble learning
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of feature tensors
        """
        
        with torch.no_grad():
            output = self.forward(x)
            
            # Extract main features for Stage 3 compatibility
            if 'features' in output.__dict__ and output.features:
                features = output.features.copy()
                features['final_features'] = features.get('combined_features', 
                                                        features.get('hybrid_features'))
                return features
            else:
                # Fallback feature extraction
                hybrid_features = self.feature_extractor(x)
                return {
                    'final_features': hybrid_features,
                    'hybrid_features': hybrid_features
                }
    
    def load_pretrained_weights(self, 
                               model_name: Optional[str] = None,
                               strict: bool = False) -> Dict[str, Any]:
        """
        Load pretrained weights from Hugging Face
        
        Args:
            model_name: Optional override for model name
            strict: Strict loading mode
            
        Returns:
            Loading result information
        """
        
        model_name = model_name or self.model_name
        
        print(f"Loading pretrained weights for {model_name}...")
        
        try:
            loading_result = self.weight_loader.load_weights(
                self, 
                model_name, 
                strict=strict,
                device=str(next(self.parameters()).device)
            )
            
            if loading_result['success']:
                print(f"✅ Successfully loaded pretrained weights")
                self.pretrained_loaded = True
            else:
                print(f"❌ Failed to load pretrained weights: {loading_result.get('error')}")
                self.pretrained_loaded = False
            
            return loading_result
            
        except Exception as e:
            print(f"❌ Error loading pretrained weights: {e}")
            self.pretrained_loaded = False
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        return {
            'model_type': 'original_genconvit',
            'variant': self.variant.value,
            'model_name': self.model_name,
            'pretrained_loaded': getattr(self, 'pretrained_loaded', False),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'parameters': sum(p.numel() for p in self.parameters()),
            'device': str(next(self.parameters()).device)
        }

def create_original_genconvit(model_name: str = 'Deressa/GenConViT',
                             variant: str = 'ED',
                             config: Optional[Dict[str, Any]] = None,
                             auto_load_weights: bool = True) -> OriginalGenConViT:
    """
    Factory function to create original GenConViT model
    
    Args:
        model_name: HuggingFace model identifier
        variant: Model variant ('ED' or 'VAE')  
        config: Optional configuration overrides
        auto_load_weights: Whether to load pretrained weights
        
    Returns:
        OriginalGenConViT model instance
    """
    
    from ..common.base import GenConViTVariant
    
    # Get configuration
    variant_enum = GenConViTVariant(variant.upper())
    model_config = get_huggingface_config(model_name, variant_enum, config)
    
    # Create model
    model = OriginalGenConViT(
        config=model_config,
        model_name=model_name,
        auto_load_weights=auto_load_weights
    )
    
    return model