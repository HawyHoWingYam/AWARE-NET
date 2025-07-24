"""
GenConViT Model Implementation
=============================

Implements the GenConViT architecture with ConvNeXt-Swin hybrid layers
and autoencoder/VAE components for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class HybridEmbed(nn.Module):
    """
    Hybrid embedding combining ConvNeXt backbone with Swin Transformer embedder
    """
    
    def __init__(self, backbone='convnext_tiny', embedder='swin_tiny_patch4_window7_224', 
                 img_size=256, embed_dim=96):
        super(HybridEmbed, self).__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # ConvNeXt backbone for local feature extraction
        self.backbone = timm.create_model(
            backbone, 
            pretrained=True, 
            features_only=True,
            out_indices=[0, 1, 2, 3]  # Multi-scale features
        )
        
        # Swin Transformer embedder for global attention
        self.embedder = timm.create_model(
            embedder,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            backbone_features = self.backbone(dummy_input)
            embedder_features = self.embedder(dummy_input)
            
            # Calculate combined feature dimensions
            self.backbone_dim = sum(f.shape[1] for f in backbone_features)
            self.embedder_dim = embedder_features.shape[1]
            self.combined_dim = self.backbone_dim + self.embedder_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
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
        
        # Swin Transformer global features
        embedder_features = self.embedder(x)
        
        # Combine all features
        combined = torch.cat([backbone_combined, embedder_features], dim=1)
        
        # Fusion
        fused_features = self.fusion(combined)
        
        return fused_features

class Encoder(nn.Module):
    """
    Encoder for GenConViT with progressive downsampling
    """
    
    def __init__(self, in_channels=3, channels=[64, 128, 256, 512, 1024]):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            prev_ch = ch
            
    def forward(self, x):
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

class Decoder(nn.Module):
    """
    Decoder for GenConViT with progressive upsampling
    """
    
    def __init__(self, out_channels=3, channels=[1024, 512, 256, 128, 64]):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            if i == len(channels) - 1:
                # Final layer to RGB
                self.layers.append(nn.Sequential(
                    nn.ConvTranspose2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()  # Output in [-1, 1] range
                ))
            else:
                # Intermediate layers
                next_ch = channels[i + 1] if i + 1 < len(channels) else channels[-1]
                self.layers.append(nn.Sequential(
                    nn.ConvTranspose2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_ch),
                    nn.ReLU(inplace=True)
                ))
                
    def forward(self, x):
        """
        Decode latent features to reconstructed image
        
        Args:
            x: Encoded features [B, channels[0], H, W]
            
        Returns:
            Reconstructed image [B, 3, H_orig, W_orig]
        """
        for layer in self.layers:
            x = layer(x)
        return x

class GenConViTED(nn.Module):
    """
    GenConViT Encoder-Decoder variant
    
    Combines ConvNeXt-Swin hybrid features with autoencoder reconstruction
    for deepfake detection through both classification and reconstruction losses.
    """
    
    def __init__(self, config):
        super(GenConViTED, self).__init__()
        
        self.config = config
        self.img_size = config['img_size']
        self.num_classes = config['num_classes']
        
        # Hybrid embedding layer
        self.hybrid_embed = HybridEmbed(
            backbone=config['backbone'],
            embedder=config['embedder'],
            img_size=config['img_size'],
            embed_dim=config['embed_dim']
        )
        
        # Encoder-Decoder for reconstruction
        self.encoder = Encoder(
            in_channels=3,
            channels=config['encoder_channels']
        )
        
        self.decoder = Decoder(
            out_channels=3,
            channels=config['decoder_channels']
        )
        
        # Calculate latent dimensions after encoding
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size)
            encoded = self.encoder(dummy)
            self.latent_size = encoded.shape[1] * encoded.shape[2] * encoded.shape[3]
        
        # Classification head combining hybrid features and reconstruction features
        self.classifier = nn.Sequential(
            nn.Linear(config['embed_dim'] + self.latent_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through GenConViTED
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            dict: {
                'classification': Classification logits [B, num_classes],
                'reconstruction': Reconstructed images [B, 3, H, W]
            }
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
        
        return {
            'classification': classification_logits,
            'reconstruction': reconstructed
        }

class GenConViTVAE(nn.Module):
    """
    GenConViT Variational Autoencoder variant
    
    Extends GenConViTED with VAE latent space and KL divergence loss
    for enhanced generative modeling of deepfake artifacts.
    """
    
    def __init__(self, config):
        super(GenConViTVAE, self).__init__()
        
        self.config = config
        self.img_size = config['img_size']
        self.num_classes = config['num_classes']
        self.latent_dim = 512  # VAE latent dimension
        
        # Hybrid embedding layer
        self.hybrid_embed = HybridEmbed(
            backbone=config['backbone'],
            embedder=config['embedder'],
            img_size=config['img_size'],
            embed_dim=config['embed_dim']
        )
        
        # VAE Encoder
        self.encoder = Encoder(
            in_channels=3,
            channels=config['encoder_channels']
        )
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size)
            encoded = self.encoder(dummy)
            self.encoder_output_size = encoded.shape[1] * encoded.shape[2] * encoded.shape[3]
        
        # VAE latent space
        self.fc_mu = nn.Linear(self.encoder_output_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, self.latent_dim)
        
        # VAE Decoder input
        self.fc_decode = nn.Linear(self.latent_dim, self.encoder_output_size)
        
        # VAE Decoder
        self.decoder = Decoder(
            out_channels=3,
            channels=config['decoder_channels']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['embed_dim'] + self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
    def reparameterize(self, mu, logvar):
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
        
    def forward(self, x):
        """
        Forward pass through GenConViTVAE
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            dict: {
                'classification': Classification logits [B, num_classes],
                'reconstruction': Reconstructed images [B, 3, H, W],
                'mu': Latent means [B, latent_dim],
                'logvar': Latent log variances [B, latent_dim]
            }
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
        
        return {
            'classification': classification_logits,
            'reconstruction': reconstructed,
            'mu': mu,
            'logvar': logvar
        }