"""
GenConViT Loss Functions
=======================

Shared loss function implementations for both hybrid and pretrained modes.
Implements multi-component losses: classification + reconstruction + KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .base import GenConViTOutput, GenConViTVariant

class GenConViTLoss(nn.Module):
    """
    Multi-component loss for GenConViT models
    
    Combines:
    - Classification loss (Binary Cross Entropy)
    - Reconstruction loss (MSE)
    - KL divergence loss (VAE only)
    """
    
    def __init__(self,
                 classification_weight: float = 0.9,
                 reconstruction_weight: float = 0.1,
                 kl_weight: float = 0.01,
                 variant: GenConViTVariant = GenConViTVariant.ED):
        """
        Args:
            classification_weight: Weight for classification loss
            reconstruction_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence loss (VAE only)
            variant: Model variant (ED or VAE)
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.variant = variant
        
        # Loss functions
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        
    def forward(self, 
                outputs: GenConViTOutput,
                targets: torch.Tensor,
                original_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component loss
        
        Args:
            outputs: Model outputs
            targets: Ground truth labels [B]
            original_images: Original input images [B, C, H, W]
            
        Returns:
            Dictionary of loss components
        """
        
        losses = {}
        
        # Classification loss
        classification_logits = outputs.classification.squeeze(1)
        losses['classification'] = self.classification_loss(classification_logits, targets)
        
        # Reconstruction loss
        losses['reconstruction'] = self.reconstruction_loss(
            outputs.reconstruction, 
            original_images
        )
        
        # KL divergence loss (VAE only)
        if self.variant == GenConViTVariant.VAE and outputs.mu is not None and outputs.logvar is not None:
            losses['kl_divergence'] = self._kl_divergence_loss(outputs.mu, outputs.logvar)
        
        # Total weighted loss
        total_loss = (self.classification_weight * losses['classification'] + 
                     self.reconstruction_weight * losses['reconstruction'])
        
        if 'kl_divergence' in losses:
            total_loss += self.kl_weight * losses['kl_divergence']
        
        losses['total'] = total_loss
        
        return losses
    
    def _kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss for VAE
        
        Args:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
            
        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1
        )
        return kl_loss.mean()  # Average over batch

class ReconstructionLoss(nn.Module):
    """Standalone reconstruction loss with multiple variants"""
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Args:
            loss_type: Type of reconstruction loss ('mse', 'l1', 'ssim', 'perceptual')
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'ssim':
            self.loss_fn = self._ssim_loss
        elif loss_type == 'perceptual':
            self.loss_fn = self._perceptual_loss
            self._init_perceptual_net()
        else:
            raise ValueError(f"Unknown reconstruction loss type: {loss_type}")
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss"""
        return self.loss_fn(reconstructed, original)
    
    def _ssim_loss(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """SSIM-based reconstruction loss"""
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(original, 3, 1, 1)
        mu2 = F.avg_pool2d(reconstructed, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(original * original, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(reconstructed * reconstructed, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(original * reconstructed, 3, 1, 1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def _init_perceptual_net(self):
        """Initialize VGG network for perceptual loss"""
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]  # Up to relu3_3
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
        except ImportError:
            raise ImportError("torchvision required for perceptual loss")
    
    def _perceptual_loss(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Perceptual loss using VGG features"""
        if not hasattr(self, 'vgg'):
            raise RuntimeError("VGG network not initialized")
        
        # Normalize to ImageNet mean/std
        def normalize(x):
            return F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        original_features = self.vgg(normalize(original))
        reconstructed_features = self.vgg(normalize(reconstructed))
        
        return F.mse_loss(reconstructed_features, original_features)

class ClassificationLoss(nn.Module):
    """Standalone classification loss with class balancing"""
    
    def __init__(self, 
                 loss_type: str = 'bce',
                 pos_weight: Optional[float] = None,
                 label_smoothing: float = 0.0):
        """
        Args:
            loss_type: Type of classification loss ('bce', 'focal', 'balanced')
            pos_weight: Positive class weight for imbalanced datasets
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        
        if loss_type == 'bce':
            pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        elif loss_type == 'focal':
            self.loss_fn = self._focal_loss
            self.alpha = pos_weight or 1.0
            self.gamma = 2.0
        elif loss_type == 'balanced':
            self.loss_fn = self._balanced_loss
            self.pos_weight = pos_weight or 1.0
        else:
            raise ValueError(f"Unknown classification loss type: {loss_type}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss"""
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets)
        
        return self.loss_fn(logits, targets)
    
    def _apply_label_smoothing(self, targets: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing"""
        return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
    
    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def _balanced_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Balanced loss with dynamic weighting"""
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_loss = F.binary_cross_entropy_with_logits(
            logits[pos_mask], targets[pos_mask], reduction='mean'
        ) if pos_mask.any() else 0
        
        neg_loss = F.binary_cross_entropy_with_logits(
            logits[neg_mask], targets[neg_mask], reduction='mean'
        ) if neg_mask.any() else 0
        
        return self.pos_weight * pos_loss + neg_loss

class AdversarialLoss(nn.Module):
    """Adversarial loss for enhanced training"""
    
    def __init__(self, loss_type: str = 'bce'):
        """
        Args:
            loss_type: Type of adversarial loss ('bce', 'wasserstein', 'hinge')
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'wasserstein':
            self.loss_fn = self._wasserstein_loss
        elif loss_type == 'hinge':
            self.loss_fn = self._hinge_loss
        else:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}")
    
    def forward(self, 
                real_logits: torch.Tensor,
                fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial loss for discriminator and generator
        
        Returns:
            (discriminator_loss, generator_loss)
        """
        if self.loss_type == 'bce':
            return self._bce_adversarial_loss(real_logits, fake_logits)
        else:
            return self.loss_fn(real_logits, fake_logits)
    
    def _bce_adversarial_loss(self, 
                             real_logits: torch.Tensor,
                             fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BCE-based adversarial loss"""
        batch_size = real_logits.size(0)
        real_labels = torch.ones(batch_size, 1, device=real_logits.device)
        fake_labels = torch.zeros(batch_size, 1, device=fake_logits.device)
        
        # Discriminator loss
        d_loss_real = self.loss_fn(real_logits, real_labels)
        d_loss_fake = self.loss_fn(fake_logits, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Generator loss (fool discriminator)
        g_loss = self.loss_fn(fake_logits, real_labels)
        
        return d_loss, g_loss
    
    def _wasserstein_loss(self, 
                         real_logits: torch.Tensor,
                         fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wasserstein GAN loss"""
        d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
        g_loss = -torch.mean(fake_logits)
        return d_loss, g_loss
    
    def _hinge_loss(self, 
                   real_logits: torch.Tensor,
                   fake_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hinge loss for adversarial training"""
        d_loss = torch.mean(torch.relu(1 - real_logits)) + torch.mean(torch.relu(1 + fake_logits))
        g_loss = -torch.mean(fake_logits)
        return d_loss, g_loss

def compute_genconvit_losses(outputs: GenConViTOutput,
                           targets: torch.Tensor,
                           original_images: torch.Tensor,
                           config: Dict[str, Any],
                           variant: GenConViTVariant = GenConViTVariant.ED) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute GenConViT losses
    
    Args:
        outputs: Model outputs
        targets: Ground truth labels
        original_images: Original input images
        config: Loss configuration
        variant: Model variant
        
    Returns:
        Dictionary of computed losses
    """
    
    loss_fn = GenConViTLoss(
        classification_weight=config.get('classification_weight', 0.9),
        reconstruction_weight=config.get('reconstruction_weight', 0.1),
        kl_weight=config.get('kl_weight', 0.01),
        variant=variant
    )
    
    return loss_fn(outputs, targets, original_images)