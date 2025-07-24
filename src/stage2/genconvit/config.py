"""
GenConViT Configuration
======================

Configuration parameters for GenConViT model variants
"""

GENCONVIT_CONFIG = {
    # Model Architecture
    'backbone': 'convnext_tiny',
    'embedder': 'swin_tiny_patch4_window7_224',
    'img_size': 256,  # AWARE-NET uses 256x256 images
    'patch_size': 4,
    'embed_dim': 96,
    'latent_dims': 12544,
    'num_classes': 1,  # Binary classification
    
    # Training Parameters
    'batch_size': 16,  # Conservative for autoencoder + transformer
    'learning_rate': 1e-4,  # Paper recommendation
    'weight_decay': 1e-4,
    'epochs': 50,
    
    # Loss Weights
    'classification_weight': 0.9,
    'reconstruction_weight': 0.1,
    'kl_weight': 0.01,  # For VAE variant only
    
    # Encoder-Decoder Architecture
    'encoder_channels': [64, 128, 256, 512, 1024],
    'decoder_channels': [1024, 512, 256, 128, 64],
    'encoder_layers': 5,
    'decoder_layers': 5,
    
    # Scheduler Parameters
    'scheduler_type': 'StepLR',
    'step_size': 10,
    'gamma': 0.1,
    
    # Data Augmentation
    'augmentation_prob': 0.9,  # Strong augmentation as in paper
    'horizontal_flip_prob': 0.5,
    'rotation_limit': 30,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
}

# Model variant configurations
GENCONVIT_ED_CONFIG = {
    **GENCONVIT_CONFIG,
    'variant': 'ED',
    'use_vae': False,
    'losses': ['classification', 'reconstruction']
}

GENCONVIT_VAE_CONFIG = {
    **GENCONVIT_CONFIG,
    'variant': 'VAE', 
    'use_vae': True,
    'losses': ['classification', 'reconstruction', 'kl_divergence']
}