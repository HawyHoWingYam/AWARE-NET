#!/usr/bin/env python3
"""
Stage 2 Feature Extraction System - feature_extractor.py
=======================================================

Unified feature extraction interface for Stage 2 precision analyzer models:
- EfficientNetV2-B3 (CNN local texture expert)
- GenConViT (Generative-discriminative hybrid expert)

Key Features:
- Multi-modal feature extraction from GenConViT
- Dimension alignment to 256-D unified space  
- Batch processing with GPU acceleration
- Memory-efficient embedding extraction
- Clean interface for Stage 3 meta-model training

Usage:
    # Extract EfficientNetV2-B3 features
    extractor = EfficientNetFeatureExtractor('efficientnetv2_b3.in21k_ft_in1k', 
                                           'output/stage2_effnet/best_model.pth')
    embeddings, labels = extractor.extract(val_dataloader)
    
    # Extract GenConViT features
    extractor = GenConViTFeatureExtractor('output/stage2_genconvit/best_model.pth',
                                        variant='ED', mode='hybrid')
    embeddings, labels = extractor.extract_features(val_dataloader)
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
try:
    from src.stage2.genconvit_manager import GenConViTManager
    from src.stage2.genconvit.common.base import GenConViTVariant, GenConViTOutput
    from src.stage1.utils import load_model_checkpoint
except ImportError as e:
    logging.warning(f"Import warning: {e}")

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_dim = None
        
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load and prepare model for feature extraction"""
        pass
    
    @abstractmethod
    def _extract_features_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from a single batch"""
        pass
    
    def extract(self, dataloader: DataLoader, 
                align_dim: Optional[int] = 256,
                return_raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from dataloader
        
        Args:
            dataloader: PyTorch DataLoader
            align_dim: Target dimension for feature alignment (None to skip)
            return_raw: Return raw features without alignment
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
                try:
                    images = images.to(self.device)
                    
                    # Extract features
                    features = self._extract_features_batch(images)
                    
                    # Ensure features are 2D [batch_size, feature_dim]
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    
                    all_features.append(features.cpu().numpy())
                    all_labels.append(labels.numpy())
                    
                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        if not all_features:
            raise RuntimeError("No features extracted. Check dataloader and model.")
            
        # Concatenate all features and labels
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Align dimensions if requested
        if align_dim is not None and not return_raw:
            features = self._align_dimensions(features, align_dim)
            
        logging.info(f"Extracted features shape: {features.shape}, Labels shape: {labels.shape}")
        return features, labels
    
    def _align_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Align feature dimensions to target size"""
        current_dim = features.shape[1]
        
        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            # Use PCA-like reduction or simple truncation
            logging.info(f"Reducing dimensions from {current_dim} to {target_dim}")
            return features[:, :target_dim]
        else:
            # Pad with zeros or repeat features
            logging.info(f"Expanding dimensions from {current_dim} to {target_dim}")
            padding = np.zeros((features.shape[0], target_dim - current_dim))
            return np.concatenate([features, padding], axis=1)

class EfficientNetFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for EfficientNetV2-B3 models"""
    
    def __init__(self, model_name: str, checkpoint_path: str, device: Optional[str] = None):
        super().__init__(device)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self._load_model()
    
    def _load_model(self):
        """Load EfficientNetV2-B3 model without classification head"""
        try:
            # Create model without classification head
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=0,  # Remove classification head
                global_pool='avg'
            )
            
            # Load trained weights
            if os.path.exists(self.checkpoint_path):
                # First load with classification head to get state dict
                temp_model = timm.create_model(
                    self.model_name,
                    pretrained=False,
                    num_classes=1
                )
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                temp_model.load_state_dict(checkpoint['model_state_dict'])
                
                # Copy weights except classifier
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in temp_model.state_dict().items() 
                                 if k in model_dict and 'classifier' not in k}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                
                logging.info(f"Loaded EfficientNetV2 weights from {self.checkpoint_path}")
            else:
                logging.warning(f"Checkpoint not found: {self.checkpoint_path}")
                
            self.model = self.model.to(self.device)
            
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                dummy_output = self.model(dummy_input)
                self.feature_dim = dummy_output.shape[1]
                
            logging.info(f"EfficientNetV2 feature dimension: {self.feature_dim}")
            
        except Exception as e:
            logging.error(f"Error loading EfficientNetV2 model: {e}")
            raise
    
    def _extract_features_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from EfficientNetV2-B3"""
        return self.model(batch)

class GenConViTFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor for GenConViT models with multi-modal features"""
    
    def __init__(self, checkpoint_path: str, variant: str = 'ED', 
                 mode: str = 'hybrid', device: Optional[str] = None):
        super().__init__(device)
        self.checkpoint_path = checkpoint_path
        self.variant = GenConViTVariant(variant)
        self.mode = mode
        self.manager = None
        self._load_model()
    
    def _load_model(self):
        """Load GenConViT model and prepare for feature extraction"""
        try:
            # Initialize GenConViT manager
            self.manager = GenConViTManager(mode=self.mode, device=self.device)
            
            if self.mode == 'hybrid':
                self.model = self.manager.create_model(variant=self.variant.value)
                if os.path.exists(self.checkpoint_path):
                    checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded GenConViT hybrid weights from {self.checkpoint_path}")
            elif self.mode == 'pretrained':
                self.model = self.manager.load_pretrained(self.checkpoint_path)
                logging.info(f"Loaded GenConViT pretrained weights")
            else:  # auto mode
                self.model = self.manager.get_best_model()
                logging.info(f"Using GenConViT auto mode")
                
            self.model = self.model.to(self.device)
            
            # Get feature dimensions by running dummy input
            self._analyze_feature_dimensions()
            
        except Exception as e:
            logging.error(f"Error loading GenConViT model: {e}")
            raise
    
    def _analyze_feature_dimensions(self):
        """Analyze GenConViT output structure and feature dimensions"""
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                output = self.model(dummy_input)
                
                if isinstance(output, GenConViTOutput):
                    # Extract feature dimensions from different components
                    self.classification_dim = output.classification.shape[1] if output.classification is not None else 0
                    
                    # Calculate reconstruction feature dimensions
                    if output.reconstruction is not None:
                        recon_features = F.adaptive_avg_pool2d(output.reconstruction, (1, 1))
                        self.reconstruction_dim = recon_features.view(recon_features.size(0), -1).shape[1]
                    else:
                        self.reconstruction_dim = 0
                    
                    # VAE features
                    self.vae_dim = 0
                    if output.mu is not None:
                        self.vae_dim += output.mu.shape[1]
                    if output.logvar is not None:
                        self.vae_dim += output.logvar.shape[1]
                    
                    # Additional features from attention/CNN layers
                    self.attention_dim = 0
                    if output.features:
                        for key, feat in output.features.items():
                            if 'attention' in key.lower() or 'swin' in key.lower():
                                if feat.dim() > 2:
                                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
                                self.attention_dim += feat.shape[1]
                    
                    # Total feature dimension
                    self.feature_dim = (self.classification_dim + self.reconstruction_dim + 
                                      self.vae_dim + self.attention_dim)
                    
                    logging.info(f"GenConViT feature dimensions - "
                               f"Classification: {self.classification_dim}, "
                               f"Reconstruction: {self.reconstruction_dim}, "
                               f"VAE: {self.vae_dim}, "
                               f"Attention: {self.attention_dim}, "
                               f"Total: {self.feature_dim}")
                else:
                    # Fallback for simple tensor output
                    if output.dim() > 2:
                        output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
                    self.feature_dim = output.shape[1]
                    logging.info(f"GenConViT simple feature dimension: {self.feature_dim}")
                    
        except Exception as e:
            logging.error(f"Error analyzing GenConViT features: {e}")
            self.feature_dim = 512  # Fallback
    
    def _extract_features_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract multi-modal features from GenConViT"""
        output = self.model(batch)
        
        if isinstance(output, GenConViTOutput):
            features = []
            
            # Classification features (typically from final layers before classification)
            if output.classification is not None:
                # Extract pre-classification features if available
                # Otherwise use classification logits directly
                features.append(output.classification)
            
            # Reconstruction features (compressed representation)
            if output.reconstruction is not None:
                recon_features = F.adaptive_avg_pool2d(output.reconstruction, (1, 1))
                recon_features = recon_features.view(recon_features.size(0), -1)
                features.append(recon_features)
            
            # VAE latent features
            if output.mu is not None:
                features.append(output.mu)
            if output.logvar is not None:
                features.append(output.logvar)
            
            # Attention/CNN hybrid features
            if output.features:
                for key, feat in output.features.items():
                    if 'attention' in key.lower() or 'swin' in key.lower() or 'convnext' in key.lower():
                        if feat.dim() > 2:
                            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
                        features.append(feat)
            
            # Concatenate all features
            if features:
                return torch.cat(features, dim=1)
            else:
                logging.warning("No features extracted from GenConViT output")
                return torch.zeros(batch.size(0), 512).to(batch.device)
        else:
            # Simple tensor output
            if output.dim() > 2:
                output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
            return output
    
    def extract_features(self, dataloader: DataLoader, 
                        align_dim: Optional[int] = 256,
                        return_components: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                                Dict[str, np.ndarray]]:
        """
        Extract GenConViT features with optional component separation
        
        Args:
            dataloader: PyTorch DataLoader
            align_dim: Target dimension for alignment
            return_components: Return separate feature components
            
        Returns:
            If return_components=False: (features, labels)
            If return_components=True: Dictionary with separate components
        """
        if return_components:
            return self._extract_components(dataloader, align_dim)
        else:
            return self.extract(dataloader, align_dim)
    
    def _extract_components(self, dataloader: DataLoader, 
                           align_dim: Optional[int]) -> Dict[str, np.ndarray]:
        """Extract separate feature components for analysis"""
        self.model.eval()
        components = {
            'classification': [],
            'reconstruction': [],
            'vae_mu': [],
            'vae_logvar': [],
            'attention': [],
            'labels': []
        }
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting components"):
                images = images.to(self.device)
                output = self.model(images)
                
                if isinstance(output, GenConViTOutput):
                    # Classification component
                    if output.classification is not None:
                        components['classification'].append(output.classification.cpu().numpy())
                    
                    # Reconstruction component
                    if output.reconstruction is not None:
                        recon_feat = F.adaptive_avg_pool2d(output.reconstruction, (1, 1))
                        recon_feat = recon_feat.view(recon_feat.size(0), -1)
                        components['reconstruction'].append(recon_feat.cpu().numpy())
                    
                    # VAE components
                    if output.mu is not None:
                        components['vae_mu'].append(output.mu.cpu().numpy())
                    if output.logvar is not None:
                        components['vae_logvar'].append(output.logvar.cpu().numpy())
                    
                    # Attention features
                    if output.features:
                        attention_feats = []
                        for key, feat in output.features.items():
                            if 'attention' in key.lower() or 'swin' in key.lower():
                                if feat.dim() > 2:
                                    feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
                                attention_feats.append(feat)
                        if attention_feats:
                            components['attention'].append(torch.cat(attention_feats, dim=1).cpu().numpy())
                
                components['labels'].append(labels.numpy())
        
        # Concatenate and align components
        result = {}
        for key, component_list in components.items():
            if component_list:
                concatenated = np.concatenate(component_list, axis=0)
                if key != 'labels' and align_dim is not None:
                    concatenated = self._align_dimensions(concatenated, align_dim // len([k for k in components if k != 'labels' and components[k]]))
                result[key] = concatenated
        
        return result

def create_combined_features(effnet_features: np.ndarray, 
                           genconvit_features: np.ndarray,
                           target_dim: int = 512) -> np.ndarray:
    """
    Combine EfficientNetV2 and GenConViT features for meta-model training
    
    Args:
        effnet_features: EfficientNetV2-B3 features [N, D1]
        genconvit_features: GenConViT features [N, D2]  
        target_dim: Target dimension for each component
        
    Returns:
        Combined features [N, target_dim * 2]
    """
    # Align both to same dimension
    half_dim = target_dim // 2
    
    # Align EfficientNet features
    if effnet_features.shape[1] != half_dim:
        if effnet_features.shape[1] > half_dim:
            effnet_aligned = effnet_features[:, :half_dim]
        else:
            padding = np.zeros((effnet_features.shape[0], half_dim - effnet_features.shape[1]))
            effnet_aligned = np.concatenate([effnet_features, padding], axis=1)
    else:
        effnet_aligned = effnet_features
    
    # Align GenConViT features  
    if genconvit_features.shape[1] != half_dim:
        if genconvit_features.shape[1] > half_dim:
            genconvit_aligned = genconvit_features[:, :half_dim]
        else:
            padding = np.zeros((genconvit_features.shape[0], half_dim - genconvit_features.shape[1]))
            genconvit_aligned = np.concatenate([genconvit_features, padding], axis=1)
    else:
        genconvit_aligned = genconvit_features
    
    # Combine features
    combined = np.concatenate([effnet_aligned, genconvit_aligned], axis=1)
    
    logging.info(f"Combined features shape: {combined.shape}")
    return combined

# Example usage and testing
if __name__ == "__main__":
    import argparse
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    import pandas as pd
    
    # Simple dataset class for testing
    class SimpleDataset(Dataset):
        def __init__(self, manifest_path, transform=None):
            self.df = pd.read_csv(manifest_path)
            self.transform = transform or transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, float(row['label'])
    
    def main():
        parser = argparse.ArgumentParser(description='Feature Extraction Testing')
        parser.add_argument('--effnet_checkpoint', type=str, 
                          default='output/stage2_effnet/best_model.pth',
                          help='EfficientNetV2 checkpoint path')
        parser.add_argument('--genconvit_checkpoint', type=str,
                          default='output/stage2_genconvit/best_model.pth', 
                          help='GenConViT checkpoint path')
        parser.add_argument('--val_manifest', type=str,
                          default='processed_data/manifests/val_manifest.csv',
                          help='Validation manifest file')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--align_dim', type=int, default=256)
        
        args = parser.parse_args()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create dataset and dataloader
        dataset = SimpleDataset(args.val_manifest)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        print("Testing EfficientNetV2-B3 feature extraction...")
        try:
            effnet_extractor = EfficientNetFeatureExtractor(
                'efficientnetv2_b3.in21k_ft_in1k',
                args.effnet_checkpoint
            )
            effnet_features, labels = effnet_extractor.extract(dataloader, args.align_dim)
            print(f"✅ EfficientNetV2 features: {effnet_features.shape}")
        except Exception as e:
            print(f"❌ EfficientNetV2 extraction failed: {e}")
            effnet_features = None
        
        print("\nTesting GenConViT feature extraction...")
        try:
            genconvit_extractor = GenConViTFeatureExtractor(
                args.genconvit_checkpoint,
                variant='ED',
                mode='hybrid'
            )
            genconvit_features, labels = genconvit_extractor.extract_features(dataloader, args.align_dim)
            print(f"✅ GenConViT features: {genconvit_features.shape}")
        except Exception as e:
            print(f"❌ GenConViT extraction failed: {e}")
            genconvit_features = None
        
        # Test combined features
        if effnet_features is not None and genconvit_features is not None:
            print("\nTesting combined feature creation...")
            combined = create_combined_features(effnet_features, genconvit_features, 512)
            print(f"✅ Combined features: {combined.shape}")
            
            # Save features for Stage 3
            output_dir = Path('output/stage2_features')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / 'effnet_features.npy', effnet_features)
            np.save(output_dir / 'genconvit_features.npy', genconvit_features)
            np.save(output_dir / 'combined_features.npy', combined)
            np.save(output_dir / 'labels.npy', labels)
            
            print(f"✅ Features saved to {output_dir}")
        
        print("\n🎉 Feature extraction testing complete!")

    if __name__ == "__main__":
        main()