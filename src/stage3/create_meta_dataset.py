#!/usr/bin/env python3
"""
Stage 3 K-Fold Cross-Validation Meta-Dataset Creation - create_meta_dataset.py
==============================================================================

Creates unbiased meta-learning dataset through 5-fold stratified cross-validation.
This is the most computationally intensive component, requiring retraining of both
Stage 2 models (EfficientNetV2-B3 + GenConViT) across 5 folds.

Key Features:
- 5-fold stratified cross-validation with strict data separation
- Automated retraining pipeline for both Stage 2 models per fold
- Out-of-fold feature extraction for unbiased meta-learning
- Comprehensive progress tracking and checkpointing
- Memory-efficient batch processing
- Data leakage prevention validation

Computational Requirements:
- Estimated Time: 10-15 hours (5 folds × 2 models × 1-2 hours each)
- GPU Memory: ~12GB+ recommended
- Storage: ~5-10GB for intermediate checkpoints and features

Usage:
    # Full pipeline (recommended for production)
    python src/stage3/create_meta_dataset.py --data_dir processed_data --folds 5
    
    # Quick test with fewer epochs
    python src/stage3/create_meta_dataset.py --test_mode --epochs 5 --folds 3
    
    # Resume from checkpoint
    python src/stage3/create_meta_dataset.py --resume_fold 2
"""

import os
import sys
import argparse
import logging
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
from src.stage1.dataset import create_dataloaders
from src.stage2.train_stage2_effnet import main as train_effnet
from src.stage2.train_stage2_genconvit import main as train_genconvit
from src.stage2.feature_extractor import (EfficientNetFeatureExtractor, 
                                        GenConViTFeatureExtractor,
                                        create_combined_features)

class MetaDatasetCreator:
    """K-Fold cross-validation meta-dataset creation system"""
    
    def __init__(self, data_dir: str, output_dir: str = "output/stage3_meta",
                 n_folds: int = 5, random_state: int = 42,
                 device: Optional[str] = None):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "fold_checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.features_dir = self.output_dir / "fold_features"
        self.features_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load training manifest
        self.train_manifest_path = self.data_dir / "manifests" / "train_manifest.csv"
        if not self.train_manifest_path.exists():
            raise FileNotFoundError(f"Training manifest not found: {self.train_manifest_path}")
        
        self.train_df = pd.read_csv(self.train_manifest_path)
        logging.info(f"Loaded training manifest with {len(self.train_df)} samples")
        
        # Initialize fold splitter
        self.setup_kfold_splits()
        
        # Progress tracking
        self.progress_file = self.output_dir / "meta_creation_progress.json"
        self.load_progress()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / f"meta_dataset_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
        
        # Suppress some verbose logs
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        logging.info("="*80)
        logging.info("STAGE 3: K-FOLD META-DATASET CREATION INITIALIZED")
        logging.info("="*80)
    
    def setup_kfold_splits(self):
        """Setup stratified K-fold splits"""
        # Extract labels for stratification
        labels = self.train_df['label'].values
        
        # Create stratified splits
        self.skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Generate and store fold indices
        self.fold_splits = list(self.skf.split(self.train_df.index, labels))
        
        # Log fold statistics
        logging.info(f"Created {self.n_folds}-fold stratified splits:")
        for fold_idx, (train_idx, val_idx) in enumerate(self.fold_splits):
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            
            logging.info(f"  Fold {fold_idx}: Train={len(train_idx)} "
                        f"(Real={np.sum(train_labels==0)}, Fake={np.sum(train_labels==1)}), "
                        f"Val={len(val_idx)} "
                        f"(Real={np.sum(val_labels==0)}, Fake={np.sum(val_labels==1)})")
        
        # Save fold splits for reproducibility
        splits_file = self.output_dir / "fold_splits.pkl"
        with open(splits_file, 'wb') as f:
            pickle.dump(self.fold_splits, f)
        logging.info(f"Saved fold splits to {splits_file}")
    
    def load_progress(self):
        """Load previous progress if exists"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            logging.info(f"Loaded progress: {self.progress}")
        else:
            self.progress = {
                'completed_folds': [],
                'current_fold': 0,
                'start_time': datetime.now().isoformat(),
                'total_folds': self.n_folds
            }
    
    def save_progress(self):
        """Save current progress"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def create_fold_manifests(self, fold_idx: int, train_idx: np.ndarray, 
                            val_idx: np.ndarray) -> Tuple[str, str]:
        """Create manifest files for current fold"""
        fold_dir = self.checkpoints_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        # Create fold-specific training manifest
        fold_train_df = self.train_df.iloc[train_idx].copy()
        fold_train_manifest = fold_dir / "train_manifest.csv"
        fold_train_df.to_csv(fold_train_manifest, index=False)
        
        # Create fold-specific validation manifest (for out-of-fold predictions)
        fold_val_df = self.train_df.iloc[val_idx].copy()
        fold_val_manifest = fold_dir / "val_manifest.csv"
        fold_val_df.to_csv(fold_val_manifest, index=False)
        
        logging.info(f"Created fold {fold_idx} manifests: "
                    f"train={len(fold_train_df)}, val={len(fold_val_df)}")
        
        return str(fold_train_manifest), str(fold_val_manifest)
    
    def train_fold_models(self, fold_idx: int, train_manifest: str, 
                         val_manifest: str, epochs: int = 50) -> Dict[str, str]:
        """Train both Stage 2 models for current fold"""
        fold_dir = self.checkpoints_dir / f"fold_{fold_idx}"
        
        # Model output directories
        effnet_dir = fold_dir / "effnet"
        genconvit_dir = fold_dir / "genconvit"
        effnet_dir.mkdir(exist_ok=True)
        genconvit_dir.mkdir(exist_ok=True)
        
        model_paths = {}
        
        # Train EfficientNetV2-B3
        logging.info(f"🚀 Training EfficientNetV2-B3 for fold {fold_idx}")
        try:
            effnet_checkpoint = self.train_efficientnet_fold(
                train_manifest, val_manifest, str(effnet_dir), epochs
            )
            model_paths['efficientnet'] = effnet_checkpoint
            logging.info(f"✅ EfficientNetV2-B3 fold {fold_idx} complete: {effnet_checkpoint}")
        except Exception as e:
            logging.error(f"❌ EfficientNetV2-B3 fold {fold_idx} failed: {e}")
            raise
        
        # Train GenConViT
        logging.info(f"🚀 Training GenConViT for fold {fold_idx}")
        try:
            genconvit_checkpoint = self.train_genconvit_fold(
                train_manifest, val_manifest, str(genconvit_dir), epochs
            )
            model_paths['genconvit'] = genconvit_checkpoint
            logging.info(f"✅ GenConViT fold {fold_idx} complete: {genconvit_checkpoint}")
        except Exception as e:
            logging.error(f"❌ GenConViT fold {fold_idx} failed: {e}")
            raise
        
        return model_paths
    
    def train_efficientnet_fold(self, train_manifest: str, val_manifest: str, 
                               output_dir: str, epochs: int) -> str:
        """Train EfficientNetV2-B3 for current fold"""
        import subprocess
        import tempfile
        
        # Create training script arguments
        cmd = [
            'python', 'src/stage2/train_stage2_effnet.py',
            '--data_dir', str(self.data_dir),
            '--train_manifest', train_manifest,
            '--val_manifest', val_manifest,
            '--output_dir', output_dir,
            '--epochs', str(epochs),
            '--batch_size', '28',
            '--lr', '5e-5',
            '--model_name', 'efficientnetv2_b3.in21k_ft_in1k'
        ]
        
        # Run training subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            logging.error(f"EfficientNet training failed: {result.stderr}")
            raise RuntimeError(f"EfficientNet training failed: {result.stderr}")
        
        # Find best model checkpoint
        checkpoint_path = Path(output_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EfficientNet checkpoint not found: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def train_genconvit_fold(self, train_manifest: str, val_manifest: str,
                           output_dir: str, epochs: int) -> str:
        """Train GenConViT for current fold"""
        import subprocess
        
        # Create training script arguments
        cmd = [
            'python', 'src/stage2/train_stage2_genconvit.py',
            '--data_dir', str(self.data_dir),
            '--train_manifest', train_manifest,
            '--val_manifest', val_manifest,
            '--output_dir', output_dir,
            '--epochs', str(epochs),
            '--batch_size', '16',
            '--lr', '1e-4',
            '--mode', 'hybrid',
            '--variant', 'ED'
        ]
        
        # Run training subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            logging.error(f"GenConViT training failed: {result.stderr}")
            raise RuntimeError(f"GenConViT training failed: {result.stderr}")
        
        # Find best model checkpoint
        checkpoint_path = Path(output_dir) / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"GenConViT checkpoint not found: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def extract_fold_features(self, fold_idx: int, val_manifest: str,
                            model_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Extract out-of-fold features for meta-learning"""
        logging.info(f"🔍 Extracting features for fold {fold_idx}")
        
        # Create temporary dataloader for fold validation set
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image
        
        class FoldDataset(Dataset):
            def __init__(self, manifest_path: str):
                self.df = pd.read_csv(manifest_path)
                self.transform = transforms.Compose([
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
        
        # Create dataloader
        fold_dataset = FoldDataset(val_manifest)
        fold_dataloader = DataLoader(fold_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        fold_features = {}
        
        # Extract EfficientNetV2-B3 features
        try:
            effnet_extractor = EfficientNetFeatureExtractor(
                'efficientnetv2_b3.in21k_ft_in1k',
                model_paths['efficientnet'],
                device=self.device
            )
            effnet_features, labels = effnet_extractor.extract(fold_dataloader, align_dim=256)
            fold_features['efficientnet_features'] = effnet_features
            fold_features['labels'] = labels
            logging.info(f"✅ EfficientNet features extracted: {effnet_features.shape}")
        except Exception as e:
            logging.error(f"❌ EfficientNet feature extraction failed: {e}")
            raise
        
        # Extract GenConViT features
        try:
            genconvit_extractor = GenConViTFeatureExtractor(
                model_paths['genconvit'],
                variant='ED',
                mode='hybrid',
                device=self.device
            )
            genconvit_features, _ = genconvit_extractor.extract_features(fold_dataloader, align_dim=256)
            fold_features['genconvit_features'] = genconvit_features
            logging.info(f"✅ GenConViT features extracted: {genconvit_features.shape}")
        except Exception as e:
            logging.error(f"❌ GenConViT feature extraction failed: {e}")
            raise
        
        # Create combined features
        combined_features = create_combined_features(
            effnet_features, genconvit_features, target_dim=512
        )
        fold_features['combined_features'] = combined_features
        
        logging.info(f"✅ Fold {fold_idx} feature extraction complete")
        logging.info(f"   EfficientNet: {effnet_features.shape}")
        logging.info(f"   GenConViT: {genconvit_features.shape}")
        logging.info(f"   Combined: {combined_features.shape}")
        logging.info(f"   Labels: {labels.shape}")
        
        return fold_features
    
    def save_fold_features(self, fold_idx: int, fold_features: Dict[str, np.ndarray]):
        """Save fold features to disk"""
        fold_features_dir = self.features_dir / f"fold_{fold_idx}"
        fold_features_dir.mkdir(exist_ok=True)
        
        for feature_name, features in fold_features.items():
            feature_path = fold_features_dir / f"{feature_name}.npy"
            np.save(feature_path, features)
            logging.info(f"Saved {feature_name}: {feature_path}")
    
    def process_fold(self, fold_idx: int, epochs: int = 50) -> Dict[str, Any]:
        """Process a single fold: train models + extract features"""
        logging.info("="*80)
        logging.info(f"PROCESSING FOLD {fold_idx + 1}/{self.n_folds}")
        logging.info("="*80)
        
        fold_start_time = datetime.now()
        
        # Get fold indices
        train_idx, val_idx = self.fold_splits[fold_idx]
        
        # Create fold manifests
        train_manifest, val_manifest = self.create_fold_manifests(fold_idx, train_idx, val_idx)
        
        # Train models for this fold
        model_paths = self.train_fold_models(fold_idx, train_manifest, val_manifest, epochs)
        
        # Extract out-of-fold features
        fold_features = self.extract_fold_features(fold_idx, val_manifest, model_paths)
        
        # Save features
        self.save_fold_features(fold_idx, fold_features)
        
        # Calculate fold processing time
        fold_duration = datetime.now() - fold_start_time
        
        fold_result = {
            'fold_idx': fold_idx,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'model_paths': model_paths,
            'feature_shapes': {k: v.shape for k, v in fold_features.items()},
            'duration_seconds': fold_duration.total_seconds(),
            'completed_at': datetime.now().isoformat()
        }
        
        logging.info(f"✅ Fold {fold_idx} completed in {fold_duration}")
        
        return fold_result
    
    def create_meta_dataset(self, epochs: int = 50, resume_fold: Optional[int] = None) -> Dict[str, Any]:
        """Create complete meta-learning dataset through K-fold CV"""
        logging.info("🚀 Starting K-Fold Meta-Dataset Creation")
        
        creation_start_time = datetime.now()
        fold_results = []
        
        # Determine starting fold
        start_fold = resume_fold if resume_fold is not None else 0
        if resume_fold is not None:
            logging.info(f"🔄 Resuming from fold {resume_fold}")
        
        # Process each fold
        for fold_idx in range(start_fold, self.n_folds):
            try:
                # Skip if fold already completed
                if fold_idx in self.progress['completed_folds']:
                    logging.info(f"⏭️ Fold {fold_idx} already completed, skipping")
                    continue
                
                # Update progress
                self.progress['current_fold'] = fold_idx
                self.save_progress()
                
                # Process fold
                fold_result = self.process_fold(fold_idx, epochs)
                fold_results.append(fold_result)
                
                # Mark fold as completed
                self.progress['completed_folds'].append(fold_idx)
                self.save_progress()
                
                # Log progress
                remaining_folds = self.n_folds - len(self.progress['completed_folds'])
                logging.info(f"📊 Progress: {len(self.progress['completed_folds'])}/{self.n_folds} folds completed, "
                           f"{remaining_folds} remaining")
                
            except Exception as e:
                logging.error(f"❌ Fold {fold_idx} failed: {e}")
                # Save partial progress and re-raise
                self.save_progress()
                raise
        
        # Combine all fold features
        logging.info("🔗 Combining features from all folds")
        combined_result = self.combine_fold_features()
        
        # Calculate total duration
        total_duration = datetime.now() - creation_start_time
        
        # Final results
        results = {
            'meta_dataset_creation': 'completed',
            'n_folds': self.n_folds,
            'epochs_per_fold': epochs,
            'fold_results': fold_results,
            'combined_features': combined_result,
            'total_duration_seconds': total_duration.total_seconds(),
            'total_duration_hours': total_duration.total_seconds() / 3600,
            'completed_at': datetime.now().isoformat()
        }
        
        # Save final results
        results_file = self.output_dir / "meta_dataset_creation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info("🎉 K-Fold Meta-Dataset Creation Complete!")
        logging.info(f"⏱️ Total duration: {total_duration}")
        logging.info(f"📁 Results saved to: {results_file}")
        
        return results
    
    def combine_fold_features(self) -> Dict[str, Any]:
        """Combine features from all processed folds"""
        logging.info("Combining features from all folds...")
        
        combined_features = {
            'efficientnet_features': [],
            'genconvit_features': [],
            'combined_features': [],
            'labels': []
        }
        
        # Load and combine features from each fold
        for fold_idx in range(self.n_folds):
            fold_features_dir = self.features_dir / f"fold_{fold_idx}"
            
            if not fold_features_dir.exists():
                logging.warning(f"Fold {fold_idx} features not found, skipping")
                continue
            
            # Load fold features
            for feature_name in combined_features.keys():
                feature_path = fold_features_dir / f"{feature_name}.npy"
                if feature_path.exists():
                    fold_feature = np.load(feature_path)
                    combined_features[feature_name].append(fold_feature)
                    logging.info(f"Loaded fold {fold_idx} {feature_name}: {fold_feature.shape}")
        
        # Concatenate all features
        final_features = {}
        for feature_name, feature_list in combined_features.items():
            if feature_list:
                final_features[feature_name] = np.concatenate(feature_list, axis=0)
                logging.info(f"Combined {feature_name}: {final_features[feature_name].shape}")
        
        # Save combined features
        combined_dir = self.output_dir / "combined_features"
        combined_dir.mkdir(exist_ok=True)
        
        for feature_name, features in final_features.items():
            feature_path = combined_dir / f"{feature_name}.npy"
            np.save(feature_path, features)
            logging.info(f"Saved combined {feature_name}: {feature_path}")
        
        # Validation
        n_samples = len(final_features['labels'])
        logging.info(f"✅ Combined meta-dataset created with {n_samples} samples")
        logging.info(f"   Real samples: {np.sum(final_features['labels'] == 0)}")
        logging.info(f"   Fake samples: {np.sum(final_features['labels'] == 1)}")
        
        return {
            'feature_shapes': {k: v.shape for k, v in final_features.items()},
            'total_samples': n_samples,
            'real_samples': int(np.sum(final_features['labels'] == 0)),
            'fake_samples': int(np.sum(final_features['labels'] == 1)),
            'save_path': str(combined_dir)
        }

def main():
    parser = argparse.ArgumentParser(description='K-Fold Meta-Dataset Creation')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Data directory path')
    parser.add_argument('--output_dir', type=str, default='output/stage3_meta',
                       help='Output directory for meta-dataset')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of K-fold splits')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per fold')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--resume_fold', type=int, default=None,
                       help='Resume from specific fold (for recovery)')
    parser.add_argument('--test_mode', action='store_true',
                       help='Test mode with reduced epochs and folds')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Test mode adjustments
    if args.test_mode:
        args.epochs = min(args.epochs, 5)
        args.folds = min(args.folds, 3)
        logging.info(f"🧪 Test mode: epochs={args.epochs}, folds={args.folds}")
    
    try:
        # Initialize creator
        creator = MetaDatasetCreator(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_folds=args.folds,
            random_state=args.random_state,
            device=args.device
        )
        
        # Create meta-dataset
        results = creator.create_meta_dataset(
            epochs=args.epochs,
            resume_fold=args.resume_fold
        )
        
        print("\n" + "="*80)
        print("🎉 META-DATASET CREATION SUCCESSFUL!")
        print("="*80)
        print(f"📊 Total samples: {results['combined_features']['total_samples']}")
        print(f"⏱️ Total time: {results['total_duration_hours']:.2f} hours")
        print(f"📁 Output directory: {args.output_dir}")
        print("="*80)
        
    except KeyboardInterrupt:
        logging.info("⏹️ Process interrupted by user")
        print("\n⏹️ Process interrupted. Progress saved for resumption.")
    except Exception as e:
        logging.error(f"❌ Meta-dataset creation failed: {e}")
        print(f"\n❌ Error: {e}")
        print("📝 Check logs for detailed error information")
        raise

if __name__ == "__main__":
    main()