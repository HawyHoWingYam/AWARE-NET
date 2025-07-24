#!/usr/bin/env python3
"""
AWARE-NET Stage 2 GenConViT Training Script
==========================================

Unified training script for GenConViT models supporting both hybrid and 
pretrained modes with seamless switching capabilities.

Usage:
    # Hybrid mode training
    python train_stage2_genconvit.py --mode hybrid --variant ED --epochs 25
    
    # Pretrained mode fine-tuning
    python train_stage2_genconvit.py --mode pretrained --variant VAE --epochs 10
    
    # Auto mode with intelligent selection
    python train_stage2_genconvit.py --mode auto --epochs 20
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
from src.stage1.dataset import create_dataloaders
from src.stage2.genconvit_manager import GenConViTManager
from src.stage2.genconvit.common.losses import GenConViTLoss
from src.stage2.genconvit.common.base import GenConViTVariant

class GenConViTTrainer:
    """Unified trainer for GenConViT models"""
    
    def __init__(self, mode: str = 'auto', variant: str = 'ED', 
                 data_dir: str = 'processed_data', device: Optional[str] = None):
        
        self.data_dir = Path(data_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GenConViT manager
        self.manager = GenConViTManager(mode=mode, variant=variant, device=self.device)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        print(f"GenConViT Trainer initialized: {self.manager.mode.value} mode, {self.manager.variant.value} variant")
    
    def setup_model(self, config: Optional[Dict[str, Any]] = None):
        """Setup GenConViT model with training configuration"""
        
        print("Setting up GenConViT model...")
        
        # Create model through manager
        self.model = self.manager.create_model(config)
        
        # Setup loss function
        self.loss_fn = GenConViTLoss(
            classification_weight=0.9,
            reconstruction_weight=0.1,
            kl_weight=0.01,
            variant=self.manager.variant
        ).to(self.device)
        
        print(f"✅ Model setup complete - {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        return self.model
    
    def train(self, epochs: int = 25, batch_size: int = 16, 
              learning_rate: float = 1e-4, save_dir: str = 'stage2_genconvit_models'):
        """Full training pipeline"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting GenConViT training: {epochs} epochs, batch size {batch_size}")
        
        # Setup model
        self.setup_model()
        
        # Setup training components
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=str(self.data_dir), batch_size=batch_size, num_workers=4, shuffle=True
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        
        # Training loop
        best_auc = 0.0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate  
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                
                model_path = save_dir / f'genconvit_{self.manager.mode.value}_{self.manager.variant.value}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': val_metrics,
                    'mode': self.manager.mode.value,
                    'variant': self.manager.variant.value
                }, model_path)
                
                print(f"💾 New best model saved: AUC {best_auc:.4f}")
            
            # Log progress
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}")
        
        total_time = time.time() - start_time
        
        results = {
            'best_auc': best_auc,
            'total_time': total_time,
            'mode': self.manager.mode.value,
            'variant': self.manager.variant.value
        }
        
        print(f"🎉 Training completed! Best AUC: {best_auc:.4f}, Time: {total_time:.1f}s")
        return results
    
    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Train {epoch}'):
            images, labels = images.to(self.device), labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            losses = self.loss_fn(outputs, labels, images)
            loss = losses['total']
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs.classification.squeeze()) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
        
        return {'loss': total_loss / len(train_loader), 'accuracy': correct / total}
    
    def _validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Val {epoch}'):
                images, labels = images.to(self.device), labels.to(self.device).float()
                
                outputs = self.model(images)
                losses = self.loss_fn(outputs, labels, images)
                
                total_loss += losses['total'].item()
                
                preds = torch.sigmoid(outputs.classification.squeeze())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5
        
        return {'loss': total_loss / len(val_loader), 'accuracy': accuracy, 'auc': auc}

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='AWARE-NET Stage 2 GenConViT Training')
    
    parser.add_argument('--mode', type=str, default='auto', 
                       choices=['hybrid', 'pretrained', 'auto'], help='GenConViT integration mode')
    parser.add_argument('--variant', type=str, default='ED', choices=['ED', 'VAE'], help='Model variant')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory path')
    parser.add_argument('--save_dir', type=str, default='stage2_genconvit_models', help='Model save directory')
    parser.add_argument('--device', type=str, default=None, help='Training device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = GenConViTTrainer(
        mode=args.mode, variant=args.variant, data_dir=args.data_dir, device=args.device
    )
    
    # Start training
    results = trainer.train(
        epochs=args.epochs, batch_size=args.batch_size, 
        learning_rate=args.learning_rate, save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Best Validation AUC: {results['best_auc']:.4f}")
    print(f"⏱️  Total Training Time: {results['total_time']:.1f}s")
    print("="*60)

if __name__ == '__main__':
    main()