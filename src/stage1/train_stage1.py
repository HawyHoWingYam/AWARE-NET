#!/usr/bin/env python3
"""
Stage 1 Model Training Script - train_stage1.py
===============================================

This script fine-tunes a MobileNetV4-Hybrid-Medium model for the fast filter
component of the cascade detection system.

Key Features:
- MobileNetV4-Hybrid-Medium from timm library
- Binary classification for deepfake detection
- Comprehensive data augmentation
- AUC-based model selection
- Detailed logging and metrics tracking

Usage:
    python src/stage1/train_stage1.py --data_dir processed_data --epochs 50
"""

import os
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Setup logging
def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection
    
    Loads images from manifest CSV files and applies transformations
    """
    
    def __init__(self, manifest_path, data_root, transform=None):
        """
        Args:
            manifest_path (str): Path to manifest CSV file
            data_root (str): Root directory of processed data
            transform (callable): Optional transform to be applied
        """
        self.data_root = Path(data_root)
        self.transform = transform
        
        # Load manifest
        self.manifest = pd.read_csv(manifest_path)
        
        # Verify data paths exist
        valid_indices = []
        for idx, row in self.manifest.iterrows():
            img_path = self.data_root / row['image_path']
            if img_path.exists():
                valid_indices.append(idx)
        
        self.manifest = self.manifest.iloc[valid_indices].reset_index(drop=True)
        
        logging.info(f"Loaded {len(self.manifest)} samples from {manifest_path}")
        logging.info(f"  Real samples: {len(self.manifest[self.manifest['label'] == 0])}")
        logging.info(f"  Fake samples: {len(self.manifest[self.manifest['label'] == 1])}")
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        row = self.manifest.iloc[idx]
        
        # Load image
        img_path = self.data_root / row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (256, 256), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label (0 for real, 1 for fake)
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return image, label

def get_transforms(input_size=(256, 256)):
    """
    Get training and validation transforms
    
    Args:
        input_size (tuple): Input image size
        
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1)
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def create_model(model_name="mobilenetv4_hybrid_medium.ix_e550_r256_in1k", num_classes=1, pretrained=True):
    """
    Create MobileNetV4 model for binary classification
    
    Args:
        model_name (str): Model name from timm
        num_classes (int): Number of output classes (1 for binary)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: The model
    """
    
    # Create model
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )
    
    logging.info(f"Created model: {model_name}")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).squeeze(1)  # Remove last dimension for binary classification
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Store predictions and labels for metrics
        with torch.no_grad():
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Avg Loss': f"{running_loss/(batch_idx+1):.4f}"
        })
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Convert probabilities to binary predictions for accuracy and F1
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)
    
    return {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1
    }

def validate_epoch(model, dataloader, criterion, device, epoch, total_epochs):
    """
    Validate for one epoch
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            
            # Store predictions and labels for metrics
            predictions = torch.sigmoid(outputs).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{running_loss/(batch_idx+1):.4f}"
            })
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_predictions)
    
    # Convert probabilities to binary predictions for accuracy and F1
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)
    
    return {
        'loss': avg_loss,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1
    }

def save_training_curves(train_history, val_history, output_dir):
    """Save training curves plots"""
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves - Stage 1 Fast Filter', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, val_history['loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # AUC
    axes[0, 1].plot(epochs, train_history['auc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, val_history['auc'], 'r-', label='Validation')
    axes[0, 1].set_title('AUC Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[1, 0].plot(epochs, train_history['accuracy'], 'b-', label='Train')
    axes[1, 0].plot(epochs, val_history['accuracy'], 'r-', label='Validation')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(epochs, train_history['f1'], 'b-', label='Train')
    axes[1, 1].plot(epochs, val_history['f1'], 'r-', label='Validation')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Fast Filter Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--train_manifest', type=str, default='processed_data/manifests/train_manifest.csv',
                        help='Path to training manifest CSV')
    parser.add_argument('--val_manifest', type=str, default='processed_data/manifests/val_manifest.csv',
                        help='Path to validation manifest CSV')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                        default='mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                        help='Model name from timm library')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                        help='Input image size (height width)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/stage1',
                        help='Output directory for models and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=== Stage 1 Fast Filter Training ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create transforms
    train_transform, val_transform = get_transforms(tuple(args.input_size))
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = DeepfakeDataset(
        args.train_manifest, 
        args.data_dir, 
        transform=train_transform
    )
    val_dataset = DeepfakeDataset(
        args.val_manifest, 
        args.data_dir, 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args.model_name, num_classes=1, pretrained=True)
    model = model.to(device)
    
    # Create loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Training history
    train_history = {'loss': [], 'auc': [], 'accuracy': [], 'f1': []}
    val_history = {'loss': [], 'auc': [], 'accuracy': [], 'f1': []}
    
    best_val_auc = 0.0
    best_epoch = 0
    
    logger.info("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, args.epochs
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        for key in train_history.keys():
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
            f"LR: {current_lr:.2e} - "
            f"Train Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}"
        )
        
        # Save best model based on validation AUC
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            
            # Save best model
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, best_model_path)
            
            logger.info(f"New best model saved! Val AUC: {best_val_auc:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'args': vars(args)
            }, checkpoint_path)
    
    # Training completed
    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_history': train_history,
            'val_history': val_history,
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'total_training_time': total_training_time,
            'args': vars(args)
        }, f, indent=2)
    
    # Save training curves
    logger.info("Generating training curves...")
    save_training_curves(train_history, val_history, output_dir)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'val_history': val_history,
        'args': vars(args)
    }, final_model_path)
    
    # Save training summary
    summary = {
        'model_name': args.model_name,
        'input_size': args.input_size,
        'total_epochs': args.epochs,
        'best_epoch': best_epoch + 1,
        'best_val_auc': best_val_auc,
        'final_train_auc': train_history['auc'][-1],
        'final_val_auc': val_history['auc'][-1],
        'total_training_time_hours': total_training_time / 3600,
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'device': str(device)
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== TRAINING SUMMARY ===")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("Stage 1 training completed successfully!")

if __name__ == "__main__":
    main()