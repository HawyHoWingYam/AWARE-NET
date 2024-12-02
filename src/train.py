import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from pathlib import Path
import json
import numpy as np
import logging
from tqdm import tqdm
from model import EnsembleDeepfakeDetector
import time

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader, variant_name=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), 
                                   lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.WARMUP_EPOCHS,
            T_mult=2,
            eta_min=config.LR_MIN
        )
        
        self.best_val_auc = 0
        self.best_epoch = 0
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 
            'train_auc': [], 'val_auc': [],
            'learning_rates': [], 'epoch_times': []
        }
        self.logger = logging.getLogger(__name__)
        self.variant_name = variant_name
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.patience = config.PATIENCE
        self.min_epochs = config.MIN_EPOCHS
        self.early_stop_counter = 0
        self.best_metrics = {
            'val_auc': 0,
            'val_loss': float('inf'),
            'epoch': 0
        }
        
        # Training time tracking
        self.start_time = None
        self.epoch_times = []
        
        # GPU optimization
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.MIXED_PRECISION)
        self.accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.long().to(self.device, non_blocking=True)
            
            # Use automatic mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.MIXED_PRECISION):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps  # Normalize loss for accumulation
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            total_loss += loss.item()
            predictions.extend(probs.detach().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(true_labels, predictions)
        
        return epoch_loss, epoch_auc
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        # Create progress bar for validation
        pbar = tqdm(loader, desc='Validating', leave=False)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.long().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                total_loss += loss.item()
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        epoch_loss = total_loss / len(loader)
        epoch_auc = roc_auc_score(true_labels, predictions)
        
        return epoch_loss, epoch_auc, predictions, true_labels
    
    def save_model(self, epoch, val_loss):
        try:
            # Parse variant name correctly
            parts = self.variant_name.split('_')
            dataset = parts[0]  # ff++ or celebdf
            model_type = parts[1]  # xception, res2net, etc.
            aug_type = 'with_aug' if 'with' in parts[-1] else 'no_aug'
            
            # Create save path with proper structure
            save_dir = self.config.RESULTS_DIR / 'weights' / dataset / model_type / aug_type
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with epoch and loss info
            filename = f'epoch_{epoch+1}_loss_{val_loss:.4f}.pth'
            save_path = save_dir / filename
            
            self.logger.info(f"Saving model checkpoint to {save_path}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_auc': self.best_val_auc,
                'metrics_history': self.metrics_history
            }
            
            # Remove previous model files with higher validation loss
            for old_file in save_dir.glob('epoch_*.pth'):
                old_loss = float(str(old_file).split('loss_')[-1].split('.pth')[0])
                if old_loss > val_loss:
                    self.logger.info(f"Removing previous model with higher loss: {old_file}")
                    old_file.unlink()
            
            # Save new model
            torch.save(checkpoint, save_path)
            self.logger.info(f"Successfully saved checkpoint with validation loss: {val_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def train(self):
        self.start_time = time.time()
        self.logger.info(f"Starting training for {self.variant_name}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        
        epoch_pbar = tqdm(range(self.config.MAX_EPOCHS), desc='Epochs')
        for epoch in epoch_pbar:
            # Training
            train_loss, train_auc = self.train_epoch()
            
            # Validation
            val_loss, val_auc, _, _ = self.validate(self.val_loader)
            
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                self.logger.info(f"New best validation loss: {val_loss:.4f} at epoch {epoch+1}")
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['train_auc'].append(train_auc)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_auc'].append(val_auc)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_auc': f'{train_auc:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_auc': f'{val_auc:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}'
            })
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.MAX_EPOCHS}:\n"
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n"
                f"Best Val Loss: {best_val_loss:.4f}\n"
                f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            
            # Update learning rate
            self.scheduler.step()
        
        # Training summary
        total_time = time.time() - self.start_time
        self.logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final metrics
        self.save_metrics()
    
    def save_metrics(self):
        # Parse variant name more robustly
        variant_parts = self.variant_name.split('_')
        dataset = variant_parts[0]
        aug_type = variant_parts[-1]
        # Join middle parts for model type in case it has underscores
        model_type = '_'.join(variant_parts[1:-1])
        
        aug_folder = 'with_aug' if 'with' in aug_type else 'no_aug'
        
        # Create metrics directory
        metrics_dir = self.config.RESULTS_DIR / 'metrics' / dataset / model_type / aug_folder
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        history_path = metrics_dir / 'training_history.json'
        self.logger.info(f"Saving training history to {history_path}")
        
        # Add more detailed metrics
        detailed_metrics = {
            'training_history': self.metrics_history,
            'final_metrics': {
                'best_val_auc': self.best_val_auc,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'total_epochs': self.config.MAX_EPOCHS,
                'training_params': {
                    'batch_size': self.config.BATCH_SIZE,
                    'learning_rate': self.config.LEARNING_RATE,
                    'weight_decay': self.config.WEIGHT_DECAY,
                    'augmentations_used': self.train_loader.dataset.augment
                }
            },
            'model_info': {
                'type': model_type,
                'dataset': dataset,
                'augmentation': aug_folder,
                'parameters': sum(p.numel() for p in self.model.parameters())
            }
        }
        
        self.logger.info(f"Saving metrics for {self.variant_name}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Dataset: {dataset}")
        self.logger.info(f"Augmentation: {aug_folder}")
        
        try:
            with open(history_path, 'w') as f:
                json.dump(detailed_metrics, f, indent=4)
            self.logger.info(f"Successfully saved metrics to {history_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise
    
    def log_gpu_stats(self):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            
            self.logger.info(
                f"GPU Memory Stats:\n"
                f"Currently allocated: {memory_allocated:.2f}GB\n"
                f"Reserved: {memory_reserved:.2f}GB\n"
                f"Peak allocation: {max_memory:.2f}GB"
            )