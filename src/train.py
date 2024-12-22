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
        
        # Log model weights if it's an ensemble
        if hasattr(self.model, 'get_model_weights'):
            weights = self.model.get_model_weights()
            if weights is not None:  # Only log weights for ensemble model
                self.logger.info("\nEnsemble Model Weights:")
                model_names = ['Xception', 'Res2Net101', 'EfficientNet']
                for name, weight in zip(model_names, weights):
                    self.logger.info(f"{name}: {weight:.3f}")
        
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
            
            # Use timm-specific model names for directory structure
            model_name_map = {
                'xception': 'legacy_xception',
                'res2net101': 'res2net101_26w_4s',
                'efficientnet': 'tf_efficientnet_b7_ns',
                'ensemble': 'ensemble'
            }
            
            # Get the model type and map to timm name if needed
            model_type = parts[1]  # xception, res2net101, etc.
            model_dir_name = model_name_map.get(model_type, model_type)
            
            # Determine augmentation type
            is_no_aug = 'no_augmentation' in self.variant_name
            aug_type = 'no_aug' if is_no_aug else 'with_aug'
            
            # Use config's WEIGHTS_DIR
            save_dir = self.config.WEIGHTS_DIR / dataset / model_dir_name / aug_type
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Log the save location for verification
            self.logger.info(f"Current experiment: {self.variant_name}")
            self.logger.info(f"Using timm model name: {model_dir_name}")
            self.logger.info(f"Augmentation status: {'disabled' if is_no_aug else 'enabled'}")
            self.logger.info(f"Saving to directory: {save_dir}")
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_auc': self.best_val_auc,
                'metrics_history': self.metrics_history,
                'augmentation_status': not is_no_aug,
                'model_name': model_dir_name
            }
            
            # Save model if it's the best so far (based on validation loss)
            model_path = save_dir / f'loss_{val_loss:.4f}.pth'
            if not list(save_dir.glob('*.pth')) or val_loss < self.best_val_loss:
                # Remove previous model file if it exists
                for old_file in save_dir.glob('*.pth'):
                    old_file.unlink()
                    self.logger.info(f"Removed previous model: {old_file.name}")
                
                # Save new best model
                self.best_val_loss = val_loss
                torch.save(checkpoint, model_path)
                self.logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def train(self):
        self.start_time = time.time()
        self.logger.info(f"Starting training for {self.variant_name}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Initialize metrics history with empty lists
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        best_val_loss = float('inf')
        
        epoch_pbar = tqdm(range(self.config.MAX_EPOCHS), desc='Epochs')
        for epoch in epoch_pbar:
            epoch_start = time.time()
            
            # Training
            train_loss, train_auc = self.train_epoch()
            
            # Validation
            val_loss, val_auc, _, _ = self.validate(self.val_loader)
            
            # Store current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update metrics history
            self.metrics_history['train_loss'].append(float(train_loss))
            self.metrics_history['val_loss'].append(float(val_loss))
            self.metrics_history['train_auc'].append(float(train_auc))
            self.metrics_history['val_auc'].append(float(val_auc))
            self.metrics_history['learning_rates'].append(float(current_lr))
            self.metrics_history['epoch_times'].append(time.time() - epoch_start)
            
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                self.logger.info(f"New best validation loss: {val_loss:.4f} at epoch {epoch+1}")
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_auc': f'{train_auc:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_auc': f'{val_auc:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.MAX_EPOCHS}:\n"
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n"
                f"Best Val Loss: {best_val_loss:.4f}\n"
                f"Learning Rate: {current_lr:.2e}"
            )
            
            # Update learning rate
            self.scheduler.step()
        
        # Training summary
        total_time = time.time() - self.start_time
        self.logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Verify metrics were collected
        self.logger.info("\nMetrics collection summary:")
        for metric_name, values in self.metrics_history.items():
            self.logger.info(f"{metric_name}: {len(values)} values collected")
        
        # Save final metrics
        self.save_metrics()
    
    def save_metrics(self):
        try:
            # Parse variant name correctly
            parts = self.variant_name.split('_')
            dataset = parts[0]  # ff++ or celebdf
            model_type = parts[1]  # xception, res2net, etc.
            
            # Determine augmentation type from variant name
            is_no_aug = 'no_augmentation' in self.variant_name
            aug_type = 'no_aug' if is_no_aug else 'with_aug'
            
            # Log the current experiment details
            self.logger.info(f"\nSaving metrics for experiment:")
            self.logger.info(f"Dataset: {dataset}")
            self.logger.info(f"Model: {model_type}")
            self.logger.info(f"Augmentation: {'disabled' if is_no_aug else 'enabled'}")
            
            # Create metrics directory with correct structure
            metrics_dir = self.config.RESULTS_DIR / 'metrics' / dataset / model_type / aug_type
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
                        'augmentations_used': not is_no_aug
                    }
                },
                'model_info': {
                    'type': model_type,
                    'dataset': dataset,
                    'augmentation': aug_type,
                    'parameters': sum(p.numel() for p in self.model.parameters())
                }
            }
            
            # Double check we're saving in the right place
            self.logger.info(f"Saving to directory: {metrics_dir}")
            self.logger.info(f"Current experiment is {'without' if is_no_aug else 'with'} augmentation")
            
            try:
                with open(history_path, 'w') as f:
                    json.dump(detailed_metrics, f, indent=4)
                self.logger.info(f"Successfully saved metrics to {history_path}")
            except Exception as e:
                self.logger.error(f"Error saving metrics: {str(e)}")
                raise
            
        except Exception as e:
            self.logger.error(f"Error in save_metrics: {str(e)}")
            self.logger.error(f"Variant name: {self.variant_name}")
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