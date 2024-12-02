import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from pathlib import Path
import json
import numpy as np
import logging

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, test_loader, subset_name=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(model.parameters(), 
                                   lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.MAX_EPOCHS)
        
        self.best_val_auc = 0
        self.best_epoch = 0
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 
            'train_auc': [], 'val_auc': []
        }
        self.logger = logging.getLogger(__name__)
        self.subset_name = subset_name
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(true_labels, predictions)
        
        return epoch_loss, epoch_auc
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(loader)
        epoch_auc = roc_auc_score(true_labels, predictions)
        
        return epoch_loss, epoch_auc, predictions, true_labels
    
    def train(self):
        self.logger.info(f"Starting training for {self.subset_name if self.subset_name else 'all data'}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config.MAX_EPOCHS):
            # Training
            train_loss, train_auc = self.train_epoch()
            
            # Validation
            val_loss, val_auc, _, _ = self.validate(self.val_loader)
            
            # Save on both AUC and loss improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.save_model(f'best_auc_model_{self.subset_name}.pth')
                self.logger.info(f"New best AUC: {val_auc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f'best_loss_model_{self.subset_name}.pth')
                self.logger.info(f"New best loss: {val_loss:.4f}")
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.MAX_EPOCHS}:\n"
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n"
                f"Model Weights: {self.model.get_model_weights()}\n"
                f"Memory Usage: {torch.cuda.memory_allocated()/1e9:.2f}GB"
            )
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_auc'].append(train_auc)
            self.metrics_history['val_auc'].append(val_auc)
            
        # Save final metrics
        self.save_metrics()
    
    def save_model(self, filename):
        save_path = self.config.RESULTS_DIR / 'weights' / filename
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
        }, save_path)
    
    def save_metrics(self):
        metrics_path = self.config.RESULTS_DIR / 'metrics' / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)