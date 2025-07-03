import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from pathlib import Path
import json
import numpy as np
import logging
from tqdm import tqdm
from model import EnsembleDeepfakeDetector
import time


class Trainer:
    def __init__(
        self, model, train_loader, val_loader, config, variant_name, test_loader=None, rank=0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.variant_name = variant_name
        self.rank = rank  # Store the rank of the current process
        self.logger = logging.getLogger(__name__)
        self.best_val_loss = float("inf")

        # The device is the rank itself in DDP
        self.device = self.rank

        self.criterion = nn.CrossEntropyLoss()
        
        # For ensemble models, we only want to train the learnable weights, not the frozen backbones.
        if isinstance(model.module, EnsembleDeepfakeDetector):
            if self.rank == 0:
                self.logger.info("Configuring optimizer for Ensemble: Training only learnable weights.")
            # Access the original model through the .module attribute of the DDP wrapper
            params_to_train = filter(lambda p: p.requires_grad, model.module.parameters())
        else:
            if self.rank == 0:
                self.logger.info("Configuring optimizer for Single Model: Training all parameters.")
            params_to_train = model.parameters()
        
        self.optimizer = optim.AdamW(params_to_train, 
                                   lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
        
        if config.LR_SCHEDULE_TYPE == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.WARMUP_EPOCHS,
                T_mult=2,
                eta_min=config.LR_MIN,
            )
        elif config.LR_SCHEDULE_TYPE == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1,
            )

        self.best_val_auc = 0
        self.best_epoch = 0
        self.metrics_history = {
            "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [],
            "learning_rates": [], "epoch_times": [],
        }

        # Early stopping
        self.patience = config.PATIENCE
        self.min_epochs = config.MIN_EPOCHS
        self.early_stop_counter = 0
        self.best_metrics = {"val_auc": 0, "val_loss": float("inf"), "epoch": 0}

        # GPU optimization
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.MIXED_PRECISION)
        self.accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        # No need to gather predictions/labels during training, as loss is the primary driver
        
        self.optimizer.zero_grad()

        # Only show progress bar on the main process
        pbar = tqdm(self.train_loader, desc="Training", leave=False, disable=(self.rank != 0))
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.long().to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.config.MIXED_PRECISION):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

            if self.rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # We need to average the loss across all processes
        total_loss_tensor = torch.tensor(total_loss).to(self.device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / (len(self.train_loader.dataset))
        
        return avg_loss, 0 # AUC is not calculated per epoch during training for simplicity

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        predictions_list = []
        true_labels_list = []

        pbar = tqdm(loader, desc="Validating", leave=False, disable=(self.rank != 0))
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                total_loss += loss.item()
                predictions_list.append(probs)
                true_labels_list.append(labels)

        # Gather results from all processes
        gathered_preds = [torch.zeros_like(predictions_list[0]) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(true_labels_list[0]) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_preds, torch.cat(predictions_list))
        dist.all_gather(gathered_labels, torch.cat(true_labels_list))

        epoch_loss = total_loss / len(loader) # This is the loss for this process's shard

        # On rank 0, compute the final metrics
        if self.rank == 0:
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
            epoch_auc = roc_auc_score(all_labels, all_preds)
            return epoch_loss, epoch_auc, all_preds.tolist(), all_labels.tolist()
        
        return epoch_loss, 0, [], []


    def save_model(self, epoch, val_loss):
        # Only the main process (rank 0) should save the model
        if self.rank != 0:
            return

        try:
            # When using DDP, we need to save the `module`'s state dict
            state_dict = self.model.module.state_dict()
            
            model_key = "ensemble" if isinstance(self.model.module, EnsembleDeepfakeDetector) else self.model.module.timm_name
            
            parts = self.variant_name.split('_')
            dataset = parts[0]
            variant = "with_aug" if "with_augmentation" in self.variant_name else "no_aug"
            
            save_dir = self.config.get_model_weights_dir(model_key, dataset, variant)
            save_dir.mkdir(parents=True, exist_ok=True)

            model_path = save_dir / f"loss_{val_loss:.4f}.pth"

            if not list(save_dir.glob("*.pth")) or val_loss < self.best_val_loss:
                for old_file in save_dir.glob("*.pth"):
                    old_file.unlink()
                
                self.best_val_loss = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": state_dict, # Save the unwrapped model
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, model_path)
                self.logger.info(f"Saved new best model to: {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model on rank {self.rank}: {str(e)}")
            raise

    def train(self):
        if self.rank == 0:
            self.logger.info(f"Starting training for {self.variant_name}")

        for epoch in range(self.config.MAX_EPOCHS):
            # Set the epoch for the sampler, which is crucial for shuffling
            self.train_loader.sampler.set_epoch(epoch)
            
            train_loss, _ = self.train_epoch()
            val_loss, val_auc, _, _ = self.validate(self.val_loader)

            # Logging and saving should only be done by the main process
            if self.rank == 0:
                self.metrics_history["train_loss"].append(train_loss)
                self.metrics_history["val_loss"].append(val_loss)
                self.metrics_history["val_auc"].append(val_auc)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(epoch, val_loss)

        if self.rank == 0:
            self.logger.info(f"Training completed for {self.variant_name}")
            self.save_metrics()
            
    def save_metrics(self):
        if self.rank != 0:
            return
        # This function is now only called by rank 0, so it's safe.
        try:
            parts = self.variant_name.split("_")
            dataset = parts[0]
            model_type = parts[1]
            aug_type = "with_aug" if "with_augmentation" in self.variant_name else "no_aug"
            metrics_dir = self.config.RESULTS_DIR / "metrics" / dataset / model_type / aug_type
            metrics_dir.mkdir(parents=True, exist_ok=True)
            history_path = metrics_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(self.metrics_history, f, indent=4)
            self.logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            self.logger.error(f"Error in save_metrics: {str(e)}")

