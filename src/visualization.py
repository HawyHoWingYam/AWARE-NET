import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import logging
import pandas as pd

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def plot_training_history(self, metrics_history, dataset_name, variant_name):
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(metrics_history['train_loss'], label='Train Loss')
        plt.plot(metrics_history['val_loss'], label='Val Loss')
        plt.title(f'{dataset_name} - {variant_name}\nLoss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # AUC plot
        plt.subplot(1, 2, 2)
        plt.plot(metrics_history['train_auc'], label='Train AUC')
        plt.plot(metrics_history['val_auc'], label='Val AUC')
        plt.title(f'{dataset_name} - {variant_name}\nAUC History')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        save_path = self.config.RESULTS_DIR / 'plots' / f'{dataset_name}_{variant_name}_training_history.png'
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved training history plot to {save_path}")
    
    def plot_roc_curves(self, results, dataset_name, variant_name):
        plt.figure(figsize=(8, 8))
        
        for subset, metrics in results.items():
            fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['predictions'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{subset} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{dataset_name} - {variant_name}\nROC Curves')
        plt.legend(loc="lower right")
        
        save_path = self.config.RESULTS_DIR / 'plots' / f'{dataset_name}_{variant_name}_roc_curves.png'
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved ROC curves to {save_path}")
    
    def plot_confusion_matrices(self, results, dataset_name, variant_name):
        n_subsets = len(results)
        plt.figure(figsize=(5*n_subsets, 4))
        
        for i, (subset, metrics) in enumerate(results.items(), 1):
            plt.subplot(1, n_subsets, i)
            cm = confusion_matrix(metrics['true_labels'], 
                                (np.array(metrics['predictions']) > 0.5).astype(int))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{subset} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
        
        plt.tight_layout()
        save_path = self.config.RESULTS_DIR / 'plots' / f'{dataset_name}_{variant_name}_confusion_matrices.png'
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved confusion matrices to {save_path}")
    
    def plot_model_weights(self, weights, dataset_name, variant_name):
        plt.figure(figsize=(8, 6))
        model_names = ['Xception', 'Res2Net101', 'EfficientNet-B7']
        sns.barplot(x=model_names, y=weights)
        plt.title(f'{dataset_name} - {variant_name}\nModel Contribution Weights')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_path = self.config.RESULTS_DIR / 'plots' / f'{dataset_name}_{variant_name}_model_weights.png'
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved model weights plot to {save_path}")
    
    def generate_all_plots(self, metrics_history, test_results, model_weights, dataset_name, variant_name):
        """Generate and save all visualization plots"""
        # Create proper directory structure
        dataset, model_name, aug_type = variant_name.split('_')
        aug_folder = 'with_augmentation' if 'with' in aug_type else 'no_augmentation'
        plots_dir = self.config.RESULTS_DIR / 'plots' / dataset / model_name / aug_folder
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating plots for {variant_name} in {plots_dir}")
        
        # Training history
        self.plot_per_epoch_metrics(metrics_history, plots_dir / 'training_metrics.png')
        
        # ROC curve
        self.plot_roc_curves(test_results, plots_dir / 'roc_curve.png')
        
        # Confusion matrix
        self.plot_confusion_matrices(test_results, plots_dir / 'confusion_matrix.png')
        
        # Model weights (for ensemble only)
        if model_weights is not None:
            self.plot_model_weights(model_weights, plots_dir / 'model_weights.png')
        
        self.logger.info(f"All plots generated and saved in {plots_dir}")
    
    def plot_learning_curves(self, metrics_history, save_path):
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics_history['train_auc'], 'b-', label='Training AUC')
        plt.plot(epochs, metrics_history['val_auc'], 'r-', label='Validation AUC')
        plt.title('AUC Curves')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_metric_correlations(self, metrics_history, save_path):
        metrics_df = pd.DataFrame(metrics_history)
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlations')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_per_epoch_metrics(self, metrics_history, save_path):
        """Plot detailed per-epoch metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(metrics_history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss Over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # AUC plot
        axes[0, 1].plot(epochs, metrics_history['train_auc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, metrics_history['val_auc'], 'r-', label='Validation')
        axes[0, 1].set_title('AUC Over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        
        # Loss vs AUC scatter
        axes[1, 0].scatter(metrics_history['train_loss'], metrics_history['train_auc'], 
                          label='Train', alpha=0.6)
        axes[1, 0].scatter(metrics_history['val_loss'], metrics_history['val_auc'], 
                          label='Validation', alpha=0.6)
        axes[1, 0].set_title('Loss vs AUC')
        axes[1, 0].set_xlabel('Loss')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Correlation heatmap
        metrics_df = pd.DataFrame(metrics_history)
        sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved per-epoch metrics plot to {save_path}")