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
    
    def plot_roc_curves(self, test_results, save_path):
        """Plot ROC curves without needing dataset_name and variant_name"""
        plt.figure(figsize=(8, 8))
        
        # If test_results is a dictionary with 'performance_metrics'
        if isinstance(test_results, dict) and 'performance_metrics' in test_results:
            fpr, tpr, _ = roc_curve(
                test_results['raw_data']['true_labels'], 
                test_results['raw_data']['predictions']
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        else:
            # Original behavior for multiple subsets
            for subset, metrics in test_results.items():
                fpr, tpr, _ = roc_curve(metrics['true_labels'], metrics['predictions'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{subset} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved ROC curves to {save_path}")
    
    def plot_confusion_matrices(self, test_results, save_path):
        """Plot confusion matrix without needing dataset_name and variant_name"""
        plt.figure(figsize=(8, 6))
        
        # Get confusion matrix from test results
        cm = np.array(test_results['confusion_matrix'])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Saved confusion matrix to {save_path}")
    
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
    
    def generate_all_plots(self, metrics_history, test_results, model_weights, dataset_name, model_type, is_no_aug):
        """Generate and save all visualization plots"""
        # Use consistent augmentation type
        aug_type = 'no_aug' if is_no_aug else 'with_aug'
        
        # Create plots directory with correct structure
        plots_dir = self.config.RESULTS_DIR / 'plots' / dataset_name / model_type / aug_type
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating plots for {dataset_name}_{model_type} ({aug_type}) in {plots_dir}")
        
        try:
            # Training metrics plot
            metrics_plot_path = plots_dir / 'training_metrics.png'
            self.plot_per_epoch_metrics(metrics_history, metrics_plot_path)
            self.logger.info(f"Saved training metrics plot to {metrics_plot_path}")
            
            # ROC curve
            roc_plot_path = plots_dir / 'roc_curve.png'
            self.plot_roc_curves(test_results, roc_plot_path)
            self.logger.info(f"Saved ROC curve to {roc_plot_path}")
            
            # Confusion matrix
            conf_matrix_path = plots_dir / 'confusion_matrix.png'
            self.plot_confusion_matrices(test_results, conf_matrix_path)
            self.logger.info(f"Saved confusion matrix to {conf_matrix_path}")
            
            # Learning curves
            learning_curves_path = plots_dir / 'learning_curves.png'
            self.plot_learning_curves(metrics_history, learning_curves_path)
            self.logger.info(f"Saved learning curves to {learning_curves_path}")
            
            # Metric correlations
            correlations_path = plots_dir / 'metric_correlations.png'
            self.plot_metric_correlations(metrics_history, correlations_path)
            
            # Model weights plot (for ensemble only)
            if model_weights is not None:
                weights_path = plots_dir / 'model_weights.png'
                plt.figure(figsize=(8, 6))
                model_names = ['Xception', 'Res2Net101', 'EfficientNet-B7']
                sns.barplot(x=model_names, y=model_weights)
                plt.title('Model Contribution Weights')
                plt.ylabel('Weight')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(weights_path)
                plt.close()
                self.logger.info(f"Saved model weights plot to {weights_path}")
            
            self.logger.info(f"All plots generated and saved in {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            self.logger.error(f"metrics_history keys: {metrics_history.keys() if metrics_history else 'None'}")
            self.logger.error(f"test_results keys: {test_results.keys() if test_results else 'None'}")
            raise
    
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
        try:
            # Get only numeric arrays of same length
            valid_metrics = {}
            min_length = min(len(v) for v in metrics_history.values() if isinstance(v, list))
            
            for key, values in metrics_history.items():
                if isinstance(values, list):
                    valid_metrics[key] = values[:min_length]
            
            if valid_metrics:
                metrics_df = pd.DataFrame(valid_metrics)
                plt.figure(figsize=(10, 8))
                sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Metric Correlations')
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                self.logger.info(f"Saved metric correlations to {save_path}")
            else:
                self.logger.warning("No valid metrics for correlation plot")
                
        except Exception as e:
            self.logger.error(f"Error in plot_metric_correlations: {str(e)}")
            self.logger.error(f"Metrics keys: {metrics_history.keys()}")
            self.logger.error(f"Metrics types: {[(k, type(v)) for k, v in metrics_history.items()]}")
    
    def plot_per_epoch_metrics(self, metrics_history, save_path):
        """Plot detailed per-epoch metrics"""
        try:
            # Ensure all arrays are of the same length
            min_length = min(len(metrics_history[key]) for key in metrics_history.keys())
            
            # Truncate all arrays to the same length
            metrics_history_equal = {
                key: values[:min_length] 
                for key, values in metrics_history.items()
                if isinstance(values, list)  # Only include list metrics
            }
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epochs = range(1, min_length + 1)
            
            # Loss plot
            axes[0, 0].plot(epochs, metrics_history_equal['train_loss'], 'b-', label='Train')
            axes[0, 0].plot(epochs, metrics_history_equal['val_loss'], 'r-', label='Validation')
            axes[0, 0].set_title('Loss Over Epochs')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # AUC plot
            axes[0, 1].plot(epochs, metrics_history_equal['train_auc'], 'b-', label='Train')
            axes[0, 1].plot(epochs, metrics_history_equal['val_auc'], 'r-', label='Validation')
            axes[0, 1].set_title('AUC Over Epochs')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
            
            # Loss vs AUC scatter
            axes[1, 0].scatter(metrics_history_equal['train_loss'], metrics_history_equal['train_auc'], 
                              label='Train', alpha=0.6)
            axes[1, 0].scatter(metrics_history_equal['val_loss'], metrics_history_equal['val_auc'], 
                              label='Validation', alpha=0.6)
            axes[1, 0].set_title('Loss vs AUC')
            axes[1, 0].set_xlabel('Loss')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            
            # Correlation heatmap
            metrics_df = pd.DataFrame(metrics_history_equal)
            sns.heatmap(metrics_df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
            axes[1, 1].set_title('Metrics Correlation')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Saved per-epoch metrics plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error in plot_per_epoch_metrics: {str(e)}")
            self.logger.error(f"Metrics history keys: {metrics_history.keys()}")
            self.logger.error(f"Metrics history lengths: {[(k, len(v) if isinstance(v, list) else 'not list') for k, v in metrics_history.items()]}")
            raise