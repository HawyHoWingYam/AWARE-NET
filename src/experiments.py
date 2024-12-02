import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

class ExperimentRunner:
    def __init__(self, config, model, trainer):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.results_dir = config.RESULTS_DIR
        self.logger = logging.getLogger(__name__)
        
    def run_experiments(self, test_loader):
        try:
            # Train the model first
            self.trainer.train()
            
            # Parse variant name safely
            parts = self.trainer.variant_name.split('_')
            dataset = parts[0]
            model_type = parts[1]
            aug_type = 'with_aug' if 'with' in parts[-1] else 'no_aug'
            
            # Test on the dataset
            metrics = self._evaluate_dataset(test_loader, self.trainer.variant_name)
            
            # Save results in the correct directory
            results_dir = self.results_dir / 'metrics' / dataset / model_type / aug_type
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            self._save_results(metrics, results_dir)
            
            # Generate visualizations
            plots_dir = self.results_dir / 'plots' / dataset / model_type / aug_type
            plots_dir.mkdir(parents=True, exist_ok=True)
            self._generate_visualizations(metrics, plots_dir)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in run_experiments: {str(e)}")
            raise
    
    def _evaluate_dataset(self, loader, variant_name):
        self.logger.info(f"Starting evaluation for {variant_name}")
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.trainer.device)
                outputs = self.model(images)
                
                # Handle both single model and ensemble outputs
                if isinstance(outputs, torch.Tensor) and outputs.shape[1] == 2:
                    # For models outputting class probabilities
                    probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probability of fake class
                else:
                    # For models already outputting single probability
                    probs = outputs.squeeze()
                
                predictions.extend(probs.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Ensure predictions are 1D
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        
        # Calculate all metrics
        try:
            auc = roc_auc_score(true_labels, predictions)
            pred_labels = (predictions > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
            accuracy = accuracy_score(true_labels, pred_labels)
            
            metrics = {
                'performance_metrics': {
                    'auc': float(auc),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                },
                'raw_data': {
                    'predictions': predictions.tolist(),
                    'true_labels': true_labels.tolist()
                },
                'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
            }
            
            self.logger.info(f"Evaluation metrics for {variant_name}:")
            self.logger.info(f"AUC: {auc:.4f}")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            self.logger.error(f"Predictions shape: {predictions.shape}")
            self.logger.error(f"Labels shape: {true_labels.shape}")
            raise
        
        return metrics
    
    def _save_results(self, metrics, results_dir):
        # Save test metrics
        test_results_path = results_dir / 'test_results.json'
        self.logger.info(f"Saving test results to {test_results_path}")
        
        # Add timestamp and configuration to results
        full_results = {
            'test_metrics': metrics,
            'configuration': {
                'model_type': type(self.model).__name__,
                'dataset_fraction': self.config.DATASET_FRACTION,
                'image_size': self.config.IMAGE_SIZE,
                'batch_size': self.config.BATCH_SIZE
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(test_results_path, 'w') as f:
            json.dump(full_results, f, indent=4)
        
        # Save model weights if it's an ensemble
        if hasattr(self.model, 'get_model_weights'):
            weights = self.model.get_model_weights()
            if weights is not None:
                weights_path = results_dir / 'ensemble_weights.json'
                with open(weights_path, 'w') as f:
                    json.dump({
                        'xception': float(weights[0]),
                        'res2net': float(weights[1]),
                        'efficientnet': float(weights[2])
                    }, f, indent=4)
                self.logger.info(f"Saved ensemble weights to {weights_path}")
    
    def _generate_visualizations(self, metrics, plots_dir):
        self.logger.info(f"Generating visualizations in {plots_dir}")
        
        # Create Visualizer instance
        visualizer = Visualizer(self.config)
        
        # Generate all plots
        visualizer.generate_all_plots(
            metrics_history=self.trainer.metrics_history,
            test_results=metrics,
            model_weights=self.model.get_model_weights() if hasattr(self.model, 'get_model_weights') else None,
            dataset_name=self.trainer.variant_name.split('_')[0],
            variant_name=self.trainer.variant_name
        )
    
    def _plot_model_weights(self, save_path):
        weights = self.model.get_model_weights()
        plt.figure(figsize=(8, 6))
        model_names = ['Xception', 'Res2Net101', 'EfficientNet-B7']
        sns.barplot(x=model_names, y=weights)
        plt.title('Architecture Contribution Weights')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 