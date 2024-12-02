import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentRunner:
    def __init__(self, config, model, trainer):
        self.config = config
        self.model = model
        self.trainer = trainer
        self.results_dir = config.RESULTS_DIR
        
    def run_experiments(self, test_loader_ff, test_loader_celeb):
        # Train the model
        self.trainer.train()
        
        # Load best model weights
        checkpoint = torch.load(self.results_dir / 'weights' / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test on both datasets
        results = {}
        
        # FF++ results
        ff_metrics = self._evaluate_dataset(test_loader_ff, "FF++")
        results['ff++'] = ff_metrics
        
        # CelebDF results
        celeb_metrics = self._evaluate_dataset(test_loader_celeb, "CelebDF")
        results['celebdf'] = celeb_metrics
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
    
    def _evaluate_dataset(self, loader, dataset_name):
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.trainer.device)
                outputs = self.model(images)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        auc = roc_auc_score(true_labels, predictions)
        pred_labels = (predictions > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        accuracy = accuracy_score(true_labels, pred_labels)
        
        return {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'predictions': predictions.tolist(),
            'true_labels': true_labels.tolist()
        }
    
    def _save_results(self, results):
        # Save metrics
        results_path = self.results_dir / 'metrics' / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save model weights
        weights = self.model.get_model_weights()
        weights_path = self.results_dir / 'metrics' / 'model_weights.json'
        with open(weights_path, 'w') as f:
            json.dump({
                'xception': float(weights[0]),
                'efficientnet': float(weights[1]),
                'swin': float(weights[2])
            }, f, indent=4)
    
    def _generate_visualizations(self, results):
        plots_dir = self.results_dir / 'plots'
        
        # Plot ROC curves
        self._plot_roc_curves(results, plots_dir / 'roc_curves.png')
        
        # Plot confusion matrices
        self._plot_confusion_matrices(results, plots_dir / 'confusion_matrices.png')
        
        # Plot model weights
        self._plot_model_weights(plots_dir / 'model_weights.png')
    
    def _plot_model_weights(self, save_path):
        weights = self.model.get_model_weights()[0]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=['Xception', 'EfficientNet', 'Swin'], y=weights)
        plt.title('Model Contribution Weights')
        plt.ylabel('Weight')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 