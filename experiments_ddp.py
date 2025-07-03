import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import json
import logging
from datetime import datetime
from visualization import Visualizer
from tqdm import tqdm

class ExperimentRunner:
    def __init__(self, config, model, trainer):
        self.config = config
        self.model = model # This is the DDP-wrapped model
        self.trainer = trainer
        self.results_dir = config.RESULTS_DIR
        self.logger = logging.getLogger(__name__)
        # Get rank from the trainer
        self.rank = trainer.rank

    def run_experiments(self, test_loader):
        # The trainer's train method is now DDP-aware
        self.trainer.train()

        # Ensure all processes have finished training before starting evaluation
        dist.barrier()

        # The main process (rank 0) will handle the final evaluation and reporting
        if self.rank == 0:
            self.logger.info("\nTraining finished. Starting final evaluation on test set.")
        
        metrics = self._evaluate_dataset(test_loader, self.trainer.variant_name)
        
        # Only rank 0 gets the full metrics and should save results/visualizations
        if self.rank == 0 and metrics is not None:
            variant_parts = self.trainer.variant_name.split('_')
            dataset = variant_parts[0]
            model_type = variant_parts[1]
            aug_type = 'with_aug' if 'with_augmentation' in self.trainer.variant_name else 'no_aug'
            
            results_dir = self.results_dir / 'metrics' / dataset / model_type / aug_type
            results_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_results(metrics, results_dir)
            
            plots_dir = self.results_dir / 'plots' / dataset / model_type / aug_type
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            self._generate_visualizations(
                metrics=metrics, 
                plots_dir=plots_dir,
                is_no_aug=(aug_type == 'no_aug'),
                dataset=dataset,
                model_type=model_type
            )
        
        # Wait for rank 0 to finish before the next experiment starts
        dist.barrier()

    def _evaluate_dataset(self, loader, variant_name):
        self.model.eval()
        
        # Each process will store its own predictions and labels
        predictions_shard = []
        labels_shard = []
        
        pbar = tqdm(loader, desc=f'Rank {self.rank} Evaluating', leave=False, disable=(self.rank != 0))
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.rank)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                predictions_shard.extend(probs.cpu().numpy())
                labels_shard.extend(labels.cpu().numpy())

        # Convert shard lists to tensors for gathering
        predictions_tensor = torch.tensor(predictions_shard).to(self.rank)
        labels_tensor = torch.tensor(labels_shard).to(self.rank)

        # Prepare lists to gather tensors from all processes
        world_size = dist.get_world_size()
        gathered_predictions = [torch.zeros_like(predictions_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]

        # Use all_gather to collect results from all GPUs on all GPUs
        dist.all_gather(gathered_predictions, predictions_tensor)
        dist.all_gather(gathered_labels, labels_tensor)

        # The main process (rank 0) will compute and return the final metrics
        if self.rank == 0:
            # Concatenate the gathered tensors
            all_predictions = np.concatenate([t.cpu().numpy() for t in gathered_predictions])
            all_labels = np.concatenate([t.cpu().numpy() for t in gathered_labels])

            self.logger.info("Calculating final metrics on rank 0...")
            auc = roc_auc_score(all_labels, all_predictions)
            pred_labels = (all_predictions > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_labels, average='binary')
            accuracy = accuracy_score(all_labels, pred_labels)
            
            metrics = {
                'performance_metrics': {
                    'auc': float(auc), 'accuracy': float(accuracy),
                    'precision': float(precision), 'recall': float(recall), 'f1': float(f1)
                },
                'raw_data': {
                    'predictions': all_predictions.tolist(), 'true_labels': all_labels.tolist()
                },
                'confusion_matrix': confusion_matrix(all_labels, pred_labels).tolist()
            }
            
            self.logger.info(f"\nEvaluation metrics for {variant_name}:")
            self.logger.info(f"{'='*50}")
            self.logger.info(f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            self.logger.info(f"{'='*50}")
            
            return metrics
        
        # Other processes don't need to return metrics
        return None

    def _save_results(self, metrics, results_dir):
        # This function is now only called by rank 0
        test_results_path = results_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Saved test results to {test_results_path}")

    def _generate_visualizations(self, metrics, plots_dir, is_no_aug, dataset, model_type):
        # This function is now only called by rank 0
        self.logger.info(f"Generating visualizations in {plots_dir}")
        visualizer = Visualizer(self.config)
        visualizer.generate_all_plots(
            metrics_history=self.trainer.metrics_history,
            test_results=metrics,
            model_weights=self.model.module.get_model_weights() if hasattr(self.model.module, 'get_model_weights') else None,
            dataset_name=dataset,
            model_type=model_type,
            is_no_aug=is_no_aug
        )
