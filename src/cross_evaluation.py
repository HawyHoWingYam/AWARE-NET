import torch
from torch.utils.data import DataLoader
import logging
import json
from tqdm import tqdm
from pathlib import Path

from model import EnsembleDeepfakeDetector, SingleModelDetector
from dataset import DeepfakeDataset, create_data_splits
from experiments import ExperimentRunner

def cross_dataset_evaluate(config, transform, source_dataset, target_dataset, model_name, model_path):
    """Evaluate model trained on source dataset on target dataset"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\nCross-Dataset Evaluation:")
    logger.info(f"Source Dataset: {source_dataset}")
    logger.info(f"Target Dataset: {target_dataset}")
    logger.info(f"Model: {model_name}")
    
    # Load target dataset
    _, _, test_df = create_data_splits(config, target_dataset, force_new=True)
    
    # Create test dataset
    test_dataset = DeepfakeDataset(
        dataframe=test_df,
        transform=transform,
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load trained model
    if model_name == 'ensemble':
        model = EnsembleDeepfakeDetector(config).cuda()
    else:
        model = SingleModelDetector(model_name, config).cuda()
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create experimenter for evaluation
    experimenter = ExperimentRunner(config, model, None)  # No trainer needed for evaluation
    
    # Evaluate
    metrics = experimenter.evaluate_model(
        model=model,
        loader=test_loader,
        experiment_name=f"{source_dataset}_to_{target_dataset}_{model_name}"
    )
    
    # Save cross-evaluation results
    results_dir = config.RESULTS_DIR / 'cross_evaluation' / source_dataset / model_name / target_dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"\nCross-Evaluation Results ({source_dataset} → {target_dataset}):")
    logger.info(json.dumps(metrics['performance_metrics'], indent=2))
    
    return metrics

def run_cross_evaluations(config, transform, models_config):
    """Run all cross-dataset evaluations"""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50 + "\nPerforming Cross-Dataset Evaluation\n" + "="*50)
    
    for model_key, model_config in models_config.items():
        model_name = model_config['name'] if 'name' in model_config else model_key
        
        # FF++ → CelebDF
        model_path = config.RESULTS_DIR / 'weights' / 'ff++' / model_name / 'no_aug' / 'best_model.pth'
        if model_path.exists():
            cross_dataset_evaluate(config, transform, 'ff++', 'celebdf', model_name, model_path)
        
        # CelebDF → FF++
        model_path = config.RESULTS_DIR / 'weights' / 'celebdf' / model_name / 'no_aug' / 'best_model.pth'
        if model_path.exists():
            cross_dataset_evaluate(config, transform, 'celebdf', 'ff++', model_name, model_path)
        
        # Same for augmented models
        model_path = config.RESULTS_DIR / 'weights' / 'ff++' / model_name / 'with_aug' / 'best_model.pth'
        if model_path.exists():
            cross_dataset_evaluate(config, transform, 'ff++', 'celebdf', model_name, model_path)
        
        model_path = config.RESULTS_DIR / 'weights' / 'celebdf' / model_name / 'with_aug' / 'best_model.pth'
        if model_path.exists():
            cross_dataset_evaluate(config, transform, 'celebdf', 'ff++', model_name, model_path) 