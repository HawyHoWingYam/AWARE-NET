import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import logging
import json

from config import Config
from dataset import DeepfakeDataset, create_data_splits
from model import HybridDeepfakeDetector
from train import Trainer
from experiments import ExperimentRunner

def train_ff_subsets(config):
    logger = logging.getLogger(__name__)
    
    for subset in config.FF_SUBSETS:
        logger.info(f"\n{'='*50}\nTraining on FF++ {subset}\n{'='*50}")
        
        # Filter data for current subset
        train_df_subset = train_df[
            (train_df.subset == 'real') | 
            (train_df.subset == subset)
        ]
        
        # Create dataset and loader for subset
        train_dataset = DeepfakeDataset(
            train_df_subset.path.values,
            train_df_subset.label.values,
            transform=transform,
            augment=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        
        # Initialize model and trainer
        model = HybridDeepfakeDetector(config).cuda()
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader_ff,
            subset_name=subset
        )
        
        # Train and evaluate
        experimenter = ExperimentRunner(config, model, trainer)
        results = experimenter.run_experiments(test_loader_ff, test_loader_celeb)
        
        logger.info(f"Results for {subset}:")
        logger.info(json.dumps(results, indent=2))

def main():
    config = Config()
    config.setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting experiment with following configuration:")
    logger.info(f"Batch Size: {config.BATCH_SIZE}")
    logger.info(f"Dataset Fraction: {config.DATASET_FRACTION}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(config)
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        train_df.path.values, 
        train_df.label.values,
        transform=transform,
        augment=True
    )
    
    val_dataset = DeepfakeDataset(
        val_df.path.values,
        val_df.label.values,
        transform=transform
    )
    
    test_dataset_ff = DeepfakeDataset(
        test_df[test_df.path.str.contains('FF++')].path.values,
        test_df[test_df.path.str.contains('FF++')].label.values,
        transform=transform
    )
    
    test_dataset_celeb = DeepfakeDataset(
        test_df[test_df.path.str.contains('CelebDF')].path.values,
        test_df[test_df.path.str.contains('CelebDF')].label.values,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    test_loader_ff = DataLoader(
        test_dataset_ff,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    test_loader_celeb = DataLoader(
        test_dataset_celeb,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Train on FF++ subsets
    train_ff_subsets(config)
    
    print("Experiment completed. Results saved in:", config.RESULTS_DIR)

if __name__ == "__main__":
    main() 