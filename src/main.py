import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import logging
import json

from config import Config
from dataset import DeepfakeDataset, create_data_splits
from model import EnsembleDeepfakeDetector, SingleModelDetector
from train import Trainer
from experiments import ExperimentRunner

def train_all_models(config, transform):
    logger = logging.getLogger(__name__)
    
    # Model configurations with exact names
    models_config = {
        'xception': {'name': 'xception', 'type': 'single'},
        'res2net101_26w_4s': {'name': 'res2net101_26w_4s', 'type': 'single'},
        'tf_efficientnet_b7_ns': {'name': 'tf_efficientnet_b7_ns', 'type': 'single'},
        'ensemble': {'type': 'ensemble'}
    }
    
    # Training variants with consistent naming
    variants = [
        {'name': 'no_augmentation', 'augment': False},
        {'name': 'with_augmentation', 'augment': True}
    ]
    
    # Datasets
    datasets = ['ff++', 'celebdf']
    
    # First run without augmentation for all models and datasets
    logger.info("\n" + "="*50 + "\nStarting training without augmentation\n" + "="*50)
    for dataset_name in datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Create data splits for this dataset
        train_df, val_df, test_df = create_data_splits(config, dataset_name, force_new=True)
        
        # Train each model
        for model_key, model_config in models_config.items():
            variant_name = f"{dataset_name}_{model_key}_no_augmentation"
            logger.info(f"\nTraining {variant_name}")
            
            # Create datasets without augmentation
            train_dataset = DeepfakeDataset(
                dataframe=train_df,
                transform=transform,
                augment=False
            )
            
            val_dataset = DeepfakeDataset(
                dataframe=val_df,
                transform=transform
            )
            
            test_dataset = DeepfakeDataset(
                dataframe=test_df,
                transform=transform
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4
            )
            
            # Initialize model
            if model_config['type'] == 'single':
                model = SingleModelDetector(model_config['name'], config).cuda()
            else:
                model = EnsembleDeepfakeDetector(config).cuda()
            
            # Train and evaluate
            trainer = Trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                variant_name=variant_name
            )
            
            experimenter = ExperimentRunner(config, model, trainer)
            results = experimenter.run_experiments(test_loader)
            
            logger.info(f"Results for {variant_name}:")
            logger.info(json.dumps(results, indent=2))
    
    # Then run with augmentation for all models and datasets
    logger.info("\n" + "="*50 + "\nStarting training with augmentation\n" + "="*50)
    for dataset_name in datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Create data splits for this dataset
        train_df, val_df, test_df = create_data_splits(config, dataset_name, force_new=True)
        
        # Train each model
        for model_key, model_config in models_config.items():
            variant_name = f"{dataset_name}_{model_key}_with_augmentation"
            logger.info(f"\nTraining {variant_name}")
            
            # Create datasets with augmentation
            train_dataset = DeepfakeDataset(
                dataframe=train_df,
                transform=transform,
                augment=True
            )
            
            val_dataset = DeepfakeDataset(
                dataframe=val_df,
                transform=transform
            )
            
            test_dataset = DeepfakeDataset(
                dataframe=test_df,
                transform=transform
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4
            )
            
            # Initialize model
            if model_config['type'] == 'single':
                model = SingleModelDetector(model_config['name'], config).cuda()
            else:
                model = EnsembleDeepfakeDetector(config).cuda()
            
            # Train and evaluate
            trainer = Trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                variant_name=variant_name
            )
            
            experimenter = ExperimentRunner(config, model, trainer)
            results = experimenter.run_experiments(test_loader)
            
            logger.info(f"Results for {variant_name}:")
            logger.info(json.dumps(results, indent=2))

def main():
    try:
        # Initialize config and create directories
        config = Config()
        config.create_directories()
        config.setup_logging()
        
        logger = logging.getLogger(__name__)
        
        logger.info("Starting experiment with following configuration:")
        logger.info(f"Batch Size: {config.BATCH_SIZE}")
        logger.info(f"Dataset Fraction: {config.DATASET_FRACTION}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        # Create data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Run all experiments
        train_all_models(config, transform)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 