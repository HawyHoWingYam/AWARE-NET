import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import argparse

from config import get_config
from dataset import DeepfakeDataset, create_data_splits
from model import EnsembleDeepfakeDetector, SingleModelDetector
from train import Trainer
from experiments import ExperimentRunner
# from cross_evaluation import run_cross_evaluations # (Optional) Uncomment if you want to run cross-evaluation

def get_model(model_name, config, dataset_name, augment):
    """Initializes the correct model based on name."""
    if model_name == 'ensemble':
        return EnsembleDeepfakeDetector(
            config=config,
            dataset=dataset_name,
            augment=augment,
        ).cuda()
    else:
        return SingleModelDetector(model_name, config).cuda()

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Framework")
    parser.add_argument(
        "--datasets",
        type=str,
        default="celebdf,ff++",
        help="Comma-separated list of datasets to train on (e.g., 'celebdf,ff++').",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="xception,res2net101_26w_4s,tf_efficientnet_b7_ns,ensemble",
        help="Comma-separated list of models to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the configuration profile to use (default, ff++, celebdf, ensemble, debug).",
    )
    parser.add_argument(
        "--skip-aug", action="store_true", help="Skip training with augmentation."
    )
    parser.add_argument(
        "--skip-no-aug", action="store_true", help="Skip training without augmentation."
    )

    args = parser.parse_args()

    # Get the selected configuration object
    config = get_config(args.config)
    config.create_directories()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment with configuration: {args.config}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets_to_run = args.datasets.split(',')
    models_to_run = args.models.split(',')

    training_variants = []
    if not args.skip_no_aug:
        training_variants.append({'name': 'no_augmentation', 'augment': False})
    if not args.skip_aug:
        training_variants.append({'name': 'with_augmentation', 'augment': True})

    for dataset_name in datasets_to_run:
        logger.info(f"\n{'='*20} PROCESSING DATASET: {dataset_name.upper()} {'='*20}")
        
        train_df, val_df, test_df = create_data_splits(config, dataset_name)
        
        for model_key in models_to_run:
            if model_key not in config.MODELS:
                logger.warning(f"Model key '{model_key}' not found in config. Skipping.")
                continue

            for variant in training_variants:
                variant_name = f"{dataset_name}_{config.MODELS[model_key]['variant_key']}_{variant['name']}"
                
                logger.info(f"\n--- Training: {variant_name} ---")

                # Create datasets
                train_dataset = DeepfakeDataset(train_df, transform, augment=variant['augment'], variant_name=variant_name, config=config)
                val_dataset = DeepfakeDataset(val_df, transform)
                test_dataset = DeepfakeDataset(test_df, transform)

                # Create dataloaders
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
                
                # Initialize model
                model = get_model(model_key, config, dataset_name, variant['augment'])

                # Initialize Trainer and ExperimentRunner
                trainer = Trainer(model, train_loader, val_loader, config, variant_name, test_loader)
                experimenter = ExperimentRunner(config, model, trainer)
                
                # Run experiment
                experimenter.run_experiments(test_loader)

    # (Optional) Uncomment the following lines to run cross-dataset evaluation after all training is complete
    # logger.info("\n" + "="*50 + "\nPerforming Cross-Dataset Evaluation\n" + "="*50)
    # run_cross_evaluations(config, transform, config.MODELS)

if __name__ == "__main__":
    main()