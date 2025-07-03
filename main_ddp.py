import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os
import argparse
import logging

from config import get_config
from dataset import DeepfakeDataset, create_data_splits
from model import EnsembleDeepfakeDetector, SingleModelDetector
from train import Trainer
from experiments import ExperimentRunner
from torchvision import transforms

def setup(rank, world_size):
    """Initializes the process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroys the process group."""
    dist.destroy_process_group()

def get_model(model_name, config, dataset_name, augment):
    """Initializes the correct model based on name."""
    # Note: Model is moved to the correct device inside the run_training function
    if model_name == 'ensemble':
        return EnsembleDeepfakeDetector(
            config=config,
            dataset=dataset_name,
            augment=augment,
        )
    else:
        return SingleModelDetector(model_name, config)

def run_training(rank, world_size, args):
    """
    This function contains the core training logic that will be executed by each process.
    """
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    # --- Configuration and Logging ---
    # Only the main process should handle directory creation and logging configuration
    # to avoid race conditions.
    config = get_config(args.config)
    if rank == 0:
        config.create_directories()

    logger = logging.getLogger(__name__)
    
    # --- Dataset and Sampler ---
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
        if rank == 0:
            logger.info(f"\n{'='*20} PROCESSING DATASET: {dataset_name.upper()} {'='*20}")
        
        # All processes will call this, but it should be deterministic.
        # Alternatively, have rank 0 create splits and then load them in all processes.
        train_df, val_df, test_df = create_data_splits(config, dataset_name)
        
        for model_key in models_to_run:
            if model_key not in config.MODELS:
                if rank == 0:
                    logger.warning(f"Model key '{model_key}' not found in config. Skipping.")
                continue

            for variant in training_variants:
                variant_name = f"{dataset_name}_{config.MODELS[model_key]['variant_key']}_{variant['name']}"
                
                if rank == 0:
                    logger.info(f"\n--- Training: {variant_name} ---")

                # --- Create Datasets ---
                train_dataset = DeepfakeDataset(train_df, transform, augment=variant['augment'], variant_name=variant_name, config=config)
                val_dataset = DeepfakeDataset(val_df, transform)
                test_dataset = DeepfakeDataset(test_df, transform)

                # --- Create Distributed Samplers ---
                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

                # --- Create DataLoaders ---
                # shuffle must be False when using a DistributedSampler
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, sampler=train_sampler)
                val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, sampler=val_sampler)
                test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, sampler=test_sampler)
                
                # --- Model Initialization and DDP Wrapping ---
                model = get_model(model_key, config, dataset_name, variant['augment']).to(rank)
                ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

                # --- Trainer and ExperimentRunner ---
                # Pass the DDP-wrapped model to the Trainer
                trainer = Trainer(ddp_model, train_loader, val_loader, config, variant_name, test_loader, rank=rank)
                experimenter = ExperimentRunner(config, ddp_model, trainer)
                
                # --- Run Experiment ---
                # The trainer's train loop will now run in a distributed fashion.
                # Evaluation should also be handled correctly.
                experimenter.run_experiments(test_loader)

    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Distributed Deepfake Detection Training Framework")
    parser.add_argument(
        "--datasets", type=str, default="celebdf,ff++", help="Comma-separated list of datasets."
    )
    parser.add_argument(
        "--models", type=str, default="xception,res2net101_26w_4s,tf_efficientnet_b7_ns,ensemble", help="Comma-separated list of models."
    )
    parser.add_argument(
        "--config", type=str, default="default", help="Configuration profile."
    )
    parser.add_argument(
        "--skip-aug", action="store_true", help="Skip training with augmentation."
    )
    parser.add_argument(
        "--skip-no-aug", action="store_true", help="Skip training without augmentation."
    )
    args = parser.parse_args()

    # Set the number of GPUs you have
    world_size = 2
    
    # Spawn the training processes
    mp.spawn(run_training,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
