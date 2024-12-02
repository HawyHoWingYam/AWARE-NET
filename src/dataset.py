import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import Augmentor
import json
import logging

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        if augment:
            self.augmentor = Augmentor.Pipeline()
            self.augmentor.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
            self.augmentor.flip_left_right(probability=0.5)
            self.augmentor.random_brightness(probability=0.7, min_factor=0.8, max_factor=1.2)
            self.augmentor.random_contrast(probability=0.7, min_factor=0.8, max_factor=1.2)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.augment:
            image = self.augmentor.torch_transform(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

def create_data_splits(config):
    logger = logging.getLogger(__name__)
    
    # Load or create annotations
    annotation_file = config.ANNOTATIONS_DIR / 'splits.json'
    if annotation_file.exists():
        logger.info("Loading existing annotations...")
        with open(annotation_file, 'r') as f:
            splits = json.load(f)
            return pd.DataFrame(splits['train']), pd.DataFrame(splits['val']), pd.DataFrame(splits['test'])
    
    logger.info("Creating new annotations...")
    all_data = []
    
    # Process FF++ subsets
    for subset in config.FF_SUBSETS:
        real_paths = list(Path(config.DATA_DIR).rglob(f"ff++/real/**/*.jpg"))
        fake_paths = list(Path(config.DATA_DIR).rglob(f"ff++/{subset}/**/*.jpg"))
        
        # Take 50% of data
        real_paths = real_paths[:int(len(real_paths) * config.DATASET_FRACTION)]
        fake_paths = fake_paths[:int(len(fake_paths) * config.DATASET_FRACTION)]
        
        logger.info(f"Subset {subset}: {len(real_paths)} real, {len(fake_paths)} fake images")
        
        all_data.extend([{'path': str(p), 'label': 0, 'subset': 'real'} for p in real_paths])
        all_data.extend([{'path': str(p), 'label': 1, 'subset': subset} for p in fake_paths])
    
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    train_idx = int(len(df) * config.TRAIN_SPLIT)
    val_idx = train_idx + int(len(df) * config.VAL_SPLIT)
    
    # Split data
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    # Save annotations
    splits = {
        'train': train_df.to_dict('records'),
        'val': val_df.to_dict('records'),
        'test': test_df.to_dict('records')
    }
    
    with open(config.ANNOTATIONS_DIR / 'splits.json', 'w') as f:
        json.dump(splits, f)
    
    return train_df, val_df, test_df 