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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
    
class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, transform=None, augment=False, variant_name=None, config=None):
        self.data = dataframe
        self.transform = transform  # 保留原始 transform 用于非增强处理
        self.augment = augment
        self.variant_name = variant_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if augment:
            self.logger.info("Initializing Albumentations augmentation pipeline:")
            params = self.config.AUGMENTATION_PARAMS
            
            # 创建 Albumentations 转换管道
            self.albumentations_transform = A.Compose([
                # 几何变换 (保留原有的)
                A.Rotate(limit=params['rotation']['max_left'], 
                         p=params['rotation']['probability']),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.05, 
                    rotate_limit=params['rotation']['max_right'],
                    p=params['shear']['probability']
                ),
                A.HorizontalFlip(p=params['flip']['probability']),
                
                # 新增的真实世界退化模拟
                A.ImageCompression(
                    quality_lower=60, quality_upper=100, 
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.4),
                ], p=0.4),
                A.Downscale(
                    scale_min=0.6, scale_max=0.9,
                    p=0.3
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
                ], p=0.4),
            ])
            
            self.logger.info("Albumentations augmentation pipeline:")
            self.logger.info("- Rotation: p={:.1f}".format(params['rotation']['probability']))
            self.logger.info("- ShiftScaleRotate: p={:.1f}".format(params['shear']['probability']))
            self.logger.info("- HorizontalFlip: p={:.1f}".format(params['flip']['probability']))
            self.logger.info("- ImageCompression: p=0.5")
            self.logger.info("- Blur (Gaussian/Median): p=0.4")
            self.logger.info("- Downscale: p=0.3")
            self.logger.info("- Noise (Gaussian/ISO): p=0.4")
            
            # 创建增强数据集
            self.augmented_data = self._create_augmented_dataset()
            self.data = pd.concat([self.data, self.augmented_data], ignore_index=True)
            self.logger.info(f"Dataset size increased from {len(dataframe)} to {len(self.data)} with augmentations")
            
            # 保存样本增强图像
            self.save_augmented_samples()
    
    def _create_augmented_dataset(self):
        """Create augmented versions of the dataset using Albumentations"""
        augmented_data = []
        
        # 计算需要增强的图像数量
        num_images = len(self.data)
        target_aug_images = int(num_images * self.config.AUGMENTATION_RATIO)
        
        self.logger.info(f"Creating {target_aug_images} augmented images ({self.config.AUGMENTATION_RATIO*100}% of {num_images})")
        
        # 随机选择图像进行增强
        indices = np.random.choice(num_images, size=target_aug_images, replace=True)
        
        for idx in indices:
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            image_np = np.array(image)
            
            # 使用 Albumentations 进行增强
            augmented = self.albumentations_transform(image=image_np)
            aug_image_np = augmented['image']
            
            # 创建新行记录增强图像信息
            aug_row = row.copy()
            aug_row['is_augmented'] = True
            augmented_data.append(aug_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        self.logger.info(f"Created {len(augmented_df)} augmented images")
        
        return augmented_df
    
    def save_augmented_samples(self):
        """Save sample augmented images for visualization"""
        try:
            # 随机选择 3 张图像
            sample_indices = np.random.choice(len(self.data), size=3, replace=False)
            
            for i, idx in enumerate(sample_indices):
                row = self.data.iloc[idx]
                orig_image = Image.open(row['image_path']).convert('RGB')
                orig_image_np = np.array(orig_image)
                
                # 创建 3x3 网格展示增强图像
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))
                fig.suptitle('Albumentations - Real-World Degradations', fontsize=16)
                
                # 原始图像放在中间
                axes[1, 1].imshow(orig_image)
                axes[1, 1].set_title('Original')
                axes[1, 1].axis('off')
                
                # 8 个增强版本围绕原图
                aug_positions = [(i,j) for i in range(3) for j in range(3) if (i,j) != (1,1)]
                for pos in aug_positions:
                    # 使用 Albumentations 进行增强
                    augmented = self.albumentations_transform(image=orig_image_np.copy())
                    aug_image_np = augmented['image']
                    
                    axes[pos[0], pos[1]].imshow(aug_image_np)
                    axes[pos[0], pos[1]].axis('off')
                
                # 保存图像
                dataset = 'ff++' if 'ff++' in row['image_path'] else 'celebdf'
                model_type = self.variant_name.split('_')[1]
                save_dir = self.config.RESULTS_DIR / 'plots' / dataset / model_type / 'with_aug' / 'augmented_samples'
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f'augmented_grid_{i+1}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved augmented sample grid to {save_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving augmented samples: {str(e)}")
            self.logger.error(f"Error details: {str(e.__class__.__name__)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image_np = np.array(image)
        
        if self.augment:
            # 使用 Albumentations 进行实时增强
            augmented = self.albumentations_transform(image=image_np)
            image_np = augmented['image']
            
            # 如果原始 transform 为 None，需要转换为张量
            if self.transform is None:
                to_tensor = ToTensorV2()
                augmented = to_tensor(image=image_np)
                image_tensor = augmented['image']
                return image_tensor, row['label']
        
        # 如果有原始 transform，使用它
        if self.transform:
            # 将 numpy 数组转回 PIL Image
            if isinstance(image_np, np.ndarray):
                image = Image.fromarray(image_np.astype('uint8'))
            image = self.transform(image)
            
        return image, row['label']

def create_data_splits(config, dataset_name, force_new=None):
    logger = logging.getLogger(__name__)

    
    # Check if we should use force_new from config
    if force_new is None:
        force_new = config.FORCE_NEW_ANNOTATIONS
    
    # Check for existing annotations if caching is enabled
    if config.ANNOTATION_CACHE:
        annotation_file = config.ANNOTATIONS_DIR / f'{dataset_name}_splits.json'
        
        if annotation_file.exists() and not force_new:
            logger.info(f"Loading cached {dataset_name} annotations...")
            with open(annotation_file, 'r') as f:
                splits = json.load(f)
                return pd.DataFrame(splits['train']), pd.DataFrame(splits['val']), pd.DataFrame(splits['test'])
    
    logger.info(f"Creating new annotations for {dataset_name}")
    logger.info(f"Starting data loading process for {dataset_name} dataset")
    logger.info(f"Looking for data in {config.DATA_DIR}")
    logger.info(f"Using {config.DATASET_FRACTION * 100}% of total data")
    
    all_data = []
    
    if dataset_name == 'ff++':
        # Process FF++ dataset
        # Real images
        for real_source, real_path in config.DATASET_STRUCTURE['ff++']['real_dirs'].items():
            real_dir = config.DATA_DIR / real_path
            logger.debug(f"Looking for FF++ real images in: {real_dir}")
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
            logger.info(f"Found {len(real_paths)} real FF++ images in {real_source}")
            
            all_data.extend([{
                'image_path': str(p),
                'label': 0,
                'subset': f'real_{real_source}',
                'dataset': 'ff++'
            } for p in real_paths[:int(len(real_paths) * config.DATASET_FRACTION)]])
        
        # Fake images
        for fake_type, fake_path in config.DATASET_STRUCTURE['ff++']['fake_dirs'].items():
            fake_dir = config.DATA_DIR / fake_path
            logger.debug(f"Looking for {fake_type} fake images in: {fake_dir}")
            fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
            logger.info(f"Found {len(fake_paths)} fake images in {fake_type}")
            
            all_data.extend([{
                'image_path': str(p),
                'label': 1,
                'subset': fake_type,
                'dataset': 'ff++'
            } for p in fake_paths[:int(len(fake_paths) * config.DATASET_FRACTION)]])
            
    else:  # celebdf
        # Real images
        real_dir = config.DATA_DIR / config.DATASET_STRUCTURE['celebdf']['real_dir']
        real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        logger.info(f"Found {len(real_paths)} real CelebDF images")
        
        all_data.extend([{
            'image_path': str(p),
            'label': 0,
            'dataset': 'celebdf'
        } for p in real_paths[:int(len(real_paths) * config.DATASET_FRACTION)]])
        
        # Fake images
        fake_dir = config.DATA_DIR / config.DATASET_STRUCTURE['celebdf']['fake_dir']
        fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
        logger.info(f"Found {len(fake_paths)} fake CelebDF images")
        
        all_data.extend([{
            'image_path': str(p),
            'label': 1,
            'dataset': 'celebdf'
        } for p in fake_paths[:int(len(fake_paths) * config.DATASET_FRACTION)]])
    
    if not all_data:
        raise ValueError(f"No images found for {dataset_name}")
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(all_data)
    
    # 使用 stratified splits 確保類別比例保持一致
    # 首先分出測試集
    train_val_df, test_df = train_test_split(
        df, 
        test_size=1-config.TRAIN_SPLIT-config.VAL_SPLIT,
        random_state=42,
        stratify=df['label']
    )
    
    # 然後從剩餘數據中分出驗證集
    val_size = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=42,
        stratify=train_val_df['label']
    )
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Real samples: {len(df[df['label'] == 0])}")
    logger.info(f"Fake samples: {len(df[df['label'] == 1])}")
    if 'subset' in df.columns:
        for subset in df['subset'].unique():
            logger.info(f"Samples in {subset}: {len(df[df['subset'] == subset])}")
    
    logger.info(f"Data Split Details:")
    logger.info(f"Training set: {len(train_df)} samples ({len(train_df[train_df['label'] == 0])} real, {len(train_df[train_df['label'] == 1])} fake)")
    logger.info(f"Validation set: {len(val_df)} samples ({len(val_df[val_df['label'] == 0])} real, {len(val_df[val_df['label'] == 1])} fake)")
    logger.info(f"Test set: {len(test_df)} samples ({len(test_df[test_df['label'] == 0])} real, {len(test_df[test_df['label'] == 1])} fake)")
    
    # Save annotations if caching is enabled
    if config.ANNOTATION_CACHE:
        splits = {
            'train': train_df.to_dict('records'),
            'val': val_df.to_dict('records'),
            'test': test_df.to_dict('records')
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(splits, f, indent=4)
        logger.info(f"Cached annotations to {annotation_file}")
    
    return train_df, val_df, test_df 