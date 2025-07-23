#!/usr/bin/env python3
"""
Manifest Generation Script for AWARE-NET
Regenerates all manifest files based on processed data structure

This script scans the processed_data directory and creates CSV manifest files
for training, validation, and test sets according to the project structure.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scan_directory(directory_path, label):
    """
    Scan a directory for images and return file paths with labels
    
    Args:
        directory_path (Path): Path to the directory to scan
        label (int): Label for the images (0 for real, 1 for fake)
    
    Returns:
        list: List of tuples (image_path, label, dataset, split)
    """
    files = []
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory_path}")
        return files
    
    # Get image files (PNG format as specified in project docs)
    image_files = list(directory_path.glob("*.png"))
    
    for img_path in image_files:
        # Extract dataset and split info from path structure
        # Path structure: processed_data/{split}/{dataset}/{real|fake}/image.png
        path_parts = img_path.parts
        if len(path_parts) >= 4:
            split = path_parts[-4]  # train/val/final_test_sets
            dataset = path_parts[-3]  # celebdf_v2/df40/dfdc/ffpp
            
            files.append((
                str(img_path.relative_to(Path("/workspace/AWARE-NET/processed_data"))), 
                label, 
                dataset, 
                split
            ))
    
    return files

def generate_manifest_for_split(processed_data_path, split_name):
    """
    Generate manifest for a specific split (train/val)
    
    Args:
        processed_data_path (Path): Path to processed_data directory
        split_name (str): Name of the split (train/val)
    
    Returns:
        pd.DataFrame: Manifest dataframe
    """
    logger.info(f"Generating manifest for {split_name} split...")
    
    all_files = []
    split_path = processed_data_path / split_name
    
    if not split_path.exists():
        logger.error(f"Split directory does not exist: {split_path}")
        return pd.DataFrame()
    
    # Scan all datasets in this split
    datasets = [d for d in split_path.iterdir() if d.is_dir()]
    
    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        logger.info(f"  Processing dataset: {dataset_name}")
        
        # Scan real images (label = 0)
        real_dir = dataset_dir / "real"
        if real_dir.exists():
            real_files = scan_directory(real_dir, 0)
            all_files.extend(real_files)
            logger.info(f"    Found {len(real_files)} real images")
        
        # Scan fake images (label = 1)
        fake_dir = dataset_dir / "fake"
        if fake_dir.exists():
            fake_files = scan_directory(fake_dir, 1)
            all_files.extend(fake_files)
            logger.info(f"    Found {len(fake_files)} fake images")
    
    # Create DataFrame
    df = pd.DataFrame(all_files, columns=['image_path', 'label', 'dataset', 'split'])
    
    logger.info(f"Total files in {split_name}: {len(df)}")
    logger.info(f"  Real images: {len(df[df['label'] == 0])}")
    logger.info(f"  Fake images: {len(df[df['label'] == 1])}")
    
    return df

def generate_test_manifests(processed_data_path):
    """
    Generate manifest files for all test sets
    
    Args:
        processed_data_path (Path): Path to processed_data directory
    
    Returns:
        dict: Dictionary of test manifests {test_name: dataframe}
    """
    logger.info("Generating test set manifests...")
    
    test_manifests = {}
    test_sets_path = processed_data_path / "final_test_sets"
    
    if not test_sets_path.exists():
        logger.error(f"Test sets directory does not exist: {test_sets_path}")
        return test_manifests
    
    # Scan each test dataset
    test_datasets = [d for d in test_sets_path.iterdir() if d.is_dir()]
    
    for dataset_dir in test_datasets:
        dataset_name = dataset_dir.name
        logger.info(f"  Processing test dataset: {dataset_name}")
        
        all_files = []
        
        # Scan real images (label = 0)
        real_dir = dataset_dir / "real"
        if real_dir.exists():
            real_files = scan_directory(real_dir, 0)
            all_files.extend(real_files)
            logger.info(f"    Found {len(real_files)} real images")
        
        # Scan fake images (label = 1)
        fake_dir = dataset_dir / "fake"
        if fake_dir.exists():
            fake_files = scan_directory(fake_dir, 1)
            all_files.extend(fake_files)
            logger.info(f"    Found {len(fake_files)} fake images")
        
        # Create DataFrame for this test set
        if all_files:
            df = pd.DataFrame(all_files, columns=['image_path', 'label', 'dataset', 'split'])
            test_manifests[f"test_{dataset_name}"] = df
            
            logger.info(f"Test set {dataset_name}: {len(df)} total files")
            logger.info(f"  Real images: {len(df[df['label'] == 0])}")
            logger.info(f"  Fake images: {len(df[df['label'] == 1])}")
    
    return test_manifests

def save_manifest(df, output_path, manifest_name):
    """
    Save manifest DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): Manifest dataframe
        output_path (Path): Output directory path
        manifest_name (str): Name of the manifest file
    """
    if df.empty:
        logger.warning(f"Empty manifest for {manifest_name}, skipping save")
        return
    
    output_file = output_path / f"{manifest_name}_manifest.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {manifest_name} manifest: {output_file} ({len(df)} entries)")

def main():
    """Main function to regenerate all manifest files"""
    logger.info("=== AWARE-NET Manifest Regeneration ===")
    
    # Define paths
    processed_data_path = Path("/workspace/AWARE-NET/processed_data")
    manifests_output_path = processed_data_path / "manifests"
    
    # Create manifests directory if it doesn't exist
    manifests_output_path.mkdir(exist_ok=True)
    
    # Check if processed_data exists
    if not processed_data_path.exists():
        logger.error(f"Processed data directory does not exist: {processed_data_path}")
        return
    
    logger.info(f"Processing data from: {processed_data_path}")
    logger.info(f"Output manifests to: {manifests_output_path}")
    
    # Generate training manifest
    train_df = generate_manifest_for_split(processed_data_path, "train")
    if not train_df.empty:
        save_manifest(train_df, manifests_output_path, "train")
    
    # Generate validation manifest
    val_df = generate_manifest_for_split(processed_data_path, "val")
    if not val_df.empty:
        save_manifest(val_df, manifests_output_path, "val")
    
    # Generate test manifests
    test_manifests = generate_test_manifests(processed_data_path)
    for test_name, test_df in test_manifests.items():
        save_manifest(test_df, manifests_output_path, test_name)
    
    # Generate summary statistics
    logger.info("\n=== MANIFEST GENERATION SUMMARY ===")
    
    total_train = len(train_df) if not train_df.empty else 0
    total_val = len(val_df) if not val_df.empty else 0
    total_test = sum(len(df) for df in test_manifests.values())
    
    logger.info(f"Training samples: {total_train}")
    logger.info(f"Validation samples: {total_val}")
    logger.info(f"Test samples: {total_test}")
    logger.info(f"Total samples: {total_train + total_val + total_test}")
    
    # Dataset breakdown
    if not train_df.empty:
        logger.info(f"\nTraining set by dataset:")
        for dataset in train_df['dataset'].unique():
            count = len(train_df[train_df['dataset'] == dataset])
            real_count = len(train_df[(train_df['dataset'] == dataset) & (train_df['label'] == 0)])
            fake_count = len(train_df[(train_df['dataset'] == dataset) & (train_df['label'] == 1)])
            logger.info(f"  {dataset}: {count} total ({real_count} real, {fake_count} fake)")
    
    if not val_df.empty:
        logger.info(f"\nValidation set by dataset:")
        for dataset in val_df['dataset'].unique():
            count = len(val_df[val_df['dataset'] == dataset])
            real_count = len(val_df[(val_df['dataset'] == dataset) & (val_df['label'] == 0)])
            fake_count = len(val_df[(val_df['dataset'] == dataset) & (val_df['label'] == 1)])
            logger.info(f"  {dataset}: {count} total ({real_count} real, {fake_count} fake)")
    
    logger.info(f"\nTest sets:")
    for test_name, test_df in test_manifests.items():
        dataset = test_df['dataset'].iloc[0] if not test_df.empty else 'unknown'
        real_count = len(test_df[test_df['label'] == 0])
        fake_count = len(test_df[test_df['label'] == 1])
        logger.info(f"  {test_name}: {len(test_df)} total ({real_count} real, {fake_count} fake)")
    
    logger.info(f"\nManifest files saved to: {manifests_output_path}")
    logger.info("=== MANIFEST GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()