#!/usr/bin/env python3
"""
Enhanced Manifest Generation Script for AWARE-NET
=================================================

Regenerates all manifest files based on processed data structure with improved
error handling, validation, and flexible configuration options.

This script scans the processed_data directory and creates CSV manifest files
for training, validation, and test sets according to the project structure.

Features:
- Comprehensive data validation
- Flexible path configuration
- Progress tracking with tqdm
- Detailed statistics and reporting
- Backup of existing manifests
- Support for multiple image formats

Usage:
    python scripts/regenerate_manifests.py
    python scripts/regenerate_manifests.py --data-dir /custom/path --output-dir /custom/manifests
    python scripts/regenerate_manifests.py --backup --validate --verbose
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import shutil
from datetime import datetime
import json

def setup_logging(verbose=False):
    """Setup enhanced logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'manifest_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def scan_directory(directory_path, label, base_path, supported_formats=None, validate_files=False):
    """
    Enhanced directory scanning with validation and multiple format support
    
    Args:
        directory_path (Path): Path to the directory to scan
        label (int): Label for the images (0 for real, 1 for fake)
        base_path (Path): Base path for relative path calculation
        supported_formats (list): List of supported image formats
        validate_files (bool): Whether to validate file accessibility
    
    Returns:
        tuple: (files_list, validation_stats)
    """
    if supported_formats is None:
        supported_formats = ['.png', '.jpg', '.jpeg']
    
    files = []
    validation_stats = {'total': 0, 'valid': 0, 'invalid': 0, 'errors': []}
    
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory_path}")
        return files, validation_stats
    
    # Get image files with multiple format support
    image_files = []
    for format_ext in supported_formats:
        image_files.extend(directory_path.glob(f"*{format_ext}"))
    
    logger.debug(f"Found {len(image_files)} image files in {directory_path}")
    
    for img_path in tqdm(image_files, desc=f"Scanning {directory_path.name}", leave=False):
        validation_stats['total'] += 1
        
        try:
            # Extract dataset and split info from path structure
            path_parts = img_path.parts
            if len(path_parts) >= 4:
                split = path_parts[-4]  # train/val/final_test_sets
                dataset = path_parts[-3]  # celebdf_v2/df40/dfdc/ffpp
                
                # Calculate relative path
                relative_path = str(img_path.relative_to(base_path))
                
                # Validate file if requested
                if validate_files:
                    if not img_path.exists() or img_path.stat().st_size == 0:
                        validation_stats['invalid'] += 1
                        validation_stats['errors'].append(f"Invalid file: {relative_path}")
                        continue
                
                files.append((relative_path, label, dataset, split))
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
                validation_stats['errors'].append(f"Invalid path structure: {img_path}")
                
        except Exception as e:
            validation_stats['invalid'] += 1
            validation_stats['errors'].append(f"Error processing {img_path}: {str(e)}")
            logger.debug(f"Error processing {img_path}: {e}")
    
    return files, validation_stats

def generate_manifest_for_split(processed_data_path, split_name, validate_files=False, supported_formats=None):
    """
    Enhanced manifest generation for a specific split with validation
    
    Args:
        processed_data_path (Path): Path to processed_data directory
        split_name (str): Name of the split (train/val)
        validate_files (bool): Whether to validate file accessibility
        supported_formats (list): List of supported image formats
    
    Returns:
        tuple: (pd.DataFrame, validation_summary)
    """
    logger.info(f"Generating manifest for {split_name} split...")
    
    all_files = []
    validation_summary = {'datasets': {}, 'total_errors': []}
    split_path = processed_data_path / split_name
    
    if not split_path.exists():
        logger.error(f"Split directory does not exist: {split_path}")
        return pd.DataFrame(), validation_summary
    
    # Scan all datasets in this split
    datasets = [d for d in split_path.iterdir() if d.is_dir()]
    
    if not datasets:
        logger.warning(f"No datasets found in {split_path}")
        return pd.DataFrame(), validation_summary
    
    logger.info(f"Found {len(datasets)} datasets: {[d.name for d in datasets]}")
    
    for dataset_dir in tqdm(datasets, desc=f"Processing {split_name} datasets"):
        dataset_name = dataset_dir.name
        logger.info(f"  Processing dataset: {dataset_name}")
        
        dataset_stats = {'real': {}, 'fake': {}, 'total_files': 0, 'valid_files': 0}
        
        # Scan real images (label = 0)
        real_dir = dataset_dir / "real"
        if real_dir.exists():
            real_files, real_stats = scan_directory(
                real_dir, 0, processed_data_path, supported_formats, validate_files
            )
            all_files.extend(real_files)
            dataset_stats['real'] = real_stats
            dataset_stats['total_files'] += real_stats['total']
            dataset_stats['valid_files'] += real_stats['valid']
            
            logger.info(f"    Real images: {real_stats['valid']}/{real_stats['total']} valid")
            if real_stats['errors']:
                validation_summary['total_errors'].extend(real_stats['errors'])
        else:
            logger.warning(f"    Real directory not found: {real_dir}")
        
        # Scan fake images (label = 1)
        fake_dir = dataset_dir / "fake"
        if fake_dir.exists():
            fake_files, fake_stats = scan_directory(
                fake_dir, 1, processed_data_path, supported_formats, validate_files
            )
            all_files.extend(fake_files)
            dataset_stats['fake'] = fake_stats
            dataset_stats['total_files'] += fake_stats['total']
            dataset_stats['valid_files'] += fake_stats['valid']
            
            logger.info(f"    Fake images: {fake_stats['valid']}/{fake_stats['total']} valid")
            if fake_stats['errors']:
                validation_summary['total_errors'].extend(fake_stats['errors'])
        else:
            logger.warning(f"    Fake directory not found: {fake_dir}")
        
        validation_summary['datasets'][dataset_name] = dataset_stats
        
        total_real = dataset_stats['real'].get('valid', 0)
        total_fake = dataset_stats['fake'].get('valid', 0)
        logger.info(f"    Dataset {dataset_name} total: {total_real + total_fake} files ({total_real} real, {total_fake} fake)")
    
    # Create DataFrame
    df = pd.DataFrame(all_files, columns=['image_path', 'label', 'dataset', 'split'])
    
    logger.info(f"Split {split_name} summary:")
    logger.info(f"  Total files: {len(df)}")
    logger.info(f"  Real images: {len(df[df['label'] == 0])}")
    logger.info(f"  Fake images: {len(df[df['label'] == 1])}")
    
    if validation_summary['total_errors']:
        logger.warning(f"  Total validation errors: {len(validation_summary['total_errors'])}")
    
    return df, validation_summary

def generate_test_manifests(processed_data_path, validate_files=False, supported_formats=None):
    """
    Enhanced test manifest generation with validation
    
    Args:
        processed_data_path (Path): Path to processed_data directory
        validate_files (bool): Whether to validate file accessibility
        supported_formats (list): List of supported image formats
    
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
    
    if not test_datasets:
        logger.warning(f"No test datasets found in {test_sets_path}")
        return test_manifests
    
    logger.info(f"Found {len(test_datasets)} test datasets: {[d.name for d in test_datasets]}")
    
    for dataset_dir in tqdm(test_datasets, desc="Processing test datasets"):
        dataset_name = dataset_dir.name
        logger.info(f"  Processing test dataset: {dataset_name}")
        
        all_files = []
        
        # Scan real images (label = 0)
        real_dir = dataset_dir / "real"
        if real_dir.exists():
            real_files, real_stats = scan_directory(
                real_dir, 0, processed_data_path, supported_formats, validate_files
            )
            all_files.extend(real_files)
            logger.info(f"    Real images: {real_stats['valid']}/{real_stats['total']} valid")
        else:
            logger.warning(f"    Real directory not found: {real_dir}")
        
        # Scan fake images (label = 1)
        fake_dir = dataset_dir / "fake"
        if fake_dir.exists():
            fake_files, fake_stats = scan_directory(
                fake_dir, 1, processed_data_path, supported_formats, validate_files
            )
            all_files.extend(fake_files)
            logger.info(f"    Fake images: {fake_stats['valid']}/{fake_stats['total']} valid")
        else:
            logger.warning(f"    Fake directory not found: {fake_dir}")
        
        # Create DataFrame for this test set
        if all_files:
            df = pd.DataFrame(all_files, columns=['image_path', 'label', 'dataset', 'split'])
            test_manifests[f"test_{dataset_name}"] = df
            
            real_count = len(df[df['label'] == 0])
            fake_count = len(df[df['label'] == 1])
            logger.info(f"    Test dataset {dataset_name}: {len(df)} total ({real_count} real, {fake_count} fake)")
        else:
            logger.warning(f"    No valid files found for test dataset: {dataset_name}")
    
    logger.info(f"Generated {len(test_manifests)} test manifests")
    return test_manifests

def backup_existing_manifests(manifests_path):
    """
    Create backup of existing manifest files
    
    Args:
        manifests_path (Path): Path to manifests directory
    
    Returns:
        Path: Backup directory path
    """
    if not manifests_path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = manifests_path.parent / f"manifests_backup_{timestamp}"
    
    # Find existing manifest files
    manifest_files = list(manifests_path.glob("*_manifest.csv"))
    
    if manifest_files:
        backup_dir.mkdir(exist_ok=True)
        logger.info(f"Backing up {len(manifest_files)} existing manifest files to: {backup_dir}")
        
        for manifest_file in manifest_files:
            shutil.copy2(manifest_file, backup_dir / manifest_file.name)
            logger.debug(f"Backed up: {manifest_file.name}")
        
        return backup_dir
    else:
        logger.info("No existing manifest files found to backup")
        return None

def save_manifest(df, output_path, manifest_name, backup_dir=None):
    """
    Enhanced manifest saving with validation and backup
    
    Args:
        df (pd.DataFrame): Manifest dataframe
        output_path (Path): Output directory path
        manifest_name (str): Name of the manifest file
        backup_dir (Path): Backup directory for existing files
    """
    if df.empty:
        logger.warning(f"Empty manifest for {manifest_name}, skipping save")
        return
    
    output_file = output_path / f"{manifest_name}_manifest.csv"
    
    # Validate DataFrame structure
    expected_columns = ['image_path', 'label', 'dataset', 'split']
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Invalid manifest structure for {manifest_name}: missing columns")
        return
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['image_path']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate entries in {manifest_name} manifest")
        df = df.drop_duplicates(subset=['image_path'])
    
    # Save manifest
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {manifest_name} manifest: {output_file} ({len(df)} entries)")
        
        # Validate saved file
        test_df = pd.read_csv(output_file)
        if len(test_df) != len(df):
            logger.error(f"Validation failed: saved file has different length than source")
        else:
            logger.debug(f"Validation passed for {manifest_name} manifest")
            
    except Exception as e:
        logger.error(f"Failed to save {manifest_name} manifest: {e}")

def save_validation_report(validation_data, output_path):
    """
    Save detailed validation report as JSON
    
    Args:
        validation_data (dict): Validation statistics and errors
        output_path (Path): Output directory path
    """
    report_file = output_path / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(validation_data, f, indent=2, default=str)
        logger.info(f"Validation report saved: {report_file}")
    except Exception as e:
        logger.error(f"Failed to save validation report: {e}")

def analyze_manifest_statistics(df, manifest_name):
    """
    Analyze and log detailed statistics for a manifest
    
    Args:
        df (pd.DataFrame): Manifest dataframe
        manifest_name (str): Name of the manifest
    
    Returns:
        dict: Statistics dictionary
    """
    if df.empty:
        return {}
    
    stats = {
        'total_samples': len(df),
        'real_samples': len(df[df['label'] == 0]),
        'fake_samples': len(df[df['label'] == 1]),
        'datasets': {},
        'class_balance': 0.0
    }
    
    # Calculate class balance
    if stats['total_samples'] > 0:
        stats['class_balance'] = min(stats['real_samples'], stats['fake_samples']) / stats['total_samples']
    
    # Per-dataset statistics
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        stats['datasets'][dataset] = {
            'total': len(dataset_df),
            'real': len(dataset_df[dataset_df['label'] == 0]),
            'fake': len(dataset_df[dataset_df['label'] == 1])
        }
    
    logger.info(f"{manifest_name} statistics:")
    logger.info(f"  Total: {stats['total_samples']} samples")
    logger.info(f"  Real: {stats['real_samples']} ({stats['real_samples']/stats['total_samples']*100:.1f}%)")
    logger.info(f"  Fake: {stats['fake_samples']} ({stats['fake_samples']/stats['total_samples']*100:.1f}%)")
    logger.info(f"  Class balance: {stats['class_balance']:.3f}")
    
    return stats

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Manifest Generation Script for AWARE-NET',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (default paths)
    python scripts/regenerate_manifests.py
    
    # Custom paths with validation
    python scripts/regenerate_manifests.py --data-dir /custom/path --validate --backup
    
    # Verbose mode with specific formats
    python scripts/regenerate_manifests.py --verbose --formats png jpg jpeg
    
    # Dry run to check what would be processed
    python scripts/regenerate_manifests.py --dry-run --validate
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='processed_data',
                        help='Path to processed data directory (default: processed_data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for manifests (default: DATA_DIR/manifests)')
    parser.add_argument('--formats', nargs='+', default=['.png', '.jpg', '.jpeg'],
                        help='Supported image formats (default: .png .jpg .jpeg)')
    parser.add_argument('--backup', action='store_true',
                        help='Backup existing manifest files before regeneration')
    parser.add_argument('--validate', action='store_true',
                        help='Validate file accessibility during scanning')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging with debug information')
    parser.add_argument('--dry-run', action='store_true',
                        help='Perform dry run without saving manifests')
    
    return parser.parse_args()

def main():
    """Enhanced main function with argument parsing and improved features"""
    args = parse_arguments()
    
    # Initialize logger with appropriate level
    global logger
    logger = setup_logging(args.verbose)
    
    logger.info("=== Enhanced AWARE-NET Manifest Regeneration ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup paths
    processed_data_path = Path(args.data_dir).resolve()
    if args.output_dir:
        manifests_output_path = Path(args.output_dir).resolve()
    else:
        manifests_output_path = processed_data_path / "manifests"
    
    # Validate paths
    if not processed_data_path.exists():
        logger.error(f"Processed data directory does not exist: {processed_data_path}")
        return 1
    
    logger.info(f"Processing data from: {processed_data_path}")
    logger.info(f"Output manifests to: {manifests_output_path}")
    logger.info(f"Supported formats: {args.formats}")
    
    # Create manifests directory if it doesn't exist (unless dry run)
    if not args.dry_run:
        manifests_output_path.mkdir(parents=True, exist_ok=True)
    
    # Backup existing manifests if requested
    backup_dir = None
    if args.backup and not args.dry_run:
        backup_dir = backup_existing_manifests(manifests_output_path)
    
    # Initialize validation tracking
    all_validation_data = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'splits': {},
        'summary': {}
    }
    
    # Generate training manifest
    logger.info("\n" + "="*50)
    train_df, train_validation = generate_manifest_for_split(
        processed_data_path, "train", args.validate, args.formats
    )
    train_stats = analyze_manifest_statistics(train_df, "Training")
    all_validation_data['splits']['train'] = {
        'validation': train_validation,
        'statistics': train_stats
    }
    
    if not args.dry_run and not train_df.empty:
        save_manifest(train_df, manifests_output_path, "train", backup_dir)
    
    # Generate validation manifest
    logger.info("\n" + "="*50)
    val_df, val_validation = generate_manifest_for_split(
        processed_data_path, "val", args.validate, args.formats
    )
    val_stats = analyze_manifest_statistics(val_df, "Validation")
    all_validation_data['splits']['val'] = {
        'validation': val_validation,
        'statistics': val_stats
    }
    
    if not args.dry_run and not val_df.empty:
        save_manifest(val_df, manifests_output_path, "val", backup_dir)
    
    # Generate test manifests
    logger.info("\n" + "="*50)
    test_manifests = generate_test_manifests(processed_data_path, args.validate, args.formats)
    
    for test_name, test_df in test_manifests.items():
        test_stats = analyze_manifest_statistics(test_df, f"Test-{test_name}")
        all_validation_data['splits'][test_name] = {
            'statistics': test_stats
        }
        
        if not args.dry_run:
            save_manifest(test_df, manifests_output_path, test_name, backup_dir)
    
    # Generate comprehensive summary
    logger.info("\n" + "="*60)
    logger.info("=== COMPREHENSIVE MANIFEST GENERATION SUMMARY ===")
    
    total_train = len(train_df) if not train_df.empty else 0
    total_val = len(val_df) if not val_df.empty else 0
    total_test = sum(len(df) for df in test_manifests.values())
    total_all = total_train + total_val + total_test
    
    summary_stats = {
        'total_samples': total_all,
        'training_samples': total_train,
        'validation_samples': total_val,
        'test_samples': total_test,
        'manifests_generated': 0,
        'validation_errors': 0
    }
    
    logger.info(f"📊 Dataset Overview:")
    logger.info(f"   Training samples: {total_train:,}")
    logger.info(f"   Validation samples: {total_val:,}")
    logger.info(f"   Test samples: {total_test:,}")
    logger.info(f"   Total samples: {total_all:,}")
    
    # Count manifests that would be generated
    manifests_count = 0
    if not train_df.empty: manifests_count += 1
    if not val_df.empty: manifests_count += 1
    manifests_count += len(test_manifests)
    summary_stats['manifests_generated'] = manifests_count
    
    logger.info(f"📁 Manifests: {manifests_count} files {'would be ' if args.dry_run else ''}generated")
    
    # Validation summary
    if args.validate:
        total_errors = (len(train_validation.get('total_errors', [])) + 
                       len(val_validation.get('total_errors', [])))
        summary_stats['validation_errors'] = total_errors
        
        if total_errors > 0:
            logger.warning(f"⚠️  Validation: {total_errors} total errors found")
        else:
            logger.info(f"✅ Validation: No errors found")
    
    # Save validation report
    all_validation_data['summary'] = summary_stats
    if not args.dry_run and (args.validate or args.verbose):
        save_validation_report(all_validation_data, manifests_output_path)
    
    # Final status
    if args.dry_run:
        logger.info(f"\n🔍 DRY RUN COMPLETE - No files were modified")
    else:
        logger.info(f"\n✅ MANIFEST GENERATION COMPLETE")
        logger.info(f"   Output directory: {manifests_output_path}")
        if backup_dir:
            logger.info(f"   Backup directory: {backup_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())