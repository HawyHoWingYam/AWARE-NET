#!/usr/bin/env python3
"""
Test Script for train_stage1.py
===============================

This script tests the Stage 1 training pipeline with a small subset of data
to verify functionality before running full training.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from pathlib import Path
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def create_test_manifests(original_train_manifest, original_val_manifest, 
                         test_output_dir, samples_per_class=100):
    """
    Create small test manifests with limited samples per class
    
    Args:
        original_train_manifest (str): Path to original training manifest
        original_val_manifest (str): Path to original validation manifest
        test_output_dir (Path): Output directory for test manifests
        samples_per_class (int): Number of samples per class (real/fake)
    """
    print(f"Creating test manifests with {samples_per_class} samples per class...")
    
    # Load original manifests
    train_df = pd.read_csv(original_train_manifest)
    val_df = pd.read_csv(original_val_manifest)
    
    print(f"Original train samples: {len(train_df)}")
    print(f"Original val samples: {len(val_df)}")
    
    # Create balanced test sets
    def create_balanced_subset(df, samples_per_class):
        """Create balanced subset with equal real/fake samples"""
        real_samples = df[df['label'] == 0].head(samples_per_class)
        fake_samples = df[df['label'] == 1].head(samples_per_class)
        return pd.concat([real_samples, fake_samples]).sample(frac=1).reset_index(drop=True)
    
    # Create test subsets
    test_train_df = create_balanced_subset(train_df, samples_per_class)
    test_val_df = create_balanced_subset(val_df, samples_per_class // 2)  # Smaller validation set
    
    print(f"Test train samples: {len(test_train_df)} (Real: {len(test_train_df[test_train_df['label']==0])}, Fake: {len(test_train_df[test_train_df['label']==1])})")
    print(f"Test val samples: {len(test_val_df)} (Real: {len(test_val_df[test_val_df['label']==0])}, Fake: {len(test_val_df[test_val_df['label']==1])})")
    
    # Save test manifests
    test_train_path = test_output_dir / 'test_train_manifest.csv'
    test_val_path = test_output_dir / 'test_val_manifest.csv'
    
    test_train_df.to_csv(test_train_path, index=False)
    test_val_df.to_csv(test_val_path, index=False)
    
    print(f"Test manifests saved:")
    print(f"  Train: {test_train_path}")
    print(f"  Val: {test_val_path}")
    
    return str(test_train_path), str(test_val_path)

def verify_data_accessibility(manifest_path, data_root, max_check=10):
    """
    Verify that image files in manifest are accessible
    
    Args:
        manifest_path (str): Path to manifest file
        data_root (str): Root directory of data
        max_check (int): Maximum number of files to check
    
    Returns:
        bool: True if all checked files exist
    """
    print(f"Verifying data accessibility for {manifest_path}...")
    
    df = pd.read_csv(manifest_path)
    data_root = Path(data_root)
    
    accessible_count = 0
    check_count = min(len(df), max_check)
    
    for i in range(check_count):
        img_path = data_root / df.iloc[i]['image_path']
        if img_path.exists():
            accessible_count += 1
        else:
            print(f"  Missing: {img_path}")
    
    success_rate = accessible_count / check_count
    print(f"  Accessibility: {accessible_count}/{check_count} ({success_rate:.1%})")
    
    return success_rate > 0.8  # Allow some missing files

def run_training_test(train_manifest, val_manifest, data_dir, epochs=2, batch_size=8):
    """
    Run the actual training test
    
    Args:
        train_manifest (str): Path to test training manifest
        val_manifest (str): Path to test validation manifest
        data_dir (str): Root data directory
        epochs (int): Number of epochs for testing
        batch_size (int): Batch size for testing
    
    Returns:
        bool: True if training completed successfully
    """
    print(f"\n=== Running Training Test ===")
    print(f"Train manifest: {train_manifest}")
    print(f"Val manifest: {val_manifest}")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Create test output directory
    test_output_dir = Path(__file__).parent / 'output' / 'stage1_test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    train_script = Path(__file__).parent.parent / 'src' / 'stage1' / 'train_stage1.py'
    
    cmd = [
        'python', str(train_script),
        '--train_manifest', train_manifest,
        '--val_manifest', val_manifest,
        '--data_dir', data_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', '1e-4',
        '--output_dir', str(test_output_dir),
        '--num_workers', '2',  # Reduce workers for testing
        '--save_freq', '1'     # Save every epoch for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {test_output_dir}")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ Training test PASSED")
            print("\nStdout (last 20 lines):")
            print('\n'.join(result.stdout.split('\n')[-20:]))
            
            # Check outputs
            print(f"\nChecking outputs in {test_output_dir}:")
            if test_output_dir.exists():
                for file_path in test_output_dir.iterdir():
                    print(f"  📁 {file_path.name}")
            
            return True
        else:
            print("❌ Training test FAILED")
            print(f"Return code: {result.returncode}")
            print("\nStderr:")
            print(result.stderr)
            print("\nStdout:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training test TIMEOUT (>30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training test ERROR: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("=== Checking Dependencies ===")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'pandas', 'numpy', 
        'PIL', 'sklearn', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing dependencies before running tests.")
        return False
    
    print("All dependencies available!")
    return True

def main():
    """Main test function"""
    print("🧪 Stage 1 Training Script Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        return False
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'processed_data'
    manifests_dir = data_dir / 'manifests'
    test_dir = Path(__file__).parent
    
    original_train_manifest = manifests_dir / 'train_manifest.csv'
    original_val_manifest = manifests_dir / 'val_manifest.csv'
    
    # Check if original manifests exist
    if not original_train_manifest.exists():
        print(f"❌ Training manifest not found: {original_train_manifest}")
        return False
    
    if not original_val_manifest.exists():
        print(f"❌ Validation manifest not found: {original_val_manifest}")
        return False
    
    print(f"✅ Found original manifests")
    
    # Create test output directory
    test_output_dir = test_dir / 'manifests'
    test_output_dir.mkdir(exist_ok=True)
    
    # Create test manifests (small subset)
    try:
        test_train_manifest, test_val_manifest = create_test_manifests(
            original_train_manifest, 
            original_val_manifest,
            test_output_dir,
            samples_per_class=50  # Small test set
        )
    except Exception as e:
        print(f"❌ Failed to create test manifests: {e}")
        return False
    
    # Verify data accessibility
    if not verify_data_accessibility(test_train_manifest, data_dir):
        print("❌ Training data not accessible")
        return False
    
    if not verify_data_accessibility(test_val_manifest, data_dir):
        print("❌ Validation data not accessible")
        return False
    
    print("✅ Data accessibility verified")
    
    # Run training test
    success = run_training_test(
        test_train_manifest, 
        test_val_manifest, 
        str(data_dir),
        epochs=2,      # Quick test
        batch_size=4   # Small batch for testing
    )
    
    if success:
        print("\n🎉 All tests PASSED!")
        print("train_stage1.py is ready for full training.")
        
        # Provide next steps
        print("\n📋 Next Steps:")
        print("1. For full training, use:")
        print(f"   python src/stage1/train_stage1.py")
        print("2. Monitor output in: output/stage1/")
        print("3. Adjust hyperparameters as needed")
        
    else:
        print("\n💥 Tests FAILED!")
        print("Please check the error messages above and fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)