#!/usr/bin/env python3
"""
Test Script for calibrate_model.py
==================================

This script tests the Stage 1 probability calibration pipeline with a trained model
to verify functionality before running full calibration.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_test_model_availability():
    """Check if test model from train_stage1.py test is available"""
    print("=== Checking Test Model Availability ===")
    
    test_model_path = Path(__file__).parent / 'output' / 'stage1_test' / 'best_model.pth'
    
    if test_model_path.exists():
        print(f"✅ Test model found: {test_model_path}")
        
        # Check model file
        try:
            checkpoint = torch.load(test_model_path, map_location='cpu')
            print(f"✅ Model checkpoint loaded successfully")
            print(f"   Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
            if 'best_val_auc' in checkpoint:
                print(f"   Best validation AUC: {checkpoint['best_val_auc']:.4f}")
            return str(test_model_path)
        except Exception as e:
            print(f"❌ Error loading model checkpoint: {e}")
            return None
    else:
        print(f"❌ Test model not found: {test_model_path}")
        print("   Please run 'python test/test_train_stage1.py' first to create a test model")
        return None

def check_validation_data():
    """Check if validation manifest and data are available"""
    print("\n=== Checking Validation Data ===")
    
    base_dir = Path(__file__).parent.parent
    val_manifest = base_dir / 'processed_data' / 'manifests' / 'val_manifest.csv'
    data_dir = base_dir / 'processed_data'
    
    if not val_manifest.exists():
        print(f"❌ Validation manifest not found: {val_manifest}")
        return None, None
    
    print(f"✅ Validation manifest found: {val_manifest}")
    
    # Check first few data paths
    try:
        with open(val_manifest, 'r') as f:
            lines = f.readlines()
        
        print(f"   Total validation samples: {len(lines) - 1}")  # -1 for header
        
        # Check a few sample paths
        sample_count = 0
        for i, line in enumerate(lines[1:6]):  # Check first 5 samples
            parts = line.strip().split(',')
            if len(parts) >= 1:
                img_path = data_dir / parts[0]
                if img_path.exists():
                    sample_count += 1
        
        print(f"   Sample accessibility: {sample_count}/5 checked files exist")
        
        if sample_count < 3:
            print("⚠️  Warning: Many validation files may be missing")
        
        return str(val_manifest), str(data_dir)
        
    except Exception as e:
        print(f"❌ Error reading validation manifest: {e}")
        return None, None

def run_calibration_test(model_path, val_manifest, data_dir, batch_size=8):
    """
    Run the actual calibration test
    
    Args:
        model_path (str): Path to test model
        val_manifest (str): Path to validation manifest
        data_dir (str): Root data directory
        batch_size (int): Batch size for testing
    
    Returns:
        bool: True if calibration completed successfully
    """
    print(f"\n=== Running Calibration Test ===")
    print(f"Model: {model_path}")
    print(f"Validation manifest: {val_manifest}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    
    # Create test output directory
    test_output_dir = Path(__file__).parent / 'output' / 'calibration_test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    calibrate_script = Path(__file__).parent.parent / 'src' / 'stage1' / 'calibrate_model.py'
    
    cmd = [
        'python', str(calibrate_script),
        '--model_path', model_path,
        '--val_manifest', val_manifest,
        '--data_dir', data_dir,
        '--batch_size', str(batch_size),
        '--output_dir', str(test_output_dir),
        '--num_workers', '2',  # Reduce workers for testing
        '--method', 'nll',     # Use NLL method for testing
        '--temp_bounds', '0.1', '5.0'  # Smaller range for testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {test_output_dir}")
    
    # Run calibration
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        calibration_time = time.time() - start_time
        print(f"\nCalibration completed in {calibration_time:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ Calibration test PASSED")
            print("\nStdout (last 15 lines):")
            print('\n'.join(result.stdout.split('\n')[-15:]))
            
            # Check outputs
            print(f"\nChecking outputs in {test_output_dir}:")
            expected_files = [
                'calibration_temp.json',
                'calibration_summary.json', 
                'reliability_diagram.png',
                'probability_distributions.png'
            ]
            
            all_files_exist = True
            for expected_file in expected_files:
                file_path = test_output_dir / expected_file
                if file_path.exists():
                    print(f"  ✅ {expected_file}")
                    
                    # Show some content for JSON files
                    if expected_file.endswith('.json'):
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                if expected_file == 'calibration_temp.json':
                                    temp = data.get('temperature', 'unknown')
                                    ece_before = data.get('ece_before', 'unknown')
                                    ece_after = data.get('ece_after', 'unknown')
                                    print(f"      Temperature: {temp}")
                                    print(f"      ECE: {ece_before} → {ece_after}")
                        except Exception as e:
                            print(f"      ⚠️  Could not read content: {e}")
                else:
                    print(f"  ❌ {expected_file} - MISSING")
                    all_files_exist = False
            
            return all_files_exist
            
        else:
            print("❌ Calibration test FAILED")
            print(f"Return code: {result.returncode}")
            print("\nStderr:")
            print(result.stderr)
            print("\nStdout:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Calibration test TIMEOUT (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Calibration test ERROR: {e}")
        return False

def check_dependencies():
    """Check if calibration-specific dependencies are available"""
    print("=== Checking Calibration Dependencies ===")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'pandas', 'numpy', 
        'scipy', 'matplotlib', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    # Check specific scipy.optimize
    try:
        from scipy.optimize import minimize
        print("✅ scipy.optimize.minimize")
    except ImportError:
        print("❌ scipy.optimize.minimize - MISSING")
        missing_packages.append('scipy.optimize')
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing dependencies:")
        print("pip install scipy scikit-learn matplotlib")
        return False
    
    print("All calibration dependencies available!")
    return True

def verify_calibration_results(output_dir):
    """Verify the quality of calibration results"""
    print("\n=== Verifying Calibration Results ===")
    
    calibration_summary_path = output_dir / 'calibration_summary.json'
    
    if not calibration_summary_path.exists():
        print("❌ Calibration summary not found")
        return False
    
    try:
        with open(calibration_summary_path, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        cal_summary = results.get('calibration_summary', {})
        cal_quality = results.get('calibration_quality', {})
        
        temperature = cal_summary.get('optimal_temperature', 'unknown')
        ece_improvement = cal_summary.get('ece_improvement_percent', 0)
        ece_before = cal_quality.get('ece_before', 'unknown')
        ece_after = cal_quality.get('ece_after', 'unknown')
        
        print(f"Optimal Temperature: {temperature}")
        print(f"ECE Before: {ece_before}")
        print(f"ECE After: {ece_after}")
        print(f"ECE Improvement: {ece_improvement:.1f}%")
        
        # Basic sanity checks
        success = True
        
        if isinstance(temperature, float) and (temperature < 0.1 or temperature > 10.0):
            print("⚠️  Warning: Temperature outside expected range")
            success = False
        
        if ece_improvement < 0:
            print("⚠️  Warning: ECE got worse after calibration")
        elif ece_improvement > 5:
            print("✅ Good ECE improvement achieved")
        
        return success
        
    except Exception as e:
        print(f"❌ Error reading calibration results: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Stage 1 Calibration Script Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        return False
    
    # Check if test model is available
    model_path = check_test_model_availability()
    if not model_path:
        print("\n💡 To create a test model, run:")
        print("   python test/test_train_stage1.py")
        return False
    
    # Check validation data
    val_manifest, data_dir = check_validation_data()
    if not val_manifest or not data_dir:
        return False
    
    # Run calibration test
    success = run_calibration_test(
        model_path, 
        val_manifest, 
        data_dir,
        batch_size=8  # Small batch for testing
    )
    
    if success:
        # Verify results quality
        test_output_dir = Path(__file__).parent / 'output' / 'calibration_test'
        results_ok = verify_calibration_results(test_output_dir)
        
        if results_ok:
            print("\n🎉 All calibration tests PASSED!")
            print("calibrate_model.py is ready for full calibration.")
            
            # Provide next steps
            print("\n📋 Next Steps:")
            print("1. For full calibration after training, use:")
            print("   python src/stage1/calibrate_model.py --model_path output/stage1/best_model.pth")
            print("2. Check calibration results in: output/stage1/")
            print("3. Temperature value will be used in evaluation and cascade")
            
        else:
            print("\n⚠️  Calibration completed but results may need review")
            
    else:
        print("\n💥 Calibration tests FAILED!")
        print("Please check the error messages above and fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)