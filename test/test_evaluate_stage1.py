#!/usr/bin/env python3
"""
Test Script for evaluate_stage1.py
==================================

This script tests the Stage 1 evaluation pipeline to verify functionality
before running full evaluation.
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

def check_calibration_file():
    """Check if calibration file from calibration test is available"""
    print("\n=== Checking Calibration File ===")
    
    cal_file_path = Path(__file__).parent / 'output' / 'calibration_test_final' / 'calibration_temp.json'
    
    if cal_file_path.exists():
        print(f"✅ Calibration file found: {cal_file_path}")
        try:
            with open(cal_file_path, 'r') as f:
                cal_data = json.load(f)
            temp = cal_data.get('temperature', 1.0)
            print(f"   Temperature: {temp}")
            return str(cal_file_path)
        except Exception as e:
            print(f"⚠️  Could not read calibration file: {e}")
            return None
    else:
        print(f"⚠️  Calibration file not found: {cal_file_path}")
        print("   Evaluation will proceed without calibration")
        return None

def check_test_data():
    """Check if test data is available"""
    print("\n=== Checking Test Data ===")
    
    base_dir = Path(__file__).parent.parent
    
    # Check for test manifest first
    test_manifest = base_dir / 'processed_data' / 'manifests' / 'test_manifest.csv'
    val_manifest = base_dir / 'processed_data' / 'manifests' / 'val_manifest.csv'
    data_dir = base_dir / 'processed_data'
    
    manifest_to_use = None
    
    if test_manifest.exists():
        print(f"✅ Test manifest found: {test_manifest}")
        manifest_to_use = str(test_manifest)
        manifest_type = "test"
    elif val_manifest.exists():
        print(f"⚠️  Test manifest not found, using validation manifest: {val_manifest}")
        manifest_to_use = str(val_manifest)
        manifest_type = "validation"
    else:
        print(f"❌ Neither test nor validation manifest found")
        return None, None, None
    
    # Check first few data paths
    try:
        with open(manifest_to_use, 'r') as f:
            lines = f.readlines()
        
        print(f"   Total {manifest_type} samples: {len(lines) - 1}")  # -1 for header
        
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
            print("⚠️  Warning: Many test files may be missing")
        
        return str(manifest_to_use), str(data_dir), manifest_type
        
    except Exception as e:
        print(f"❌ Error reading test manifest: {e}")
        return None, None, None

def run_evaluation_test(model_path, test_manifest, data_dir, calibration_file=None, batch_size=8):
    """
    Run the actual evaluation test
    
    Args:
        model_path (str): Path to test model
        test_manifest (str): Path to test manifest
        data_dir (str): Root data directory
        calibration_file (str): Path to calibration file (optional)
        batch_size (int): Batch size for testing
    
    Returns:
        bool: True if evaluation completed successfully
    """
    print(f"\n=== Running Evaluation Test ===")
    print(f"Model: {model_path}")
    print(f"Test manifest: {test_manifest}")
    print(f"Data directory: {data_dir}")
    print(f"Calibration file: {calibration_file or 'None (no calibration)'}")
    print(f"Batch size: {batch_size}")
    
    # Create test output directory
    test_output_dir = Path(__file__).parent / 'output' / 'evaluation_test'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct command
    evaluate_script = Path(__file__).parent.parent / 'src' / 'stage1' / 'evaluate_stage1.py'
    
    cmd = [
        'python', str(evaluate_script),
        '--model_path', model_path,
        '--test_manifest', test_manifest,
        '--data_dir', data_dir,
        '--batch_size', str(batch_size),
        '--output_dir', str(test_output_dir),
        '--num_workers', '2'  # Reduce workers for testing
    ]
    
    # Add calibration if available
    if calibration_file:
        cmd.extend(['--calibration_file', calibration_file, '--use_calibration'])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {test_output_dir}")
    
    # Run evaluation
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        evaluation_time = time.time() - start_time
        print(f"\nEvaluation completed in {evaluation_time:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ Evaluation test PASSED")
            print("\nStdout (last 15 lines):")
            print('\n'.join(result.stdout.split('\n')[-15:]))
            
            # Check outputs
            print(f"\nChecking outputs in {test_output_dir}:")
            expected_files = [
                'evaluation_results.json',
                'roc_curve.png',
                'threshold_analysis.png',
                'confusion_matrix.png',
                'probability_distribution.png',
                'cascade_analysis.png'
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
                                if expected_file == 'evaluation_results.json':
                                    auc = data['performance_metrics']['auc']
                                    f1 = data['performance_metrics']['f1']
                                    samples = data['dataset_info']['total_samples']
                                    print(f"      AUC: {auc:.4f}, F1: {f1:.4f}, Samples: {samples}")
                        except Exception as e:
                            print(f"      ⚠️  Could not read content: {e}")
                else:
                    print(f"  ❌ {expected_file} - MISSING")
                    all_files_exist = False
            
            return all_files_exist
            
        else:
            print("❌ Evaluation test FAILED")
            print(f"Return code: {result.returncode}")
            print("\nStderr:")
            print(result.stderr)
            print("\nStdout:")
            print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation test TIMEOUT (>10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Evaluation test ERROR: {e}")
        return False

def check_dependencies():
    """Check if evaluation-specific dependencies are available"""
    print("=== Checking Evaluation Dependencies ===")
    
    required_packages = [
        'torch', 'torchvision', 'timm', 'pandas', 'numpy', 
        'sklearn', 'matplotlib', 'seaborn'
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
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing dependencies:")
        print("pip install scikit-learn seaborn")
        return False
    
    print("All evaluation dependencies available!")
    return True

def verify_evaluation_results(output_dir):
    """Verify the quality of evaluation results"""
    print("\n=== Verifying Evaluation Results ===")
    
    results_path = output_dir / 'evaluation_results.json'
    
    if not results_path.exists():
        print("❌ Evaluation results not found")
        return False
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        metrics = results.get('performance_metrics', {})
        model_info = results.get('model_info', {})
        dataset_info = results.get('dataset_info', {})
        
        auc = metrics.get('auc', 0)
        f1 = metrics.get('f1', 0)
        accuracy = metrics.get('accuracy', 0)
        samples = dataset_info.get('total_samples', 0)
        
        print(f"Model: {model_info.get('model_name', 'unknown')}")
        print(f"Dataset: {samples} samples")
        print(f"AUC: {auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Basic sanity checks
        success = True
        
        if auc < 0.5:
            print("⚠️  Warning: AUC below random performance")
            success = False
        elif auc > 0.8:
            print("✅ Good AUC performance")
        
        if f1 < 0.3:
            print("⚠️  Warning: Low F1-Score")
            success = False
        elif f1 > 0.6:
            print("✅ Good F1-Score")
            
        if samples < 10:
            print("⚠️  Warning: Very small test set")
        
        return success
        
    except Exception as e:
        print(f"❌ Error reading evaluation results: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Stage 1 Evaluation Script Test")
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
    
    # Check calibration file
    calibration_file = check_calibration_file()
    
    # Check test data
    test_manifest, data_dir, manifest_type = check_test_data()
    if not test_manifest or not data_dir:
        return False
    
    # Run evaluation test
    success = run_evaluation_test(
        model_path, 
        test_manifest, 
        data_dir,
        calibration_file,
        batch_size=8  # Small batch for testing
    )
    
    if success:
        # Verify results quality
        test_output_dir = Path(__file__).parent / 'output' / 'evaluation_test'
        results_ok = verify_evaluation_results(test_output_dir)
        
        if results_ok:
            print("\n🎉 All evaluation tests PASSED!")
            print("evaluate_stage1.py is ready for full evaluation.")
            
            # Provide next steps
            print("\n📋 Next Steps:")
            print("1. For full evaluation after training, use:")
            print("   python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth")
            print("2. For calibrated evaluation:")
            print("   python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --use_calibration")
            print("3. Check evaluation results in: output/stage1/evaluation/")
            
        else:
            print("\n⚠️  Evaluation completed but results may need review")
            
    else:
        print("\n💥 Evaluation tests FAILED!")
        print("Please check the error messages above and fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)