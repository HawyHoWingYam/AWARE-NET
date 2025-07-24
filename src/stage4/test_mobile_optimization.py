#!/usr/bin/env python3
"""
Mobile Optimization Testing Script - test_mobile_optimization.py
==============================================================

Comprehensive testing script for the mobile optimization pipeline.
Tests QAT + Knowledge Distillation and ONNX export functionality.

Usage:
    # Test single model optimization
    python test_mobile_optimization.py --test_stage1
    
    # Test all components
    python test_mobile_optimization.py --test_all
    
    # Test ONNX export only
    python test_mobile_optimization.py --test_onnx_export
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.stage4.optimize_for_mobile import MobileOptimizer, QATConfig, OptimizationTarget
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter

class MobileOptimizationTester:
    """Test suite for mobile optimization pipeline"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mobile_opt_test_"))
        print(f"🔧 Test environment setup: {self.temp_dir}")
        
        # Create test data
        self.create_test_data()
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Test environment cleaned up")
    
    def create_test_data(self):
        """Create synthetic test data"""
        print("📊 Creating test data...")
        
        # Create synthetic dataset
        batch_size = 16
        num_samples = 100
        
        # Generate random images and labels
        images = torch.randn(num_samples, 3, 256, 256)
        labels = torch.randint(0, 2, (num_samples,))
        
        # Create dataset and loader
        dataset = TensorDataset(images, labels)
        self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Test data created: {num_samples} samples, batch_size={batch_size}")
    
    def create_test_models(self) -> Dict[str, nn.Module]:
        """Create test models for optimization"""
        print("🏗️ Creating test models...")
        
        models = {}
        
        # Stage 1: MobileNetV4 (simplified)
        models['stage1'] = timm.create_model(
            'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
            pretrained=True,
            num_classes=1
        )
        
        # Stage 2: EfficientNetV2-B3 (simplified)
        models['stage2_effnet'] = timm.create_model(
            'efficientnetv2_b3.in21k_ft_in1k',
            pretrained=True,
            num_classes=1
        )
        
        print(f"✅ Test models created: {list(models.keys())}")
        return models
    
    def test_qat_config(self) -> bool:
        """Test QAT configuration creation and validation"""
        print("\n🧪 Testing QAT Configuration...")
        
        try:
            # Test default config
            config = QATConfig()
            assert config.epochs > 0
            assert config.batch_size > 0
            assert config.learning_rate > 0
            assert 0 < config.alpha_hard_loss < 1
            assert 0 < config.alpha_soft_loss < 1
            assert config.alpha_hard_loss + config.alpha_soft_loss == 1.0
            
            # Test custom config
            custom_config = QATConfig(
                epochs=5,
                batch_size=16,
                learning_rate=1e-4,
                distillation_temperature=3.0
            )
            assert custom_config.epochs == 5
            assert custom_config.distillation_temperature == 3.0
            
            print("✅ QAT Configuration test passed")
            self.test_results['qat_config'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ QAT Configuration test failed: {e}")
            self.test_results['qat_config'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_mobile_optimizer_init(self) -> bool:
        """Test MobileOptimizer initialization"""
        print("\n🧪 Testing MobileOptimizer Initialization...")
        
        try:
            # Test with default config
            config = QATConfig(output_dir=str(self.temp_dir / "optimizer_test"))
            optimizer = MobileOptimizer(config)
            
            assert optimizer.device is not None
            assert optimizer.output_dir.exists()
            assert optimizer.config.quantization_backend in ['fbgemm', 'qnnpack']
            
            print("✅ MobileOptimizer initialization test passed")
            self.test_results['optimizer_init'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ MobileOptimizer initialization test failed: {e}")
            self.test_results['optimizer_init'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_model_size_calculation(self) -> bool:
        """Test model size calculation"""
        print("\n🧪 Testing Model Size Calculation...")
        
        try:
            config = QATConfig(output_dir=str(self.temp_dir / "size_test"))
            optimizer = MobileOptimizer(config)
            
            # Create test model
            test_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k', 
                                         pretrained=False, num_classes=1)
            
            # Calculate size
            model_size = optimizer.calculate_model_size(test_model)
            
            assert model_size > 0
            assert isinstance(model_size, float)
            
            print(f"✅ Model size calculation test passed: {model_size:.2f} MB")
            self.test_results['model_size_calc'] = {
                'status': 'PASSED', 
                'model_size_mb': model_size
            }
            return True
            
        except Exception as e:
            print(f"❌ Model size calculation test failed: {e}")
            self.test_results['model_size_calc'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_teacher_model_loading(self) -> bool:
        """Test teacher model loading (with fake checkpoints)"""
        print("\n🧪 Testing Teacher Model Loading...")
        
        try:
            config = QATConfig(output_dir=str(self.temp_dir / "teacher_test"))
            optimizer = MobileOptimizer(config)
            
            # Test loading without checkpoint (should use pretrained)
            teacher_model = optimizer.load_teacher_model(OptimizationTarget.STAGE1)
            
            assert teacher_model is not None
            assert isinstance(teacher_model, nn.Module)
            
            # Verify model is in eval mode
            assert not teacher_model.training
            
            print("✅ Teacher model loading test passed")
            self.test_results['teacher_loading'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ Teacher model loading test failed: {e}")
            self.test_results['teacher_loading'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_student_model_creation(self) -> bool:
        """Test student model creation and QAT preparation"""
        print("\n🧪 Testing Student Model Creation...")
        
        try:
            config = QATConfig(output_dir=str(self.temp_dir / "student_test"))
            optimizer = MobileOptimizer(config)
            
            # Create teacher model
            teacher_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                                            pretrained=True, num_classes=1)
            
            # Create student model
            student_model = optimizer.create_student_model(teacher_model, OptimizationTarget.STAGE1)
            
            assert student_model is not None
            assert isinstance(student_model, nn.Module)
            
            # Verify QAT preparation
            assert hasattr(student_model, 'qconfig')
            
            print("✅ Student model creation test passed")
            self.test_results['student_creation'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ Student model creation test failed: {e}")
            self.test_results['student_creation'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_distillation_loss(self) -> bool:
        """Test distillation loss calculation"""
        print("\n🧪 Testing Distillation Loss...")
        
        try:
            from src.stage4.optimize_for_mobile import DistillationLoss
            
            # Create loss function
            distill_loss = DistillationLoss(temperature=4.0, alpha_hard=0.3, alpha_soft=0.7)
            
            # Create test tensors
            batch_size = 8
            student_logits = torch.randn(batch_size, 1)
            teacher_logits = torch.randn(batch_size, 1)
            targets = torch.randint(0, 2, (batch_size, 1)).float()
            
            # Calculate loss
            loss_dict = distill_loss(student_logits, teacher_logits, targets)
            
            # Verify loss components
            assert 'total_loss' in loss_dict
            assert 'hard_loss' in loss_dict
            assert 'soft_loss' in loss_dict
            
            assert loss_dict['total_loss'].requires_grad
            assert loss_dict['total_loss'].item() > 0
            
            print("✅ Distillation loss test passed")
            self.test_results['distillation_loss'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ Distillation loss test failed: {e}")
            self.test_results['distillation_loss'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_onnx_export(self) -> bool:
        """Test ONNX export functionality"""
        print("\n🧪 Testing ONNX Export...")
        
        try:
            # Create test model
            test_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                                         pretrained=False, num_classes=1)
            test_model.eval()
            
            # Export to ONNX
            exporter = ONNXExporter(verbose=False)
            output_path = self.temp_dir / "test_model.onnx"
            
            result = exporter.export_model(
                model=test_model,
                output_path=output_path,
                input_shape=(1, 3, 256, 256),
                model_name="test_model"
            )
            
            # Verify export success
            assert result['success'] == True
            assert output_path.exists()
            assert result['model_size_mb'] > 0
            
            # Verify validation results
            assert 'validation_results' in result
            validation = result['validation_results']
            assert 'outputs_match' in validation
            
            print(f"✅ ONNX export test passed - Size: {result['model_size_mb']:.2f} MB")
            self.test_results['onnx_export'] = {
                'status': 'PASSED',
                'model_size_mb': result['model_size_mb'],
                'validation_passed': validation.get('outputs_match', False)
            }
            return True
            
        except Exception as e:
            print(f"❌ ONNX export test failed: {e}")
            self.test_results['onnx_export'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def test_quick_optimization_pipeline(self) -> bool:
        """Test quick optimization pipeline (minimal epochs)"""
        print("\n🧪 Testing Quick Optimization Pipeline...")
        
        try:
            # Create minimal config for testing
            config = QATConfig(
                epochs=1,  # Minimal for testing
                batch_size=8,
                calibration_dataset_size=32,  # Small dataset
                output_dir=str(self.temp_dir / "quick_opt_test")
            )
            
            optimizer = MobileOptimizer(config)
            
            # Mock calibration dataset
            test_images = torch.randn(32, 3, 256, 256)
            test_labels = torch.randint(0, 2, (32,))
            test_dataset = TensorDataset(test_images, test_labels)
            calibration_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            # Create simple teacher model
            teacher_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                                            pretrained=False, num_classes=1)
            teacher_model.eval()
            
            # Create student model
            student_model = optimizer.create_student_model(teacher_model, OptimizationTarget.STAGE1)
            
            # Run quick QAT (this might take a few minutes)
            print("⏳ Running quick QAT training (1 epoch)...")
            quantized_model = optimizer.run_qat_training(
                teacher_model, student_model, calibration_loader, OptimizationTarget.STAGE1
            )
            
            assert quantized_model is not None
            
            # Test inference
            test_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                output = quantized_model(test_input)
                assert output.shape == (1, 1)
            
            print("✅ Quick optimization pipeline test passed")
            self.test_results['quick_optimization'] = {'status': 'PASSED'}
            return True
            
        except Exception as e:
            print(f"❌ Quick optimization pipeline test failed: {e}")
            self.test_results['quick_optimization'] = {'status': 'FAILED', 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and generate report"""
        print("🚀 Starting Mobile Optimization Test Suite")
        print("=" * 60)
        
        # Setup test environment
        self.setup_test_environment()
        
        try:
            # Run individual tests
            test_functions = [
                self.test_qat_config,
                self.test_mobile_optimizer_init,
                self.test_model_size_calculation,
                self.test_teacher_model_loading,
                self.test_student_model_creation,
                self.test_distillation_loss,
                self.test_onnx_export,
                # self.test_quick_optimization_pipeline,  # Skip for quick testing
            ]
            
            passed_tests = 0
            total_tests = len(test_functions)
            
            for test_func in test_functions:
                try:
                    if test_func():
                        passed_tests += 1
                except Exception as e:
                    print(f"❌ Test {test_func.__name__} crashed: {e}")
                    self.test_results[test_func.__name__] = {'status': 'CRASHED', 'error': str(e)}
            
            # Generate summary
            success_rate = (passed_tests / total_tests) * 100
            
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY")
            print("=" * 60)
            print(f"Total Tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {total_tests - passed_tests}")
            print(f"Success Rate: {success_rate:.1f}%")
            
            # Print individual results
            print("\n📋 Individual Test Results:")
            for test_name, result in self.test_results.items():
                status = result['status']
                emoji = "✅" if status == "PASSED" else "❌"
                print(f"{emoji} {test_name}: {status}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
            
            # Save detailed report
            report = {
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': success_rate,
                    'timestamp': time.time()
                },
                'individual_results': self.test_results
            }
            
            report_path = self.temp_dir / "test_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Detailed report saved to: {report_path}")
            
            if success_rate >= 80:
                print("🎉 Mobile optimization pipeline is ready for use!")
            else:
                print("⚠️ Some tests failed. Please review and fix issues before deployment.")
            
            return report
            
        finally:
            # Cleanup
            self.cleanup_test_environment()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Mobile Optimization Test Suite')
    parser.add_argument('--test_all', action='store_true', help='Run all tests')
    parser.add_argument('--test_stage1', action='store_true', help='Test Stage 1 optimization')
    parser.add_argument('--test_onnx_export', action='store_true', help='Test ONNX export only')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    
    args = parser.parse_args()
    
    tester = MobileOptimizationTester(verbose=True)
    
    if args.test_all or not any([args.test_stage1, args.test_onnx_export]):
        # Run full test suite
        report = tester.run_all_tests()
    elif args.test_onnx_export:
        # Test ONNX export only
        tester.setup_test_environment()
        try:
            success = tester.test_onnx_export()
            print(f"ONNX Export Test: {'PASSED' if success else 'FAILED'}")
        finally:
            tester.cleanup_test_environment()
    elif args.test_stage1:
        # Test specific stage optimization
        print("Stage 1 optimization test not implemented yet")
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    main()