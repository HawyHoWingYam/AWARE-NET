#!/usr/bin/env python3
"""
AWARE-NET Stage 2 Testing Suite
===============================

Comprehensive testing suite for Stage 2 components:
- EfficientNetV2-B3 training and evaluation
- GenConViT dual-mode system (hybrid + pretrained)
- Model performance comparison
- Integration validation

Usage:
    # Test EfficientNetV2-B3
    python test_stage2.py --test effnet --epochs 2
    
    # Test GenConViT hybrid mode
    python test_stage2.py --test genconvit --mode hybrid --epochs 2
    
    # Test GenConViT pretrained mode
    python test_stage2.py --test genconvit --mode pretrained --epochs 2
    
    # Test dual-mode switching
    python test_stage2.py --test switching
    
    # Run all tests
    python test_stage2.py --test all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
from src.stage1.dataset import create_dataloaders
from src.stage2.train_stage2_effnet import EfficientNetTrainer
from src.stage2.train_stage2_genconvit import GenConViTTrainer
from src.stage2.genconvit_manager import GenConViTManager

class Stage2TestSuite:
    """Comprehensive testing suite for Stage 2 components"""
    
    def __init__(self, data_dir: str = 'processed_data', device: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('Stage2TestSuite')
        
        print(f"🧪 Stage 2 Test Suite initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖥️  Device: {self.device}")
    
    def test_data_availability(self) -> bool:
        """Test if required data is available"""
        print("\n" + "="*50)
        print("🔍 Testing Data Availability")
        print("="*50)
        
        required_dirs = ['train', 'val', 'manifests']
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
                print(f"❌ Missing: {dir_path}")
            else:
                # Count samples
                if dir_name in ['train', 'val']:
                    real_count = len(list((dir_path / 'real').glob('*.jpg'))) if (dir_path / 'real').exists() else 0
                    fake_count = len(list((dir_path / 'fake').glob('*.jpg'))) if (dir_path / 'fake').exists() else 0
                    print(f"✅ {dir_name}: {real_count} real, {fake_count} fake samples")
                else:
                    print(f"✅ {dir_path} exists")
        
        if missing_dirs:
            print(f"\n❌ Data check failed. Missing directories: {missing_dirs}")
            return False
        
        print(f"\n✅ Data availability check passed")
        return True
    
    def test_effnet_training(self, epochs: int = 2, batch_size: int = 8) -> Dict[str, Any]:
        """Test EfficientNetV2-B3 training"""
        print("\n" + "="*50)
        print("🔬 Testing EfficientNetV2-B3 Training")
        print("="*50)
        
        try:
            # Create trainer
            trainer = EfficientNetTrainer(
                data_dir=str(self.data_dir),
                device=self.device
            )
            
            print(f"📊 Starting EfficientNet training: {epochs} epochs, batch size {batch_size}")
            start_time = time.time()
            
            # Run training
            results = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=5e-5,
                save_dir='test_effnet_models'
            )
            
            training_time = time.time() - start_time
            
            # Validate results
            assert 'best_auc' in results, "Missing best_auc in results"
            assert results['best_auc'] > 0, f"Invalid AUC: {results['best_auc']}"
            
            test_results = {
                'status': 'success',
                'best_auc': results['best_auc'],
                'training_time': training_time,
                'epochs': epochs,
                'model_path': results.get('best_model_path', 'Unknown')
            }
            
            print(f"✅ EfficientNet test completed successfully")
            print(f"📊 Best AUC: {results['best_auc']:.4f}")
            print(f"⏱️  Training time: {training_time:.1f}s")
            
            return test_results
            
        except Exception as e:
            print(f"❌ EfficientNet test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'training_time': 0
            }
    
    def test_genconvit_hybrid(self, epochs: int = 2, batch_size: int = 4) -> Dict[str, Any]:
        """Test GenConViT hybrid mode"""
        print("\n" + "="*50)
        print("🔬 Testing GenConViT Hybrid Mode")
        print("="*50)
        
        try:
            # Create trainer
            trainer = GenConViTTrainer(
                mode='hybrid',
                variant='ED',
                data_dir=str(self.data_dir),
                device=self.device
            )
            
            print(f"📊 Starting GenConViT hybrid training: {epochs} epochs, batch size {batch_size}")
            start_time = time.time()
            
            # Run training
            results = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=1e-4,
                save_dir='test_genconvit_hybrid_models'
            )
            
            training_time = time.time() - start_time
            
            # Validate results
            assert 'best_auc' in results, "Missing best_auc in results"
            assert results['best_auc'] > 0, f"Invalid AUC: {results['best_auc']}"
            
            test_results = {
                'status': 'success',
                'mode': 'hybrid',
                'variant': 'ED',
                'best_auc': results['best_auc'],
                'training_time': training_time,
                'epochs': epochs
            }
            
            print(f"✅ GenConViT hybrid test completed successfully")
            print(f"📊 Best AUC: {results['best_auc']:.4f}")
            print(f"⏱️  Training time: {training_time:.1f}s")
            
            return test_results
            
        except Exception as e:
            print(f"❌ GenConViT hybrid test failed: {e}")
            return {
                'status': 'failed',
                'mode': 'hybrid',
                'error': str(e),
                'training_time': 0
            }
    
    def test_genconvit_pretrained(self, epochs: int = 2, batch_size: int = 4) -> Dict[str, Any]:
        """Test GenConViT pretrained mode"""
        print("\n" + "="*50)
        print("🔬 Testing GenConViT Pretrained Mode")
        print("="*50)
        
        try:
            # Create trainer
            trainer = GenConViTTrainer(
                mode='pretrained',
                variant='ED',
                data_dir=str(self.data_dir),
                device=self.device
            )
            
            print(f"📊 Starting GenConViT pretrained training: {epochs} epochs, batch size {batch_size}")
            start_time = time.time()
            
            # Run training
            results = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=1e-4,
                save_dir='test_genconvit_pretrained_models'
            )
            
            training_time = time.time() - start_time
            
            # Validate results
            assert 'best_auc' in results, "Missing best_auc in results"
            assert results['best_auc'] > 0, f"Invalid AUC: {results['best_auc']}"
            
            test_results = {
                'status': 'success',
                'mode': 'pretrained',
                'variant': 'ED',
                'best_auc': results['best_auc'],
                'training_time': training_time,
                'epochs': epochs
            }
            
            print(f"✅ GenConViT pretrained test completed successfully")
            print(f"📊 Best AUC: {results['best_auc']:.4f}")
            print(f"⏱️  Training time: {training_time:.1f}s")
            
            return test_results
            
        except Exception as e:
            print(f"❌ GenConViT pretrained test failed: {e}")
            return {
                'status': 'failed',
                'mode': 'pretrained',
                'error': str(e),
                'training_time': 0
            }
    
    def test_mode_switching(self) -> Dict[str, Any]:
        """Test GenConViT mode switching capabilities"""
        print("\n" + "="*50)
        print("🔄 Testing GenConViT Mode Switching")
        print("="*50)
        
        try:
            # Test manager creation
            manager = GenConViTManager(mode='auto', variant='ED', device=self.device)
            print(f"✅ Manager created in {manager.mode.value} mode")
            
            # Test model creation
            model1 = manager.create_model()
            print(f"✅ Model created successfully")
            
            # Test mode switching
            original_mode = manager.mode
            new_mode = 'hybrid' if original_mode.value == 'pretrained' else 'pretrained'
            
            print(f"🔄 Switching from {original_mode.value} to {new_mode}")
            
            try:
                new_model = manager.switch_mode(new_mode)
                print(f"✅ Mode switching successful: {original_mode.value} → {manager.mode.value}")
                switching_success = True
            except Exception as switch_error:
                print(f"⚠️  Mode switching failed: {switch_error}")
                switching_success = False
            
            # Test model inference
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                output = model1(dummy_input)
                
            assert hasattr(output, 'classification'), "Missing classification output"
            assert hasattr(output, 'reconstruction'), "Missing reconstruction output"
            
            print(f"✅ Model inference successful")
            print(f"📊 Output shapes: classification {output.classification.shape}, reconstruction {output.reconstruction.shape}")
            
            return {
                'status': 'success',
                'initial_mode': original_mode.value,
                'switching_success': switching_success,
                'final_mode': manager.mode.value,
                'inference_success': True
            }
            
        except Exception as e:
            print(f"❌ Mode switching test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_feature_extraction(self) -> Dict[str, Any]:
        """Test feature extraction for Stage 3 compatibility"""
        print("\n" + "="*50)
        print("🧬 Testing Feature Extraction")
        print("="*50)
        
        try:
            results = {}
            
            # Test EfficientNet feature extraction
            print("Testing EfficientNet feature extraction...")
            from src.stage2.train_stage2_effnet import EfficientNetTrainer
            
            effnet_trainer = EfficientNetTrainer(data_dir=str(self.data_dir), device=self.device)
            effnet_model = effnet_trainer.setup_model()
            
            dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                effnet_features = effnet_model.extract_features(dummy_input)
            
            results['effnet'] = {
                'feature_dim': effnet_features.shape[1],
                'batch_size': effnet_features.shape[0],
                'success': True
            }
            
            print(f"✅ EfficientNet features: {effnet_features.shape}")
            
            # Test GenConViT feature extraction
            print("Testing GenConViT feature extraction...")
            
            genconvit_trainer = GenConViTTrainer(mode='auto', variant='ED', data_dir=str(self.data_dir), device=self.device)
            genconvit_model = genconvit_trainer.setup_model()
            
            with torch.no_grad():
                genconvit_features = genconvit_model.extract_features(dummy_input)
            
            # Validate feature structure
            assert 'final_features' in genconvit_features, "Missing final_features for Stage 3"
            
            final_features = genconvit_features['final_features']
            
            results['genconvit'] = {
                'feature_keys': list(genconvit_features.keys()),
                'final_feature_dim': final_features.shape[1],
                'batch_size': final_features.shape[0],
                'success': True
            }
            
            print(f"✅ GenConViT features: {final_features.shape}")
            print(f"📊 Available features: {list(genconvit_features.keys())}")
            
            results['status'] = 'success'
            return results
            
        except Exception as e:
            print(f"❌ Feature extraction test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_performance_comparison(self, epochs: int = 5) -> Dict[str, Any]:
        """Compare performance between different Stage 2 models"""
        print("\n" + "="*50)
        print("🏁 Performance Comparison")
        print("="*50)
        
        comparison_results = {}
        
        # Test configurations
        configs = [
            {'name': 'EfficientNetV2-B3', 'type': 'effnet'},
            {'name': 'GenConViT-Hybrid-ED', 'type': 'genconvit', 'mode': 'hybrid', 'variant': 'ED'},
            {'name': 'GenConViT-Pretrained-ED', 'type': 'genconvit', 'mode': 'pretrained', 'variant': 'ED'}
        ]
        
        for config in configs:
            print(f"\n🔬 Testing {config['name']}...")
            
            try:
                start_time = time.time()
                
                if config['type'] == 'effnet':
                    result = self.test_effnet_training(epochs=epochs, batch_size=8)
                else:
                    if config['mode'] == 'hybrid':
                        result = self.test_genconvit_hybrid(epochs=epochs, batch_size=4)
                    else:
                        result = self.test_genconvit_pretrained(epochs=epochs, batch_size=4)
                
                result['total_time'] = time.time() - start_time
                comparison_results[config['name']] = result
                
                if result['status'] == 'success':
                    print(f"✅ {config['name']}: AUC {result['best_auc']:.4f}")
                else:
                    print(f"❌ {config['name']}: Failed")
                    
            except Exception as e:
                print(f"❌ {config['name']}: Error - {e}")
                comparison_results[config['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Generate comparison summary
        successful_models = {k: v for k, v in comparison_results.items() if v['status'] == 'success'}
        
        if successful_models:
            print(f"\n📊 Performance Summary:")
            print("-" * 40)
            
            for name, result in successful_models.items():
                print(f"{name:25} | AUC: {result['best_auc']:.4f} | Time: {result.get('total_time', 0):.1f}s")
            
            # Find best model
            best_model = max(successful_models.items(), key=lambda x: x[1]['best_auc'])
            print(f"\n🏆 Best Model: {best_model[0]} (AUC: {best_model[1]['best_auc']:.4f})")
        
        return comparison_results
    
    def run_all_tests(self, epochs: int = 2) -> Dict[str, Any]:
        """Run complete test suite"""
        print("\n" + "🚀"*20)
        print("🧪 RUNNING COMPLETE STAGE 2 TEST SUITE")
        print("🚀"*20)
        
        all_results = {}
        start_time = time.time()
        
        # 1. Data availability
        print(f"\n[1/6] Data Availability Check")
        data_ok = self.test_data_availability()
        all_results['data_availability'] = {'success': data_ok}
        
        if not data_ok:
            print(f"❌ Cannot proceed without data. Please ensure processed_data directory exists.")
            return all_results
        
        # 2. EfficientNet test
        print(f"\n[2/6] EfficientNet Testing")
        all_results['effnet'] = self.test_effnet_training(epochs=epochs)
        
        # 3. GenConViT hybrid test
        print(f"\n[3/6] GenConViT Hybrid Testing")
        all_results['genconvit_hybrid'] = self.test_genconvit_hybrid(epochs=epochs)
        
        # 4. GenConViT pretrained test
        print(f"\n[4/6] GenConViT Pretrained Testing")
        all_results['genconvit_pretrained'] = self.test_genconvit_pretrained(epochs=epochs)
        
        # 5. Mode switching test
        print(f"\n[5/6] Mode Switching Testing")
        all_results['mode_switching'] = self.test_mode_switching()
        
        # 6. Feature extraction test
        print(f"\n[6/6] Feature Extraction Testing")
        all_results['feature_extraction'] = self.test_feature_extraction()
        
        total_time = time.time() - start_time
        all_results['total_test_time'] = total_time
        
        # Generate final report
        self._generate_test_report(all_results)
        
        return all_results
    
    def _generate_test_report(self, results: Dict[str, Any]):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📋 STAGE 2 TEST REPORT")
        print("="*60)
        
        # Test summary
        total_tests = len([k for k in results.keys() if k != 'total_test_time'])
        successful_tests = len([k for k, v in results.items() if 
                               k != 'total_test_time' and 
                               (v.get('status') == 'success' or v.get('success') == True)])
        
        print(f"📊 Test Summary: {successful_tests}/{total_tests} tests passed")
        print(f"⏱️  Total test time: {results.get('total_test_time', 0):.1f}s")
        print()
        
        # Individual test results
        test_names = {
            'data_availability': 'Data Availability',
            'effnet': 'EfficientNetV2-B3',
            'genconvit_hybrid': 'GenConViT Hybrid',
            'genconvit_pretrained': 'GenConViT Pretrained',
            'mode_switching': 'Mode Switching',
            'feature_extraction': 'Feature Extraction'
        }
        
        for key, name in test_names.items():
            if key in results:
                result = results[key]
                status = result.get('status', 'unknown')
                success = result.get('success', False)
                
                if status == 'success' or success:
                    icon = "✅"
                    status_text = "PASSED"
                else:
                    icon = "❌"
                    status_text = "FAILED"
                
                print(f"{icon} {name:25} | {status_text}")
                
                # Additional info
                if key in ['effnet', 'genconvit_hybrid', 'genconvit_pretrained']:
                    if result.get('best_auc'):
                        print(f"   └─ Best AUC: {result['best_auc']:.4f}")
        
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        
        if results.get('effnet', {}).get('status') == 'success':
            print("   ✅ EfficientNetV2-B3 is working correctly")
        
        hybrid_success = results.get('genconvit_hybrid', {}).get('status') == 'success'
        pretrained_success = results.get('genconvit_pretrained', {}).get('status') == 'success'
        
        if hybrid_success and pretrained_success:
            print("   ✅ Both GenConViT modes are functional - use auto mode for best results")
        elif hybrid_success:
            print("   ⚠️  Only hybrid mode working - consider hybrid mode for production")
        elif pretrained_success:
            print("   ⚠️  Only pretrained mode working - use pretrained mode")
        else:
            print("   ❌ GenConViT integration needs attention")
        
        if results.get('mode_switching', {}).get('switching_success'):
            print("   ✅ Mode switching is functional - runtime switching available")
        
        print()
        print("🎯 Next Steps:")
        print("   1. Review any failed tests and address issues")
        print("   2. Run longer training sessions for performance validation")
        print("   3. Proceed to Stage 3 meta-model integration")
        print("="*60)
        
        # Save results to file
        results_path = Path('stage2_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Test results saved to: {results_path}")

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='AWARE-NET Stage 2 Test Suite')
    
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'effnet', 'genconvit', 'switching', 'features', 'compare'],
                       help='Test type to run')
    parser.add_argument('--mode', type=str, default='auto',
                       choices=['hybrid', 'pretrained', 'auto'],
                       help='GenConViT mode for testing')
    parser.add_argument('--variant', type=str, default='ED',
                       choices=['ED', 'VAE'],
                       help='GenConViT variant for testing')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs for testing')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Data directory path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device for testing')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = Stage2TestSuite(data_dir=args.data_dir, device=args.device)
    
    # Run requested tests
    if args.test == 'all':
        results = test_suite.run_all_tests(epochs=args.epochs)
    elif args.test == 'effnet':
        results = test_suite.test_effnet_training(epochs=args.epochs)
    elif args.test == 'genconvit':
        if args.mode == 'hybrid':
            results = test_suite.test_genconvit_hybrid(epochs=args.epochs)
        elif args.mode == 'pretrained':
            results = test_suite.test_genconvit_pretrained(epochs=args.epochs)
        else:
            # Test both modes
            results = {
                'hybrid': test_suite.test_genconvit_hybrid(epochs=args.epochs),
                'pretrained': test_suite.test_genconvit_pretrained(epochs=args.epochs)
            }
    elif args.test == 'switching':
        results = test_suite.test_mode_switching()
    elif args.test == 'features':
        results = test_suite.test_feature_extraction()
    elif args.test == 'compare':
        results = test_suite.run_performance_comparison(epochs=args.epochs)
    
    print(f"\n🎉 Testing completed!")

if __name__ == '__main__':
    main()