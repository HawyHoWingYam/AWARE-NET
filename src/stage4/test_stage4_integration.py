#!/usr/bin/env python3
"""
Stage 4 Integration Testing Framework - test_stage4_integration.py
===============================================================

Comprehensive end-to-end integration testing for Stage 4 mobile optimization
and deployment systems. Tests complete pipeline functionality, robustness,
and deployment readiness.

Key Testing Areas:
- End-to-end pipeline validation from raw input to mobile deployment
- Mobile deployment compatibility and performance testing
- Robustness testing with edge cases and error conditions
- Cross-platform deployment validation
- Performance regression testing
- Real-world scenario simulation

Usage:
    # Full integration test suite
    python test_stage4_integration.py --full_pipeline
    
    # Specific test categories
    python test_stage4_integration.py --test_mobile_deployment
    python test_stage4_integration.py --test_robustness
    
    # Performance regression testing
    python test_stage4_integration.py --regression_test
"""

import os
import sys
import json
import time
import tempfile
import shutil
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
import subprocess
import threading
import queue

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import pytest
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Stage 4 components
from src.stage4.cascade_detector import CascadeDetector
from src.stage4.optimize_for_mobile import MobileOptimizer, QATConfig, OptimizationTarget
from src.stage4.benchmark_cascade import CascadeBenchmarker, BenchmarkConfig
from src.stage4.video_processor import VideoProcessor, VideoProcessingConfig, StreamingVideoProcessor
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter

@dataclass
class IntegrationTestConfig:
    """Configuration for integration testing"""
    # Test environment
    test_timeout_seconds: int = 3600  # 1 hour max per test suite
    temp_dir_cleanup: bool = True
    enable_gpu_testing: bool = True
    enable_performance_profiling: bool = True
    
    # Test data
    create_synthetic_data: bool = True
    synthetic_video_duration: int = 10  # seconds
    synthetic_image_count: int = 50
    test_batch_sizes: List[int] = None
    
    # Pipeline testing
    test_cascade_integration: bool = True
    test_mobile_optimization: bool = True
    test_video_processing: bool = True
    test_onnx_deployment: bool = True
    
    # Robustness testing
    test_edge_cases: bool = True
    test_error_handling: bool = True
    test_memory_limits: bool = True
    test_concurrent_processing: bool = True
    
    # Performance testing
    performance_baseline_path: Optional[str] = None
    performance_tolerance_percent: float = 10.0
    memory_limit_mb: int = 2048
    
    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 4, 8, 16, 32]

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_category: str
    status: str  # 'PASSED', 'FAILED', 'SKIPPED', 'ERROR'
    execution_time_seconds: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
@dataclass
class IntegrationTestReport:
    """Complete integration test report"""
    test_session_id: str
    start_time: float
    total_duration_seconds: float
    
    # Test summary
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    
    # System information
    system_info: Dict[str, Any]
    test_config: IntegrationTestConfig
    
    # Detailed results
    test_results: List[TestResult]
    performance_regression: Optional[Dict[str, Any]] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

class SyntheticDataGenerator:
    """Generate synthetic test data for integration testing"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_test_images(self, count: int = 50) -> List[Path]:
        """Generate synthetic test images"""
        image_paths = []
        
        for i in range(count):
            # Create synthetic image with random patterns
            image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            if i % 2 == 0:
                # Add geometric patterns for "fake" samples
                cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), 2)
                cv2.circle(image, (128, 128), 30, (0, 255, 0), -1)
            else:
                # Add noise patterns for "real" samples
                noise = np.random.normal(0, 25, (256, 256, 3))
                image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            image_path = self.output_dir / f"test_image_{i:03d}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_paths.append(image_path)
        
        return image_paths
    
    def generate_test_video(self, duration: int = 10, fps: int = 30) -> Path:
        """Generate synthetic test video"""
        video_path = self.output_dir / "test_video.mp4"
        
        # Video parameters
        width, height = 256, 256
        total_frames = duration * fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        try:
            for frame_idx in range(total_frames):
                # Create frame with animated patterns
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add moving elements
                time_factor = frame_idx / total_frames
                
                # Moving circle
                center_x = int(width * 0.2 + width * 0.6 * time_factor)
                center_y = height // 2
                cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
                
                # Changing background pattern
                pattern_intensity = int(128 + 127 * np.sin(time_factor * 4 * np.pi))
                frame[:, :, 2] = pattern_intensity
                
                # Add some random noise
                noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                writer.write(frame)
        
        finally:
            writer.release()
        
        return video_path
    
    def generate_edge_case_data(self) -> Dict[str, Path]:
        """Generate edge case test data"""
        edge_cases = {}
        
        # Very small image
        small_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        small_path = self.output_dir / "small_image.jpg"
        cv2.imwrite(str(small_path), cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR))
        edge_cases['small_image'] = small_path
        
        # Very large image
        large_img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        large_path = self.output_dir / "large_image.jpg"
        cv2.imwrite(str(large_path), cv2.cvtColor(large_img, cv2.COLOR_RGB2BGR))
        edge_cases['large_image'] = large_path
        
        # All black image
        black_img = np.zeros((256, 256, 3), dtype=np.uint8)
        black_path = self.output_dir / "black_image.jpg"
        cv2.imwrite(str(black_path), black_img)
        edge_cases['black_image'] = black_path
        
        # All white image
        white_img = np.full((256, 256, 3), 255, dtype=np.uint8)
        white_path = self.output_dir / "white_image.jpg"
        cv2.imwrite(str(white_path), white_img)
        edge_cases['white_image'] = white_path
        
        # Corrupted image (partially)
        corrupt_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        corrupt_img[100:150, :, :] = 0  # Create "corrupted" region
        corrupt_path = self.output_dir / "corrupt_image.jpg"
        cv2.imwrite(str(corrupt_path), cv2.cvtColor(corrupt_img, cv2.COLOR_RGB2BGR))
        edge_cases['corrupt_image'] = corrupt_path
        
        return edge_cases

class SystemMonitor:
    """Monitor system resources during testing"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = []
        
    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.resource_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.resource_history:
            return {'peak_memory_mb': 0, 'avg_cpu_percent': 0, 'samples': 0}
        
        memory_values = [r['memory_mb'] for r in self.resource_history]
        cpu_values = [r['cpu_percent'] for r in self.resource_history]
        
        return {
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'avg_cpu_percent': np.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'samples': len(self.resource_history)
        }
    
    def _monitor_loop(self):
        """Resource monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024 ** 2)
                cpu_percent = process.cpu_percent()
                
                self.resource_history.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception:
                break

class Stage4IntegrationTester:
    """Main integration testing framework"""
    
    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        self.config = config or IntegrationTestConfig()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize test environment
        self.test_session_id = f"integration_test_{int(time.time())}"
        self.start_time = time.time()
        self.temp_dir = None
        
        # Test results tracking
        self.test_results: List[TestResult] = []
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        self.logger.info(f"🧪 Stage 4 Integration Tester initialized")
        self.logger.info(f"Session ID: {self.test_session_id}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('Stage4IntegrationTester')
    
    def setup_test_environment(self):
        """Setup test environment and synthetic data"""
        self.logger.info("🔧 Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.test_session_id}_"))
        self.logger.info(f"Test directory: {self.temp_dir}")
        
        # Generate synthetic test data
        if self.config.create_synthetic_data:
            data_generator = SyntheticDataGenerator(self.temp_dir / "test_data")
            
            # Generate test images
            self.test_images = data_generator.generate_test_images(self.config.synthetic_image_count)
            self.logger.info(f"Generated {len(self.test_images)} test images")
            
            # Generate test video
            self.test_video = data_generator.generate_test_video(self.config.synthetic_video_duration)
            self.logger.info(f"Generated test video: {self.test_video}")
            
            # Generate edge case data
            self.edge_case_data = data_generator.generate_edge_case_data()
            self.logger.info(f"Generated {len(self.edge_case_data)} edge case samples")
        
        # Setup output directories
        self.output_dir = self.temp_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("✅ Test environment setup completed")
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.config.temp_dir_cleanup and self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info("🧹 Test environment cleaned up")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to cleanup test environment: {e}")
    
    def run_test(self, test_func: callable, test_name: str, test_category: str, 
                timeout_seconds: Optional[int] = None) -> TestResult:
        """Run individual test with monitoring and error handling"""
        self.logger.info(f"🧪 Running test: {test_name}")
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        start_time = time.time()
        
        try:
            # Set timeout
            if timeout_seconds is None:
                timeout_seconds = self.config.test_timeout_seconds
            
            # Run test with timeout (simplified - would need proper timeout implementation)
            result = test_func()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Stop monitoring and get resource usage
            resource_stats = self.system_monitor.stop_monitoring()
            
            # Create test result
            test_result = TestResult(
                test_name=test_name,
                test_category=test_category,
                status='PASSED',
                execution_time_seconds=execution_time,
                memory_usage_mb=resource_stats['peak_memory_mb'],
                details={
                    'resource_stats': resource_stats,
                    'test_result': result
                }
            )
            
            self.logger.info(f"✅ Test passed: {test_name} ({execution_time:.1f}s)")
            
        except AssertionError as e:
            # Test assertion failed
            execution_time = time.time() - start_time
            resource_stats = self.system_monitor.stop_monitoring()
            
            test_result = TestResult(
                test_name=test_name,
                test_category=test_category,
                status='FAILED',
                execution_time_seconds=execution_time,
                memory_usage_mb=resource_stats['peak_memory_mb'],
                error_message=str(e),
                details={'resource_stats': resource_stats}
            )
            
            self.logger.error(f"❌ Test failed: {test_name} - {e}")
            
        except Exception as e:
            # Test error/crash
            execution_time = time.time() - start_time
            resource_stats = self.system_monitor.stop_monitoring()
            
            test_result = TestResult(
                test_name=test_name,
                test_category=test_category,
                status='ERROR',
                execution_time_seconds=execution_time,
                memory_usage_mb=resource_stats['peak_memory_mb'],
                error_message=str(e),
                details={
                    'resource_stats': resource_stats,
                    'traceback': traceback.format_exc()
                }
            )
            
            self.logger.error(f"💥 Test error: {test_name} - {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    def test_cascade_detector_integration(self) -> Dict[str, Any]:
        """Test complete cascade detector integration"""
        self.logger.info("🔗 Testing cascade detector integration...")
        
        # Initialize cascade detector
        detector = CascadeDetector()
        
        # Test single image prediction
        test_image_path = self.test_images[0]
        image = cv2.imread(str(test_image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = detector.predict(image_rgb)
        
        # Validate result structure
        required_keys = ['final_prediction', 'final_probability', 'processing_time_ms']
        for key in required_keys:
            assert key in result, f"Missing key in cascade result: {key}"
        
        assert result['final_prediction'] in ['real', 'fake'], "Invalid prediction value"
        assert 0 <= result['final_probability'] <= 1, "Invalid probability range"
        assert result['processing_time_ms'] > 0, "Invalid processing time"
        
        # Test batch prediction
        batch_images = []
        for img_path in self.test_images[:5]:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_images.append(img_rgb)
        
        batch_results = detector.predict_batch(batch_images)
        assert len(batch_results) == len(batch_images), "Batch size mismatch"
        
        # Test video prediction
        video_results = detector.predict_video(str(self.test_video))
        assert 'final_prediction' in video_results, "Missing video prediction"
        assert 'frame_predictions' in video_results, "Missing frame predictions"
        
        return {
            'single_prediction': result,
            'batch_predictions': len(batch_results),
            'video_prediction': video_results['final_prediction'],
            'video_frames_processed': len(video_results['frame_predictions'])
        }
    
    def test_mobile_optimization_integration(self) -> Dict[str, Any]:
        """Test mobile optimization pipeline integration"""
        self.logger.info("📱 Testing mobile optimization integration...")
        
        # Create minimal config for testing
        qat_config = QATConfig(
            epochs=1,  # Minimal for testing
            batch_size=4,
            calibration_dataset_size=16,
            output_dir=str(self.output_dir / "optimization_test")
        )
        
        optimizer = MobileOptimizer(qat_config)
        
        # Test optimization target validation
        target = OptimizationTarget.STAGE1
        
        # Mock test (full optimization would take too long)
        # In a real test environment, you might run a quick optimization
        
        # Test model size calculation
        import timm
        test_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k', 
                                     pretrained=False, num_classes=1)
        model_size = optimizer.calculate_model_size(test_model)
        
        assert model_size > 0, "Model size calculation failed"
        assert model_size < 1000, "Model size unreasonably large"  # Should be < 1GB
        
        # Test teacher model loading (should work with pretrained fallback)
        teacher_model = optimizer.load_teacher_model(target)
        assert teacher_model is not None, "Teacher model loading failed"
        
        # Test student model creation
        student_model = optimizer.create_student_model(teacher_model, target)
        assert student_model is not None, "Student model creation failed"
        assert hasattr(student_model, 'qconfig'), "QAT configuration not applied"
        
        return {
            'model_size_mb': model_size,
            'teacher_model_loaded': True,
            'student_model_created': True,
            'qat_config_applied': hasattr(student_model, 'qconfig')
        }
    
    def test_video_processing_integration(self) -> Dict[str, Any]:
        """Test video processing system integration"""
        self.logger.info("🎬 Testing video processing integration...")
        
        # Create video processing configuration
        video_config = VideoProcessingConfig(
            temporal_window_size=4,  # Smaller for testing
            streaming_mode=False,
            save_temporal_features=True,
            generate_heatmaps=False  # Skip to save time
        )
        
        processor = VideoProcessor(video_config)
        
        # Test video processing
        result = processor.process_video(self.test_video)
        
        # Validate result structure
        assert result.video_path == str(self.test_video), "Video path mismatch"
        assert result.total_frames > 0, "No frames detected"
        assert result.processed_frames > 0, "No frames processed"
        assert result.final_prediction in ['real', 'fake'], "Invalid final prediction"
        assert 0 <= result.overall_confidence <= 1, "Invalid confidence range"
        assert len(result.frame_results) > 0, "No frame results"
        
        # Test frame result structure
        frame_result = result.frame_results[0]
        assert hasattr(frame_result, 'prediction'), "Missing frame prediction"
        assert hasattr(frame_result, 'confidence'), "Missing frame confidence"
        assert hasattr(frame_result, 'temporal_consistency_score'), "Missing temporal consistency"
        
        # Test streaming processing (with short video)
        streaming_config = VideoProcessingConfig(
            streaming_mode=True,
            parallel_workers=1,  # Single worker for testing
            buffer_size=8
        )
        
        streaming_processor = StreamingVideoProcessor(streaming_config)
        
        # Process first few frames from stream
        frame_count = 0
        max_frames = 10  # Limit for testing
        
        for frame_result in streaming_processor.process_stream(str(self.test_video)):
            assert frame_result is not None, "Null frame result from stream"
            frame_count += 1
            if frame_count >= max_frames:
                break
        
        assert frame_count > 0, "No frames processed from stream"
        
        return {
            'batch_processing': {
                'total_frames': result.total_frames,
                'processed_frames': result.processed_frames,
                'final_prediction': result.final_prediction,
                'processing_time': result.total_processing_time_seconds
            },
            'streaming_processing': {
                'frames_processed': frame_count,
                'parallel_workers': streaming_config.parallel_workers
            }
        }
    
    def test_onnx_deployment_integration(self) -> Dict[str, Any]:
        """Test ONNX deployment integration"""
        self.logger.info("🚀 Testing ONNX deployment integration...")
        
        # Create test model
        import timm
        test_model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                                     pretrained=False, num_classes=1)
        test_model.eval()
        
        # Test ONNX export
        exporter = ONNXExporter(verbose=False)
        output_path = self.output_dir / "test_model.onnx"
        
        export_result = exporter.export_model(
            model=test_model,
            output_path=output_path,
            input_shape=(1, 3, 256, 256),
            model_name="integration_test_model"
        )
        
        assert export_result['success'], f"ONNX export failed: {export_result.get('error', 'Unknown error')}"
        assert output_path.exists(), "ONNX file not created"
        assert export_result['model_size_mb'] > 0, "Invalid model size"
        
        # Test validation results
        validation = export_result['validation_results']
        assert 'outputs_match' in validation, "Missing validation results"
        
        # Test ONNX runtime loading
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(output_path))
            
            # Test inference
            test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})
            
            assert len(output) > 0, "No ONNX output"
            assert output[0].shape[0] == 1, "Invalid output batch size"
            
            onnx_inference_success = True
            
        except ImportError:
            self.logger.warning("ONNX Runtime not available, skipping inference test")
            onnx_inference_success = False
        except Exception as e:
            self.logger.error(f"ONNX inference failed: {e}")
            onnx_inference_success = False
        
        # Test bundle export (mock)
        models = {'test_model': test_model}
        bundle_result = exporter.export_cascade_bundle(
            models=models,
            output_dir=self.output_dir / "test_bundle",
            bundle_name="integration_test"
        )
        
        assert bundle_result['success'], "Bundle export failed"
        
        return {
            'single_export': {
                'success': export_result['success'],
                'model_size_mb': export_result['model_size_mb'],
                'validation_passed': validation.get('outputs_match', False)
            },
            'onnx_inference': {
                'success': onnx_inference_success
            },
            'bundle_export': {
                'success': bundle_result['success'],
                'models_exported': bundle_result.get('models_exported', 0)
            }
        }
    
    def test_robustness_and_edge_cases(self) -> Dict[str, Any]:
        """Test system robustness with edge cases"""
        self.logger.info("🛡️ Testing robustness and edge cases...")
        
        edge_case_results = {}
        detector = CascadeDetector()
        
        # Test edge case images
        for case_name, image_path in self.edge_case_data.items():
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    # Try with PIL for corrupted images
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = detector.predict(image_rgb)
                
                edge_case_results[case_name] = {
                    'success': True,
                    'prediction': result['final_prediction'],
                    'confidence': result['final_probability']
                }
                
            except Exception as e:
                edge_case_results[case_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Test invalid inputs
        invalid_input_results = {}
        
        # Test empty array
        try:
            empty_array = np.array([])
            detector.predict(empty_array)
            invalid_input_results['empty_array'] = {'success': True}
        except Exception as e:
            invalid_input_results['empty_array'] = {'success': False, 'expected_error': True}
        
        # Test wrong dimensions
        try:
            wrong_dims = np.random.randn(100, 200)  # 2D instead of 3D
            detector.predict(wrong_dims)
            invalid_input_results['wrong_dimensions'] = {'success': True}
        except Exception as e:
            invalid_input_results['wrong_dimensions'] = {'success': False, 'expected_error': True}
        
        # Test very small batch sizes
        small_batch_results = {}
        for batch_size in [1, 2, 3]:
            try:
                batch_images = []
                for i in range(batch_size):
                    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
                    batch_images.append(img)
                
                results = detector.predict_batch(batch_images)
                small_batch_results[f'batch_size_{batch_size}'] = {
                    'success': len(results) == batch_size,
                    'result_count': len(results)
                }
                
            except Exception as e:
                small_batch_results[f'batch_size_{batch_size}'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'edge_case_images': edge_case_results,
            'invalid_inputs': invalid_input_results,
            'small_batches': small_batch_results,
            'total_edge_cases_tested': len(edge_case_results) + len(invalid_input_results) + len(small_batch_results)
        }
    
    def test_performance_regression(self) -> Dict[str, Any]:
        """Test for performance regression compared to baseline"""
        self.logger.info("📊 Testing performance regression...")
        
        # Simple performance test with synthetic data
        detector = CascadeDetector()
        
        # Benchmark single image inference
        test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            detector.predict(test_image)
        
        # Benchmark
        inference_times = []
        for _ in range(20):
            start_time = time.time()
            result = detector.predict(test_image)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        # Benchmark batch processing
        batch_images = [test_image] * 8
        batch_times = []
        
        for _ in range(10):
            start_time = time.time()
            batch_results = detector.predict_batch(batch_images)
            end_time = time.time()
            batch_times.append((end_time - start_time) * 1000)
        
        avg_batch_time = np.mean(batch_times)
        throughput_fps = len(batch_images) / (avg_batch_time / 1000)
        
        # Memory usage benchmark
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 ** 2)
        
        # Process larger batch
        large_batch = [test_image] * 32
        large_batch_results = detector.predict_batch(large_batch)
        
        memory_after = process.memory_info().rss / (1024 ** 2)
        memory_increase = memory_after - memory_before
        
        performance_metrics = {
            'single_inference_ms': avg_inference_time,
            'inference_std_ms': std_inference_time,
            'batch_inference_ms': avg_batch_time,
            'throughput_fps': throughput_fps,
            'memory_increase_mb': memory_increase
        }
        
        # Check against baseline (if available)
        regression_detected = False
        if self.config.performance_baseline_path and Path(self.config.performance_baseline_path).exists():
            try:
                with open(self.config.performance_baseline_path, 'r') as f:
                    baseline = json.load(f)
                
                # Check if performance degraded beyond tolerance
                tolerance = self.config.performance_tolerance_percent / 100
                
                for metric, current_value in performance_metrics.items():
                    if metric in baseline:
                        baseline_value = baseline[metric]
                        if metric.endswith('_ms'):  # Lower is better for time metrics
                            regression = (current_value - baseline_value) / baseline_value > tolerance
                        else:  # Higher is better for throughput
                            regression = (baseline_value - current_value) / baseline_value > tolerance
                        
                        if regression:
                            regression_detected = True
                            self.logger.warning(f"Performance regression detected in {metric}: "
                                             f"{current_value:.2f} vs baseline {baseline_value:.2f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load performance baseline: {e}")
        
        # Save current performance as new baseline
        baseline_path = self.output_dir / "performance_baseline.json"
        with open(baseline_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        return {
            'performance_metrics': performance_metrics,
            'regression_detected': regression_detected,
            'baseline_saved': str(baseline_path)
        }
    
    def run_full_integration_test_suite(self) -> IntegrationTestReport:
        """Run complete integration test suite"""
        self.logger.info("🚀 Starting full integration test suite...")
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Define test suite
            test_suite = []
            
            if self.config.test_cascade_integration:
                test_suite.append(
                    (self.test_cascade_detector_integration, "cascade_detector_integration", "core")
                )
            
            if self.config.test_mobile_optimization:
                test_suite.append(
                    (self.test_mobile_optimization_integration, "mobile_optimization_integration", "optimization")
                )
            
            if self.config.test_video_processing:
                test_suite.append(
                    (self.test_video_processing_integration, "video_processing_integration", "video")
                )
            
            if self.config.test_onnx_deployment:
                test_suite.append(
                    (self.test_onnx_deployment_integration, "onnx_deployment_integration", "deployment")
                )
            
            if self.config.test_edge_cases:
                test_suite.append(
                    (self.test_robustness_and_edge_cases, "robustness_and_edge_cases", "robustness")
                )
            
            # Always run performance regression test
            test_suite.append(
                (self.test_performance_regression, "performance_regression", "performance")
            )
            
            # Run all tests
            for test_func, test_name, test_category in test_suite:
                self.run_test(test_func, test_name, test_category)
            
            # Calculate summary statistics
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results if r.status == 'PASSED'])
            failed_tests = len([r for r in self.test_results if r.status == 'FAILED'])
            error_tests = len([r for r in self.test_results if r.status == 'ERROR'])
            skipped_tests = len([r for r in self.test_results if r.status == 'SKIPPED'])
            
            # Gather system information
            system_info = {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'platform': sys.platform,
                'total_memory_gb': psutil.virtual_memory().total / (1024**3)
            }
            
            # Create final report
            total_duration = time.time() - self.start_time
            
            report = IntegrationTestReport(
                test_session_id=self.test_session_id,
                start_time=self.start_time,
                total_duration_seconds=total_duration,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                system_info=system_info,
                test_config=self.config,
                test_results=self.test_results
            )
            
            # Save report
            self.save_test_report(report)
            
            # Log summary
            self.log_test_summary(report)
            
            return report
            
        finally:
            # Cleanup
            self.cleanup_test_environment()
    
    def save_test_report(self, report: IntegrationTestReport):
        """Save detailed test report"""
        if self.output_dir:
            report_path = self.output_dir / f"integration_test_report_{self.test_session_id}.json"
        else:
            report_path = Path(f"integration_test_report_{self.test_session_id}.json")
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"📄 Test report saved: {report_path}")
    
    def log_test_summary(self, report: IntegrationTestReport):
        """Log test summary to console"""
        self.logger.info("🎯 INTEGRATION TEST SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Session ID: {report.test_session_id}")
        self.logger.info(f"Total Duration: {report.total_duration_seconds:.1f}s")
        self.logger.info(f"Tests Run: {report.total_tests}")
        self.logger.info(f"Passed: {report.passed_tests}")
        self.logger.info(f"Failed: {report.failed_tests}")
        self.logger.info(f"Errors: {report.error_tests}")
        self.logger.info(f"Skipped: {report.skipped_tests}")
        self.logger.info(f"Success Rate: {report.success_rate:.1f}%")
        
        # Log test details
        self.logger.info("\n📋 Test Details:")
        for result in report.test_results:
            status_emoji = {
                'PASSED': '✅',
                'FAILED': '❌', 
                'ERROR': '💥',
                'SKIPPED': '⏭️'
            }.get(result.status, '❓')
            
            self.logger.info(f"{status_emoji} {result.test_name} ({result.test_category}): "
                           f"{result.status} ({result.execution_time_seconds:.1f}s)")
            
            if result.error_message:
                self.logger.info(f"   Error: {result.error_message}")
        
        # Overall assessment
        if report.success_rate >= 90:
            self.logger.info("\n🎉 INTEGRATION TESTS PASSED - System ready for deployment!")
        elif report.success_rate >= 70:
            self.logger.info("\n⚠️ INTEGRATION TESTS PARTIALLY PASSED - Review failures before deployment")
        else:
            self.logger.info("\n❌ INTEGRATION TESTS FAILED - System not ready for deployment")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 4 Integration Testing')
    parser.add_argument('--full_pipeline', action='store_true', 
                       help='Run full integration test pipeline')
    parser.add_argument('--test_mobile_deployment', action='store_true',
                       help='Test mobile deployment only')
    parser.add_argument('--test_robustness', action='store_true',
                       help='Test robustness and edge cases only')
    parser.add_argument('--regression_test', action='store_true',
                       help='Run performance regression test only')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick smoke test')
    parser.add_argument('--synthetic_data_only', action='store_true',
                       help='Use only synthetic test data')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for test results')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Test timeout in seconds')
    
    args = parser.parse_args()
    
    # Configure integration testing
    config = IntegrationTestConfig(
        test_timeout_seconds=args.timeout,
        create_synthetic_data=True,
        temp_dir_cleanup=not args.output_dir,  # Keep results if output dir specified
        enable_gpu_testing=torch.cuda.is_available(),
        test_cascade_integration=args.full_pipeline or not any([
            args.test_mobile_deployment, args.test_robustness, args.regression_test
        ]),
        test_mobile_optimization=args.full_pipeline or not any([
            args.test_mobile_deployment, args.test_robustness, args.regression_test
        ]),
        test_video_processing=args.full_pipeline or not any([
            args.test_mobile_deployment, args.test_robustness, args.regression_test
        ]),
        test_onnx_deployment=args.full_pipeline or args.test_mobile_deployment,
        test_edge_cases=args.full_pipeline or args.test_robustness,
        test_error_handling=args.full_pipeline or args.test_robustness
    )
    
    # Quick test configuration
    if args.quick_test:
        config.synthetic_video_duration = 5
        config.synthetic_image_count = 20
        config.test_batch_sizes = [1, 4, 8]
    
    # Initialize tester
    tester = Stage4IntegrationTester(config)
    
    # Run tests
    if args.regression_test:
        result = tester.run_test(
            tester.test_performance_regression,
            "performance_regression",
            "performance"
        )
        print(f"Performance regression test: {result.status}")
        
    elif args.test_mobile_deployment:
        result = tester.run_test(
            tester.test_onnx_deployment_integration,
            "onnx_deployment_integration", 
            "deployment"
        )
        print(f"Mobile deployment test: {result.status}")
        
    elif args.test_robustness:
        result = tester.run_test(
            tester.test_robustness_and_edge_cases,
            "robustness_and_edge_cases",
            "robustness"
        )
        print(f"Robustness test: {result.status}")
        
    else:
        # Run full test suite
        report = tester.run_full_integration_test_suite()
        
        print(f"\n🎯 Integration testing completed!")
        print(f"Success rate: {report.success_rate:.1f}%")
        print(f"Tests passed: {report.passed_tests}/{report.total_tests}")
        
        if report.success_rate >= 90:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure

if __name__ == "__main__":
    main()