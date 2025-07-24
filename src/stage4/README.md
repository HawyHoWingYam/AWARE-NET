# Stage 4: Mobile Optimization & Deployment

## Overview

Stage 4 implements mobile optimization and deployment capabilities for the AWARE-NET cascade deepfake detection system. This stage transforms the research prototype into a production-ready mobile application through quantization, knowledge distillation, and comprehensive performance optimization.

## 🎯 Core Objectives

- **Mobile Model Optimization**: >75% size reduction, >3x speedup, <2% accuracy loss
- **Deployment Package Creation**: ONNX/TensorFlow Lite export with validation
- **Performance Benchmarking**: Comprehensive testing across devices and datasets
- **Production Readiness**: End-to-end mobile deployment framework

## 📁 Directory Structure

```
src/stage4/
├── cascade_detector.py              # ✅ Unified cascade detection system
├── optimize_for_mobile.py           # ✅ QAT + Knowledge Distillation pipeline
├── benchmark_cascade.py             # ✅ Performance benchmarking system
├── test_mobile_optimization.py      # ✅ Testing framework
├── video_processor.py               # 🔄 Advanced video processing (Next)
├── test_stage4_integration.py       # 🔄 Integration testing (Next)
├── mobile_deployment/               # ✅ Deployment utilities
│   ├── __init__.py                  # Package initialization
│   ├── onnx_exporter.py            # ONNX export with validation
│   ├── tflite_converter.py         # TensorFlow Lite conversion (Next)
│   ├── mobile_inference.py         # Mobile inference wrapper (Next)
│   └── deployment_validator.py     # Deployment testing (Next)
├── configs/                         # Configuration templates
│   ├── qat_config.json             # QAT training parameters
│   ├── benchmark_config.json       # Benchmarking settings
│   └── deployment_config.json      # Deployment parameters
├── TESTING.md                       # ✅ Comprehensive testing guide
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Mobile Optimization Pipeline

**Optimize all cascade models:**
```bash
# Full optimization (45-90 minutes)
python src/stage4/optimize_for_mobile.py --model all --epochs 10

# Quick test optimization (15-30 minutes)
python src/stage4/optimize_for_mobile.py --model stage1 --epochs 3 --batch_size 16
```

**Configuration options:**
```bash
# Custom configuration
python src/stage4/optimize_for_mobile.py \
    --model all \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --temperature 4.0
```

### 2. Performance Benchmarking

**Full benchmark suite:**
```bash
# Complete benchmarking (30-60 minutes)
python src/stage4/benchmark_cascade.py --benchmark_all

# Specific benchmarks
python src/stage4/benchmark_cascade.py --accuracy_only
python src/stage4/benchmark_cascade.py --speed_only --device cpu
python src/stage4/benchmark_cascade.py --mobile_simulation
```

### 3. Testing & Validation

**Component testing:**
```bash
# Quick component tests (5-10 minutes)
python src/stage4/test_mobile_optimization.py --test_all

# Specific tests
python src/stage4/test_mobile_optimization.py --test_onnx_export
```

**Integration testing:**
```bash
# End-to-end validation (when available)
python src/stage4/test_stage4_integration.py --full_pipeline
```

## 📊 Performance Targets & Results

### Optimization Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Model Size Reduction** | >75% | FP32 → INT8 quantization |
| **Inference Speedup** | >3x | Mobile hardware performance |
| **Accuracy Preservation** | <2% degradation | Knowledge distillation effectiveness |
| **Memory Footprint** | <512MB | Mobile deployment constraints |

### Expected Results

**Model Sizes (After Optimization):**
- Stage 1 (MobileNetV4): ~8MB → ~2MB
- Stage 2 (EfficientNetV2): ~45MB → ~11MB  
- Stage 2 (GenConViT): ~35MB → ~9MB
- **Total System**: ~88MB → ~22MB

**Performance Improvements:**
- **Inference Speed**: 150ms → 50ms (3x speedup)
- **Throughput**: 6.7 FPS → 20 FPS
- **Memory Usage**: 1.8GB → 450MB

## 🔧 Core Components

### 1. Mobile Optimization (`optimize_for_mobile.py`)

**Quantization-Aware Training + Knowledge Distillation pipeline:**

```python
from src.stage4.optimize_for_mobile import MobileOptimizer, QATConfig

# Configure optimization
config = QATConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-5,
    distillation_temperature=4.0,
    alpha_hard_loss=0.3,
    alpha_soft_loss=0.7
)

# Run optimization
optimizer = MobileOptimizer(config)
results = optimizer.optimize_all_models()
```

**Key Features:**
- **Teacher-Student Training**: FP32 teachers → INT8 students
- **Combined Loss Function**: Hard targets + soft knowledge distillation
- **Model-Specific Optimization**: Custom strategies for each cascade stage
- **Target Achievement Tracking**: Automatic validation of optimization goals

### 2. ONNX Export System (`mobile_deployment/onnx_exporter.py`)

**Cross-platform model deployment:**

```python
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter

# Export single model
exporter = ONNXExporter()
result = exporter.export_model(
    model=trained_model,
    output_path="cascade_stage1.onnx",
    input_shape=(1, 3, 256, 256)
)

# Export complete cascade bundle
bundle_result = exporter.export_cascade_bundle(
    models={'stage1': model1, 'stage2': model2},
    output_dir="deployment_bundle",
    bundle_name="cascade_detector"
)
```

**Key Features:**
- **Validation**: Automatic PyTorch vs ONNX accuracy verification
- **Optimization**: Model simplification and mobile-specific tuning
- **Bundle Creation**: Complete deployment packages with inference scripts
- **Metadata Embedding**: Model information and deployment instructions

### 3. Performance Benchmarking (`benchmark_cascade.py`)

**Comprehensive performance analysis:**

```python
from src.stage4.benchmark_cascade import CascadeBenchmarker, BenchmarkConfig

# Configure benchmarking
config = BenchmarkConfig(
    test_devices=['cuda', 'cpu'],
    num_test_samples=1000,
    save_plots=True
)

# Run benchmarks
benchmarker = CascadeBenchmarker(config)
report = benchmarker.run_full_benchmark()
```

**Benchmark Coverage:**
- **Accuracy Testing**: AUC, F1-Score, Precision, Recall across datasets
- **Speed Analysis**: Inference time, throughput, memory profiling
- **Cascade Efficiency**: Stage filtration rates, leakage analysis
- **Mobile Compatibility**: Size validation, export testing, deployment readiness

### 4. Testing Framework (`test_mobile_optimization.py`)

**Automated testing and validation:**

```python
from src.stage4.test_mobile_optimization import MobileOptimizationTester

# Run component tests
tester = MobileOptimizationTester()
results = tester.run_all_tests()

# Expected success rate: >80% for deployment readiness
```

**Test Coverage:**
- **Configuration Validation**: QAT parameters, device compatibility
- **Model Operations**: Loading, quantization preparation, size calculation
- **Loss Functions**: Distillation loss computation and gradients
- **Export Functionality**: ONNX conversion and validation

## 📈 Usage Examples

### Complete Mobile Optimization Workflow

```bash
# 1. Optimize models for mobile deployment
python src/stage4/optimize_for_mobile.py --model all --epochs 10

# 2. Export to ONNX format
python -c "
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter
import torch
model = torch.load('output/stage4/optimized_models/stage1_quantized_model.pth')
exporter = ONNXExporter()
exporter.export_model(model, 'stage1_mobile.onnx', model_name='stage1_mobile')
"

# 3. Benchmark performance
python src/stage4/benchmark_cascade.py --benchmark_all

# 4. Validate deployment readiness
python src/stage4/test_mobile_optimization.py --test_all
```

### Custom Configuration Workflow

```python
# Custom QAT configuration
from src.stage4.optimize_for_mobile import QATConfig, MobileOptimizer

config = QATConfig(
    epochs=5,                          # Shorter training for testing
    batch_size=16,                     # Reduce for memory constraints
    distillation_temperature=3.0,      # Sharper knowledge transfer
    calibration_dataset_size=500,      # Smaller calibration set
    output_dir="custom_optimization"
)

optimizer = MobileOptimizer(config)
result = optimizer.optimize_model(OptimizationTarget.STAGE1)

print(f"Size reduction: {result.size_reduction_percent:.1f}%")
print(f"Speedup: {result.inference_speedup:.2f}x")
print(f"Accuracy loss: {result.accuracy_degradation_percent:.2f}%")
```

### Deployment Package Creation

```python
# Create complete deployment package
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter
import torch

# Load optimized models
models = {
    'stage1': torch.load('output/stage4/optimized_models/stage1_quantized_model.pth'),
    'stage2_effnet': torch.load('output/stage4/optimized_models/stage2_effnet_quantized_model.pth'),
    'stage2_genconvit': torch.load('output/stage4/optimized_models/stage2_genconvit_quantized_model.pth')
}

# Export bundle
exporter = ONNXExporter()
bundle_result = exporter.export_cascade_bundle(
    models=models,
    output_dir="mobile_deployment_package",
    bundle_name="aware_net_mobile"
)

# Results in:
# mobile_deployment_package/
# ├── aware_net_mobile_stage1.onnx
# ├── aware_net_mobile_stage2_effnet.onnx  
# ├── aware_net_mobile_stage2_genconvit.onnx
# ├── aware_net_mobile_manifest.json
# ├── deploy_aware_net_mobile.py
# ├── requirements.txt
# └── README.md
```

## 🛠️ Configuration

### QAT Configuration (`configs/qat_config.json`)

```json
{
  "epochs": 10,
  "batch_size": 32,
  "learning_rate": 1e-5,
  "weight_decay": 1e-5,
  "distillation_temperature": 4.0,
  "alpha_hard_loss": 0.3,
  "alpha_soft_loss": 0.7,
  "quantization_backend": "fbgemm",
  "quantization_scheme": "asymmetric",
  "calibration_dataset_size": 1000
}
```

### Benchmark Configuration (`configs/benchmark_config.json`)

```json
{
  "test_batch_size": 32,
  "num_test_samples": 1000,
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "test_devices": ["cuda", "cpu"],
  "memory_profiling": true,
  "save_plots": true,
  "datasets_to_test": ["CelebDF", "FF++", "DFDC"]
}
```

## 📋 Quality Assurance Checklist

Before production deployment:

**✅ Optimization Validation:**
- [ ] All models achieve >75% size reduction
- [ ] Inference speedup >3x verified
- [ ] Accuracy degradation <2% confirmed
- [ ] Memory usage <512MB validated

**✅ Export Validation:**
- [ ] ONNX models export successfully
- [ ] PyTorch vs ONNX output difference <1e-5
- [ ] Deployment packages generated correctly
- [ ] Cross-platform compatibility verified

**✅ Performance Validation:**
- [ ] Benchmarks complete without errors
- [ ] Speed targets met on target devices
- [ ] Memory profiles within constraints
- [ ] Cascade efficiency metrics acceptable

**✅ Testing Validation:**
- [ ] Component tests pass (>80% success rate)
- [ ] Integration tests complete successfully
- [ ] Edge cases handled properly
- [ ] Error recovery mechanisms validated

## 🔍 Troubleshooting

### Common Issues

**1. Quantization Backend Issues:**
```bash
# Error: fbgemm not available
# Solution: Install proper PyTorch build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**2. CUDA Memory Issues:**
```bash
# Error: CUDA out of memory
# Solution: Reduce batch size
python optimize_for_mobile.py --batch_size 16
```

**3. Model Loading Issues:**
```bash
# Error: Model checkpoint not found
# Solution: Ensure Stage 1-3 models are trained
ls output/stage1/best_model.pth
ls output/stage2*/best_model.pth
```

**4. ONNX Export Issues:**
```bash
# Error: ONNX export failed
# Solution: Install dependencies
pip install onnx onnxruntime onnx-simplifier
```

### Performance Debugging

**Memory Profiling:**
```python
from src.stage4.benchmark_cascade import MemoryProfiler

with MemoryProfiler('cuda') as profiler:
    # Run inference
    model(input_tensor)

print(f"Peak memory: {profiler.get_peak_memory():.1f} MB")
```

**Speed Profiling:**
```python
import time
import torch

model.eval()
with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = model(test_input)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = model(test_input)
    torch.cuda.synchronize()  # If using CUDA
    
    avg_time = (time.time() - start_time) / 100 * 1000  # ms
    print(f"Average inference time: {avg_time:.2f} ms")
```

## 📊 Results Analysis

### Optimization Report Structure

```json
{
  "optimization_summary": {
    "total_models": 3,
    "successful_optimizations": 3,
    "overall_size_reduction_percent": 77.2,
    "average_accuracy_degradation_percent": 1.3,
    "average_inference_speedup": 3.4
  },
  "individual_results": [
    {
      "model_name": "stage1",
      "size_reduction_percent": 78.5,
      "accuracy_degradation_percent": 0.8,
      "inference_speedup": 3.2,
      "success": true
    }
  ]
}
```

### Benchmark Report Structure

```json
{
  "summary": {
    "accuracy": {"avg_auc": 0.954, "best_auc": 0.967},
    "speed": {"avg_inference_time_ms": 52.3, "fastest_inference_ms": 45.1}
  },
  "recommendations": [
    "✅ All models are mobile-ready for deployment",
    "🏆 Best accuracy-speed tradeoff: optimized"
  ]
}
```

## 🚀 Next Steps

After Stage 4 mobile optimization:

1. **Advanced Video Processing**: Real-time video analysis with temporal consistency
2. **Integration Testing**: End-to-end deployment validation
3. **Mobile App Integration**: Native mobile application development
4. **Performance Monitoring**: Production deployment monitoring
5. **Continuous Optimization**: Ongoing model improvement and updates

## 📞 Support

For issues or questions:
- Review **TESTING.md** for detailed testing procedures
- Check component tests: `python test_mobile_optimization.py --test_all`
- Validate configurations and model paths
- Monitor system resources during optimization
- Report issues with detailed error logs and system information