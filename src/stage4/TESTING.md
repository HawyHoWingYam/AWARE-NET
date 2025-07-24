# Stage 4 Mobile Optimization Testing Guide

## Overview

This document provides comprehensive testing instructions for the Stage 4 mobile optimization pipeline, including QAT + Knowledge Distillation and ONNX export functionality.

## Test Components

### 1. Core Pipeline Testing (`test_mobile_optimization.py`)

**Quick Testing (Recommended for development):**
```bash
# Run all core tests (5-10 minutes)
python src/stage4/test_mobile_optimization.py --test_all

# Test only ONNX export functionality 
python src/stage4/test_mobile_optimization.py --test_onnx_export

# Quick component tests
python src/stage4/test_mobile_optimization.py --quick
```

**Test Coverage:**
- ✅ QAT Configuration validation
- ✅ MobileOptimizer initialization
- ✅ Model size calculation accuracy
- ✅ Teacher model loading (with/without checkpoints)
- ✅ Student model creation and QAT preparation
- ✅ Distillation loss calculation
- ✅ ONNX export and validation
- ⏳ Quick optimization pipeline (1 epoch test)

### 2. Full Optimization Testing

**Prerequisites:**
```bash
# Ensure you have trained models available
ls output/stage1/best_model.pth  # Stage 1 checkpoint
ls output/stage2_*/best_model.pth  # Stage 2 checkpoints (if available)

# Or use pretrained weights (automatic fallback)
```

**Full Pipeline Test:**
```bash
# Test single model optimization (15-30 minutes)
python src/stage4/optimize_for_mobile.py --model stage1 --epochs 3 --batch_size 16

# Test all models (45-90 minutes)
python src/stage4/optimize_for_mobile.py --model all --epochs 5
```

### 3. ONNX Export Testing

**Individual Model Export:**
```bash
# Export single PyTorch model to ONNX
python src/stage4/mobile_deployment/onnx_exporter.py \
    --model_path output/stage1/best_model.pth \
    --output_path test_export.onnx \
    --model_name "stage1_test"
```

**Bundle Export Test:**
```python
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter
import torch
import timm

# Create test models
models = {
    'stage1': timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k', num_classes=1),
    'stage2': timm.create_model('efficientnetv2_b3.in21k_ft_in1k', num_classes=1)
}

# Export bundle
exporter = ONNXExporter()
result = exporter.export_cascade_bundle(
    models=models,
    output_dir="test_bundle",
    bundle_name="test_cascade"
)

print(f"Bundle export: {'SUCCESS' if result['success'] else 'FAILED'}")
```

## Expected Test Results

### Performance Benchmarks

**QAT Optimization Targets:**
- Model Size Reduction: **>75%** (FP32 → INT8)
- Inference Speedup: **>3x** on mobile hardware
- Accuracy Preservation: **<2%** AUC degradation
- Memory Usage: **<512MB** total footprint

**ONNX Export Validation:**
- Output Matching: **Max difference <1e-5** between PyTorch and ONNX
- Model Size: **Reasonable compression** with optimization
- Format Compatibility: **Successfully loads** in ONNX Runtime

### Test Success Criteria

**Component Tests (test_mobile_optimization.py):**
```
✅ QAT Configuration: Configuration validation passes
✅ MobileOptimizer Init: Proper initialization with device detection
✅ Model Size Calculation: Accurate size calculation in MB
✅ Teacher Model Loading: Loads pretrained or checkpoint weights
✅ Student Model Creation: QAT preparation successful
✅ Distillation Loss: Loss components calculated correctly
✅ ONNX Export: Model exports with validation passing
```

**Expected Success Rate:** **>80%** for deployment readiness

### Common Issues and Solutions

**1. CUDA/Device Issues:**
```bash
# Error: CUDA not available
# Solution: Tests will automatically fall back to CPU

# Error: GPU memory insufficient
# Solution: Reduce batch_size in config
python test_mobile_optimization.py --batch_size 8
```

**2. Missing Dependencies:**
```bash
# Error: No module named 'onnx'
pip install onnx onnxruntime onnx-simplifier

# Error: timm model not found
pip install timm>=0.9.0
```

**3. Model Loading Issues:**
```bash
# Error: Checkpoint not found
# Solution: Tests use pretrained weights as fallback
# This is expected behavior for testing

# Error: Model architecture mismatch
# Solution: Ensure consistent model creation across pipeline
```

**4. Quantization Backend Issues:**
```bash
# Error: fbgemm backend not available
# Solution: Install PyTorch with proper backend support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Debugging and Troubleshooting

### Verbose Testing

```bash
# Enable detailed logging
python src/stage4/test_mobile_optimization.py --test_all --verbose

# Check individual test results
cat /tmp/mobile_opt_test_*/test_report.json
```

### Manual Testing Steps

**1. Basic Functionality Test:**
```python
from src.stage4.optimize_for_mobile import MobileOptimizer, QATConfig

# Test basic initialization
config = QATConfig(epochs=1, batch_size=4)
optimizer = MobileOptimizer(config)
print(f"Device: {optimizer.device}")
print(f"Backend: {optimizer.config.quantization_backend}")
```

**2. Model Creation Test:**
```python
import timm
import torch

# Test model creation
model = timm.create_model('mobilenetv4_hybrid_medium.ix_e550_r256_in1k', num_classes=1)
input_tensor = torch.randn(1, 3, 256, 256)

# Test forward pass
with torch.no_grad():
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
```

**3. ONNX Validation Test:**
```python
import onnxruntime as ort

# Test ONNX loading
session = ort.InferenceSession("test_model.onnx")
input_name = session.get_inputs()[0].name
print(f"ONNX input: {input_name}")
print(f"Providers: {session.get_providers()}")
```

## Performance Monitoring

### Resource Usage

Monitor system resources during testing:
```bash
# GPU monitoring
nvidia-smi -l 1

# CPU and memory monitoring  
htop

# Disk usage monitoring
df -h /tmp
```

### Test Timing

Expected test durations:
- **Component Tests**: 5-10 minutes
- **Single Model QAT**: 15-30 minutes (depends on epochs)
- **Full Pipeline**: 45-90 minutes (all models)
- **ONNX Export Only**: 2-5 minutes

### Quality Assurance Checklist

**Before Production Use:**
- [ ] All component tests pass (>80% success rate)
- [ ] At least one full optimization completes successfully
- [ ] ONNX export validation passes with <1e-5 difference
- [ ] Models meet size reduction targets (>75%)
- [ ] Inference speed improvements verified (>3x)
- [ ] Memory usage within acceptable limits (<512MB)
- [ ] No critical errors in logs
- [ ] Deployment package generation successful

## Continuous Integration

### Automated Testing Setup

**GitHub Actions Example (.github/workflows/test-mobile-opt.yml):**
```yaml
name: Mobile Optimization Tests

on: [push, pull_request]

jobs:
  test-mobile-optimization:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install onnx onnxruntime onnx-simplifier
      - name: Run mobile optimization tests
        run: |
          python src/stage4/test_mobile_optimization.py --test_all --quick
      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: /tmp/mobile_opt_test_*/test_report.json
```

### Test Data Management

**For CI/CD:**
- Use synthetic data for component tests
- Keep test datasets small (<100MB)
- Cache model weights to reduce download time
- Use CPU-only testing for basic validation

**For Full Validation:**
- Use representative dataset samples
- Test on actual mobile hardware when available
- Validate end-to-end deployment workflows
- Monitor performance regression over time

## Next Steps

After testing passes:
1. **Deploy models** to target mobile platforms
2. **Run integration tests** with real applications
3. **Monitor performance** in production environments
4. **Collect feedback** for optimization improvements
5. **Update benchmarks** based on real-world usage

For issues or questions:
- Check logs in temp directories
- Review error messages carefully
- Ensure all dependencies are properly installed
- Test with smaller configurations first
- Contact development team with detailed error reports