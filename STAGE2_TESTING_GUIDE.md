# AWARE-NET Stage 2 Testing Guide

This guide provides comprehensive testing instructions for Phase 2 components: EfficientNetV2-B3 and GenConViT dual-mode system.

## 🚀 Quick Start Testing

### Prerequisites
```bash
# Ensure you have processed data
ls processed_data/  # Should show: train/ val/ manifests/

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1. Complete Test Suite
```bash
# Run all tests with minimal epochs for quick validation
python src/stage2/test_stage2.py --test all --epochs 2

# Run with more epochs for better validation
python src/stage2/test_stage2.py --test all --epochs 5
```

### 2. Individual Component Testing

#### Test EfficientNetV2-B3
```bash
# Quick test (2 epochs)
python src/stage2/test_stage2.py --test effnet --epochs 2

# Direct training test
python src/stage2/train_stage2_effnet.py --epochs 5 --batch_size 8
```

#### Test GenConViT Hybrid Mode
```bash
# Test hybrid mode
python src/stage2/test_stage2.py --test genconvit --mode hybrid --epochs 2

# Direct hybrid training
python src/stage2/train_stage2_genconvit.py --mode hybrid --variant ED --epochs 5
```

#### Test GenConViT Pretrained Mode
```bash
# Test pretrained mode (requires internet for model download)
python src/stage2/test_stage2.py --test genconvit --mode pretrained --epochs 2

# Direct pretrained training
python src/stage2/train_stage2_genconvit.py --mode pretrained --variant ED --epochs 5
```

### 3. Advanced Testing

#### Test Mode Switching
```bash
# Test dual-mode switching capability
python src/stage2/test_stage2.py --test switching
```

#### Test Feature Extraction (Stage 3 compatibility)
```bash
# Test feature extraction for meta-model integration
python src/stage2/test_stage2.py --test features
```

#### Performance Comparison
```bash
# Compare all models
python src/stage2/test_stage2.py --test compare --epochs 5
```

## 📊 Expected Test Results

### Data Availability Check
```
✅ train: X real, Y fake samples
✅ val: X real, Y fake samples  
✅ processed_data/manifests exists
```

### EfficientNetV2-B3 Training
```
✅ Model setup complete - ~6M parameters
📊 Best AUC: 0.85+ (expected for short training)
⏱️  Training time: ~60-120s per epoch
```

### GenConViT Hybrid Mode
```
✅ Model setup complete - ~2M parameters  
📊 Best AUC: 0.70+ (expected for short training)
⏱️  Training time: ~90-150s per epoch
```

### GenConViT Pretrained Mode
```
✅ Pretrained weights downloaded successfully
📊 Best AUC: 0.80+ (expected with pretrained weights)
⏱️  Training time: ~90-150s per epoch
```

### Mode Switching
```
✅ Manager created in auto mode
✅ Mode switching successful: auto → hybrid
✅ Model inference successful
📊 Output shapes: classification (1, 1), reconstruction (1, 3, 224, 224)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or test from project root
cd /workspace/AWARE-NET
python src/stage2/test_stage2.py --test all
```

#### 2. CUDA Out of Memory
```bash
# Reduce batch size
python src/stage2/test_stage2.py --test all --epochs 2
# Models use smaller batch sizes for testing (4-8)
```

#### 3. Data Directory Issues
```bash
# Check data structure
ls -la processed_data/
# Expected: train/real/, train/fake/, val/real/, val/fake/, manifests/

# Specify custom data directory
python src/stage2/test_stage2.py --test all --data_dir /path/to/your/data
```

#### 4. Missing Dependencies
```bash
# Install missing packages
pip install timm scikit-learn tqdm

# For pretrained mode (optional)
pip install huggingface_hub safetensors
```

#### 5. Pretrained Model Download Issues
```bash
# Test without pretrained mode
python src/stage2/test_stage2.py --test genconvit --mode hybrid

# Check internet connection for HuggingFace downloads
```

## 📈 Performance Benchmarks

### Expected Training Performance (5 epochs)

| Model | AUC Range | Training Time | Parameters |
|-------|-----------|---------------|------------|
| EfficientNetV2-B3 | 0.90-0.98 | 5-8 min | ~6M |
| GenConViT Hybrid | 0.75-0.90 | 7-10 min | ~2M |
| GenConViT Pretrained | 0.85-0.95 | 7-10 min | ~2M |

### Memory Usage (Batch Size 16)
- EfficientNetV2-B3: ~4-6GB VRAM
- GenConViT Hybrid: ~3-5GB VRAM  
- GenConViT Pretrained: ~3-5GB VRAM

## 🎯 Validation Criteria

### ✅ Test Passes If:
1. **Data availability**: All required directories exist with samples
2. **Model creation**: Models initialize without errors
3. **Training**: Loss decreases and AUC improves over epochs
4. **Feature extraction**: Models output correct feature dimensions
5. **Mode switching**: GenConViT can switch between hybrid/pretrained
6. **Integration**: Models are compatible with AWARE-NET framework

### ❌ Test Fails If:
1. Missing data directories or samples
2. Model initialization errors
3. Training crashes or produces NaN values
4. Incorrect feature dimensions for Stage 3
5. Mode switching throws exceptions
6. Performance significantly below expectations

## 🔍 Detailed Diagnostics

### Manual Model Testing
```python
# Test individual components
import torch
from src.stage2.genconvit_manager import GenConViTManager

# Create manager
manager = GenConViTManager(mode='auto', variant='ED')
print(f"Active mode: {manager.mode.value}")

# Create model
model = manager.create_model()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Test inference
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shapes: {output.classification.shape}, {output.reconstruction.shape}")

# Test feature extraction
features = model.extract_features(dummy_input)
print(f"Feature keys: {list(features.keys())}")
print(f"Final features shape: {features['final_features'].shape}")
```

### Debug Training Issues
```python
# Check data loading
from src.stage1.dataset import create_dataloaders
train_loader, val_loader, _ = create_dataloaders('processed_data', batch_size=4)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Test single batch
for images, labels in train_loader:
    print(f"Batch shapes: images {images.shape}, labels {labels.shape}")
    print(f"Label distribution: {labels.sum().item()}/{len(labels)} positive")
    break
```

## 📋 Test Report Analysis

After running tests, check `stage2_test_results.json` for detailed results:

```json
{
  "data_availability": {"success": true},
  "effnet": {
    "status": "success",
    "best_auc": 0.8234,
    "training_time": 145.2
  },
  "genconvit_hybrid": {
    "status": "success", 
    "best_auc": 0.7891,
    "mode": "hybrid"
  },
  "genconvit_pretrained": {
    "status": "success",
    "best_auc": 0.8567,
    "mode": "pretrained"
  },
  "mode_switching": {
    "status": "success",
    "switching_success": true
  }
}
```

## 🚀 Next Steps After Testing

1. **If all tests pass**: Proceed to Stage 3 meta-model integration
2. **If EfficientNet fails**: Check timm installation and model compatibility
3. **If GenConViT hybrid fails**: Review custom implementation and dependencies
4. **If GenConViT pretrained fails**: Check internet connection and HuggingFace access
5. **If switching fails**: Review manager implementation and mode compatibility

## 💡 Tips for Successful Testing

1. **Start small**: Use `--epochs 2` for initial validation
2. **Monitor GPU memory**: Reduce batch sizes if needed
3. **Check logs**: Look for detailed error messages in output
4. **Test incrementally**: Run individual tests before full suite
5. **Verify data**: Ensure processed_data has sufficient samples
6. **Use appropriate hardware**: GPU recommended for reasonable training times

---

**Remember**: These are validation tests with minimal epochs. For production use, train for 20-50 epochs depending on dataset size and convergence.