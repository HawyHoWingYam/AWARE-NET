# Optimize Stage 2 Command

Analyze and optimize Stage 2 model performance (EfficientNet/GenConViT).

## Parameters
- $1: Model type ("effnet", "genconvit", or "both") - required
- $2: Optimization target ("speed", "accuracy", "memory", or "balanced") - optional, defaults to "balanced"
- $3: Hardware constraint ("mobile", "desktop", "server") - optional, defaults to "desktop"

## Execution Steps

### 1. Current Performance Analysis
```bash
# Run current performance benchmarks
python src/stage2/test_stage2.py --test compare --epochs 3

# Profile memory usage
python -m memory_profiler src/stage2/train_stage2_genconvit.py --epochs 1

# Check GPU utilization
nvidia-smi dmon -s pmu -c 10
```

### 2. Model Analysis
- Analyze current model architecture
- Identify performance bottlenecks
- Review training configuration
- Check data loading efficiency
- Assess memory usage patterns

### 3. Optimization Strategy Selection

#### For EfficientNet:
- Model pruning for mobile deployment
- Knowledge distillation
- Quantization (INT8/FP16)
- Batch size optimization
- Learning rate scheduling

#### For GenConViT:
- Mode switching (hybrid vs pretrained)
- Architecture component optimization
- Multi-component loss balancing
- Feature extraction efficiency
- Autoencoder compression ratio

### 4. Implementation Planning
Based on target optimization:

#### Speed Optimization:
- Mixed precision training (AMP)
- DataLoader optimization (num_workers, pin_memory)
- Model architecture pruning
- Inference optimization

#### Accuracy Optimization:
- Advanced data augmentation
- Loss function improvements
- Ensemble techniques
- Hyperparameter tuning

#### Memory Optimization:
- Gradient checkpointing
- Model parallelism
- Batch size reduction
- Memory-efficient training

#### Balanced Optimization:
- Pareto optimal solutions
- Multi-objective optimization
- Trade-off analysis

### 5. Configuration Updates
- Update training scripts with optimized parameters
- Modify model configurations
- Adjust data preprocessing pipeline
- Update evaluation metrics

### 6. Performance Validation
```bash
# Test optimized configuration
python src/stage2/train_stage2_${MODEL_TYPE}.py --config optimized_config.json --epochs 5

# Compare before/after performance
python src/stage2/test_stage2.py --test compare --epochs 5

# Validate on different hardware
python benchmark_model.py --model stage2_${MODEL_TYPE} --hardware ${HARDWARE}
```

### 7. Documentation Updates
- Update performance benchmarks in documentation
- Add optimization notes to CLAUDE.md
- Update training guides with new best practices
- Create optimization troubleshooting guide

## Expected Outcomes

### Speed Optimization:
- 20-40% reduction in training time
- 30-50% reduction in inference time
- Maintained or improved accuracy

### Accuracy Optimization:
- 2-5% improvement in AUC score
- Better generalization across datasets
- Robust performance metrics

### Memory Optimization:
- 25-40% reduction in VRAM usage
- Support for larger batch sizes
- Mobile deployment feasibility

### Balanced Optimization:
- Optimal trade-offs across all metrics
- Production-ready performance
- Scalable training pipeline

## Usage Examples
```bash
# Optimize EfficientNet for speed
@claude /optimize-stage2 effnet speed mobile

# Optimize GenConViT for accuracy
@claude /optimize-stage2 genconvit accuracy desktop

# Balanced optimization for both models
@claude /optimize-stage2 both balanced server
```

## Success Criteria
- [ ] Current performance baseline established
- [ ] Optimization strategy clearly defined
- [ ] Implementation completed without breaking changes
- [ ] Performance improvements validated
- [ ] Documentation updated with new benchmarks
- [ ] Optimization is reproducible
- [ ] No regression in model accuracy (unless explicitly trading off)

## Related Files
- `src/stage2/train_stage2_effnet.py`
- `src/stage2/train_stage2_genconvit.py`
- `src/stage2/genconvit_manager.py`
- `src/stage2/test_stage2.py`
- `STAGE2_TESTING_GUIDE.md`