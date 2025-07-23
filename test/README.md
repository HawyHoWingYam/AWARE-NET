# AWARE-NET Test Suite

This directory contains test scripts for validating the functionality of AWARE-NET components.

## Test Scripts

### `test_train_stage1.py`
Tests the Stage 1 fast filter training pipeline with a small subset of data.

**Features:**
- Creates small test manifests (50 samples per class)
- Verifies data accessibility
- Runs quick training test (2 epochs, batch size 4)
- Validates output generation
- Dependency checking

**Usage:**
```bash
# Run from project root
python test/test_train_stage1.py
```

**What it tests:**
- Data loading pipeline
- Model creation and initialization
- Training and validation loops
- Metrics calculation
- Model saving and checkpointing
- Output file generation

**Expected outputs:**
```
test/
├── manifests/
│   ├── test_train_manifest.csv
│   └── test_val_manifest.csv
└── output/
    └── stage1_test/
        ├── best_model.pth
        ├── training_curves.png
        ├── training_history.json
        └── training_*.log
```

## Running Tests

### Prerequisites
Ensure all dependencies are installed:
- torch, torchvision
- timm
- pandas, numpy
- PIL, sklearn
- matplotlib, tqdm

### Quick Test
```bash
python test/test_train_stage1.py
```

### What to expect
1. **Dependency Check** - Verifies all required packages
2. **Manifest Creation** - Creates small test datasets
3. **Data Verification** - Checks if image files are accessible
4. **Training Test** - Runs 2 epochs with small batch size
5. **Output Validation** - Confirms all expected files are created

### Success Criteria
- All dependencies available ✅
- Test manifests created successfully ✅
- >80% of test images accessible ✅
- Training completes without errors ✅
- All expected output files generated ✅

### Troubleshooting

**Common Issues:**

1. **Missing Dependencies**
   ```bash
   pip install torch torchvision timm pandas scikit-learn matplotlib tqdm pillow
   ```

2. **CUDA Issues**
   - Test runs on CPU if CUDA unavailable
   - Check GPU memory if training fails

3. **Data Path Issues**
   - Verify processed_data/ directory exists
   - Check manifest files contain valid paths

4. **Memory Issues**
   - Reduce batch_size in test script
   - Reduce samples_per_class in create_test_manifests()

### Performance Expectations
- **Test Duration**: 2-5 minutes on GPU, 10-15 minutes on CPU
- **Memory Usage**: ~1-2GB GPU memory, ~2-4GB RAM
- **Disk Space**: ~100MB for test outputs

## Next Steps After Testing

If tests pass:
1. Run full training: `python src/stage1/train_stage1.py`
2. Monitor training progress in `output/stage1/`
3. Proceed to Task 1.2 (calibration) after training completes

If tests fail:
1. Check error messages in test output
2. Verify data integrity and paths
3. Install missing dependencies
4. Check available GPU memory