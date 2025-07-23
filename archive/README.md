# Archive Directory

This directory contains deprecated and legacy scripts that have been superseded by the current implementation but are preserved for reference.

## Contents

### df40_scripts/
Legacy scripts for DF40 dataset processing that have been replaced by the unified preprocessing pipeline:
- `df40_rearrange_corrected.py` - Old DF40 data reorganization script
- `df40_rearrange_with_json.py` - JSON-based DF40 processing (deprecated)

**Replacement**: Use `scripts/preprocess_datasets_v2.py` for all dataset preprocessing.

### forgerynet_download.py
Script for downloading the ForgeryNet dataset.

**Status**: Deprecated - ForgeryNet dataset is not currently used in the Stage 1 implementation.

### test_scripts/
Various experimental and testing scripts from earlier development phases.

**Status**: Archived for reference - current testing should use the Stage 1 pipeline scripts.

## Migration Notes

All functionality from these archived scripts has been integrated into the current Stage 1 cascade architecture:

- **Data Processing**: Use `scripts/preprocess_datasets_v2.py`
- **Model Training**: Use `src/stage1/train_stage1.py`
- **Evaluation**: Use `src/stage1/evaluate_stage1.py`

These archived files are maintained for historical reference and should not be used in current development.