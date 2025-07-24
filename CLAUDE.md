# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MobileDeepfakeDetection implements a multi-stage cascade architecture for efficient deepfake detection optimized for mobile deployment. The system is designed with a fast filter approach, where Stage 1 serves as a high-speed preliminary filter using MobileNetV4-Hybrid-Medium, followed by more sophisticated analysis stages for complex samples.

### Repository
- **GitHub**: https://github.com/HawyHoWingYam/MobileDeepfakeDetection
- **Claude Code Integration**: Full AI-driven development workflow enabled

### Core Architecture (Current Implementation)
1. **Stage 1 - Fast Filter**: MobileNetV4-Hybrid-Medium with temperature scaling calibration (✅ Complete)
2. **Stage 2-5 - Advanced Analyzers**: Heterogeneous ensemble, cross-attention fusion, temporal analysis (🔄 Planned)

### Current Status
- ✅ **Stage 1 Complete**: Training, calibration, and evaluation pipeline implemented
- 🔄 **Stages 2-5**: Advanced ensemble analyzers in development

## Environment Setup

### Quick Start Commands
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate aware-net

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install PyTorch nightly (required for compatibility)
pip install --pre --upgrade --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install --pre --upgrade --no-cache-dir torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Stage 1 Model (Current Implementation)
The current implementation uses MobileNetV4-Hybrid-Medium from the timm library:
- **Model**: `mobilenetv4_hybrid_medium.ix_e550_r256_in1k` (automatically downloaded by timm)
- **Input Size**: 256×256 RGB images
- **Output**: Binary classification (real/fake detection)

## Development Commands

### Stage 1 Training Pipeline
```bash
# Task 1.1: Train MobileNetV4-Hybrid-Medium model
python src/stage1/train_stage1.py --data_dir processed_data --epochs 50 --batch_size 32 --lr 1e-4

# Task 1.2: Calibrate model probabilities using temperature scaling
python src/stage1/calibrate_model.py --model_path output/stage1/best_model.pth --data_dir processed_data

# Task 1.3: Comprehensive performance evaluation 
python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --temp_file output/stage1/calibration_temp.json
```

### Data Preprocessing
```bash
# Setup dataset configuration (recommended)
python scripts/setup_dataset_config.py

# Validate dataset paths
python scripts/preprocess_datasets_v2.py --validate-only

# Preprocess datasets with flexible configuration (Task 0.2)
python scripts/preprocess_datasets_v2.py --config config/dataset_paths.json

# Check configuration summary
python scripts/preprocess_datasets_v2.py --print-config
```

### Testing and Validation
```bash
# Validate dataset paths and configuration
python scripts/preprocess_datasets_v2.py --validate-only

# Test individual Stage 1 components
python src/stage1/train_stage1.py --help  # View training options
python src/stage1/calibrate_model.py --help  # View calibration options  
python src/stage1/evaluate_stage1.py --help  # View evaluation options

# Comprehensive Stage 1 evaluation with metrics
python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --temp_file output/stage1/calibration_temp.json
```

## Code Architecture

### Key Components (Current Implementation)
- **src/stage1/train_stage1.py**: Stage 1 model training with MobileNetV4-Hybrid-Medium
- **src/stage1/calibrate_model.py**: Temperature scaling calibration for probability reliability
- **src/stage1/evaluate_stage1.py**: Comprehensive performance evaluation with reliability diagrams
- **src/stage1/utils.py**: Shared utility functions for Stage 1 pipeline
- **src/utils/dataset_config.py**: Dataset configuration management class
- **scripts/preprocess_datasets_v2.py**: Multi-threaded GPU-accelerated data preprocessing
- **scripts/setup_dataset_config.py**: Interactive dataset configuration setup
- **config/dataset_paths.json**: JSON-based dataset path configuration

### Directory Structure
```
├── src/                    # Core implementation files
│   ├── stage1/            # Stage 1 fast filter implementation
│   │   ├── train_stage1.py       # Model training script
│   │   ├── calibrate_model.py    # Probability calibration
│   │   ├── evaluate_stage1.py    # Performance evaluation
│   │   └── utils.py              # Shared utilities
│   └── utils/             # Utility modules
│       └── dataset_config.py     # Dataset configuration management
├── scripts/               # Data processing and setup scripts
│   ├── preprocess_datasets_v2.py
│   └── setup_dataset_config.py
├── config/                # Configuration files
│   └── dataset_paths.json
├── docs/                  # Documentation
│   └── setup_environment.md
├── output/                # Training outputs
│   └── stage1/           # Stage 1 training results
│       ├── best_model.pth
│       ├── calibration_temp.json
│       └── evaluation_report.json
├── dataset/               # Raw video datasets
│   ├── CelebDF-v2/
│   ├── FF++/
│   ├── DFDC/
│   └── DF40/
├── processed_data/        # Processed face images (created by preprocessing)
│   ├── train/
│   ├── val/
│   ├── final_test_sets/
│   └── manifests/
└── project_instruction/   # Project phases documentation
```

## Dataset Configuration

### Supported Datasets
- **CelebDF-v2**: Celebrity deepfake detection dataset
- **FF++ (FaceForensics++)**: Face manipulation detection dataset  
- **DFDC**: Deepfake Detection Challenge dataset
- **DF40**: Pre-processed face swap dataset (256x256 PNG format)

### Data Processing
- **Output Format**: Unified 256x256 PNG images across all datasets
- **Train/Val/Test Split**: 70/15/15 (configurable)
- **Face Detection**: Multi-backend support (InsightFace, MediaPipe, YOLOv8, MTCNN)
- **GPU Acceleration**: Multi-threaded processing with 70-85% GPU utilization

## Configuration Management

### Dataset Configuration (New)
Uses flexible JSON-based configuration system:
- **`src/utils/dataset_config.py`**: Core configuration management class
- **`config/dataset_paths.json`**: Your specific dataset path configuration
- **`scripts/setup_dataset_config.py`**: Interactive configuration setup tool

```bash
# Setup configuration for your dataset structure
python scripts/setup_dataset_config.py

# Validate paths without processing
python scripts/preprocess_datasets_v2.py --validate-only
```

### Stage 1 Model Configuration
Stage 1 training parameters are configured via command-line arguments:
- **Model**: MobileNetV4-Hybrid-Medium from timm library
- **Loss Function**: BCEWithLogitsLoss for binary classification
- **Optimizer**: AdamW with weight decay (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR for smooth learning rate decay
- **Data Augmentation**: RandomHorizontalFlip, ColorJitter, RandomAffine, GaussianBlur

## Development Guidelines

### Stage 1 Training Pipeline (Current Implementation)
1. **Training (Task 1.1)**: Fine-tune MobileNetV4-Hybrid-Medium on combined datasets
2. **Calibration (Task 1.2)**: Apply temperature scaling for probability reliability
3. **Evaluation (Task 1.3)**: Comprehensive performance analysis with reliability diagrams
4. **Cascade Design**: Analyze threshold strategies for multi-stage architecture

### Code Conventions
- Follow existing PyTorch patterns in the codebase
- Use timm library for pre-trained models (MobileNetV4, EfficientNetV2) and GenConViT from Hugging Face/GitHub
- Implement proper error handling for GPU/CPU compatibility
- Use tqdm for progress tracking during long operations

### Face Detection (Multi-Backend Support)
- **Primary**: InsightFace (GPU-accelerated, recommended for performance)
- **Alternatives**: MediaPipe, YOLOv8, OpenCV DNN, MTCNN
- **Configuration**: Configurable via command-line arguments in preprocessing scripts
- **GPU Optimization**: Multi-threaded processing with 70-85% GPU utilization

## Environment Variables

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Python path for src imports
```

## Troubleshooting

### Common Issues
- **CUDA not available**: Verify NVIDIA drivers and install PyTorch nightly builds
- **Memory errors**: Reduce batch sizes in Stage 1 training configuration
- **Import errors**: Ensure PYTHONPATH includes src directory
- **Face detection issues**: Switch between different face detection backends (InsightFace, MediaPipe, YOLOv8)

### Performance Optimization
- Use gradient accumulation for large effective batch sizes
- Enable mixed precision training when available
- Implement proper data loading with multiple workers
- Monitor GPU utilization during multi-threaded preprocessing

## Personal Memory Notes

- 之後的測試大部分都是我來運行,最好詢問一下我是否我運行.
# GitHub CLI Integration & AI-Driven Workflows

## Natural Language Development Commands

This project supports comprehensive AI-driven development through natural language commands and GitHub CLI integration. All development operations can be controlled through prompts.

### Core Workflow Patterns

#### Issue Management
```bash
# Auto-create issues from natural language
@claude create issue for "Stage 2 GenConViT integration performance optimization"
@claude analyze issue #123 and suggest implementation approach
@claude assign issue #123 to appropriate team member based on code history
```

#### Pull Request Automation
```bash
# Auto-generate PRs from issues
@claude create PR for issue #123 with implementation
@claude review PR #456 for security and performance issues
@claude merge PR #456 if all checks pass and approved
```

#### Branch Management
```bash
# Auto-create feature branches
@claude create branch for issue #123 following naming conventions
@claude switch to hybrid genconvit development branch
@claude cleanup stale branches older than 30 days
```

#### Intelligent Context Collection
For complex operations, Claude will automatically gather context using chained GitHub CLI commands:
```bash
# Automatic context gathering for issue analysis
gh issue view $ISSUE --comments
gh search code --repo $REPO --keyword "related_function"
gh pr list --state all --search "involves:$USER"
```

### Issue Templates & Labels

#### Standard Labels
- `bug` - Bug reports requiring fixes
- `feature` - New feature requests
- `stage1` - Stage 1 MobileNetV4 related issues
- `stage2` - Stage 2 EfficientNet/GenConViT related issues
- `ai-review` - Issues requiring AI analysis
- `claude-approved` - AI-approved changes ready for merge
- `needs-context` - Issues requiring additional context gathering

#### Issue Templates
- **Bug Report**: Structured template with reproduction steps, expected behavior, actual behavior
- **Feature Request**: Template with problem statement, proposed solution, acceptance criteria
- **Model Performance**: Template for reporting training/inference performance issues

### Pull Request Conventions

#### PR Requirements
1. **Auto-generated title**: `[Stage#] Brief description (fixes #issue)`
2. **AI code review**: All PRs automatically reviewed by Claude for:
   - Security vulnerabilities
   - Performance issues
   - Code style consistency
   - Architecture compliance
3. **Test automation**: PRs trigger automated testing for affected stages
4. **Documentation sync**: Changes automatically update relevant documentation

#### Safe Merge Automation
PRs are auto-merged only when:
- Labeled with `claude-approved` AND `tests-passed`
- Approved by at least 2 team members (for critical changes)
- All CI/CD checks pass
- No merge conflicts exist

### Slash Commands (.claude/commands/)

#### Available Commands
- `/fix-issue <number>` - Analyze issue and create fix PR
- `/create-feature <description>` - Create feature branch and basic implementation
- `/review-pr <number>` - Comprehensive PR review with suggestions
- `/optimize-stage1` - Performance optimization for Stage 1 MobileNetV4
- `/optimize-stage2` - Performance optimization for Stage 2 models
- `/generate-docs` - Update documentation based on recent changes
- `/cleanup-branches` - Remove stale development branches

#### Command Structure
Each command file in `.claude/commands/` contains:
```markdown
# Command Description
Brief description of what this command does

## Parameters
- $1: Required parameter description
- $2: Optional parameter description

## Execution Steps
1. Step 1 description
2. Step 2 description
3. Expected outcome
```

### Automated Knowledge Management

#### Wiki Synchronization
- PR merges automatically update relevant Wiki pages
- New features trigger documentation generation
- Performance improvements update benchmark documentation
- Architecture changes update system overview

#### Dynamic Rule Evolution
- Code review discussions automatically update CLAUDE.md rules
- Team decisions from PR comments become codified standards
- Performance learnings update optimization guidelines

### Security & Team Collaboration

#### Permission Gates
- Only maintainers can trigger certain high-impact commands
- Critical system changes require multi-person approval
- AI operations are logged for audit purposes

#### Team Integration
- @mentions trigger appropriate team member notifications
- Code owners are automatically added to relevant PRs
- Conflict resolution follows predefined escalation paths

### Model Training Integration

#### Stage 1 Automation
```bash
@claude start stage1 training with current best hyperparameters
@claude analyze stage1 performance and suggest improvements
@claude compare stage1 models and recommend best checkpoint
```

#### Stage 2 Automation
```bash
@claude switch genconvit to pretrained mode and retrain
@claude compare hybrid vs pretrained genconvit performance
@claude optimize stage2 ensemble weights automatically
```

### Asynchronous AI Operations

#### Nightly Batch Processing
- Issues labeled `ai-review` are processed overnight
- PRs with `needs-deep-analysis` get comprehensive review
- Performance benchmarks are automatically updated
- Stale issues and PRs are triaged and updated

#### Smart Notifications
- Critical issues trigger immediate notifications
- Performance degradations auto-create priority issues
- Training completions update relevant stakeholders

## Development Workflow Integration

All standard development operations work seamlessly with natural language:
1. **Planning**: "Create development plan for Stage 3 meta-model"
2. **Implementation**: "Implement the plan with proper testing"
3. **Review**: "Review implementation for performance and security"
4. **Deployment**: "Prepare deployment package with documentation"
5. **Monitoring**: "Monitor performance and create optimization tasks"

This AI-driven approach enables 100% natural language control over the entire development lifecycle while maintaining code quality, security, and team collaboration standards.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

When using GitHub CLI integration:
- Always gather full context before making changes
- Use appropriate labels and templates for consistency
- Ensure security gates are respected for automated operations
- Log all AI-driven changes for team transparency