#!/bin/bash
# Mobile Optimization Quick Start Script
# =====================================
# Automated script for running Stage 4 mobile optimization pipeline

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/output/stage4/optimization_log_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Default configuration
OPTIMIZATION_MODE="quick"  # quick, full, test
DEVICE="auto"
BATCH_SIZE=16
EPOCHS=3
ENABLE_BENCHMARKING=true
ENABLE_ONNX_EXPORT=true
SKIP_TESTING=false

# Parse command line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode MODE           Optimization mode: quick, full, test (default: quick)"
    echo "  --device DEVICE       Device: cuda, cpu, auto (default: auto)"
    echo "  --batch-size SIZE     Batch size for training (default: 16)"
    echo "  --epochs NUM          Number of epochs (default: 3)"
    echo "  --skip-benchmark      Skip benchmarking step"
    echo "  --skip-onnx           Skip ONNX export step"
    echo "  --skip-testing        Skip initial testing"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Quick optimization with defaults"
    echo "  $0 --mode full --epochs 10   # Full optimization with 10 epochs"
    echo "  $0 --mode test --skip-onnx   # Test mode without ONNX export"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            OPTIMIZATION_MODE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-benchmark)
            ENABLE_BENCHMARKING=false
            shift
            ;;
        --skip-onnx)
            ENABLE_ONNX_EXPORT=false
            shift
            ;;
        --skip-testing)
            SKIP_TESTING=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate mode
case $OPTIMIZATION_MODE in
    quick|full|test)
        ;;
    *)
        error "Invalid mode: $OPTIMIZATION_MODE. Must be quick, full, or test."
        exit 1
        ;;
esac

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create output directories
    mkdir -p "$PROJECT_ROOT/output/stage4/optimized_models"
    mkdir -p "$PROJECT_ROOT/output/stage4/benchmark_results"
    mkdir -p "$PROJECT_ROOT/output/stage4/deployment_packages"
    
    # Check Python environment
    if ! command -v python &> /dev/null; then
        error "Python not found. Please ensure Python is installed and in PATH."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/src/stage4/optimize_for_mobile.py" ]]; then
        error "Stage 4 optimization script not found. Please run from project root."
        exit 1
    fi
    
    # Check CUDA availability if requested
    if [[ "$DEVICE" == "cuda" ]] || [[ "$DEVICE" == "auto" ]]; then
        if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            DEVICE="cuda"
            success "CUDA is available and will be used"
        else
            if [[ "$DEVICE" == "cuda" ]]; then
                error "CUDA requested but not available. Please install CUDA or use --device cpu"
                exit 1
            else
                DEVICE="cpu"
                warning "CUDA not available, falling back to CPU"
            fi
        fi
    fi
    
    success "Environment setup completed"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Stage 1-3 models exist or can use pretrained
    local models_found=0
    
    if [[ -f "$PROJECT_ROOT/output/stage1/best_model.pth" ]]; then
        success "Stage 1 model found"
        models_found=$((models_found + 1))
    else
        warning "Stage 1 model not found, will use pretrained weights"
    fi
    
    if [[ -f "$PROJECT_ROOT/output/stage2_effnet/best_model.pth" ]]; then
        success "Stage 2 EfficientNet model found"
        models_found=$((models_found + 1))
    else
        warning "Stage 2 EfficientNet model not found, will use pretrained weights"
    fi
    
    if [[ -f "$PROJECT_ROOT/output/stage2_genconvit/best_model.pth" ]]; then
        success "Stage 2 GenConViT model found"  
        models_found=$((models_found + 1))
    else
        warning "Stage 2 GenConViT model not found, will use pretrained weights"
    fi
    
    if [[ $models_found -eq 0 ]]; then
        warning "No trained models found. Optimization will use pretrained weights."
        warning "For best results, train Stage 1-3 models first."
    fi
    
    # Check dataset configuration
    if [[ -f "$PROJECT_ROOT/config/dataset_paths.json" ]]; then
        success "Dataset configuration found"
    else
        warning "Dataset configuration not found. Please run setup_dataset_config.py if needed."
    fi
    
    success "Prerequisites check completed"
}

# Run component testing
run_testing() {
    if [[ "$SKIP_TESTING" == true ]]; then
        log "Skipping testing as requested"
        return 0
    fi
    
    log "Running component tests..."
    
    cd "$PROJECT_ROOT"
    
    if python src/stage4/test_mobile_optimization.py --test_all 2>&1 | tee -a "$LOG_FILE"; then
        success "Component tests passed"
    else
        error "Component tests failed. Check logs for details."
        warning "Continuing with optimization, but results may be unreliable."
    fi
}

# Run mobile optimization
run_optimization() {
    log "Starting mobile optimization (mode: $OPTIMIZATION_MODE)..."
    
    cd "$PROJECT_ROOT"
    
    # Set parameters based on mode
    local target_model="stage1"  # Default for quick/test
    local actual_epochs=$EPOCHS
    
    case $OPTIMIZATION_MODE in
        quick)
            target_model="stage1"
            actual_epochs=3
            ;;
        full)
            target_model="all"
            actual_epochs=10
            ;;
        test)
            target_model="stage1"
            actual_epochs=1
            ;;
    esac
    
    log "Optimization parameters:"
    log "  - Target model: $target_model"
    log "  - Epochs: $actual_epochs"
    log "  - Batch size: $BATCH_SIZE"
    log "  - Device: $DEVICE"
    
    # Run optimization
    local optimization_cmd="python src/stage4/optimize_for_mobile.py"
    optimization_cmd="$optimization_cmd --model $target_model"
    optimization_cmd="$optimization_cmd --epochs $actual_epochs"
    optimization_cmd="$optimization_cmd --batch_size $BATCH_SIZE"
    
    log "Running: $optimization_cmd"
    
    if eval "$optimization_cmd" 2>&1 | tee -a "$LOG_FILE"; then
        success "Mobile optimization completed successfully"
        
        # Display results summary
        local results_file="$PROJECT_ROOT/output/stage4/optimization_comparison.json"
        if [[ -f "$results_file" ]]; then
            log "Optimization results summary:"
            python -c "
import json
with open('$results_file') as f:
    data = json.load(f)
    summary = data.get('optimization_summary', {})
    print(f'  - Models optimized: {summary.get(\"successful_optimizations\", 0)}/{summary.get(\"total_models\", 0)}')
    print(f'  - Average size reduction: {summary.get(\"average_size_reduction_percent\", 0):.1f}%')
    print(f'  - Average speedup: {summary.get(\"average_inference_speedup\", 0):.2f}x')
    print(f'  - Average accuracy degradation: {summary.get(\"average_accuracy_degradation_percent\", 0):.2f}%')
" 2>/dev/null || warning "Could not parse optimization results"
        fi
    else
        error "Mobile optimization failed. Check logs for details."
        return 1
    fi
}

# Run ONNX export
run_onnx_export() {
    if [[ "$ENABLE_ONNX_EXPORT" != true ]]; then
        log "Skipping ONNX export as requested"
        return 0
    fi
    
    log "Exporting optimized models to ONNX format..."
    
    cd "$PROJECT_ROOT"
    
    # Check if optimized models exist
    local optimized_dir="$PROJECT_ROOT/output/stage4/optimized_models"
    local model_count=$(find "$optimized_dir" -name "*_quantized_model.pth" 2>/dev/null | wc -l)
    
    if [[ $model_count -eq 0 ]]; then
        warning "No optimized models found for ONNX export"
        return 0
    fi
    
    success "Found $model_count optimized models for export"
    
    # Run ONNX export (simplified - would need proper implementation)
    log "ONNX export functionality ready, but requires model-specific implementation"
    warning "Manual ONNX export may be needed for production deployment"
    
    # Create placeholder export structure
    local export_dir="$PROJECT_ROOT/output/stage4/deployment_packages"
    mkdir -p "$export_dir"
    
    # Create deployment readme
    cat > "$export_dir/DEPLOYMENT_README.md" << EOF
# Mobile Deployment Package

## Optimized Models
Quantized models are available in: \`output/stage4/optimized_models/\`

## ONNX Export
To export models to ONNX format:

\`\`\`bash
python -c "
from src.stage4.mobile_deployment.onnx_exporter import ONNXExporter
import torch

# Load optimized model
model = torch.load('output/stage4/optimized_models/stage1_quantized_model.pth')

# Export to ONNX
exporter = ONNXExporter()
result = exporter.export_model(
    model=model,
    output_path='stage1_mobile.onnx',
    input_shape=(1, 3, 256, 256)
)
print(f'Export successful: {result[\"success\"]}')
"
\`\`\`

## Mobile Integration
1. Load ONNX models in your mobile application
2. Preprocess images to 256x256 RGB format
3. Normalize with ImageNet statistics
4. Run inference and apply sigmoid to outputs
5. Use cascade thresholds for efficient processing

Generated on: $(date)
EOF
    
    success "ONNX export setup completed"
}

# Run benchmarking
run_benchmarking() {
    if [[ "$ENABLE_BENCHMARKING" != true ]]; then
        log "Skipping benchmarking as requested"
        return 0
    fi
    
    log "Running performance benchmarking..."
    
    cd "$PROJECT_ROOT"
    
    # Adjust benchmarking based on mode
    local benchmark_args="--benchmark_all"
    local num_samples=500  # Default for quick testing
    
    case $OPTIMIZATION_MODE in
        quick|test)
            num_samples=200
            ;;
        full)
            num_samples=1000
            ;;
    esac
    
    benchmark_args="$benchmark_args --num_samples $num_samples --device $DEVICE"
    
    log "Running benchmarking with: $benchmark_args"
    
    if python src/stage4/benchmark_cascade.py $benchmark_args 2>&1 | tee -a "$LOG_FILE"; then
        success "Benchmarking completed successfully"
        
        # Display benchmark summary
        local benchmark_file="$PROJECT_ROOT/output/stage4/benchmark_results/benchmark_results.json"
        if [[ -f "$benchmark_file" ]]; then
            log "Benchmark results summary:"
            python -c "
import json
with open('$benchmark_file') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    if 'accuracy' in summary:
        print(f'  - Average AUC: {summary[\"accuracy\"].get(\"avg_auc\", 0):.4f}')
    if 'speed' in summary:
        print(f'  - Average inference time: {summary[\"speed\"].get(\"avg_inference_time_ms\", 0):.1f}ms')
    
    recommendations = data.get('recommendations', [])
    if recommendations:
        print('  - Key recommendations:')
        for rec in recommendations[:3]:  # Show top 3
            print(f'    {rec}')
" 2>/dev/null || warning "Could not parse benchmark results"
        fi
    else
        warning "Benchmarking completed with issues. Check logs for details."
    fi
}

# Generate final report
generate_report() {
    log "Generating final optimization report..."
    
    local report_file="$PROJECT_ROOT/output/stage4/optimization_summary_${TIMESTAMP}.md"
    
    cat > "$report_file" << EOF
# Stage 4 Mobile Optimization Report

**Generated:** $(date)
**Mode:** $OPTIMIZATION_MODE
**Device:** $DEVICE
**Configuration:** Epochs=$EPOCHS, Batch Size=$BATCH_SIZE

## Summary

This report summarizes the mobile optimization results for the AWARE-NET cascade deepfake detection system.

## Files Generated

- **Optimized Models:** \`output/stage4/optimized_models/\`
- **Benchmark Results:** \`output/stage4/benchmark_results/\`
- **Deployment Package:** \`output/stage4/deployment_packages/\`
- **Detailed Logs:** \`$LOG_FILE\`

## Next Steps

1. **Validate Results:** Review benchmark results and optimization metrics
2. **Test Deployment:** Use deployment package for mobile integration testing
3. **Performance Tuning:** Adjust parameters if targets not met
4. **Production Integration:** Deploy to target mobile platform

## Quick Commands

\`\`\`bash
# View optimization results
cat output/stage4/optimization_comparison.json

# View benchmark results  
cat output/stage4/benchmark_results/benchmark_results.json

# Test mobile inference (when implemented)
python src/stage4/mobile_deployment/mobile_inference.py test_image.jpg
\`\`\`

## Support

For issues or questions:
- Review detailed logs: \`$LOG_FILE\`
- Check testing guide: \`src/stage4/TESTING.md\`  
- Run component tests: \`python src/stage4/test_mobile_optimization.py --test_all\`
EOF

    success "Final report generated: $report_file"
}

# Main execution
main() {
    echo "=========================================="
    echo "AWARE-NET Stage 4 Mobile Optimization"
    echo "=========================================="
    echo ""
    
    log "Starting mobile optimization pipeline..."
    log "Mode: $OPTIMIZATION_MODE"
    log "Timestamp: $TIMESTAMP"
    log "Log file: $LOG_FILE"
    echo ""
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Execute pipeline steps
    setup_environment
    check_prerequisites
    run_testing
    run_optimization
    run_onnx_export
    run_benchmarking
    generate_report
    
    echo ""
    success "=========================================="
    success "Mobile optimization pipeline completed!"
    success "=========================================="
    echo ""
    log "Results available in: output/stage4/"
    log "Detailed logs: $LOG_FILE"
    
    # Final recommendations
    echo ""
    log "🎯 Recommended next steps:"
    log "  1. Review optimization results in output/stage4/optimization_comparison.json"
    log "  2. Check benchmark results in output/stage4/benchmark_results/"
    log "  3. Test deployment package for mobile integration"
    log "  4. Run integration tests when available"
    echo ""
}

# Execute main function
main "$@"