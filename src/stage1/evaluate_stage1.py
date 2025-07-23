#!/usr/bin/env python3
"""
Stage 1 Model Evaluation Script - evaluate_stage1.py
====================================================

This script performs comprehensive evaluation of the Stage 1 fast filter including:
- Performance metrics calculation (AUC, accuracy, F1-score)
- Cascade analysis at different thresholds
- ROC curve and confusion matrix visualization
- Cross-dataset evaluation capability
- Calibration quality assessment

Usage:
    python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth
    python src/stage1/evaluate_stage1.py --model_path output/stage1/best_model.pth --test_manifest processed_data/manifests/test_manifest.csv
"""

import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import from stage1 utils
import sys
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_model_checkpoint, 
    calculate_metrics, 
    create_probability_distribution_plot,
    analyze_cascade_performance,
    print_cascade_analysis,
    plot_confusion_matrix
)
from train_stage1 import DeepfakeDataset

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_calibration_temperature(calibration_file):
    """
    Load temperature scaling parameter from calibration file
    
    Args:
        calibration_file (str): Path to calibration JSON file
        
    Returns:
        float: Temperature parameter, or 1.0 if not found
    """
    if not os.path.exists(calibration_file):
        return 1.0
        
    try:
        with open(calibration_file, 'r') as f:
            cal_data = json.load(f)
        return cal_data.get('temperature', 1.0)
    except Exception:
        return 1.0

def apply_temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits
    
    Args:
        logits (torch.Tensor): Model logits
        temperature (float): Temperature parameter
        
    Returns:
        torch.Tensor: Calibrated probabilities
    """
    return torch.sigmoid(logits / temperature)

def create_roc_curve_plot(y_true, y_pred_proba, save_path=None, title="ROC Curve"):
    """
    Create ROC curve plot
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        save_path (str): Path to save plot
        title (str): Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_threshold_analysis_plot(y_true, y_pred_proba, save_path=None):
    """
    Create threshold analysis visualization showing precision, recall, and F1 scores
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        save_path (str): Path to save plot
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores for different thresholds
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    accuracies = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision-Recall curve
    ax1.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Threshold analysis
    ax2.plot(thresholds, f1_scores, linewidth=2, label='F1-Score', color='blue')
    ax2.plot(thresholds, accuracies, linewidth=2, label='Accuracy', color='green')
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Threshold Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark optimal F1 threshold
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_thresh = thresholds[optimal_f1_idx]
    optimal_f1_score = f1_scores[optimal_f1_idx]
    
    ax2.axvline(x=optimal_f1_thresh, color='red', linestyle='--', alpha=0.7)
    ax2.text(optimal_f1_thresh + 0.02, optimal_f1_score, 
             f'Optimal F1\n({optimal_f1_thresh:.3f}, {optimal_f1_score:.3f})',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return optimal_f1_thresh, optimal_f1_score

def create_cascade_analysis_plot(cascade_results, save_path=None):
    """
    Create cascade performance visualization
    
    Args:
        cascade_results (dict): Results from analyze_cascade_performance
        save_path (str): Path to save plot
    """
    thresholds = list(cascade_results.keys())
    filtration_rates = [results['filtration_rate'] for results in cascade_results.values()]
    leakage_rates = [results['leakage_rate'] for results in cascade_results.values()]
    precisions = [results['simple_real_precision'] for results in cascade_results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filtration vs Leakage
    ax1.plot(thresholds, filtration_rates, 'o-', linewidth=2, label='Filtration Rate', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(thresholds, leakage_rates, 's-', linewidth=2, label='Leakage Rate', color='red')
    
    ax1.set_xlabel('Cascade Threshold')
    ax1.set_ylabel('Filtration Rate', color='blue')
    ax1_twin.set_ylabel('Leakage Rate', color='red')
    ax1.set_title('Cascade Performance: Filtration vs Leakage')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Precision analysis
    ax2.plot(thresholds, precisions, 'o-', linewidth=2, label='Simple Real Precision', color='green')
    ax2.set_xlabel('Cascade Threshold')
    ax2.set_ylabel('Precision')
    ax2.set_title('Cascade Performance: Simple Real Precision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataloader, device, temperature=1.0):
    """
    Evaluate model on dataset
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        device (torch.device): Device to run evaluation on
        temperature (float): Temperature scaling parameter
        
    Returns:
        tuple: (all_labels, all_logits, all_probs)
    """
    model.eval()
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images).squeeze(1)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Apply temperature scaling
    all_probs = apply_temperature_scaling(all_logits, temperature)
    
    return all_labels.numpy(), all_logits.numpy(), all_probs.numpy()

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Model Evaluation')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='output/stage1/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, 
                        default='mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                        help='Model name from timm library')
    parser.add_argument('--calibration_file', type=str, default='output/stage1/calibration_temp.json',
                        help='Path to calibration temperature file')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--test_manifest', type=str, default='processed_data/manifests/test_manifest.csv',
                        help='Path to test manifest CSV')
    parser.add_argument('--val_manifest', type=str, default='processed_data/manifests/val_manifest.csv',
                        help='Path to validation manifest CSV (fallback if test not available)')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--use_calibration', action='store_true',
                        help='Apply temperature scaling calibration')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/stage1/evaluation',
                        help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=== Stage 1 Model Evaluation ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return
    
    # Create model architecture
    model = timm.create_model(args.model_name, pretrained=False, num_classes=1)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_model_checkpoint(args.model_path, model, device)
    model.eval()
    
    # Load training args if available
    if 'args' in checkpoint:
        train_args = checkpoint['args']
        logger.info(f"Model trained with: epochs={train_args.get('epochs', 'unknown')}, "
                   f"best_val_auc={checkpoint.get('best_val_auc', 'unknown'):.4f}")
    
    # Load calibration temperature
    temperature = 1.0
    if args.use_calibration:
        temperature = load_calibration_temperature(args.calibration_file)
        logger.info(f"Using temperature scaling: {temperature:.4f}")
    else:
        logger.info("Temperature scaling disabled")
    
    # Determine which manifest to use
    test_manifest_path = args.test_manifest
    if not os.path.exists(test_manifest_path):
        logger.warning(f"Test manifest not found: {test_manifest_path}")
        logger.info(f"Using validation manifest: {args.val_manifest}")
        test_manifest_path = args.val_manifest
        
        if not os.path.exists(test_manifest_path):
            logger.error(f"Neither test nor validation manifest found!")
            return
    
    # Create transforms (same as validation transforms from training)
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    logger.info("Loading test dataset...")
    dataset = DeepfakeDataset(test_manifest_path, args.data_dir, test_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    logger.info(f"Test samples: {len(dataset)}")
    
    # Calculate real and fake counts
    real_count = len(dataset.manifest[dataset.manifest['label'] == 0])
    fake_count = len(dataset.manifest[dataset.manifest['label'] == 1])
    
    logger.info(f"  Real samples: {real_count}")
    logger.info(f"  Fake samples: {fake_count}")
    
    # Evaluate model
    logger.info("Running model evaluation...")
    y_true, logits, y_pred_proba = evaluate_model(model, dataloader, device, temperature)
    
    logger.info(f"Collected {len(y_true)} predictions")
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(y_true, y_pred_proba)
    
    logger.info("=== EVALUATION METRICS ===")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    
    # Cascade analysis
    logger.info("Performing cascade analysis...")
    cascade_results = analyze_cascade_performance(y_true, y_pred_proba)
    print_cascade_analysis(cascade_results)
    
    # Create visualizations
    logger.info("Creating evaluation visualizations...")
    
    # ROC Curve
    roc_plot_path = output_dir / 'roc_curve.png'
    create_roc_curve_plot(y_true, y_pred_proba, save_path=roc_plot_path)
    logger.info(f"ROC curve saved: {roc_plot_path}")
    
    # Threshold Analysis
    threshold_plot_path = output_dir / 'threshold_analysis.png'
    optimal_thresh, optimal_f1 = create_threshold_analysis_plot(y_true, y_pred_proba, save_path=threshold_plot_path)
    logger.info(f"Threshold analysis saved: {threshold_plot_path}")
    logger.info(f"Optimal F1 threshold: {optimal_thresh:.4f} (F1 = {optimal_f1:.4f})")
    
    # Confusion Matrix
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    cm_plot_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred_binary, save_path=cm_plot_path)
    logger.info(f"Confusion matrix saved: {cm_plot_path}")
    
    # Probability Distribution
    prob_dist_path = output_dir / 'probability_distribution.png'
    create_probability_distribution_plot(y_true, y_pred_proba, save_path=prob_dist_path)
    logger.info(f"Probability distribution saved: {prob_dist_path}")
    
    # Cascade Analysis Plot
    cascade_plot_path = output_dir / 'cascade_analysis.png'
    create_cascade_analysis_plot(cascade_results, save_path=cascade_plot_path)
    logger.info(f"Cascade analysis plot saved: {cascade_plot_path}")
    
    # Save detailed results
    evaluation_results = {
        'model_info': {
            'model_path': args.model_path,
            'model_name': args.model_name,
            'temperature': float(temperature),
            'calibration_used': args.use_calibration
        },
        'dataset_info': {
            'test_manifest': test_manifest_path,
            'total_samples': len(dataset),
            'real_samples': real_count,
            'fake_samples': fake_count
        },
        'performance_metrics': metrics,
        'optimal_threshold': {
            'f1_threshold': float(optimal_thresh),
            'f1_score': float(optimal_f1)
        },
        'cascade_analysis': {
            str(k): v for k, v in cascade_results.items()
        }
    }
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Detailed results saved: {results_file}")
    
    # Summary
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Model Performance: AUC = {metrics['auc']:.4f}, F1 = {metrics['f1']:.4f}")
    logger.info(f"Optimal Classification Threshold: {optimal_thresh:.4f}")
    logger.info(f"Best Cascade Threshold (90% filtration): {[k for k, v in cascade_results.items() if v['filtration_rate'] >= 0.9]}")
    logger.info(f"Evaluation outputs saved to: {output_dir}")
    logger.info("Stage 1 evaluation completed successfully!")

if __name__ == "__main__":
    main()