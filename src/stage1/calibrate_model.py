#!/usr/bin/env python3
"""
Stage 1 Model Calibration Script - calibrate_model.py
=====================================================

This script implements temperature scaling calibration for the Stage 1 fast filter.
Temperature scaling is a simple but effective post-hoc calibration method that 
improves the reliability of predicted probabilities.

Key Features:
- Temperature scaling optimization using scipy
- Reliability diagram visualization
- Expected Calibration Error (ECE) computation
- Calibration before/after comparison

Usage:
    python src/stage1/calibrate_model.py --model_path output/stage1/best_model.pth
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import from stage1 utils
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils import load_model_checkpoint, calculate_metrics, create_probability_distribution_plot

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = output_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class TemperatureScalingCalibrator:
    """
    Temperature scaling calibrator for binary classification
    
    Temperature scaling applies a single scalar parameter T to the logits:
    calibrated_prob = sigmoid(logits / T)
    
    The optimal T is found by minimizing the negative log-likelihood on validation data.
    """
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_fitted = False
    
    def fit(self, logits, labels, method='nll', bounds=(0.1, 10.0)):
        """
        Fit the temperature parameter
        
        Args:
            logits (torch.Tensor): Model logits
            labels (torch.Tensor): True labels
            method (str): Optimization method ('nll' or 'ece')
            bounds (tuple): Bounds for temperature parameter
            
        Returns:
            dict: Optimization results
        """
        
        # Convert to numpy for scipy optimization
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Define objective function
        if method == 'nll':
            def objective(temp):
                scaled_logits = torch.tensor(logits_np / temp, dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(scaled_logits, torch.tensor(labels_np))
                return loss.item()
        else:
            def objective(temp):
                scaled_probs = torch.sigmoid(torch.tensor(logits_np / temp))
                return self._compute_ece(scaled_probs.numpy(), labels_np)
        
        # Optimize temperature
        result = minimize(
            objective,
            x0=1.0,
            bounds=[bounds],
            method='L-BFGS-B'
        )
        
        optimal_temp = result.x[0]
        self.temperature.data = torch.tensor(optimal_temp)
        self.is_fitted = True
        
        logging.info(f"Temperature calibration completed:")
        logging.info(f"  Optimal temperature: {optimal_temp:.4f}")
        logging.info(f"  Optimization success: {result.success}")
        logging.info(f"  Final objective value: {result.fun:.6f}")
        
        return {
            'optimal_temperature': optimal_temp,
            'success': result.success,
            'final_loss': result.fun,
            'method': method,
            'bounds': bounds
        }
    
    def calibrate(self, logits):
        """
        Apply temperature scaling to logits
        
        Args:
            logits (torch.Tensor): Model logits
            
        Returns:
            torch.Tensor: Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        scaled_logits = logits / self.temperature
        return torch.sigmoid(scaled_logits)
    
    def _compute_ece(self, probs, labels, n_bins=10):
        """
        Compute Expected Calibration Error
        
        Args:
            probs (np.array): Predicted probabilities
            labels (np.array): True labels
            n_bins (int): Number of bins for calibration
            
        Returns:
            float: ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

def create_dataset_from_manifest(manifest_path, data_root, transform=None):
    """Create dataset from manifest (reuse from train_stage1.py)"""
    from train_stage1 import DeepfakeDataset
    return DeepfakeDataset(manifest_path, data_root, transform)

def create_reliability_diagram(y_true, y_prob_uncalibrated, y_prob_calibrated, 
                             save_path=None, n_bins=10):
    """
    Create reliability diagram showing calibration before and after
    
    Args:
        y_true (np.array): True labels
        y_prob_uncalibrated (np.array): Uncalibrated probabilities
        y_prob_calibrated (np.array): Calibrated probabilities
        save_path (str): Path to save plot
        n_bins (int): Number of bins
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def plot_reliability(ax, y_true, y_prob, title):
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                bin_sizes.append(in_bin.sum())
            else:
                accuracies.append(0)
                confidences.append((bin_lower + bin_upper) / 2)
                bin_sizes.append(0)
        
        # Plot bars
        bar_width = 0.8 / n_bins
        positions = [(bin_lowers[i] + bin_uppers[i]) / 2 for i in range(n_bins)]
        
        bars = ax.bar(positions, accuracies, width=bar_width, alpha=0.7, 
                     edgecolor='black', label='Accuracy')
        
        # Color bars by bin size
        max_bin_size = max(bin_sizes) if max(bin_sizes) > 0 else 1
        for i, (bar, size) in enumerate(zip(bars, bin_sizes)):
            intensity = size / max_bin_size
            bar.set_color(plt.cm.Blues(0.3 + 0.7 * intensity))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        
        # Confidence line
        ax.plot(confidences, accuracies, 'o-', color='orange', 
               linewidth=2, markersize=6, label='Model Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Compute ECE
        ece = 0
        for i in range(n_bins):
            if bin_sizes[i] > 0:
                bin_acc = accuracies[i]
                bin_conf = confidences[i]
                bin_weight = bin_sizes[i] / len(y_true)
                ece += abs(bin_acc - bin_conf) * bin_weight
        
        ax.text(0.02, 0.98, f'ECE: {ece:.4f}', transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ece
    
    # Plot uncalibrated
    ece_uncal = plot_reliability(ax1, y_true, y_prob_uncalibrated, 'Before Calibration')
    
    # Plot calibrated
    ece_cal = plot_reliability(ax2, y_true, y_prob_calibrated, 'After Calibration')
    
    plt.suptitle(f'Reliability Diagrams - ECE Improvement: {ece_uncal:.4f} → {ece_cal:.4f}', 
                fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return ece_uncal, ece_cal

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Model Calibration')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='output/stage1/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, 
                        default='mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                        help='Model name from timm library')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--val_manifest', type=str, default='processed_data/manifests/val_manifest.csv',
                        help='Path to validation manifest CSV')
    
    # Calibration arguments
    parser.add_argument('--method', type=str, default='nll', choices=['nll', 'ece'],
                        help='Calibration optimization method')
    parser.add_argument('--temp_bounds', type=float, nargs=2, default=[0.1, 10.0],
                        help='Temperature parameter bounds')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/stage1',
                        help='Output directory for calibration results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=== Stage 1 Model Calibration ===")
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
    
    # Create transforms (same as validation transforms from training)
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    logger.info("Loading validation dataset...")
    dataset = create_dataset_from_manifest(args.val_manifest, args.data_dir, val_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    logger.info(f"Validation samples: {len(dataset)}")
    
    # Get model predictions on validation set
    logger.info("Computing model predictions...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images).squeeze(1)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    logger.info(f"Collected {len(all_logits)} predictions")
    
    # Compute uncalibrated probabilities
    uncalibrated_probs = torch.sigmoid(all_logits)
    
    # Compute uncalibrated metrics
    uncal_metrics = calculate_metrics(all_labels.numpy(), uncalibrated_probs.numpy())
    
    logger.info("=== UNCALIBRATED METRICS ===")
    logger.info(f"AUC: {uncal_metrics['auc']:.4f}")
    logger.info(f"Accuracy: {uncal_metrics['accuracy']:.4f}")
    logger.info(f"F1-Score: {uncal_metrics['f1']:.4f}")
    
    # Fit temperature scaling
    logger.info("Fitting temperature scaling...")
    calibrator = TemperatureScalingCalibrator()
    cal_results = calibrator.fit(all_logits, all_labels, method=args.method, bounds=args.temp_bounds)
    
    # Apply calibration
    calibrated_probs = calibrator.calibrate(all_logits)
    
    # Compute calibrated metrics
    cal_metrics = calculate_metrics(all_labels.numpy(), calibrated_probs.detach().numpy())
    
    logger.info("=== CALIBRATED METRICS ===")
    logger.info(f"AUC: {cal_metrics['auc']:.4f}")
    logger.info(f"Accuracy: {cal_metrics['accuracy']:.4f}")
    logger.info(f"F1-Score: {cal_metrics['f1']:.4f}")
    
    # Compute ECE before and after
    calibrator_for_ece = TemperatureScalingCalibrator()
    calibrator_for_ece.temperature.data = torch.tensor(cal_results['optimal_temperature'])
    calibrator_for_ece.is_fitted = True
    
    uncal_ece = calibrator_for_ece._compute_ece(uncalibrated_probs.numpy(), all_labels.numpy())
    cal_ece = calibrator_for_ece._compute_ece(calibrated_probs.detach().numpy(), all_labels.numpy())
    
    logger.info("=== CALIBRATION QUALITY ===")
    logger.info(f"ECE Before: {uncal_ece:.4f}")
    logger.info(f"ECE After: {cal_ece:.4f}")
    logger.info(f"ECE Improvement: {uncal_ece - cal_ece:.4f} ({((uncal_ece - cal_ece) / uncal_ece * 100):.1f}%)")
    
    # Save calibration temperature
    calibration_params = {
        'temperature': cal_results['optimal_temperature'],
        'calibration_method': args.method,
        'temperature_bounds': args.temp_bounds,
        'uncalibrated_metrics': uncal_metrics,
        'calibrated_metrics': cal_metrics,
        'ece_before': float(uncal_ece),
        'ece_after': float(cal_ece),
        'ece_improvement': float(uncal_ece - cal_ece),
        'model_path': args.model_path,
        'val_manifest': args.val_manifest,
        'validation_samples': len(dataset)
    }
    
    temp_file = output_dir / 'calibration_temp.json'
    with open(temp_file, 'w') as f:
        json.dump(calibration_params, f, indent=2)
    
    logger.info(f"Calibration parameters saved: {temp_file}")
    
    # Create reliability diagram
    logger.info("Creating reliability diagram...")
    reliability_plot_path = output_dir / 'reliability_diagram.png'
    ece_uncal_plot, ece_cal_plot = create_reliability_diagram(
        all_labels.numpy(),
        uncalibrated_probs.numpy(),
        calibrated_probs.detach().numpy(),
        save_path=reliability_plot_path
    )
    
    logger.info(f"Reliability diagram saved: {reliability_plot_path}")
    
    # Create probability distribution plots
    logger.info("Creating probability distribution plots...")
    prob_dist_plot_path = output_dir / 'probability_distributions.png'
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Probability Distributions Before and After Calibration', fontsize=16)
    
    # Before calibration
    real_probs_uncal = uncalibrated_probs[all_labels == 0].numpy()
    fake_probs_uncal = uncalibrated_probs[all_labels == 1].numpy()
    
    axes[0, 0].hist(real_probs_uncal, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0, 0].hist(fake_probs_uncal, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    axes[0, 0].set_title('Before Calibration - Histogram')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot([real_probs_uncal, fake_probs_uncal], labels=['Real', 'Fake'])
    axes[0, 1].set_title('Before Calibration - Box Plot')
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].grid(True, alpha=0.3)
    
    # After calibration
    real_probs_cal = calibrated_probs[all_labels == 0].detach().numpy()
    fake_probs_cal = calibrated_probs[all_labels == 1].detach().numpy()
    
    axes[1, 0].hist(real_probs_cal, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[1, 0].hist(fake_probs_cal, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    axes[1, 0].set_title('After Calibration - Histogram')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].boxplot([real_probs_cal, fake_probs_cal], labels=['Real', 'Fake'])
    axes[1, 1].set_title('After Calibration - Box Plot')
    axes[1, 1].set_ylabel('Predicted Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(prob_dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Probability distributions saved: {prob_dist_plot_path}")
    
    # Save detailed results
    results_summary = {
        'calibration_summary': {
            'optimal_temperature': cal_results['optimal_temperature'],
            'method': args.method,
            'ece_improvement_percent': float((uncal_ece - cal_ece) / uncal_ece * 100),
            'model_performance_maintained': abs(uncal_metrics['auc'] - cal_metrics['auc']) < 0.01
        },
        'uncalibrated_metrics': uncal_metrics,
        'calibrated_metrics': cal_metrics,
        'calibration_quality': {
            'ece_before': float(uncal_ece),
            'ece_after': float(cal_ece),
            'ece_reduction': float(uncal_ece - cal_ece)
        }
    }
    
    summary_file = output_dir / 'calibration_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("=== CALIBRATION SUMMARY ===")
    logger.info(f"Temperature: {cal_results['optimal_temperature']:.4f}")
    logger.info(f"ECE Improvement: {uncal_ece:.4f} → {cal_ece:.4f} ({((uncal_ece - cal_ece) / uncal_ece * 100):.1f}%)")
    logger.info(f"AUC Maintained: {uncal_metrics['auc']:.4f} → {cal_metrics['auc']:.4f}")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("Stage 1 calibration completed successfully!")

if __name__ == "__main__":
    main()