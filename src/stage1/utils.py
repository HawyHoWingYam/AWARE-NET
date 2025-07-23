#!/usr/bin/env python3
"""
Stage 1 Utility Functions - utils.py
====================================

Shared utility functions for Stage 1 fast filter implementation.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate comprehensive metrics for binary classification
    
    Args:
        y_true (array-like): True labels (0 or 1)
        y_pred_proba (array-like): Predicted probabilities
        threshold (float): Decision threshold
        
    Returns:
        dict: Dictionary of metrics
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'threshold': threshold
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        })
    
    return metrics

def load_model_checkpoint(checkpoint_path, model, device='cpu'):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (torch.nn.Module): Model to load weights into
        device (str): Device to load model on
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def plot_confusion_matrix(y_true, y_pred, class_names=['Real', 'Fake'], 
                         title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): Class names for labels
        title (str): Plot title
        save_path (str): Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_cascade_thresholds():
    """
    Get standard cascade thresholds for analysis
    
    Returns:
        list: List of threshold values
    """
    return [0.90, 0.95, 0.98, 0.99]

def analyze_cascade_performance(y_true, y_pred_proba, thresholds=None):
    """
    Analyze cascade performance at different thresholds
    
    Args:
        y_true (array-like): True labels (0=real, 1=fake)
        y_pred_proba (array-like): Predicted probabilities for fake class
        thresholds (list): Thresholds to analyze
        
    Returns:
        dict: Analysis results
    """
    if thresholds is None:
        thresholds = get_cascade_thresholds()
    
    results = {}
    
    for thresh in thresholds:
        # Simple real samples (high confidence real)
        simple_real = y_pred_proba < (1 - thresh)  # Low fake probability = high real confidence
        
        # Calculate leakage rate (fake samples incorrectly passed as simple real)
        fake_mask = np.array(y_true) == 1
        if fake_mask.sum() > 0:
            leakage_rate = (simple_real & fake_mask).sum() / fake_mask.sum()
        else:
            leakage_rate = 0.0
        
        # Calculate filtration rate (samples handled by stage 1)
        filtration_rate = simple_real.sum() / len(y_pred_proba)
        
        # Calculate precision for simple real classification
        if simple_real.sum() > 0:
            simple_real_precision = (simple_real & ~fake_mask).sum() / simple_real.sum()
        else:
            simple_real_precision = 0.0
        
        results[thresh] = {
            'threshold': thresh,
            'leakage_rate': float(leakage_rate),
            'filtration_rate': float(filtration_rate),
            'simple_real_precision': float(simple_real_precision),
            'samples_filtered': int(simple_real.sum()),
            'false_negatives': int((simple_real & fake_mask).sum())
        }
    
    return results

def print_cascade_analysis(cascade_results):
    """
    Print cascade analysis results in a formatted table
    
    Args:
        cascade_results (dict): Results from analyze_cascade_performance
    """
    print("\n=== CASCADE PERFORMANCE ANALYSIS ===")
    print(f"{'Threshold':<10} {'Filtration':<11} {'Leakage':<8} {'Precision':<10} {'Filtered':<9} {'False Neg':<10}")
    print("-" * 70)
    
    for thresh, results in cascade_results.items():
        print(f"{thresh:<10.2f} {results['filtration_rate']:<11.3f} "
              f"{results['leakage_rate']:<8.3f} {results['simple_real_precision']:<10.3f} "
              f"{results['samples_filtered']:<9d} {results['false_negatives']:<10d}")
    
    print("\nDefinitions:")
    print("- Filtration: % of samples handled by Stage 1 (passed as 'simple real')")
    print("- Leakage: % of fake samples incorrectly passed by Stage 1")  
    print("- Precision: % of Stage 1 predictions that are actually real")
    print("- Filtered: Number of samples handled by Stage 1")
    print("- False Neg: Number of fake samples missed by Stage 1")

def create_probability_distribution_plot(y_true, y_pred_proba, save_path=None):
    """
    Create probability distribution plot for real vs fake samples
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        save_path (str): Path to save plot
    """
    real_probs = y_pred_proba[np.array(y_true) == 0]
    fake_probs = y_pred_proba[np.array(y_true) == 1]
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Histograms
    plt.subplot(1, 2, 1)
    plt.hist(real_probs, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    plt.hist(fake_probs, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    plt.xlabel('Predicted Probability (Fake)')
    plt.ylabel('Density')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plots
    plt.subplot(1, 2, 2)
    plt.boxplot([real_probs, fake_probs], labels=['Real', 'Fake'])
    plt.ylabel('Predicted Probability (Fake)')
    plt.title('Probability Box Plots')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def validate_data_paths(manifest_path, data_root):
    """
    Validate that data paths in manifest exist
    
    Args:
        manifest_path (str): Path to manifest CSV
        data_root (str): Root directory of data
        
    Returns:
        tuple: (valid_count, total_count, missing_files)
    """
    import pandas as pd
    
    manifest = pd.read_csv(manifest_path)
    data_root = Path(data_root)
    
    missing_files = []
    valid_count = 0
    
    for _, row in manifest.iterrows():
        img_path = data_root / row['image_path']
        if img_path.exists():
            valid_count += 1
        else:
            missing_files.append(str(img_path))
    
    return valid_count, len(manifest), missing_files