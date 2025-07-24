#!/usr/bin/env python3
"""
Stage 2 Comprehensive Evaluation Script - evaluate_stage2.py
===========================================================

Evaluates both EfficientNetV2-B3 and GenConViT models individually and performs
complementarity analysis to validate the heterogeneous expert approach.

Key Features:
- Individual model performance assessment
- Cross-model complementarity analysis  
- Feature quality validation for Stage 3
- Comprehensive metrics and visualizations
- Stage 3 preparation validation

Usage:
    # Evaluate both models with complementarity analysis
    python src/stage2/evaluate_stage2.py --data_dir processed_data
    
    # Evaluate specific model only
    python src/stage2/evaluate_stage2.py --model effnet --no_complementarity
    python src/stage2/evaluate_stage2.py --model genconvit --variant ED
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                           confusion_matrix, classification_report, roc_curve)
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
from src.stage1.dataset import create_dataloaders
from src.stage2.feature_extractor import (EfficientNetFeatureExtractor, 
                                        GenConViTFeatureExtractor,
                                        create_combined_features)
from src.stage1.utils import calculate_metrics, plot_confusion_matrix

class Stage2Evaluator:
    """Comprehensive evaluator for Stage 2 precision analyzer models"""
    
    def __init__(self, data_dir: str, output_dir: str = "output/stage2_evaluation",
                 device: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=str(self.data_dir),
            batch_size=32,
            num_workers=4
        )
        
        logging.info(f"Initialized Stage 2 evaluator with {len(self.val_loader.dataset)} validation samples")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"stage2_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def evaluate_efficientnet(self, checkpoint_path: str) -> Dict[str, Any]:
        """Evaluate EfficientNetV2-B3 model"""
        logging.info("=" * 60)
        logging.info("EVALUATING EFFICIENTNETV2-B3 MODEL")
        logging.info("=" * 60)
        
        try:
            # Initialize feature extractor
            extractor = EfficientNetFeatureExtractor(
                'efficientnetv2_b3.in21k_ft_in1k',
                checkpoint_path,
                device=self.device
            )
            
            # Extract features and get predictions
            features, labels = extractor.extract(self.val_loader, align_dim=None, return_raw=True)
            
            # For evaluation, we need the actual model predictions
            # Load the full model with classification head
            import timm
            model = timm.create_model('efficientnetv2_b3.in21k_ft_in1k', 
                                    pretrained=False, num_classes=1)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # Get predictions
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for images, _ in tqdm(self.val_loader, desc="Getting predictions"):
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    all_preds.extend(preds.flatten())
                    all_probs.extend(probs.flatten())
            
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(labels, all_preds, all_probs, "EfficientNetV2-B3")
            
            # Save results
            results = {
                'model': 'EfficientNetV2-B3',
                'checkpoint': checkpoint_path,
                'metrics': metrics,
                'feature_dim': features.shape[1],
                'num_samples': len(labels)
            }
            
            # Create visualizations
            self.create_model_visualizations(labels, all_preds, all_probs, "efficientnet", results)
            
            logging.info(f"✅ EfficientNetV2-B3 evaluation complete - AUC: {metrics['auc']:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"❌ EfficientNetV2-B3 evaluation failed: {e}")
            return {'model': 'EfficientNetV2-B3', 'error': str(e)}
    
    def evaluate_genconvit(self, checkpoint_path: str, variant: str = 'ED', 
                          mode: str = 'hybrid') -> Dict[str, Any]:
        """Evaluate GenConViT model"""
        logging.info("=" * 60) 
        logging.info(f"EVALUATING GENCONVIT-{variant} MODEL ({mode.upper()} MODE)")
        logging.info("=" * 60)
        
        try:
            # Initialize feature extractor
            extractor = GenConViTFeatureExtractor(
                checkpoint_path,
                variant=variant,
                mode=mode,
                device=self.device
            )
            
            # Extract features with component analysis
            components = extractor.extract_features(
                self.val_loader, 
                align_dim=None, 
                return_components=True
            )
            
            labels = components['labels']
            
            # For predictions, we need to use the actual model
            model = extractor.model
            model.eval()
            
            all_preds = []
            all_probs = []
            
            with torch.no_grad():
                for images, _ in tqdm(self.val_loader, desc="Getting predictions"):
                    images = images.to(self.device)
                    outputs = model(images)
                    
                    # Handle GenConViT output format
                    if hasattr(outputs, 'classification'):
                        logits = outputs.classification
                    else:
                        logits = outputs
                    
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    all_preds.extend(preds.flatten())
                    all_probs.extend(probs.flatten())
            
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(
                labels, all_preds, all_probs, f"GenConViT-{variant}"
            )
            
            # Analyze component contributions
            component_analysis = self.analyze_genconvit_components(components)
            
            # Save results
            results = {
                'model': f'GenConViT-{variant}',
                'mode': mode,
                'checkpoint': checkpoint_path,
                'metrics': metrics,
                'component_analysis': component_analysis,
                'num_samples': len(labels)
            }
            
            # Create visualizations
            self.create_model_visualizations(labels, all_preds, all_probs, f"genconvit_{variant.lower()}", results)
            self.create_genconvit_visualizations(components, f"genconvit_{variant.lower()}")
            
            logging.info(f"✅ GenConViT-{variant} evaluation complete - AUC: {metrics['auc']:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"❌ GenConViT-{variant} evaluation failed: {e}")
            return {'model': f'GenConViT-{variant}', 'error': str(e)}
    
    def perform_complementarity_analysis(self, effnet_results: Dict[str, Any], 
                                       genconvit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze complementarity between EfficientNetV2-B3 and GenConViT"""
        logging.info("=" * 60)
        logging.info("PERFORMING COMPLEMENTARITY ANALYSIS")
        logging.info("=" * 60)
        
        try:
            # Extract predictions from both models for comparison
            # This requires re-running inference to get aligned predictions
            # For simplicity, we'll analyze the feature space complementarity
            
            # Initialize both extractors
            effnet_extractor = EfficientNetFeatureExtractor(
                'efficientnetv2_b3.in21k_ft_in1k',
                effnet_results.get('checkpoint', 'output/stage2_effnet/best_model.pth'),
                device=self.device
            )
            
            genconvit_extractor = GenConViTFeatureExtractor(
                genconvit_results.get('checkpoint', 'output/stage2_genconvit/best_model.pth'),
                variant='ED',
                mode='hybrid',
                device=self.device
            )
            
            # Extract features
            effnet_features, labels = effnet_extractor.extract(self.val_loader, align_dim=256)
            genconvit_features, _ = genconvit_extractor.extract_features(self.val_loader, align_dim=256)
            
            # Analyze feature correlation
            correlation_analysis = self.analyze_feature_correlation(
                effnet_features, genconvit_features, labels
            )
            
            # Analyze complementary predictions
            prediction_analysis = self.analyze_prediction_complementarity(
                effnet_features, genconvit_features, labels
            )
            
            # Combined feature analysis
            combined_features = create_combined_features(effnet_features, genconvit_features, 512)
            combined_analysis = self.analyze_combined_features(combined_features, labels)
            
            results = {
                'correlation_analysis': correlation_analysis,
                'prediction_analysis': prediction_analysis,
                'combined_analysis': combined_analysis,
                'feature_dimensions': {
                    'efficientnet': effnet_features.shape,
                    'genconvit': genconvit_features.shape,
                    'combined': combined_features.shape
                }
            }
            
            # Create complementarity visualizations
            self.create_complementarity_visualizations(
                effnet_features, genconvit_features, combined_features, labels, results
            )
            
            logging.info("✅ Complementarity analysis complete")
            return results
            
        except Exception as e:
            logging.error(f"❌ Complementarity analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'auc': float(roc_auc_score(y_true, y_prob)),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1': float(f1_score(y_true, y_pred)),
            }
            
            # Confusion matrix metrics
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0.0),
                    'recall': float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
                    'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
                    'confusion_matrix': cm.tolist()
                })
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
            logging.info(f"{model_name} Metrics:")
            logging.info(f"  AUC: {metrics['auc']:.4f}")
            logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"  F1-Score: {metrics['f1']:.4f}")
            logging.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logging.info(f"  Recall: {metrics.get('recall', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics for {model_name}: {e}")
            return {'error': str(e)}
    
    def analyze_genconvit_components(self, components: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze individual GenConViT components"""
        analysis = {}
        
        for component_name, component_data in components.items():
            if component_name == 'labels' or len(component_data) == 0:
                continue
                
            try:
                # Basic statistics
                analysis[component_name] = {
                    'shape': component_data.shape,
                    'mean': float(np.mean(component_data)),
                    'std': float(np.std(component_data)),
                    'min': float(np.min(component_data)),
                    'max': float(np.max(component_data))
                }
                
                # Dimensionality analysis
                if component_data.ndim == 2:
                    # PCA analysis for feature quality
                    try:
                        pca = PCA(n_components=min(10, component_data.shape[1]))
                        pca.fit(component_data)
                        analysis[component_name]['pca_variance_ratio'] = pca.explained_variance_ratio_[:5].tolist()
                    except:
                        pass
                        
            except Exception as e:
                logging.warning(f"Error analyzing component {component_name}: {e}")
        
        return analysis
    
    def analyze_feature_correlation(self, effnet_features: np.ndarray, 
                                  genconvit_features: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation between EfficientNet and GenConViT features"""
        try:
            # Use PCA to reduce dimensions for correlation analysis
            pca_dim = min(50, effnet_features.shape[1], genconvit_features.shape[1])
            
            pca_effnet = PCA(n_components=pca_dim)
            effnet_reduced = pca_effnet.fit_transform(effnet_features)
            
            pca_genconvit = PCA(n_components=pca_dim)
            genconvit_reduced = pca_genconvit.fit_transform(genconvit_features)
            
            # Calculate correlation matrix
            correlations = []
            for i in range(pca_dim):
                corr, p_value = pearsonr(effnet_reduced[:, i], genconvit_reduced[:, i])
                correlations.append(corr)
            
            correlations = np.array(correlations)
            
            analysis = {
                'mean_correlation': float(np.mean(np.abs(correlations))),
                'max_correlation': float(np.max(np.abs(correlations))),
                'min_correlation': float(np.min(np.abs(correlations))),
                'correlation_std': float(np.std(correlations)),
                'low_correlation_ratio': float(np.mean(np.abs(correlations) < 0.3))  # Complementarity indicator
            }
            
            logging.info(f"Feature Correlation Analysis:")
            logging.info(f"  Mean absolute correlation: {analysis['mean_correlation']:.4f}")
            logging.info(f"  Low correlation ratio: {analysis['low_correlation_ratio']:.4f}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in correlation analysis: {e}")
            return {'error': str(e)}
    
    def analyze_prediction_complementarity(self, effnet_features: np.ndarray,
                                         genconvit_features: np.ndarray,
                                         labels: np.ndarray) -> Dict[str, Any]:
        """Analyze how predictions complement each other"""
        try:
            # Simple classifier on each feature set for comparison
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            # Individual model performance
            lr_effnet = LogisticRegression(random_state=42, max_iter=1000)
            effnet_scores = cross_val_score(lr_effnet, effnet_features, labels, 
                                          cv=3, scoring='roc_auc')
            
            lr_genconvit = LogisticRegression(random_state=42, max_iter=1000)
            genconvit_scores = cross_val_score(lr_genconvit, genconvit_features, labels,
                                             cv=3, scoring='roc_auc')
            
            # Combined features performance
            combined_features = np.concatenate([effnet_features, genconvit_features], axis=1)
            lr_combined = LogisticRegression(random_state=42, max_iter=1000)
            combined_scores = cross_val_score(lr_combined, combined_features, labels,
                                            cv=3, scoring='roc_auc')
            
            analysis = {
                'efficientnet_auc': float(np.mean(effnet_scores)),
                'genconvit_auc': float(np.mean(genconvit_scores)),
                'combined_auc': float(np.mean(combined_scores)),
                'improvement': float(np.mean(combined_scores) - max(np.mean(effnet_scores), np.mean(genconvit_scores)))
            }
            
            logging.info(f"Prediction Complementarity Analysis:")
            logging.info(f"  EfficientNet AUC: {analysis['efficientnet_auc']:.4f}")
            logging.info(f"  GenConViT AUC: {analysis['genconvit_auc']:.4f}")
            logging.info(f"  Combined AUC: {analysis['combined_auc']:.4f}")
            logging.info(f"  Improvement: {analysis['improvement']:.4f}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in prediction complementarity analysis: {e}")
            return {'error': str(e)}
    
    def analyze_combined_features(self, combined_features: np.ndarray, 
                                labels: np.ndarray) -> Dict[str, Any]:
        """Analyze combined feature quality for Stage 3"""
        try:
            # Feature quality metrics
            analysis = {
                'shape': combined_features.shape,
                'mean': float(np.mean(combined_features)),
                'std': float(np.std(combined_features)),
                'feature_range': [float(np.min(combined_features)), float(np.max(combined_features))]
            }
            
            # PCA analysis for dimensionality assessment
            pca = PCA()
            pca.fit(combined_features)
            
            # Find dimensions for 95% variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            dims_95 = int(np.argmax(cumsum_variance >= 0.95)) + 1
            dims_99 = int(np.argmax(cumsum_variance >= 0.99)) + 1
            
            analysis.update({
                'pca_components_95_variance': dims_95,
                'pca_components_99_variance': dims_99,
                'first_10_components_variance': pca.explained_variance_ratio_[:10].tolist()
            })
            
            logging.info(f"Combined Features Analysis:")
            logging.info(f"  Shape: {analysis['shape']}")
            logging.info(f"  Components for 95% variance: {dims_95}")
            logging.info(f"  Components for 99% variance: {dims_99}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in combined feature analysis: {e}")
            return {'error': str(e)}
    
    def create_model_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray, model_name: str, 
                                  results: Dict[str, Any]):
        """Create standard model evaluation visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{model_name.upper()} Model Evaluation', fontsize=16)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = results['metrics']['auc']
            axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
            axes[0, 1].set_title('Confusion Matrix')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
            
            # Probability Distribution
            axes[1, 0].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Real', density=True)
            axes[1, 0].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Fake', density=True)
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Probability Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Metrics Summary
            metrics_text = f"""
            AUC: {results['metrics']['auc']:.4f}
            Accuracy: {results['metrics']['accuracy']:.4f}
            F1-Score: {results['metrics']['f1']:.4f}
            Precision: {results['metrics'].get('precision', 0):.4f}
            Recall: {results['metrics'].get('recall', 0):.4f}
            Specificity: {results['metrics'].get('specificity', 0):.4f}
            """
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Metrics Summary')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating visualizations for {model_name}: {e}")
    
    def create_genconvit_visualizations(self, components: Dict[str, np.ndarray], 
                                      model_name: str):
        """Create GenConViT-specific component visualizations"""
        try:
            # Filter out empty components
            valid_components = {k: v for k, v in components.items() 
                              if k != 'labels' and len(v) > 0 and v.ndim == 2}
            
            if not valid_components:
                return
                
            n_components = len(valid_components)
            fig, axes = plt.subplots(2, (n_components + 1) // 2, figsize=(5 * n_components, 10))
            if n_components == 1:
                axes = [axes]
            elif n_components == 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{model_name.upper()} Component Analysis', fontsize=16)
            
            for idx, (comp_name, comp_data) in enumerate(valid_components.items()):
                if idx >= len(axes):
                    break
                    
                # PCA visualization
                if comp_data.shape[1] > 2:
                    pca = PCA(n_components=2)
                    comp_2d = pca.fit_transform(comp_data)
                    
                    labels = components['labels']
                    scatter = axes[idx].scatter(comp_2d[labels == 0, 0], comp_2d[labels == 0, 1], 
                                             c='blue', alpha=0.6, label='Real', s=20)
                    scatter = axes[idx].scatter(comp_2d[labels == 1, 0], comp_2d[labels == 1, 1], 
                                             c='red', alpha=0.6, label='Fake', s=20)
                    axes[idx].set_title(f'{comp_name} (PCA)')
                    axes[idx].legend()
                else:
                    axes[idx].hist(comp_data.flatten(), bins=50, alpha=0.7)
                    axes[idx].set_title(f'{comp_name} Distribution')
                
                axes[idx].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(valid_components), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_components.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating GenConViT visualizations: {e}")
    
    def create_complementarity_visualizations(self, effnet_features: np.ndarray,
                                            genconvit_features: np.ndarray,
                                            combined_features: np.ndarray,
                                            labels: np.ndarray,
                                            results: Dict[str, Any]):
        """Create complementarity analysis visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Complementarity Analysis', fontsize=16)
            
            # PCA of individual features
            pca_effnet = PCA(n_components=2)
            effnet_2d = pca_effnet.fit_transform(effnet_features)
            
            axes[0, 0].scatter(effnet_2d[labels == 0, 0], effnet_2d[labels == 0, 1], 
                              c='blue', alpha=0.6, label='Real', s=20)
            axes[0, 0].scatter(effnet_2d[labels == 1, 0], effnet_2d[labels == 1, 1], 
                              c='red', alpha=0.6, label='Fake', s=20)
            axes[0, 0].set_title('EfficientNetV2-B3 Features (PCA)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            pca_genconvit = PCA(n_components=2)
            genconvit_2d = pca_genconvit.fit_transform(genconvit_features)
            
            axes[0, 1].scatter(genconvit_2d[labels == 0, 0], genconvit_2d[labels == 0, 1], 
                              c='blue', alpha=0.6, label='Real', s=20)
            axes[0, 1].scatter(genconvit_2d[labels == 1, 0], genconvit_2d[labels == 1, 1], 
                              c='red', alpha=0.6, label='Fake', s=20)
            axes[0, 1].set_title('GenConViT Features (PCA)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Combined features
            pca_combined = PCA(n_components=2)
            combined_2d = pca_combined.fit_transform(combined_features)
            
            axes[0, 2].scatter(combined_2d[labels == 0, 0], combined_2d[labels == 0, 1], 
                              c='blue', alpha=0.6, label='Real', s=20)
            axes[0, 2].scatter(combined_2d[labels == 1, 0], combined_2d[labels == 1, 1], 
                              c='red', alpha=0.6, label='Fake', s=20)
            axes[0, 2].set_title('Combined Features (PCA)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Performance comparison
            comp_analysis = results.get('prediction_analysis', {})
            models = ['EfficientNetV2', 'GenConViT', 'Combined']
            aucs = [comp_analysis.get('efficientnet_auc', 0), 
                   comp_analysis.get('genconvit_auc', 0),
                   comp_analysis.get('combined_auc', 0)]
            
            axes[1, 0].bar(models, aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 0].set_ylabel('AUC Score')
            axes[1, 0].set_title('Model Performance Comparison')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Feature correlation heatmap (sample)
            sample_size = min(1000, len(effnet_features))
            sample_idx = np.random.choice(len(effnet_features), sample_size, replace=False)
            
            # Use PCA-reduced features for correlation
            effnet_sample = pca_effnet.transform(effnet_features[sample_idx])
            genconvit_sample = pca_genconvit.transform(genconvit_features[sample_idx])
            
            corr_matrix = np.corrcoef(effnet_sample.T, genconvit_sample.T)[:2, 2:4]
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Cross-Model Feature Correlation')
            
            # Summary statistics
            corr_analysis = results.get('correlation_analysis', {})
            summary_text = f"""
            Mean Correlation: {corr_analysis.get('mean_correlation', 0):.4f}
            Low Correlation Ratio: {corr_analysis.get('low_correlation_ratio', 0):.4f}
            Combined AUC Improvement: {comp_analysis.get('improvement', 0):.4f}
            
            Feature Dimensions:
            EfficientNet: {effnet_features.shape[1]}
            GenConViT: {genconvit_features.shape[1]}
            Combined: {combined_features.shape[1]}
            """
            axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
            axes[1, 2].set_title('Analysis Summary')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'complementarity_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating complementarity visualizations: {e}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            results_serializable = convert_numpy(results)
            
            output_file = self.output_dir / 'stage2_evaluation_results.json'
            with open(output_file, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            logging.info(f"✅ Results saved to {output_file}")
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Stage 2 Comprehensive Evaluation')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Data directory path')
    parser.add_argument('--effnet_checkpoint', type=str,
                       default='output/stage2_effnet/best_model.pth',
                       help='EfficientNetV2-B3 checkpoint path')
    parser.add_argument('--genconvit_checkpoint', type=str,
                       default='output/stage2_genconvit/best_model.pth',
                       help='GenConViT checkpoint path')
    parser.add_argument('--genconvit_variant', type=str, default='ED',
                       choices=['ED', 'VAE'], help='GenConViT variant')
    parser.add_argument('--genconvit_mode', type=str, default='hybrid',
                       choices=['hybrid', 'pretrained', 'auto'], help='GenConViT mode')
    parser.add_argument('--model', type=str, choices=['effnet', 'genconvit', 'both'],
                       default='both', help='Which model to evaluate')
    parser.add_argument('--no_complementarity', action='store_true',
                       help='Skip complementarity analysis')
    parser.add_argument('--output_dir', type=str, default='output/stage2_evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Stage2Evaluator(args.data_dir, args.output_dir)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'arguments': vars(args)
    }
    
    # Evaluate models
    if args.model in ['effnet', 'both']:
        if os.path.exists(args.effnet_checkpoint):
            results['efficientnet'] = evaluator.evaluate_efficientnet(args.effnet_checkpoint)
        else:
            logging.warning(f"EfficientNet checkpoint not found: {args.effnet_checkpoint}")
    
    if args.model in ['genconvit', 'both']:
        if os.path.exists(args.genconvit_checkpoint):
            results['genconvit'] = evaluator.evaluate_genconvit(
                args.genconvit_checkpoint, 
                args.genconvit_variant,
                args.genconvit_mode
            )
        else:
            logging.warning(f"GenConViT checkpoint not found: {args.genconvit_checkpoint}")
    
    # Complementarity analysis
    if (not args.no_complementarity and args.model == 'both' and 
        'efficientnet' in results and 'genconvit' in results):
        results['complementarity'] = evaluator.perform_complementarity_analysis(
            results['efficientnet'], results['genconvit']
        )
    
    # Save results
    evaluator.save_results(results)
    
    print("\n" + "="*80)
    print("🎉 STAGE 2 EVALUATION COMPLETE!")
    print("="*80)
    
    # Print summary
    if 'efficientnet' in results and 'metrics' in results['efficientnet']:
        eff_metrics = results['efficientnet']['metrics']
        print(f"📊 EfficientNetV2-B3: AUC={eff_metrics['auc']:.4f}, F1={eff_metrics['f1']:.4f}")
    
    if 'genconvit' in results and 'metrics' in results['genconvit']:
        gen_metrics = results['genconvit']['metrics']
        print(f"📊 GenConViT: AUC={gen_metrics['auc']:.4f}, F1={gen_metrics['f1']:.4f}")
    
    if 'complementarity' in results and 'prediction_analysis' in results['complementarity']:
        comp_analysis = results['complementarity']['prediction_analysis']
        improvement = comp_analysis.get('improvement', 0)
        print(f"📊 Combined Improvement: +{improvement:.4f} AUC")
    
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 See visualizations in: {args.output_dir}")

if __name__ == "__main__":
    main()