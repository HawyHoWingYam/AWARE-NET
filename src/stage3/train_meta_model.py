#!/usr/bin/env python3
"""
Stage 3 LightGBM Meta-Model Training - train_meta_model.py
=========================================================

Trains the final meta-model using features extracted from K-fold cross-validation.
This meta-model learns to optimally combine predictions from EfficientNetV2-B3 
and GenConViT to achieve superior performance.

Key Features:
- LightGBM gradient boosting for optimal ensemble learning
- Comprehensive hyperparameter optimization (GridSearch/RandomizedSearch)
- Multiple feature combination strategies
- Cross-validation performance assessment
- Model interpretability analysis
- Production-ready model serialization

Usage:
    # Standard training with hyperparameter optimization
    python src/stage3/train_meta_model.py --data_dir output/stage3_meta
    
    # Quick training for testing
    python src/stage3/train_meta_model.py --quick_mode --cv_folds 3
    
    # Custom hyperparameter search
    python src/stage3/train_meta_model.py --search_type randomized --n_iter 50
"""

import os
import sys
import argparse
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                   cross_val_score, StratifiedKFold)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                           classification_report, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class MetaModelTrainer:
    """LightGBM meta-model training and optimization system"""
    
    def __init__(self, data_dir: str, output_dir: str = "output/stage3_meta_model",
                 random_state: int = 42):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load meta-dataset
        self.load_meta_dataset()
        
        # Initialize best model tracker
        self.best_model = None
        self.best_score = 0.0
        self.best_params = None
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / f"meta_model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Suppress LightGBM verbose output
        logging.getLogger('lightgbm').setLevel(logging.WARNING)
        
        logging.info("="*80)
        logging.info("STAGE 3: LIGHTGBM META-MODEL TRAINING INITIALIZED")
        logging.info("="*80)
    
    def load_meta_dataset(self):
        """Load the K-fold generated meta-learning dataset"""
        combined_features_dir = self.data_dir / "combined_features"
        
        if not combined_features_dir.exists():
            raise FileNotFoundError(f"Combined features directory not found: {combined_features_dir}")
        
        # Load features
        feature_files = {
            'efficientnet': combined_features_dir / "efficientnet_features.npy",
            'genconvit': combined_features_dir / "genconvit_features.npy", 
            'combined': combined_features_dir / "combined_features.npy",
            'labels': combined_features_dir / "labels.npy"
        }
        
        # Verify all files exist
        missing_files = [str(path) for path in feature_files.values() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing feature files: {missing_files}")
        
        # Load data
        self.features = {}
        for name, path in feature_files.items():
            self.features[name] = np.load(path)
            logging.info(f"Loaded {name}: {self.features[name].shape}")
        
        # Validate data consistency
        n_samples = len(self.features['labels'])
        for name, features in self.features.items():
            if len(features) != n_samples:
                raise ValueError(f"Inconsistent sample count in {name}: {len(features)} vs {n_samples}")
        
        # Dataset statistics
        n_real = np.sum(self.features['labels'] == 0)
        n_fake = np.sum(self.features['labels'] == 1)
        
        logging.info(f"Meta-dataset loaded successfully:")
        logging.info(f"  Total samples: {n_samples}")
        logging.info(f"  Real samples: {n_real} ({n_real/n_samples*100:.1f}%)")
        logging.info(f"  Fake samples: {n_fake} ({n_fake/n_samples*100:.1f}%)")
        logging.info(f"  Feature dimensions:")
        logging.info(f"    EfficientNet: {self.features['efficientnet'].shape[1]}")
        logging.info(f"    GenConViT: {self.features['genconvit'].shape[1]}")
        logging.info(f"    Combined: {self.features['combined'].shape[1]}")
    
    def get_feature_combinations(self) -> Dict[str, np.ndarray]:
        """Generate different feature combination strategies"""
        combinations = {}
        
        # Individual model features
        combinations['efficientnet_only'] = self.features['efficientnet']
        combinations['genconvit_only'] = self.features['genconvit']
        
        # Simple concatenation (as created by K-fold)
        combinations['simple_concat'] = self.features['combined']
        
        # Weighted concatenation (experimental)
        effnet_weighted = self.features['efficientnet'] * 0.6  # CNN weight
        genconvit_weighted = self.features['genconvit'] * 0.4  # Generative weight
        combinations['weighted_concat'] = np.concatenate([effnet_weighted, genconvit_weighted], axis=1)
        
        # Normalized concatenation
        from sklearn.preprocessing import StandardScaler
        scaler_effnet = StandardScaler()
        scaler_genconvit = StandardScaler()
        
        effnet_normalized = scaler_effnet.fit_transform(self.features['efficientnet'])
        genconvit_normalized = scaler_genconvit.fit_transform(self.features['genconvit'])
        combinations['normalized_concat'] = np.concatenate([effnet_normalized, genconvit_normalized], axis=1)
        
        # Feature selection-ready (top features from each)
        # Use variance as a simple feature selection criterion
        effnet_vars = np.var(self.features['efficientnet'], axis=0)
        genconvit_vars = np.var(self.features['genconvit'], axis=0)
        
        # Select top 50% features by variance
        effnet_top_idx = np.argsort(effnet_vars)[-len(effnet_vars)//2:]
        genconvit_top_idx = np.argsort(genconvit_vars)[-len(genconvit_vars)//2:]
        
        effnet_selected = self.features['efficientnet'][:, effnet_top_idx]
        genconvit_selected = self.features['genconvit'][:, genconvit_top_idx]
        combinations['variance_selected'] = np.concatenate([effnet_selected, genconvit_selected], axis=1)
        
        logging.info("Feature combination strategies prepared:")
        for name, features in combinations.items():
            logging.info(f"  {name}: {features.shape}")
        
        return combinations
    
    def get_hyperparameter_space(self, search_type: str = 'grid') -> Dict[str, List]:
        """Define hyperparameter search space for LightGBM"""
        if search_type == 'grid':
            # Comprehensive grid search (slower but thorough)
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 100, 150],
                'max_depth': [3, 5, 7, 10],
                'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.0, 0.1, 0.5, 1.0],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:  # randomized search
            # Broader range for randomized search
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 500, 800],
                'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
                'num_leaves': [15, 31, 50, 70, 100, 150, 200],
                'max_depth': [3, 5, 7, 10, 15, -1],
                'reg_alpha': [0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0],
                'reg_lambda': [0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_samples': [10, 20, 30, 50],
                'min_child_weight': [0.001, 0.01, 0.1, 1, 10]
            }
        
        return param_grid
    
    def create_base_model(self) -> lgb.LGBMClassifier:
        """Create base LightGBM model with good defaults"""
        return lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
    
    def train_with_hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
                                       search_type: str = 'grid',
                                       cv_folds: int = 5,
                                       n_iter: int = 100) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """Train model with hyperparameter optimization"""
        logging.info(f"🔍 Starting {search_type} hyperparameter search")
        
        # Create base model
        base_model = self.create_base_model()
        
        # Get hyperparameter space
        param_grid = self.get_hyperparameter_space(search_type)
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Choose search strategy
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            logging.info(f"Grid search with {len(param_grid)} parameter combinations")
        else:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            logging.info(f"Randomized search with {n_iter} iterations")
        
        # Perform search
        logging.info("Training in progress...")
        search.fit(X, y)
        
        # Extract results
        best_model = search.best_estimator_
        search_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_,
            'search_type': search_type,
            'n_iterations': len(search.cv_results_['mean_test_score'])
        }
        
        logging.info(f"✅ Hyperparameter search complete!")
        logging.info(f"   Best AUC: {search.best_score_:.4f}")
        logging.info(f"   Best parameters: {search.best_params_}")
        
        return best_model, search_results
    
    def evaluate_model(self, model: lgb.LGBMClassifier, X: np.ndarray, y: np.ndarray,
                      feature_name: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logging.info(f"📊 Evaluating model on {feature_name} features")
        
        # Cross-validation scores
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'auc': cross_val_score(model, X, y, cv=cv_splitter, scoring='roc_auc'),
            'accuracy': cross_val_score(model, X, y, cv=cv_splitter, scoring='accuracy'),
            'f1': cross_val_score(model, X, y, cv=cv_splitter, scoring='f1')
        }
        
        # Fit model for additional metrics
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Detailed metrics
        evaluation = {
            'feature_combination': feature_name,
            'cv_scores': {
                'auc_mean': float(np.mean(cv_scores['auc'])),
                'auc_std': float(np.std(cv_scores['auc'])),
                'accuracy_mean': float(np.mean(cv_scores['accuracy'])),
                'accuracy_std': float(np.std(cv_scores['accuracy'])),
                'f1_mean': float(np.mean(cv_scores['f1'])),
                'f1_std': float(np.std(cv_scores['f1']))
            },
            'train_metrics': {
                'auc': float(roc_auc_score(y, y_pred_proba)),
                'accuracy': float(accuracy_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred))
            },
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
        
        logging.info(f"   CV AUC: {evaluation['cv_scores']['auc_mean']:.4f} ± {evaluation['cv_scores']['auc_std']:.4f}")
        logging.info(f"   CV F1: {evaluation['cv_scores']['f1_mean']:.4f} ± {evaluation['cv_scores']['f1_std']:.4f}")
        
        return evaluation
    
    def analyze_feature_importance(self, model: lgb.LGBMClassifier, 
                                 feature_combination: str) -> Dict[str, Any]:
        """Analyze feature importance for model interpretability"""
        try:
            # Get feature importance
            importance = model.feature_importances_
            
            # Create importance analysis
            importance_analysis = {
                'feature_combination': feature_combination,
                'total_features': len(importance),
                'importance_stats': {
                    'mean': float(np.mean(importance)),
                    'std': float(np.std(importance)),
                    'max': float(np.max(importance)),
                    'min': float(np.min(importance))
                },
                'top_10_features': {
                    'indices': np.argsort(importance)[-10:].tolist(),
                    'values': importance[np.argsort(importance)[-10:]].tolist()
                }
            }
            
            # Analyze feature distribution by source (if combined features)
            if 'concat' in feature_combination:
                # Assuming first half is EfficientNet, second half is GenConViT
                mid_point = len(importance) // 2
                effnet_importance = importance[:mid_point]
                genconvit_importance = importance[mid_point:]
                
                importance_analysis['source_analysis'] = {
                    'efficientnet_contribution': float(np.sum(effnet_importance)),
                    'genconvit_contribution': float(np.sum(genconvit_importance)),
                    'efficientnet_mean': float(np.mean(effnet_importance)),
                    'genconvit_mean': float(np.mean(genconvit_importance))
                }
            
            return importance_analysis
            
        except Exception as e:
            logging.warning(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}
    
    def create_visualizations(self, model: lgb.LGBMClassifier, X: np.ndarray, y: np.ndarray,
                            feature_name: str, evaluation: Dict[str, Any]):
        """Create comprehensive visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Meta-Model Analysis: {feature_name}', fontsize=16)
            
            # Model predictions
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)
            
            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_score = evaluation['train_metrics']['auc']
            axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.4f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Prediction Distribution
            axes[0, 1].hist(y_pred_proba[y == 0], bins=30, alpha=0.7, label='Real', density=True)
            axes[0, 1].hist(y_pred_proba[y == 1], bins=30, alpha=0.7, label='Fake', density=True)
            axes[0, 1].set_xlabel('Predicted Probability')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Prediction Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
            axes[0, 2].set_title('Confusion Matrix')
            axes[0, 2].set_xlabel('Predicted')
            axes[0, 2].set_ylabel('Actual')
            
            # 4. Feature Importance (top 20)
            importance = model.feature_importances_
            top_indices = np.argsort(importance)[-20:]
            axes[1, 0].barh(range(len(top_indices)), importance[top_indices])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 20 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. CV Score Distribution
            cv_scores = evaluation['cv_scores']
            score_names = ['AUC', 'Accuracy', 'F1']
            means = [cv_scores['auc_mean'], cv_scores['accuracy_mean'], cv_scores['f1_mean']]
            stds = [cv_scores['auc_std'], cv_scores['accuracy_std'], cv_scores['f1_std']]
            
            x_pos = np.arange(len(score_names))
            axes[1, 1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            axes[1, 1].set_xlabel('Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Cross-Validation Scores')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(score_names)
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Learning Curve (if available)
            try:
                # Use validation curve as proxy for learning curve
                from sklearn.model_selection import validation_curve
                param_range = [50, 100, 200, 300, 500]
                train_scores, val_scores = validation_curve(
                    model, X, y, param_name='n_estimators', param_range=param_range,
                    cv=3, scoring='roc_auc', n_jobs=-1
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                axes[1, 2].plot(param_range, train_mean, 'o-', label='Training Score')
                axes[1, 2].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
                axes[1, 2].plot(param_range, val_mean, 'o-', label='Cross-validation Score')
                axes[1, 2].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)
                axes[1, 2].set_xlabel('Number of Estimators')
                axes[1, 2].set_ylabel('AUC Score')
                axes[1, 2].set_title('Validation Curve')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            except Exception as e:
                axes[1, 2].text(0.5, 0.5, f'Learning curve\nnot available\n({str(e)})', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Learning Curve (Not Available)')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / f'meta_model_analysis_{feature_name}.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"✅ Visualizations saved: {viz_path}")
            
        except Exception as e:
            logging.error(f"❌ Visualization creation failed: {e}")
    
    def compare_feature_combinations(self, combinations: Dict[str, np.ndarray],
                                   search_type: str = 'randomized',
                                   cv_folds: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """Compare different feature combination strategies"""
        logging.info("🔬 Comparing feature combination strategies")
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'search_type': search_type,
            'cv_folds': cv_folds,
            'feature_combinations': {}
        }
        
        for combo_name, features in combinations.items():
            logging.info(f"  Testing {combo_name} ({features.shape})")
            
            try:
                # Train model with hyperparameter search
                model, search_results = self.train_with_hyperparameter_search(
                    features, self.features['labels'], search_type, cv_folds, n_iter
                )
                
                # Evaluate model
                evaluation = self.evaluate_model(model, features, self.features['labels'], combo_name, cv_folds)
                
                # Feature importance analysis
                importance_analysis = self.analyze_feature_importance(model, combo_name)
                
                # Create visualizations
                self.create_visualizations(model, features, self.features['labels'], combo_name, evaluation)
                
                # Store results
                combo_results = {
                    'search_results': search_results,
                    'evaluation': evaluation,
                    'importance_analysis': importance_analysis,
                    'model_params': model.get_params()
                }
                
                comparison_results['feature_combinations'][combo_name] = combo_results
                
                # Track best model
                cv_auc = evaluation['cv_scores']['auc_mean']
                if cv_auc > self.best_score:
                    self.best_score = cv_auc
                    self.best_model = model
                    self.best_params = combo_results
                    logging.info(f"🏆 New best model: {combo_name} (AUC: {cv_auc:.4f})")
                
            except Exception as e:
                logging.error(f"❌ Failed to process {combo_name}: {e}")
                comparison_results['feature_combinations'][combo_name] = {'error': str(e)}
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save comprehensive results and best model"""
        # Save results JSON
        results_file = self.output_dir / 'meta_model_training_results.json'
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        results_serializable = convert_types(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logging.info(f"✅ Results saved: {results_file}")
        
        # Save best model
        if self.best_model is not None:
            model_file = self.output_dir / 'best_meta_model.pkl'
            joblib.dump(self.best_model, model_file)
            logging.info(f"✅ Best model saved: {model_file}")
            
            # Save model info
            model_info = {
                'best_score': self.best_score,
                'best_combination': None,
                'model_params': self.best_model.get_params(),
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Find best combination name
            for combo_name, combo_results in results['feature_combinations'].items():
                if 'evaluation' in combo_results:
                    cv_auc = combo_results['evaluation']['cv_scores']['auc_mean']
                    if abs(cv_auc - self.best_score) < 1e-6:
                        model_info['best_combination'] = combo_name
                        break
            
            info_file = self.output_dir / 'best_model_info.json'
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logging.info(f"✅ Model info saved: {info_file}")
    
    def train_meta_model(self, search_type: str = 'randomized', cv_folds: int = 5,
                        n_iter: int = 100, quick_mode: bool = False) -> Dict[str, Any]:
        """Main meta-model training pipeline"""
        logging.info("🚀 Starting meta-model training pipeline")
        
        # Quick mode adjustments
        if quick_mode:
            cv_folds = min(cv_folds, 3)
            n_iter = min(n_iter, 20)
            logging.info(f"🧪 Quick mode: cv_folds={cv_folds}, n_iter={n_iter}")
        
        training_start_time = datetime.now()
        
        # Get feature combinations
        combinations = self.get_feature_combinations()
        
        # Compare all combinations
        results = self.compare_feature_combinations(combinations, search_type, cv_folds, n_iter)
        
        # Add summary statistics
        results['summary'] = {
            'total_combinations': len(combinations),
            'successful_combinations': len([c for c in results['feature_combinations'].values() if 'error' not in c]),
            'best_score': self.best_score,
            'best_combination': None,
            'training_duration_seconds': (datetime.now() - training_start_time).total_seconds()
        }
        
        # Find best combination
        for combo_name, combo_results in results['feature_combinations'].items():
            if 'evaluation' in combo_results:
                cv_auc = combo_results['evaluation']['cv_scores']['auc_mean']
                if abs(cv_auc - self.best_score) < 1e-6:
                    results['summary']['best_combination'] = combo_name
                    break
        
        # Save results
        self.save_results(results)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='LightGBM Meta-Model Training')
    parser.add_argument('--data_dir', type=str, default='output/stage3_meta',
                       help='Meta-dataset directory path')
    parser.add_argument('--output_dir', type=str, default='output/stage3_meta_model',
                       help='Output directory for meta-model')
    parser.add_argument('--search_type', type=str, choices=['grid', 'randomized'],
                       default='randomized', help='Hyperparameter search type')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Cross-validation folds')
    parser.add_argument('--n_iter', type=int, default=100,
                       help='Iterations for randomized search')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--quick_mode', action='store_true',
                       help='Quick mode with reduced parameters')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = MetaModelTrainer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            random_state=args.random_state
        )
        
        # Train meta-model
        results = trainer.train_meta_model(
            search_type=args.search_type,
            cv_folds=args.cv_folds,
            n_iter=args.n_iter,
            quick_mode=args.quick_mode
        )
        
        # Print results summary
        print("\n" + "="*80)
        print("🎉 META-MODEL TRAINING COMPLETE!")
        print("="*80)
        
        summary = results['summary']
        print(f"📊 Results Summary:")
        print(f"   Best AUC Score: {summary['best_score']:.4f}")
        print(f"   Best Combination: {summary['best_combination']}")
        print(f"   Successful Combinations: {summary['successful_combinations']}/{summary['total_combinations']}")
        print(f"   Training Time: {summary['training_duration_seconds']/3600:.2f} hours")
        print(f"📁 Output Directory: {args.output_dir}")
        print(f"🤖 Best Model: {args.output_dir}/best_meta_model.pkl")
        print("="*80)
        
    except Exception as e:
        logging.error(f"❌ Meta-model training failed: {e}")
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()