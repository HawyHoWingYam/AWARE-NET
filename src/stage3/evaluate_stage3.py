#!/usr/bin/env python3
"""
Stage 3 Meta-Model Evaluation - evaluate_stage3.py
==================================================

Comprehensive evaluation of the trained meta-model including performance analysis,
Stage 4 integration testing, and comparison with individual Stage 2 models.

Usage:
    python src/stage3/evaluate_stage3.py --model_path output/stage3_meta_model/best_meta_model.pkl
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class Stage3Evaluator:
    """Comprehensive Stage 3 meta-model evaluation"""
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str = "output/stage3_evaluation"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load meta-model
        self.load_meta_model()
        
        # Load validation data
        self.load_validation_data()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f"stage3_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("="*80)
        logging.info("STAGE 3: META-MODEL EVALUATION")
        logging.info("="*80)
    
    def load_meta_model(self):
        """Load trained meta-model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Meta-model not found: {self.model_path}")
        
        self.meta_model = joblib.load(self.model_path)
        logging.info(f"✅ Meta-model loaded: {self.model_path}")
        logging.info(f"   Model type: {type(self.meta_model).__name__}")
        logging.info(f"   Parameters: {len(self.meta_model.get_params())} parameters")
    
    def load_validation_data(self):
        """Load validation data for evaluation"""
        # Try to load separate validation set first
        val_features_dir = self.data_dir / "val_features"
        if val_features_dir.exists():
            logging.info("Loading separate validation set")
            self.load_features_from_dir(val_features_dir)
        else:
            # Use part of training data for evaluation
            logging.info("Using training data subset for evaluation")
            combined_dir = self.data_dir / "combined_features"
            self.load_features_from_dir(combined_dir)
    
    def load_features_from_dir(self, features_dir: Path):
        """Load features from directory"""
        feature_files = {
            'combined': features_dir / "combined_features.npy",
            'labels': features_dir / "labels.npy"
        }
        
        self.features = {}
        for name, path in feature_files.items():
            if path.exists():
                self.features[name] = np.load(path)
                logging.info(f"Loaded {name}: {self.features[name].shape}")
        
        if 'combined' not in self.features or 'labels' not in self.features:
            raise FileNotFoundError("Required feature files not found")
    
    def evaluate_meta_model(self) -> Dict[str, Any]:
        """Comprehensive meta-model evaluation"""
        logging.info("🔍 Evaluating meta-model performance")
        
        X = self.features['combined']
        y = self.features['labels']
        
        # Cross-validation evaluation
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {
            'auc': cross_val_score(self.meta_model, X, y, cv=cv_splitter, scoring='roc_auc'),
            'accuracy': cross_val_score(self.meta_model, X, y, cv=cv_splitter, scoring='accuracy'),
            'f1': cross_val_score(self.meta_model, X, y, cv=cv_splitter, scoring='f1')
        }
        
        # Full dataset evaluation
        self.meta_model.fit(X, y)
        y_pred_proba = self.meta_model.predict_proba(X)[:, 1]
        y_pred = self.meta_model.predict(X)
        
        evaluation = {
            'cv_scores': {
                'auc_mean': float(np.mean(cv_scores['auc'])),
                'auc_std': float(np.std(cv_scores['auc'])),
                'accuracy_mean': float(np.mean(cv_scores['accuracy'])),
                'accuracy_std': float(np.std(cv_scores['accuracy'])),
                'f1_mean': float(np.mean(cv_scores['f1'])),
                'f1_std': float(np.std(cv_scores['f1']))
            },
            'full_dataset': {
                'auc': float(roc_auc_score(y, y_pred_proba)),
                'accuracy': float(accuracy_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred))
            },
            'sample_size': len(y),
            'feature_dim': X.shape[1]
        }
        
        logging.info(f"✅ Meta-model evaluation complete:")
        logging.info(f"   CV AUC: {evaluation['cv_scores']['auc_mean']:.4f} ± {evaluation['cv_scores']['auc_std']:.4f}")
        logging.info(f"   CV F1: {evaluation['cv_scores']['f1_mean']:.4f} ± {evaluation['cv_scores']['f1_std']:.4f}")
        
        return evaluation
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        results_file = self.output_dir / 'stage3_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"✅ Results saved: {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Stage 3 Meta-Model Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained meta-model')
    parser.add_argument('--data_dir', type=str, default='output/stage3_meta',
                       help='Meta-dataset directory')
    parser.add_argument('--output_dir', type=str, default='output/stage3_evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    try:
        evaluator = Stage3Evaluator(args.model_path, args.data_dir, args.output_dir)
        results = evaluator.evaluate_meta_model()
        evaluator.save_results(results)
        
        print("\n" + "="*60)
        print("🎉 STAGE 3 EVALUATION COMPLETE!")
        print("="*60)
        print(f"📊 Meta-Model Performance:")
        print(f"   AUC: {results['cv_scores']['auc_mean']:.4f} ± {results['cv_scores']['auc_std']:.4f}")
        print(f"   F1: {results['cv_scores']['f1_mean']:.4f} ± {results['cv_scores']['f1_std']:.4f}")
        print(f"📁 Results: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()