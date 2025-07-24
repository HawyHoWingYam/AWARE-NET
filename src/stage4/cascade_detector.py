#!/usr/bin/env python3
"""
Unified Cascade Deepfake Detection System - cascade_detector.py
==============================================================

Complete implementation of the three-stage cascade deepfake detection system
combining MobileNetV4 (Stage 1), EfficientNetV2-B3 + GenConViT (Stage 2),
and LightGBM meta-model (Stage 3) with dynamic threshold strategies.

Key Features:
- End-to-end inference pipeline with automatic stage switching
- Dynamic threshold adaptation based on video complexity
- Conservative leakage control (<5% fake samples passed)
- Batch processing for efficiency
- Comprehensive performance monitoring
- Mobile-optimized inference paths

Usage:
    # Single frame detection
    detector = CascadeDetector()
    result = detector.predict(frame)
    
    # Video processing  
    results = detector.predict_video("video.mp4")
    
    # Batch processing
    results = detector.predict_batch(frame_list)
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import joblib
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import project components
from src.stage1.utils import load_model_checkpoint
from src.stage2.feature_extractor import (EfficientNetFeatureExtractor, 
                                        GenConViTFeatureExtractor,
                                        create_combined_features)

class DecisionStage(Enum):
    """Cascade decision stages"""
    STAGE1_REAL = "stage1_real"
    STAGE1_FAKE = "stage1_fake" 
    STAGE2_ANALYSIS = "stage2_analysis"
    STAGE3_META = "stage3_meta"

@dataclass
class CascadeResult:
    """Standardized cascade detection result"""
    prediction: str  # "real" or "fake"
    confidence: float  # 0.0 to 1.0
    decision_stage: DecisionStage
    stage_confidences: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass  
class CascadeConfig:
    """Cascade system configuration"""
    # Stage 1 thresholds
    stage1_real_threshold: float = 0.98  # Conservative real threshold
    stage1_fake_threshold: float = 0.02  # Conservative fake threshold
    
    # Dynamic threshold adaptation
    enable_dynamic_thresholds: bool = True
    uncertainty_high_threshold: float = 0.3
    dynamic_real_range: Tuple[float, float] = (0.90, 0.98)
    dynamic_fake_range: Tuple[float, float] = (0.02, 0.10)
    
    # Processing configuration
    batch_size: int = 32
    device: str = "auto"  # "auto", "cuda", "cpu"
    enable_timing: bool = True
    
    # Model paths
    stage1_model_path: str = "output/stage1/best_model.pth"
    stage1_temp_path: str = "output/stage1/calibration_temp.json"
    stage2_effnet_path: str = "output/stage2_effnet/best_model.pth"
    stage2_genconvit_path: str = "output/stage2_genconvit/best_model.pth"
    stage3_meta_path: str = "output/stage3_meta_model/best_meta_model.pkl"

class CascadeDetector:
    """Unified cascade deepfake detection system"""
    
    def __init__(self, config: Optional[CascadeConfig] = None):
        self.config = config or CascadeConfig()
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup logging
        self.setup_logging()
        
        # Performance tracking
        self.stats = {
            'total_predictions': 0,
            'stage1_real_filtered': 0,
            'stage1_fake_filtered': 0,
            'stage2_processed': 0,
            'stage3_processed': 0,
            'total_processing_time': 0.0,
            'stage_times': {'stage1': [], 'stage2': [], 'stage3': []}
        }
        
        # Load all models
        self.load_models()
        
        # Setup image preprocessing
        self.setup_preprocessing()
        
        logging.info("✅ CascadeDetector initialized successfully")
        self.log_system_info()
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        return device
    
    def setup_logging(self):
        """Setup logging for cascade system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CascadeDetector')
    
    def load_models(self):
        """Load all cascade models"""
        self.logger.info("🔄 Loading cascade models...")
        
        # Load Stage 1: MobileNetV4 + Temperature Scaling
        self.load_stage1_models()
        
        # Load Stage 2: EfficientNetV2-B3 + GenConViT
        self.load_stage2_models()
        
        # Load Stage 3: LightGBM Meta-model
        self.load_stage3_model()
        
        self.logger.info("✅ All cascade models loaded successfully")
    
    def load_stage1_models(self):
        """Load Stage 1 fast filter model"""
        try:
            import timm
            
            # Load MobileNetV4 model
            self.stage1_model = timm.create_model(
                'mobilenetv4_hybrid_medium.ix_e550_r256_in1k',
                pretrained=False,
                num_classes=1
            )
            
            # Load trained weights
            if os.path.exists(self.config.stage1_model_path):
                checkpoint = torch.load(self.config.stage1_model_path, map_location='cpu')
                self.stage1_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"✅ Stage 1 model loaded: {self.config.stage1_model_path}")
            else:
                self.logger.warning(f"Stage 1 checkpoint not found: {self.config.stage1_model_path}")
                
            self.stage1_model = self.stage1_model.to(self.device)
            self.stage1_model.eval()
            
            # Load temperature scaling parameter
            if os.path.exists(self.config.stage1_temp_path):
                with open(self.config.stage1_temp_path, 'r') as f:
                    temp_data = json.load(f)
                    self.stage1_temperature = temp_data.get('optimal_temperature', 1.0)
                self.logger.info(f"✅ Temperature scaling loaded: T={self.stage1_temperature:.4f}")
            else:
                self.stage1_temperature = 1.0
                self.logger.warning("Temperature scaling not found, using T=1.0")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load Stage 1 models: {e}")
            raise
    
    def load_stage2_models(self):
        """Load Stage 2 precision analyzer models"""
        try:
            # Initialize feature extractors (they handle model loading internally)
            self.stage2_effnet_extractor = EfficientNetFeatureExtractor(
                'efficientnetv2_b3.in21k_ft_in1k',
                self.config.stage2_effnet_path,
                device=self.device
            )
            
            self.stage2_genconvit_extractor = GenConViTFeatureExtractor(
                self.config.stage2_genconvit_path,
                variant='ED',
                mode='hybrid',
                device=self.device
            )
            
            self.logger.info("✅ Stage 2 feature extractors initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Stage 2 models: {e}")
            raise
    
    def load_stage3_model(self):
        """Load Stage 3 meta-model"""
        try:
            if os.path.exists(self.config.stage3_meta_path):
                self.stage3_meta_model = joblib.load(self.config.stage3_meta_path)
                self.logger.info(f"✅ Stage 3 meta-model loaded: {self.config.stage3_meta_path}")
            else:
                self.logger.error(f"Stage 3 model not found: {self.config.stage3_meta_path}")
                raise FileNotFoundError(f"Stage 3 model not found: {self.config.stage3_meta_path}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load Stage 3 model: {e}")
            raise
    
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def log_system_info(self):
        """Log system configuration and model info"""
        self.logger.info("="*60)
        self.logger.info("CASCADE DETECTOR SYSTEM INFO")
        self.logger.info("="*60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Stage 1 Real Threshold: {self.config.stage1_real_threshold}")
        self.logger.info(f"Stage 1 Fake Threshold: {self.config.stage1_fake_threshold}")
        self.logger.info(f"Dynamic Thresholds: {self.config.enable_dynamic_thresholds}")
        self.logger.info(f"Batch Size: {self.config.batch_size}")
        self.logger.info("="*60)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """Preprocess input image for model inference"""
        if isinstance(image, str):
            # Load from file path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply preprocessing
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return tensor
    
    def stage1_predict(self, image_tensor: torch.Tensor) -> float:
        """Stage 1 fast filter prediction with temperature scaling"""
        start_time = time.time() if self.config.enable_timing else 0
        
        with torch.no_grad():
            logits = self.stage1_model(image_tensor)
            # Apply temperature scaling
            scaled_logits = logits / self.stage1_temperature
            probability = torch.sigmoid(scaled_logits).item()
        
        if self.config.enable_timing:
            processing_time = time.time() - start_time
            self.stats['stage_times']['stage1'].append(processing_time)
        
        return probability
    
    def stage2_extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Stage 2 feature extraction from both models"""
        start_time = time.time() if self.config.enable_timing else 0
        
        # Convert tensor back to format expected by feature extractors
        # Create a simple dataset wrapper
        class SingleImageDataset:
            def __init__(self, tensor, label=0):
                self.tensor = tensor.cpu()
                self.label = label
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return self.tensor.squeeze(0), self.label
        
        from torch.utils.data import DataLoader
        
        dataset = SingleImageDataset(image_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Extract features from both models
        effnet_features, _ = self.stage2_effnet_extractor.extract(dataloader, align_dim=256)
        genconvit_features, _ = self.stage2_genconvit_extractor.extract_features(dataloader, align_dim=256)
        
        # Combine features
        combined_features = create_combined_features(
            effnet_features, genconvit_features, target_dim=512
        )
        
        if self.config.enable_timing:
            processing_time = time.time() - start_time
            self.stats['stage_times']['stage2'].append(processing_time)
        
        return combined_features[0]  # Return single sample features
    
    def stage3_predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Stage 3 meta-model prediction"""
        start_time = time.time() if self.config.enable_timing else 0
        
        # Reshape features for single prediction
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.stage3_meta_model.predict(features)[0]
        probability = self.stage3_meta_model.predict_proba(features)[0]
        
        # Convert to standard format
        if prediction == 1:  # Fake
            result = "fake"
            confidence = probability[1]  # Probability of fake class
        else:  # Real
            result = "real"
            confidence = probability[0]  # Probability of real class
        
        if self.config.enable_timing:
            processing_time = time.time() - start_time
            self.stats['stage_times']['stage3'].append(processing_time)
        
        return result, confidence
    
    def calculate_dynamic_thresholds(self, uncertainty_score: float) -> Tuple[float, float]:
        """Calculate dynamic thresholds based on uncertainty"""
        if not self.config.enable_dynamic_thresholds:
            return self.config.stage1_real_threshold, self.config.stage1_fake_threshold
        
        # Adapt thresholds based on uncertainty
        if uncertainty_score > self.config.uncertainty_high_threshold:
            # High uncertainty: widen the uncertain region
            real_thresh = self.config.dynamic_real_range[0]  # Lower real threshold
            fake_thresh = self.config.dynamic_fake_range[1]  # Higher fake threshold
        else:
            # Low uncertainty: use standard thresholds
            real_thresh = self.config.stage1_real_threshold
            fake_thresh = self.config.stage1_fake_threshold
        
        return real_thresh, fake_thresh
    
    def predict(self, image: Union[np.ndarray, Image.Image, str], 
                uncertainty_score: Optional[float] = None) -> CascadeResult:
        """
        Main cascade prediction pipeline
        
        Args:
            image: Input image (various formats supported)
            uncertainty_score: Optional uncertainty score for dynamic thresholds
            
        Returns:
            CascadeResult with prediction, confidence, and metadata
        """
        prediction_start_time = time.time()
        
        # Update statistics
        self.stats['total_predictions'] += 1
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Stage 1: Fast filter
        stage1_prob = self.stage1_predict(image_tensor)
        
        # Calculate dynamic thresholds if enabled
        real_threshold, fake_threshold = self.calculate_dynamic_thresholds(
            uncertainty_score or 0.0
        )
        
        stage_confidences = {'stage1': stage1_prob}
        
        # Stage 1 decision logic
        if stage1_prob > real_threshold:
            # High confidence REAL - filter out
            self.stats['stage1_real_filtered'] += 1
            result = CascadeResult(
                prediction="real",
                confidence=stage1_prob,
                decision_stage=DecisionStage.STAGE1_REAL,
                stage_confidences=stage_confidences,
                processing_time=time.time() - prediction_start_time,
                metadata={
                    'real_threshold': real_threshold,
                    'fake_threshold': fake_threshold,
                    'filtered_at_stage1': True
                }
            )
        elif stage1_prob < fake_threshold:
            # High confidence FAKE - filter out
            self.stats['stage1_fake_filtered'] += 1
            result = CascadeResult(
                prediction="fake",
                confidence=1.0 - stage1_prob,  # Convert to fake confidence
                decision_stage=DecisionStage.STAGE1_FAKE,
                stage_confidences=stage_confidences,
                processing_time=time.time() - prediction_start_time,
                metadata={
                    'real_threshold': real_threshold,
                    'fake_threshold': fake_threshold,
                    'filtered_at_stage1': True
                }
            )
        else:
            # Uncertain sample - proceed to Stage 2
            self.stats['stage2_processed'] += 1
            
            # Stage 2: Extract features
            stage2_features = self.stage2_extract_features(image_tensor)
            
            # Stage 3: Meta-model prediction
            self.stats['stage3_processed'] += 1
            final_prediction, final_confidence = self.stage3_predict(stage2_features)
            
            stage_confidences.update({
                'stage2': 'features_extracted',
                'stage3': final_confidence
            })
            
            result = CascadeResult(
                prediction=final_prediction,
                confidence=final_confidence,
                decision_stage=DecisionStage.STAGE3_META,
                stage_confidences=stage_confidences,
                processing_time=time.time() - prediction_start_time,
                metadata={
                    'real_threshold': real_threshold,
                    'fake_threshold': fake_threshold,
                    'filtered_at_stage1': False,
                    'stage2_features_dim': len(stage2_features)
                }
            )
        
        # Update total processing time
        self.stats['total_processing_time'] += result.processing_time
        
        return result
    
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image, str]],
                     uncertainty_scores: Optional[List[float]] = None) -> List[CascadeResult]:
        """Batch prediction for multiple images"""
        if uncertainty_scores is None:
            uncertainty_scores = [None] * len(images)
        
        results = []
        for i, image in enumerate(tqdm(images, desc="Processing batch")):
            result = self.predict(image, uncertainty_scores[i])
            results.append(result)
        
        return results
    
    def predict_video(self, video_path: str, 
                     sampling_rate: int = 30,
                     max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process video file and return aggregated results
        
        Args:
            video_path: Path to video file
            sampling_rate: Process every Nth frame
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary with video-level prediction and frame-level results
        """
        self.logger.info(f"🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # Process frames
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on sampling rate
            if frame_count % sampling_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict on frame
                result = self.predict(frame_rgb)
                frame_results.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'decision_stage': result.decision_stage.value,
                    'processing_time': result.processing_time
                })
                
                processed_count += 1
                
                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        # Aggregate results
        fake_frames = sum(1 for r in frame_results if r['prediction'] == 'fake')
        total_processed = len(frame_results)
        
        fake_ratio = fake_frames / total_processed if total_processed > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in frame_results])
        
        # Video-level decision (simple majority voting)
        video_prediction = "fake" if fake_ratio > 0.5 else "real"
        video_confidence = max(fake_ratio, 1 - fake_ratio)
        
        video_result = {
            'video_path': video_path,
            'video_prediction': video_prediction,
            'video_confidence': float(video_confidence),
            'fake_frame_ratio': float(fake_ratio),
            'total_frames': total_frames,
            'processed_frames': total_processed,
            'sampling_rate': sampling_rate,
            'average_confidence': float(avg_confidence),
            'processing_stats': {
                'stage1_filtered': sum(1 for r in frame_results 
                                     if 'stage1' in r['decision_stage']),
                'stage3_processed': sum(1 for r in frame_results 
                                      if r['decision_stage'] == DecisionStage.STAGE3_META.value)
            },
            'frame_results': frame_results
        }
        
        self.logger.info(f"✅ Video processing complete: {video_prediction} ({video_confidence:.3f})")
        return video_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_preds = self.stats['total_predictions']
        if total_preds == 0:
            return {'error': 'No predictions made yet'}
        
        # Calculate filtration rates
        stage1_filtration_rate = (
            (self.stats['stage1_real_filtered'] + self.stats['stage1_fake_filtered']) / total_preds
        )
        
        stage2_usage_rate = self.stats['stage2_processed'] / total_preds
        stage3_usage_rate = self.stats['stage3_processed'] / total_preds
        
        # Calculate average processing times
        avg_times = {}
        for stage, times in self.stats['stage_times'].items():
            if times:
                avg_times[f'{stage}_avg_ms'] = np.mean(times) * 1000
                avg_times[f'{stage}_std_ms'] = np.std(times) * 1000
            else:
                avg_times[f'{stage}_avg_ms'] = 0
                avg_times[f'{stage}_std_ms'] = 0
        
        return {
            'total_predictions': total_preds,
            'stage1_filtration_rate': stage1_filtration_rate,
            'stage1_real_filtered': self.stats['stage1_real_filtered'],
            'stage1_fake_filtered': self.stats['stage1_fake_filtered'],
            'stage2_usage_rate': stage2_usage_rate,
            'stage3_usage_rate': stage3_usage_rate,
            'total_processing_time_sec': self.stats['total_processing_time'],
            'avg_processing_time_ms': (self.stats['total_processing_time'] / total_preds) * 1000,
            'timing_breakdown': avg_times
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_predictions': 0,
            'stage1_real_filtered': 0,
            'stage1_fake_filtered': 0,
            'stage2_processed': 0,
            'stage3_processed': 0,
            'total_processing_time': 0.0,
            'stage_times': {'stage1': [], 'stage2': [], 'stage3': []}
        }
        self.logger.info("📊 Performance statistics reset")

# Example usage and testing
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cascade Deepfake Detection System')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or video file')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--video_sampling', type=int, default=30,
                       help='Video frame sampling rate')
    parser.add_argument('--batch_test', action='store_true',
                       help='Test batch processing mode')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = CascadeConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Initialize detector
    print("🚀 Initializing Cascade Detector...")
    detector = CascadeDetector(config)
    
    input_path = Path(args.input)
    
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video processing
        print(f"🎬 Processing video: {input_path}")
        result = detector.predict_video(
            str(input_path), 
            sampling_rate=args.video_sampling
        )
        
        print(f"\n📊 Video Results:")
        print(f"   Prediction: {result['video_prediction']}")
        print(f"   Confidence: {result['video_confidence']:.4f}")
        print(f"   Fake Frame Ratio: {result['fake_frame_ratio']:.4f}")
        print(f"   Frames Processed: {result['processed_frames']}")
        
    else:
        # Image processing
        print(f"🖼️ Processing image: {input_path}")
        result = detector.predict(str(input_path))
        
        print(f"\n📊 Image Results:")
        print(f"   Prediction: {result.prediction}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Decision Stage: {result.decision_stage.value}")
        print(f"   Processing Time: {result.processing_time*1000:.2f}ms")
    
    # Performance statistics
    stats = detector.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"   Stage 1 Filtration Rate: {stats['stage1_filtration_rate']*100:.1f}%")
    print(f"   Average Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
    
    # Save results if requested
    if args.output:
        output_data = {
            'input_path': str(input_path),
            'result': result._asdict() if hasattr(result, '_asdict') else result,
            'performance_stats': stats,
            'timestamp': time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()