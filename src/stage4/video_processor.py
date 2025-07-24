#!/usr/bin/env python3
"""
Advanced Video Processing & Temporal Analysis - video_processor.py
=================================================================

Enhanced video processing system with temporal consistency analysis,
real-time streaming capabilities, and adaptive content-aware processing.

Key Features:
- Temporal consistency analysis with motion-based features
- Adaptive frame sampling based on content and scene changes
- Real-time streaming video processing with buffer management
- Optical flow integration for motion-aware detection
- Scene change detection for intelligent processing
- Memory-efficient parallel processing architecture

Usage:
    # Process single video file
    processor = VideoProcessor()
    results = processor.process_video("video.mp4")
    
    # Real-time streaming processing
    stream_processor = StreamingVideoProcessor()
    for frame_result in stream_processor.process_stream(video_source):
        print(f"Frame {frame_result['frame_id']}: {frame_result['prediction']}")
"""

import os
import sys
import cv2
import json
import time
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator, Callable
from dataclasses import dataclass, asdict
from collections import deque
import concurrent.futures

import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.stage4.cascade_detector import CascadeDetector

@dataclass
class VideoProcessingConfig:
    """Configuration for video processing"""
    # Temporal analysis
    temporal_window_size: int = 8          # Frames for temporal analysis
    temporal_stride: int = 2               # Stride between temporal windows
    optical_flow_integration: bool = True  # Enable optical flow features
    frame_consistency_threshold: float = 0.8  # Consistency scoring threshold
    
    # Adaptive processing
    dynamic_sampling_rate: bool = True     # Content-aware frame sampling
    base_sampling_rate: int = 5            # Base frames per second to process
    max_sampling_rate: int = 15            # Maximum fps during high activity
    scene_change_threshold: float = 0.3    # Scene change detection threshold
    quality_threshold: float = 0.6         # Skip frames below this quality
    
    # Real-time optimization
    streaming_mode: bool = False           # Enable streaming processing
    buffer_size: int = 32                  # Frame buffer size
    max_processing_delay_ms: float = 100   # Maximum acceptable delay
    parallel_workers: int = 2              # Number of parallel processing workers
    
    # Output configuration
    save_temporal_features: bool = False   # Save temporal analysis features
    generate_heatmaps: bool = False        # Generate attention heatmaps
    output_video_analysis: bool = True     # Save annotated video
    
    # Quality and performance
    target_resolution: Tuple[int, int] = (256, 256)  # Processing resolution
    enable_gpu_acceleration: bool = True   # Use GPU when available
    memory_optimization: bool = True       # Enable memory optimizations

@dataclass
class FrameAnalysisResult:
    """Individual frame analysis result"""
    frame_id: int
    timestamp: float
    
    # Basic detection results
    prediction: str  # 'real' or 'fake'
    confidence: float
    cascade_stages_used: List[str]
    
    # Temporal analysis
    temporal_consistency_score: float = 0.0
    motion_magnitude: float = 0.0
    scene_change_score: float = 0.0
    
    # Quality metrics
    frame_quality_score: float = 0.0
    blur_score: float = 0.0
    brightness_score: float = 0.0
    
    # Processing metadata
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class VideoAnalysisResult:
    """Complete video analysis result"""
    video_path: str
    total_frames: int
    processed_frames: int
    
    # Overall prediction
    final_prediction: str
    overall_confidence: float
    fake_frame_percentage: float
    
    # Temporal analysis
    temporal_consistency_score: float
    scene_changes_detected: int
    motion_profile: List[float]
    
    # Performance metrics
    total_processing_time_seconds: float
    average_fps: float
    peak_memory_usage_mb: float
    
    # Detailed results
    frame_results: List[FrameAnalysisResult]
    temporal_features: Optional[Dict[str, Any]] = None

class OpticalFlowAnalyzer:
    """Optical flow analysis for motion-based features"""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.prev_frame = None
        self.flow_history = deque(maxlen=config.temporal_window_size)
        
    def compute_optical_flow(self, current_frame: np.ndarray) -> Dict[str, float]:
        """Compute optical flow features between consecutive frames"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            return {'motion_magnitude': 0.0, 'motion_coherence': 0.0, 'motion_direction_std': 0.0}
        
        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, current_gray, None, None)
        
        if flow is not None:
            # Calculate motion statistics
            motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion_magnitude = np.mean(motion_magnitude)
            
            # Motion coherence (how consistent the motion is)
            flow_angles = np.arctan2(flow[..., 1], flow[..., 0])
            motion_coherence = 1.0 - np.std(flow_angles) / np.pi
            
            # Motion direction standard deviation
            motion_direction_std = np.std(flow_angles)
            
            # Store in history
            self.flow_history.append({
                'magnitude': avg_motion_magnitude,
                'coherence': motion_coherence,
                'direction_std': motion_direction_std
            })
            
        else:
            avg_motion_magnitude = 0.0
            motion_coherence = 0.0
            motion_direction_std = 0.0
        
        # Update previous frame
        self.prev_frame = current_gray.copy()
        
        return {
            'motion_magnitude': float(avg_motion_magnitude),
            'motion_coherence': float(motion_coherence),
            'motion_direction_std': float(motion_direction_std)
        }
    
    def get_temporal_motion_features(self) -> Dict[str, float]:
        """Get aggregated temporal motion features"""
        if not self.flow_history:
            return {'temporal_motion_consistency': 0.0, 'motion_trend': 0.0}
        
        # Calculate temporal consistency of motion
        magnitudes = [f['magnitude'] for f in self.flow_history]
        motion_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0.0
        temporal_consistency = 1.0 / (1.0 + motion_variance)  # Higher consistency = lower variance
        
        # Motion trend (increasing/decreasing motion over time)
        if len(magnitudes) >= 3:
            # Use simple linear trend
            x = np.arange(len(magnitudes))
            trend = np.polyfit(x, magnitudes, 1)[0]  # Slope of linear fit
        else:
            trend = 0.0
        
        return {
            'temporal_motion_consistency': float(temporal_consistency),
            'motion_trend': float(trend)
        }

class SceneChangeDetector:
    """Scene change detection for adaptive processing"""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.prev_histogram = None
        self.scene_change_history = deque(maxlen=10)
        
    def detect_scene_change(self, frame: np.ndarray) -> float:
        """Detect scene change using histogram comparison"""
        # Convert to HSV for better color representation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if self.prev_histogram is not None:
            # Compare histograms using correlation
            correlation = cv2.compareHist(self.prev_histogram, hist, cv2.HISTCMP_CORREL)
            scene_change_score = 1.0 - correlation  # Higher score = more change
            
            self.scene_change_history.append(scene_change_score)
        else:
            scene_change_score = 0.0
        
        self.prev_histogram = hist.copy()
        return float(scene_change_score)
    
    def is_scene_change(self, current_score: float) -> bool:
        """Determine if current frame represents a scene change"""
        return current_score > self.config.scene_change_threshold
    
    def get_scene_activity_level(self) -> float:
        """Get overall scene activity level from recent history"""
        if not self.scene_change_history:
            return 0.0
        return float(np.mean(self.scene_change_history))

class FrameQualityAssessor:
    """Frame quality assessment for intelligent filtering"""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        
    def assess_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Assess frame quality using multiple metrics"""
        # Convert to grayscale for some calculations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Blur detection using Laplacian variance
        blur_score = self._calculate_blur_score(gray_frame)
        
        # Brightness assessment
        brightness_score = self._calculate_brightness_score(gray_frame)
        
        # Contrast assessment
        contrast_score = self._calculate_contrast_score(gray_frame)
        
        # Noise assessment
        noise_score = self._calculate_noise_score(gray_frame)
        
        # Overall quality score (weighted combination)
        overall_quality = (
            0.3 * (1.0 - blur_score) +      # Less blur = higher quality
            0.2 * brightness_score +         # Good brightness = higher quality
            0.3 * contrast_score +           # Good contrast = higher quality
            0.2 * (1.0 - noise_score)       # Less noise = higher quality
        )
        
        return {
            'overall_quality': float(overall_quality),
            'blur_score': float(blur_score),
            'brightness_score': float(brightness_score),
            'contrast_score': float(contrast_score),
            'noise_score': float(noise_score)
        }
    
    def _calculate_blur_score(self, gray_frame: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (higher variance = less blur)
        # Typical variance values: <100 (blurry), >1000 (sharp)
        normalized = min(variance / 1000.0, 1.0)
        blur_score = 1.0 - normalized  # Convert to blur score (higher = more blurry)
        
        return blur_score
    
    def _calculate_brightness_score(self, gray_frame: np.ndarray) -> float:
        """Calculate brightness score (penalize over/under-exposed)"""
        mean_brightness = np.mean(gray_frame) / 255.0
        
        # Optimal brightness is around 0.4-0.6
        if 0.4 <= mean_brightness <= 0.6:
            brightness_score = 1.0
        elif mean_brightness < 0.4:
            brightness_score = mean_brightness / 0.4  # Penalize dark images
        else:  # mean_brightness > 0.6
            brightness_score = (1.0 - mean_brightness) / 0.4  # Penalize bright images
        
        return max(0.0, brightness_score)
    
    def _calculate_contrast_score(self, gray_frame: np.ndarray) -> float:
        """Calculate contrast score using standard deviation"""
        contrast = np.std(gray_frame) / 255.0
        
        # Good contrast is typically > 0.2
        contrast_score = min(contrast / 0.2, 1.0)
        
        return contrast_score
    
    def _calculate_noise_score(self, gray_frame: np.ndarray) -> float:
        """Calculate noise score using high-frequency content"""
        # Apply Gaussian blur and calculate difference
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        noise = np.abs(gray_frame.astype(float) - blurred.astype(float))
        noise_score = np.mean(noise) / 255.0
        
        return min(noise_score * 5.0, 1.0)  # Amplify and clamp
    
    def should_skip_frame(self, quality_metrics: Dict[str, float]) -> bool:
        """Determine if frame should be skipped based on quality"""
        return quality_metrics['overall_quality'] < self.config.quality_threshold

class TemporalConsistencyAnalyzer:
    """Temporal consistency analysis for video-level detection"""
    
    def __init__(self, config: VideoProcessingConfig):
        self.config = config
        self.prediction_history = deque(maxlen=config.temporal_window_size)
        self.confidence_history = deque(maxlen=config.temporal_window_size)
        
    def add_frame_result(self, prediction: str, confidence: float):
        """Add frame result to temporal analysis"""
        pred_value = 1.0 if prediction == 'fake' else 0.0
        self.prediction_history.append(pred_value)
        self.confidence_history.append(confidence)
    
    def calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency score"""
        if len(self.prediction_history) < 2:
            return 0.0
        
        # Calculate prediction consistency (low variance = high consistency)
        pred_variance = np.var(self.prediction_history)
        pred_consistency = 1.0 / (1.0 + pred_variance * 4.0)  # Scale and invert
        
        # Calculate confidence consistency
        conf_variance = np.var(self.confidence_history)
        conf_consistency = 1.0 / (1.0 + conf_variance * 4.0)
        
        # Combined consistency score
        overall_consistency = 0.7 * pred_consistency + 0.3 * conf_consistency
        
        return float(overall_consistency)
    
    def get_smoothed_prediction(self) -> Tuple[str, float]:
        """Get temporally smoothed prediction"""
        if not self.prediction_history:
            return 'real', 0.5
        
        # Apply temporal smoothing using moving average
        if len(self.prediction_history) >= 3:
            smoothed_values = savgol_filter(
                list(self.prediction_history), 
                window_length=min(len(self.prediction_history), 5),
                polyorder=1
            )
            smoothed_pred = smoothed_values[-1]
        else:
            smoothed_pred = np.mean(self.prediction_history)
        
        # Calculate smoothed confidence
        smoothed_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.5
        
        # Convert to prediction
        prediction = 'fake' if smoothed_pred > 0.5 else 'real'
        
        return prediction, float(smoothed_confidence)

class VideoProcessor:
    """Main video processing system with temporal analysis"""
    
    def __init__(self, config: Optional[VideoProcessingConfig] = None):
        self.config = config or VideoProcessingConfig()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize cascade detector
        self.cascade_detector = CascadeDetector()
        
        # Initialize analysis components
        self.optical_flow_analyzer = OpticalFlowAnalyzer(self.config)
        self.scene_change_detector = SceneChangeDetector(self.config)
        self.quality_assessor = FrameQualityAssessor(self.config)
        self.temporal_analyzer = TemporalConsistencyAnalyzer(self.config)
        
        # Processing state
        self.frame_buffer = deque(maxlen=self.config.buffer_size)
        self.processing_stats = {'frames_processed': 0, 'frames_skipped': 0}
        
        self.logger.info(f"🎬 VideoProcessor initialized")
        self.logger.info(f"Temporal window size: {self.config.temporal_window_size}")
        self.logger.info(f"Streaming mode: {self.config.streaming_mode}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('VideoProcessor')
    
    def process_video(self, video_path: Union[str, Path]) -> VideoAnalysisResult:
        """Process complete video file with temporal analysis"""
        self.logger.info(f"🎬 Processing video: {video_path}")
        
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.logger.info(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS")
        
        # Process frames
        frame_results = []
        scene_changes = 0
        motion_profile = []
        peak_memory = 0
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                frame_id = 0
                processed_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    frame_result = self._process_single_frame(
                        frame_rgb, frame_id, frame_id / fps
                    )
                    
                    if frame_result is not None:
                        frame_results.append(frame_result)
                        processed_count += 1
                        
                        # Track scene changes
                        if frame_result.scene_change_score > self.config.scene_change_threshold:
                            scene_changes += 1
                        
                        # Track motion profile
                        motion_profile.append(frame_result.motion_magnitude)
                        
                        # Track memory usage
                        import psutil
                        current_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                        peak_memory = max(peak_memory, current_memory)
                    
                    frame_id += 1
                    pbar.update(1)
                    
                    # Adaptive sampling (skip frames if enabled)
                    if self.config.dynamic_sampling_rate and self._should_skip_next_frames():
                        skip_count = self._calculate_skip_count()
                        for _ in range(skip_count):
                            ret, _ = cap.read()
                            if not ret:
                                break
                            frame_id += 1
                            pbar.update(1)
        
        finally:
            cap.release()
        
        # Calculate final results
        processing_time = time.time() - start_time
        
        # Aggregate predictions
        fake_predictions = [r for r in frame_results if r.prediction == 'fake']
        fake_percentage = len(fake_predictions) / len(frame_results) * 100 if frame_results else 0
        
        # Overall prediction based on fake percentage and confidence
        if fake_percentage > 30:  # If more than 30% frames are fake
            final_prediction = 'fake'
            overall_confidence = np.mean([r.confidence for r in fake_predictions])
        else:
            final_prediction = 'real'
            real_predictions = [r for r in frame_results if r.prediction == 'real']
            overall_confidence = np.mean([r.confidence for r in real_predictions]) if real_predictions else 0.5
        
        # Temporal consistency
        temporal_consistency = self.temporal_analyzer.calculate_temporal_consistency()
        
        # Create result
        result = VideoAnalysisResult(
            video_path=str(video_path),
            total_frames=total_frames,
            processed_frames=processed_count,
            final_prediction=final_prediction,
            overall_confidence=float(overall_confidence),
            fake_frame_percentage=fake_percentage,
            temporal_consistency_score=temporal_consistency,
            scene_changes_detected=scene_changes,
            motion_profile=motion_profile,
            total_processing_time_seconds=processing_time,
            average_fps=processed_count / processing_time if processing_time > 0 else 0,
            peak_memory_usage_mb=peak_memory,
            frame_results=frame_results
        )
        
        self.logger.info(f"✅ Video processing completed:")
        self.logger.info(f"  Final prediction: {final_prediction} ({overall_confidence:.3f})")
        self.logger.info(f"  Fake frames: {fake_percentage:.1f}%")
        self.logger.info(f"  Processing time: {processing_time:.1f}s")
        self.logger.info(f"  Average FPS: {result.average_fps:.1f}")
        
        return result
    
    def _process_single_frame(self, frame: np.ndarray, 
                            frame_id: int, 
                            timestamp: float) -> Optional[FrameAnalysisResult]:
        """Process individual frame with all analysis components"""
        start_time = time.time()
        
        # Resize frame for processing
        if frame.shape[:2] != self.config.target_resolution:
            frame = cv2.resize(frame, self.config.target_resolution[::-1])  # (w, h) format
        
        # Quality assessment
        quality_metrics = self.quality_assessor.assess_frame_quality(frame)
        
        # Skip frame if quality is too low
        if self.quality_assessor.should_skip_frame(quality_metrics):
            self.processing_stats['frames_skipped'] += 1
            return None
        
        # Scene change detection
        scene_change_score = self.scene_change_detector.detect_scene_change(frame)
        
        # Optical flow analysis
        motion_features = self.optical_flow_analyzer.compute_optical_flow(frame)
        
        # Run cascade detection
        detection_result = self.cascade_detector.predict(frame)
        
        # Extract results
        prediction = detection_result['final_prediction']
        confidence = detection_result['final_probability']
        stages_used = [stage for stage, used in detection_result.get('stages_used', {}).items() if used]
        
        # Add to temporal analysis
        self.temporal_analyzer.add_frame_result(prediction, confidence)
        
        # Calculate temporal consistency
        temporal_consistency = self.temporal_analyzer.calculate_temporal_consistency()
        
        # Calculate processing metrics
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create frame result
        frame_result = FrameAnalysisResult(
            frame_id=frame_id,
            timestamp=timestamp,
            prediction=prediction,
            confidence=confidence,
            cascade_stages_used=stages_used,
            temporal_consistency_score=temporal_consistency,
            motion_magnitude=motion_features['motion_magnitude'],
            scene_change_score=scene_change_score,
            frame_quality_score=quality_metrics['overall_quality'],
            blur_score=quality_metrics['blur_score'],
            brightness_score=quality_metrics['brightness_score'],
            processing_time_ms=processing_time_ms,
            memory_usage_mb=0.0  # Would need process monitoring
        )
        
        self.processing_stats['frames_processed'] += 1
        return frame_result
    
    def _should_skip_next_frames(self) -> bool:
        """Determine if next frames should be skipped for adaptive sampling"""
        if not self.config.dynamic_sampling_rate:
            return False
        
        # Skip frames in low-activity scenes
        scene_activity = self.scene_change_detector.get_scene_activity_level()
        return scene_activity < 0.1  # Low activity threshold
    
    def _calculate_skip_count(self) -> int:
        """Calculate how many frames to skip"""
        scene_activity = self.scene_change_detector.get_scene_activity_level()
        
        # Skip more frames in very static scenes
        if scene_activity < 0.05:
            return 3  # Skip 3 frames
        elif scene_activity < 0.1:
            return 1  # Skip 1 frame
        else:
            return 0  # Don't skip
    
    def generate_analysis_report(self, result: VideoAnalysisResult, 
                               output_path: Optional[Path] = None) -> Path:
        """Generate comprehensive analysis report"""
        if output_path is None:
            output_path = Path(f"video_analysis_report_{int(time.time())}.json")
        
        # Convert result to serializable format
        report_data = asdict(result)
        
        # Add processing statistics
        report_data['processing_statistics'] = self.processing_stats
        
        # Add configuration
        report_data['processing_config'] = asdict(self.config)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Analysis report saved: {output_path}")
        return output_path
    
    def create_visualization(self, result: VideoAnalysisResult, 
                           output_dir: Optional[Path] = None) -> List[Path]:
        """Create visualization plots for video analysis"""
        if output_dir is None:
            output_dir = Path("video_analysis_plots")
        
        output_dir.mkdir(exist_ok=True)
        plot_files = []
        
        try:
            # Temporal consistency plot
            consistency_plot = self._plot_temporal_consistency(result, output_dir)
            plot_files.append(consistency_plot)
            
            # Motion profile plot
            motion_plot = self._plot_motion_profile(result, output_dir)
            plot_files.append(motion_plot)
            
            # Quality assessment plot
            quality_plot = self._plot_quality_metrics(result, output_dir)
            plot_files.append(quality_plot)
            
            # Prediction timeline plot
            timeline_plot = self._plot_prediction_timeline(result, output_dir)
            plot_files.append(timeline_plot)
            
            self.logger.info(f"📊 Visualization plots created in: {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create some visualizations: {e}")
        
        return plot_files
    
    def _plot_temporal_consistency(self, result: VideoAnalysisResult, output_dir: Path) -> Path:
        """Plot temporal consistency over time"""
        timestamps = [r.timestamp for r in result.frame_results]
        consistency_scores = [r.temporal_consistency_score for r in result.frame_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, consistency_scores, linewidth=2, alpha=0.8)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temporal Consistency Score')
        plt.title(f'Temporal Consistency Analysis - {Path(result.video_path).name}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add scene change markers
        scene_changes = [(r.timestamp, r.scene_change_score) for r in result.frame_results 
                        if r.scene_change_score > self.config.scene_change_threshold]
        if scene_changes:
            change_times, _ = zip(*scene_changes)
            plt.scatter(change_times, [0.9] * len(change_times), 
                       color='red', alpha=0.6, s=50, label='Scene Changes')
            plt.legend()
        
        output_path = output_dir / 'temporal_consistency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_motion_profile(self, result: VideoAnalysisResult, output_dir: Path) -> Path:
        """Plot motion profile over time"""
        timestamps = [r.timestamp for r in result.frame_results]
        motion_magnitudes = [r.motion_magnitude for r in result.frame_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, motion_magnitudes, linewidth=2, alpha=0.8, color='green')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Motion Magnitude')
        plt.title(f'Motion Profile - {Path(result.video_path).name}')
        plt.grid(True, alpha=0.3)
        
        # Add smoothed trend line
        if len(motion_magnitudes) > 5:
            smoothed = savgol_filter(motion_magnitudes, window_length=min(11, len(motion_magnitudes)), polyorder=2)
            plt.plot(timestamps, smoothed, '--', color='red', alpha=0.7, label='Smoothed Trend')
            plt.legend()
        
        output_path = output_dir / 'motion_profile.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_quality_metrics(self, result: VideoAnalysisResult, output_dir: Path) -> Path:
        """Plot frame quality metrics"""
        timestamps = [r.timestamp for r in result.frame_results]
        quality_scores = [r.frame_quality_score for r in result.frame_results]
        blur_scores = [r.blur_score for r in result.frame_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Overall quality
        ax1.plot(timestamps, quality_scores, linewidth=2, alpha=0.8, color='blue')
        ax1.axhline(y=self.config.quality_threshold, color='red', linestyle='--', alpha=0.7, label='Quality Threshold')
        ax1.set_ylabel('Overall Quality Score')
        ax1.set_title(f'Frame Quality Assessment - {Path(result.video_path).name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Blur assessment
        ax2.plot(timestamps, blur_scores, linewidth=2, alpha=0.8, color='orange')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Blur Score')
        ax2.set_title('Blur Assessment')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        output_path = output_dir / 'quality_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_prediction_timeline(self, result: VideoAnalysisResult, output_dir: Path) -> Path:
        """Plot prediction timeline with confidence"""
        timestamps = [r.timestamp for r in result.frame_results]
        predictions = [1 if r.prediction == 'fake' else 0 for r in result.frame_results]
        confidences = [r.confidence for r in result.frame_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Prediction timeline
        colors = ['green' if p == 0 else 'red' for p in predictions]
        ax1.scatter(timestamps, predictions, c=colors, alpha=0.6, s=20)
        ax1.set_ylabel('Prediction (0=Real, 1=Fake)')
        ax1.set_title(f'Prediction Timeline - {Path(result.video_path).name}')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Confidence timeline
        confidence_colors = ['green' if p == 0 else 'red' for p in predictions]
        ax2.scatter(timestamps, confidences, c=confidence_colors, alpha=0.6, s=20)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Timeline')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add overall prediction line
        fake_threshold = 0.5
        ax2.axhline(y=fake_threshold, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        output_path = output_dir / 'prediction_timeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

class StreamingVideoProcessor:
    """Real-time streaming video processor"""
    
    def __init__(self, config: Optional[VideoProcessingConfig] = None):
        self.config = config or VideoProcessingConfig(streaming_mode=True)
        self.base_processor = VideoProcessor(self.config)
        
        # Streaming-specific components
        self.frame_queue = queue.Queue(maxsize=self.config.buffer_size)
        self.result_queue = queue.Queue()
        self.processing_workers = []
        self.is_processing = False
        
        self.logger = self.base_processor.logger
        self.logger.info(f"🔴 StreamingVideoProcessor initialized with {self.config.parallel_workers} workers")
    
    def process_stream(self, video_source: Union[str, int], 
                      callback: Optional[Callable[[FrameAnalysisResult], None]] = None) -> Iterator[FrameAnalysisResult]:
        """Process video stream in real-time"""
        self.logger.info(f"🔴 Starting stream processing: {video_source}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        
        # Start processing workers
        self._start_processing_workers()
        
        try:
            frame_id = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = time.time() - start_time
                
                # Add frame to processing queue
                try:
                    self.frame_queue.put((frame_rgb, frame_id, timestamp), timeout=0.1)
                except queue.Full:
                    self.logger.warning("Frame queue full, dropping frame")
                    continue
                
                # Get processed results
                try:
                    while True:
                        result = self.result_queue.get_nowait()
                        if callback:
                            callback(result)
                        yield result
                except queue.Empty:
                    pass
                
                frame_id += 1
                
                # Check for processing delay
                if timestamp > self.config.max_processing_delay_ms / 1000.0:
                    self.logger.warning(f"Processing delay detected: {timestamp*1000:.1f}ms")
        
        finally:
            # Cleanup
            cap.release()
            self._stop_processing_workers()
            
            # Get remaining results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if callback:
                        callback(result)
                    yield result
                except queue.Empty:
                    break
    
    def _start_processing_workers(self):
        """Start parallel processing workers"""
        self.is_processing = True
        
        for i in range(self.config.parallel_workers):
            worker = threading.Thread(target=self._processing_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.processing_workers.append(worker)
        
        self.logger.info(f"✅ Started {len(self.processing_workers)} processing workers")
    
    def _stop_processing_workers(self):
        """Stop processing workers"""
        self.is_processing = False
        
        # Wait for workers to finish
        for worker in self.processing_workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.processing_workers.clear()
        self.logger.info("🛑 Processing workers stopped")
    
    def _processing_worker(self, worker_id: int):
        """Worker thread for processing frames"""
        self.logger.info(f"🔧 Processing worker {worker_id} started")
        
        while self.is_processing:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                frame_rgb, frame_id, timestamp = frame_data
                
                # Process frame
                result = self.base_processor._process_single_frame(
                    frame_rgb, frame_id, timestamp
                )
                
                if result is not None:
                    # Add to result queue
                    try:
                        self.result_queue.put(result, timeout=0.1)
                    except queue.Full:
                        self.logger.warning(f"Worker {worker_id}: Result queue full")
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
        
        self.logger.info(f"🔧 Processing worker {worker_id} stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Video Processing')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output_dir', type=str, default='video_analysis_output',
                       help='Output directory for results')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming mode')
    parser.add_argument('--temporal_window', type=int, default=8, help='Temporal analysis window size')
    parser.add_argument('--save_plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--quality_threshold', type=float, default=0.6, help='Frame quality threshold')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VideoProcessingConfig(
        temporal_window_size=args.temporal_window,
        streaming_mode=args.streaming,
        quality_threshold=args.quality_threshold,
        save_temporal_features=True,
        generate_heatmaps=args.save_plots
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.streaming:
        # Streaming processing
        processor = StreamingVideoProcessor(config)
        
        def result_callback(result: FrameAnalysisResult):
            print(f"Frame {result.frame_id}: {result.prediction} ({result.confidence:.3f})")
        
        print("🔴 Starting streaming processing...")
        for result in processor.process_stream(args.video_path, callback=result_callback):
            pass  # Results handled by callback
        
    else:
        # Batch processing
        processor = VideoProcessor(config)
        
        print(f"🎬 Processing video: {args.video_path}")
        result = processor.process_video(args.video_path)
        
        print(f"\n✅ Processing completed:")
        print(f"Final prediction: {result.final_prediction}")
        print(f"Confidence: {result.overall_confidence:.3f}")
        print(f"Fake frames: {result.fake_frame_percentage:.1f}%")
        print(f"Processing time: {result.total_processing_time_seconds:.1f}s")
        
        # Save report
        report_path = processor.generate_analysis_report(result, output_dir / "analysis_report.json")
        print(f"📄 Report saved: {report_path}")
        
        # Create visualizations
        if args.save_plots:
            plot_files = processor.create_visualization(result, output_dir / "plots")
            print(f"📊 Plots created: {len(plot_files)} files")

if __name__ == "__main__":
    main()