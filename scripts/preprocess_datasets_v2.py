#!/usr/bin/env python3
"""
数据预处理脚本 V2 - preprocess_datasets_v2.py
使用灵活的配置系统处理多种数据集结构

功能:
1. 基于JSON配置文件的灵活路径管理
2. 支持任意数据集目录结构
3. 智能的数据集解析和处理
4. 自动生成数据清单文件
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from facenet_pytorch import MTCNN
import warnings
warnings.filterwarnings('ignore')

# 导入配置管理器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.dataset_config import DatasetPathConfig

# =============================================================================
# 核心预处理类 V2
# =============================================================================

class DatasetPreprocessorV2:
    """数据集预处理器 V2 - 使用配置系统"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化预处理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认的dataset_paths.json
        """
        self.path_config = DatasetPathConfig(config_file)
        self.processing_config = self.path_config.get_processing_config()
        
        self.setup_directories()
        self.setup_face_detector()
        self.setup_logging()
        
    def setup_directories(self):
        """创建必要的目录结构"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        
        dirs_to_create = [
            f"{processed_path}/train/real",
            f"{processed_path}/train/fake", 
            f"{processed_path}/val/real",
            f"{processed_path}/val/fake",
            f"{processed_path}/final_test_sets/celebdf_v2/real",
            f"{processed_path}/final_test_sets/celebdf_v2/fake",
            f"{processed_path}/final_test_sets/ffpp/real",
            f"{processed_path}/final_test_sets/ffpp/fake",
            f"{processed_path}/final_test_sets/dfdc",
            f"{processed_path}/manifests"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Created directory structure at {processed_path}")
    
    def setup_face_detector(self):
        """初始化人脸检测器"""
        try:
            detector_config = self.processing_config.get("face_detector", {})
            
            self.face_detector = MTCNN(
                min_face_size=detector_config.get("min_face_size", 20),
                thresholds=detector_config.get("thresholds", [0.6, 0.7, 0.7]),
                factor=detector_config.get("factor", 0.709),
                post_process=detector_config.get("post_process", True),
                device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            )
            logging.info("Face detector initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize face detector: {e}")
            raise
    
    def setup_logging(self):
        """设置日志记录"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        log_file = f"{processed_path}/preprocessing_v2.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def process_video(self, video_info: Dict) -> int:
        """
        处理单个视频文件，提取人脸图像
        
        Args:
            video_info: 包含视频路径、标签、划分等信息的字典
            
        Returns:
            提取的人脸数量
        """
        video_path = video_info["video_path"]
        label = video_info["label"]
        split = video_info["split"]
        dataset_name = video_info["dataset"]
        video_id = video_info["video_id"]
        
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}")
            return 0
        
        # 确定输出目录
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        
        if split == 'test' and dataset_name in ['celebdf_v2', 'ffpp']:
            output_dir = f"{processed_path}/final_test_sets/{dataset_name}/{label}"
        elif dataset_name == 'dfdc':
            output_dir = f"{processed_path}/final_test_sets/dfdc"
        else:
            output_dir = f"{processed_path}/{split}/{label}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        face_count = 0
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Cannot open video: {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        frame_interval = self.processing_config.get("frame_interval", 10)
        max_faces = self.processing_config.get("max_faces_per_video", 50)
        
        try:
            while face_count < max_faces:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔处理帧
                if frame_idx % frame_interval == 0:
                    faces_extracted = self._extract_faces_from_frame(
                        frame, output_dir, video_id, frame_idx, face_count
                    )
                    face_count += faces_extracted
                
                frame_idx += 1
                
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
        finally:
            cap.release()
        
        return face_count
    
    def _extract_faces_from_frame(self, frame: np.ndarray, output_dir: str, 
                                 video_id: str, frame_idx: int, face_count: int) -> int:
        """从单帧中提取人脸"""
        try:
            # 转换为RGB格式 (MTCNN期望RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            boxes, _ = self.face_detector.detect(rgb_frame)
            
            if boxes is None:
                return 0
            
            extracted_count = 0
            max_faces = self.processing_config.get("max_faces_per_video", 50)
            image_size = tuple(self.processing_config.get("image_size", [224, 224]))
            bbox_scale = self.processing_config.get("bbox_scale", 1.3)
            min_face_size = self.processing_config.get("min_face_size", 80)
            
            for i, box in enumerate(boxes):
                if face_count + extracted_count >= max_faces:
                    break
                
                # 扩展边界框
                x1, y1, x2, y2 = box.astype(int)
                
                # 计算扩展后的边界框
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                
                new_width = int(width * bbox_scale)
                new_height = int(height * bbox_scale)
                
                x1_new = max(0, center_x - new_width // 2)
                y1_new = max(0, center_y - new_height // 2)
                x2_new = min(frame.shape[1], center_x + new_width // 2)
                y2_new = min(frame.shape[0], center_y + new_height // 2)
                
                # 检查人脸尺寸
                if (x2_new - x1_new) < min_face_size or (y2_new - y1_new) < min_face_size:
                    continue
                
                # 裁剪人脸
                face_crop = frame[y1_new:y2_new, x1_new:x2_new]
                
                # 调整大小
                face_resized = cv2.resize(face_crop, image_size)
                
                # 保存图像
                filename = f"{video_id}_frame_{frame_idx:06d}_face_{i:02d}.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, face_resized)
                
                extracted_count += 1
            
            return extracted_count
            
        except Exception as e:
            logging.error(f"Error extracting faces from frame: {e}")
            return 0
    
    def process_dataset(self, dataset_name: str) -> Dict[str, int]:
        """处理指定数据集"""
        logging.info(f"Starting processing of {dataset_name} dataset...")
        
        try:
            video_paths = self.path_config.get_all_video_paths(dataset_name)
            logging.info(f"Found {len(video_paths)} videos in {dataset_name}")
            
            stats = {"processed": 0, "total_faces": 0, "errors": 0}
            
            for video_info in tqdm(video_paths, desc=f"Processing {dataset_name}"):
                try:
                    face_count = self.process_video(video_info)
                    stats["total_faces"] += face_count
                    stats["processed"] += 1
                    
                    if stats["processed"] % 50 == 0:
                        logging.info(f"Processed {stats['processed']} videos, extracted {stats['total_faces']} faces")
                        
                except Exception as e:
                    logging.error(f"Failed to process {video_info['video_path']}: {e}")
                    stats["errors"] += 1
                    continue
            
            logging.info(f"Completed {dataset_name}: {stats['processed']} videos, {stats['total_faces']} faces, {stats['errors']} errors")
            return stats
            
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {e}")
            return {"processed": 0, "total_faces": 0, "errors": 1}
    
    def generate_manifests(self):
        """生成数据清单文件"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        manifests_dir = f"{processed_path}/manifests"
        
        # 为每个数据集划分生成清单
        splits_to_process = [
            ('train', f"{processed_path}/train"),
            ('val', f"{processed_path}/val"),
            ('test_celebdf_v2', f"{processed_path}/final_test_sets/celebdf_v2"),
            ('test_ffpp', f"{processed_path}/final_test_sets/ffpp"),
            ('test_dfdc', f"{processed_path}/final_test_sets/dfdc")
        ]
        
        for split_name, split_path in splits_to_process:
            if not os.path.exists(split_path):
                continue
            
            manifest_data = []
            
            # 处理有real/fake子目录的情况
            if split_name.startswith('test_') and split_name != 'test_dfdc':
                for label in ['real', 'fake']:
                    label_path = os.path.join(split_path, label)
                    
                    if not os.path.exists(label_path):
                        continue
                    
                    for image_file in os.listdir(label_path):
                        if image_file.endswith('.png'):
                            relative_path = os.path.join(split_name.replace('test_', ''), label, image_file)
                            label_numeric = 0 if label == 'real' else 1
                            
                            manifest_data.append({
                                'filepath': relative_path,
                                'label': label_numeric,
                                'label_name': label,
                                'dataset': split_name.replace('test_', '')
                            })
            
            # 处理DFDC等混合目录的情况
            elif split_name == 'test_dfdc':
                for image_file in os.listdir(split_path):
                    if image_file.endswith('.png'):
                        # 从文件名推断标签 (需要在处理时保存这个信息)
                        relative_path = os.path.join('dfdc', image_file)
                        
                        manifest_data.append({
                            'filepath': relative_path,
                            'label': -1,  # 需要后续处理
                            'label_name': 'unknown',
                            'dataset': 'dfdc'
                        })
            
            # 处理train/val目录
            else:
                for label in ['real', 'fake']:
                    label_path = os.path.join(split_path, label)
                    
                    if not os.path.exists(label_path):
                        continue
                    
                    for image_file in os.listdir(label_path):
                        if image_file.endswith('.png'):
                            relative_path = os.path.join(split_name, label, image_file)
                            label_numeric = 0 if label == 'real' else 1
                            
                            manifest_data.append({
                                'filepath': relative_path,
                                'label': label_numeric,
                                'label_name': label,
                                'dataset': 'mixed'
                            })
            
            # 保存清单文件
            if manifest_data:
                manifest_df = pd.DataFrame(manifest_data)
                manifest_file = os.path.join(manifests_dir, f"{split_name}_manifest.csv")
                manifest_df.to_csv(manifest_file, index=False)
                logging.info(f"Generated manifest: {manifest_file} with {len(manifest_data)} samples")
    
    def run_full_preprocessing(self, datasets: Optional[List[str]] = None):
        """运行完整的数据预处理流程"""
        logging.info("Starting full dataset preprocessing...")
        
        # 验证配置路径
        validation_results = self.path_config.validate_paths()
        for path_name, exists in validation_results.items():
            status = "✓" if exists else "✗"
            logging.info(f"{status} {path_name}")
        
        # 确定要处理的数据集
        if datasets is None:
            datasets = list(self.path_config.config["datasets"].keys())
        
        total_stats = {"processed": 0, "total_faces": 0, "errors": 0}
        
        # 逐个处理数据集
        for dataset_name in datasets:
            if dataset_name in self.path_config.config["datasets"]:
                stats = self.process_dataset(dataset_name)
                for key in total_stats:
                    total_stats[key] += stats[key]
            else:
                logging.warning(f"Dataset {dataset_name} not found in configuration")
        
        logging.info(f"All datasets processed: {total_stats['processed']} videos, {total_stats['total_faces']} faces, {total_stats['errors']} errors")
        
        # 生成清单文件
        self.generate_manifests()
        logging.info("All manifest files generated successfully")
        
        return total_stats

# =============================================================================
# 主程序入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing V2 for Deepfake Detection")
    parser.add_argument("--config", default="config/dataset_paths.json", help="Path to configuration file")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to process (e.g., celebdf_v2 ffpp)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate paths without processing")
    parser.add_argument("--print-config", action="store_true", help="Print configuration summary and exit")
    
    args = parser.parse_args()
    
    # 创建预处理器
    try:
        preprocessor = DatasetPreprocessorV2(args.config)
    except Exception as e:
        print(f"Error initializing preprocessor: {e}")
        return 1
    
    # 打印配置摘要
    if args.print_config:
        preprocessor.path_config.print_config_summary()
        return 0
    
    # 验证路径
    if args.validate_only:
        validation_results = preprocessor.path_config.validate_paths()
        print("Path validation results:")
        for path_name, exists in validation_results.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {path_name}")
        return 0
    
    # 运行预处理
    try:
        stats = preprocessor.run_full_preprocessing(args.datasets)
        print(f"\nPreprocessing completed successfully!")
        print(f"Total videos processed: {stats['processed']}")
        print(f"Total faces extracted: {stats['total_faces']}")
        print(f"Errors encountered: {stats['errors']}")
        return 0
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())