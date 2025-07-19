#!/usr/bin/env python3
"""
数据集配置管理系统 - dataset_config.py
提供灵活的路径配置和数据集结构管理
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

class DatasetPathConfig:
    """数据集路径配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: JSON配置文件路径，如果为None则使用默认配置
        """
        self.config_file = config_file or "dataset_paths.json"
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logging.info(f"Loaded dataset configuration from {self.config_file}")
            except Exception as e:
                logging.error(f"Failed to load config file {self.config_file}: {e}")
                self.config = self._get_default_config()
        else:
            logging.info(f"Config file {self.config_file} not found, using default configuration")
            self.config = self._get_default_config()
            self.save_config()  # 保存默认配置供用户修改
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved dataset configuration to {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to save config file: {e}")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "base_paths": {
                "raw_datasets": "/path/to/dataset",
                "processed_data": "/path/to/processed_data"
            },
            "datasets": {
                "celebdf_v2": {
                    "name": "CelebDF-v2",
                    "base_path": "CelebDF-v2",
                    "real_videos": [
                        "Celeb-real",
                        "YouTube-real"
                    ],
                    "fake_videos": [
                        "Celeb-synthesis"
                    ],
                    "test_list_file": "List_of_testing_videos.txt",
                    "supported_extensions": [".mp4", ".avi"],
                    "description": "Celebrity deepfake detection dataset"
                },
                "ffpp": {
                    "name": "FaceForensics++",
                    "base_path": "FF++",
                    "compressions": ["c0", "c23", "c40"],
                    "real_videos": {
                        "original_sequences": [
                            "actors",
                            "youtube"
                        ]
                    },
                    "fake_videos": {
                        "manipulated_sequences": [
                            "Deepfakes",
                            "Face2Face", 
                            "FaceSwap",
                            "NeuralTextures",
                            "DeepFakeDetection",
                            "FaceShifter"
                        ]
                    },
                    "splits_file": "splits/train.json",
                    "supported_extensions": [".mp4", ".avi"],
                    "description": "Face manipulation detection dataset"
                },
                "dfdc": {
                    "name": "DFDC",
                    "base_path": "DFDC",
                    "metadata_file": "metadata.json",
                    "folder_pattern": "dfdc_train_part_*",
                    "supported_extensions": [".mp4"],
                    "description": "Deepfake Detection Challenge dataset"
                }
            },
            "processing": {
                "frame_interval": 10,
                "image_size": [224, 224],
                "bbox_scale": 1.3,
                "max_faces_per_video": 50,
                "min_face_size": 80
            }
        }
    
    def get_dataset_paths(self, dataset_name: str) -> Dict:
        """获取指定数据集的路径配置"""
        if dataset_name not in self.config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")
        
        dataset_config = self.config["datasets"][dataset_name].copy()
        base_raw_path = self.config["base_paths"]["raw_datasets"]
        dataset_config["full_base_path"] = os.path.join(base_raw_path, dataset_config["base_path"])
        
        return dataset_config
    
    def get_all_video_paths(self, dataset_name: str) -> List[Dict]:
        """获取数据集中所有视频的路径和标签信息"""
        dataset_config = self.get_dataset_paths(dataset_name)
        video_paths = []
        
        if dataset_name == "celebdf_v2":
            video_paths.extend(self._get_celebdf_paths(dataset_config))
        elif dataset_name == "ffpp":
            video_paths.extend(self._get_ffpp_paths(dataset_config))
        elif dataset_name == "dfdc":
            video_paths.extend(self._get_dfdc_paths(dataset_config))
        
        return video_paths
    
    def _get_celebdf_paths(self, config: Dict) -> List[Dict]:
        """获取CelebDF-v2数据集的所有视频路径"""
        video_paths = []
        base_path = config["full_base_path"]
        
        # 处理真实视频
        for real_dir in config["real_videos"]:
            real_path = os.path.join(base_path, real_dir)
            if os.path.exists(real_path):
                for video_file in os.listdir(real_path):
                    if any(video_file.endswith(ext) for ext in config["supported_extensions"]):
                        video_paths.append({
                            "video_path": os.path.join(real_path, video_file),
                            "video_id": f"celebdf_{real_dir}_{os.path.splitext(video_file)[0]}",
                            "label": "real",
                            "dataset": "celebdf_v2",
                            "source_dir": real_dir,
                            "split": self._determine_celebdf_split(video_file, config)
                        })
        
        # 处理伪造视频
        for fake_dir in config["fake_videos"]:
            fake_path = os.path.join(base_path, fake_dir)
            if os.path.exists(fake_path):
                for video_file in os.listdir(fake_path):
                    if any(video_file.endswith(ext) for ext in config["supported_extensions"]):
                        video_paths.append({
                            "video_path": os.path.join(fake_path, video_file),
                            "video_id": f"celebdf_{fake_dir}_{os.path.splitext(video_file)[0]}",
                            "label": "fake",
                            "dataset": "celebdf_v2", 
                            "source_dir": fake_dir,
                            "split": self._determine_celebdf_split(video_file, config)
                        })
        
        return video_paths
    
    def _get_ffpp_paths(self, config: Dict) -> List[Dict]:
        """获取FF++数据集的所有视频路径"""
        video_paths = []
        base_path = config["full_base_path"]
        
        # 处理真实视频 (original_sequences)
        for compression in config["compressions"]:
            for real_category in config["real_videos"]["original_sequences"]:
                real_path = os.path.join(base_path, "original_sequences", real_category, compression, "videos")
                if os.path.exists(real_path):
                    for video_file in os.listdir(real_path):
                        if any(video_file.endswith(ext) for ext in config["supported_extensions"]):
                            video_id = os.path.splitext(video_file)[0]
                            video_paths.append({
                                "video_path": os.path.join(real_path, video_file),
                                "video_id": f"ffpp_original_{real_category}_{compression}_{video_id}",
                                "label": "real",
                                "dataset": "ffpp",
                                "compression": compression,
                                "method": "original",
                                "category": real_category,
                                "split": self._determine_ffpp_split(video_id, config)
                            })
        
        # 处理伪造视频 (manipulated_sequences)
        for compression in config["compressions"]:
            for fake_method in config["fake_videos"]["manipulated_sequences"]:
                fake_path = os.path.join(base_path, "manipulated_sequences", fake_method, compression, "videos")
                if os.path.exists(fake_path):
                    for video_file in os.listdir(fake_path):
                        if any(video_file.endswith(ext) for ext in config["supported_extensions"]):
                            video_id = os.path.splitext(video_file)[0]
                            video_paths.append({
                                "video_path": os.path.join(fake_path, video_file),
                                "video_id": f"ffpp_{fake_method}_{compression}_{video_id}",
                                "label": "fake",
                                "dataset": "ffpp",
                                "compression": compression,
                                "method": fake_method,
                                "split": self._determine_ffpp_split(video_id, config)
                            })
        
        return video_paths
    
    def _get_dfdc_paths(self, config: Dict) -> List[Dict]:
        """获取DFDC数据集的所有视频路径"""
        video_paths = []
        base_path = config["full_base_path"]
        
        # 查找所有匹配的文件夹
        import glob
        pattern = os.path.join(base_path, config["folder_pattern"])
        for folder_path in glob.glob(pattern):
            if os.path.isdir(folder_path):
                metadata_file = os.path.join(folder_path, config["metadata_file"])
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        for video_file, info in metadata.items():
                            video_path = os.path.join(folder_path, video_file)
                            if os.path.exists(video_path):
                                folder_name = os.path.basename(folder_path)
                                video_id = os.path.splitext(video_file)[0]
                                
                                video_paths.append({
                                    "video_path": video_path,
                                    "video_id": f"dfdc_{folder_name}_{video_id}",
                                    "label": "real" if info.get("label") == "REAL" else "fake",
                                    "dataset": "dfdc",
                                    "folder": folder_name,
                                    "split": "train"  # DFDC主要用于训练
                                })
                    except Exception as e:
                        logging.error(f"Error reading metadata from {metadata_file}: {e}")
        
        return video_paths
    
    def _determine_celebdf_split(self, video_file: str, config: Dict) -> str:
        """确定CelebDF视频的数据集划分"""
        test_list_file = os.path.join(config["full_base_path"], config["test_list_file"])
        
        if os.path.exists(test_list_file):
            try:
                with open(test_list_file, 'r') as f:
                    test_videos = set()
                    for line in f:
                        if line.strip():
                            # 处理不同格式的测试列表文件
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                test_videos.add(parts[1])  # 通常第二列是文件名
                            else:
                                test_videos.add(parts[0])
                
                return "test" if video_file in test_videos else "train"
            except Exception as e:
                logging.error(f"Error reading test list file: {e}")
        
        return "train"  # 默认为训练集
    
    def _determine_ffpp_split(self, video_id: str, config: Dict) -> str:
        """确定FF++视频的数据集划分"""
        splits_file = os.path.join(config["full_base_path"], config["splits_file"])
        
        if os.path.exists(splits_file):
            try:
                with open(splits_file, 'r') as f:
                    train_videos = json.load(f)
                
                return "train" if video_id in train_videos else "test"
            except Exception as e:
                logging.error(f"Error reading splits file: {e}")
        
        return "train"  # 默认为训练集
    
    def update_base_paths(self, raw_datasets_path: str, processed_data_path: str):
        """更新基础路径"""
        self.config["base_paths"]["raw_datasets"] = raw_datasets_path
        self.config["base_paths"]["processed_data"] = processed_data_path
        self.save_config()
    
    def get_processing_config(self) -> Dict:
        """获取处理参数配置"""
        return self.config.get("processing", {})
    
    def update_processing_config(self, **kwargs):
        """更新处理参数"""
        if "processing" not in self.config:
            self.config["processing"] = {}
        
        self.config["processing"].update(kwargs)
        self.save_config()
    
    def validate_paths(self) -> Dict[str, bool]:
        """验证所有配置的路径是否存在"""
        validation_results = {}
        
        # 验证基础路径
        base_raw_path = self.config["base_paths"]["raw_datasets"]
        validation_results["base_raw_path"] = os.path.exists(base_raw_path)
        
        # 验证各数据集路径
        for dataset_name in self.config["datasets"]:
            dataset_config = self.get_dataset_paths(dataset_name)
            validation_results[f"{dataset_name}_base_path"] = os.path.exists(dataset_config["full_base_path"])
        
        return validation_results
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("=== Dataset Configuration Summary ===")
        print(f"Raw datasets path: {self.config['base_paths']['raw_datasets']}")
        print(f"Processed data path: {self.config['base_paths']['processed_data']}")
        print("\nDatasets:")
        
        for dataset_name, dataset_config in self.config["datasets"].items():
            full_path = os.path.join(self.config["base_paths"]["raw_datasets"], dataset_config["base_path"])
            exists = "✓" if os.path.exists(full_path) else "✗"
            print(f"  {exists} {dataset_name}: {full_path}")
        
        print(f"\nProcessing parameters:")
        for key, value in self.config["processing"].items():
            print(f"  {key}: {value}")


def create_example_config():
    """创建示例配置文件"""
    config_manager = DatasetPathConfig("dataset_paths_example.json")
    config_manager.save_config()
    print("Created example configuration file: dataset_paths_example.json")
    print("Please copy it to dataset_paths.json and modify the paths according to your setup.")


if __name__ == "__main__":
    # 创建示例配置
    create_example_config()
    
    # 演示用法
    config = DatasetPathConfig()
    config.print_config_summary()