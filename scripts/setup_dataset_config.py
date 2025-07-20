#!/usr/bin/env python3
"""
数据集配置设置助手 - setup_dataset_config.py
帮助用户快速配置数据集路径和处理参数
"""

import os
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.dataset_config import DatasetPathConfig

def interactive_setup():
    """交互式配置设置"""
    print("=== AWARE-NET 数据集配置设置助手 ===\n")
    
    # 获取基础路径
    print("1. 基础路径配置")
    print("-" * 50)
    
    raw_datasets_path = input("请输入原始数据集根目录路径 (例: /path/to/dataset): ").strip()
    if not raw_datasets_path:
        raw_datasets_path = "/path/to/dataset"
    
    processed_data_path = input("请输入处理后数据输出目录路径 (例: /path/to/processed_data): ").strip()
    if not processed_data_path:
        processed_data_path = "/path/to/processed_data"
    
    # 验证路径是否存在
    raw_exists = os.path.exists(raw_datasets_path)
    print(f"原始数据集路径: {raw_datasets_path} {'✓' if raw_exists else '✗ (不存在)'}")
    
    if not raw_exists:
        create_raw = input("原始数据集路径不存在，是否继续？ (y/n): ").lower() == 'y'
        if not create_raw:
            print("配置取消。")
            return
    
    # 处理参数配置
    print("\n2. 处理参数配置")
    print("-" * 50)
    
    frame_interval = input("帧采样间隔 (默认: 10): ").strip()
    frame_interval = int(frame_interval) if frame_interval.isdigit() else 10
    
    image_size = input("输出图像尺寸 (默认: 224): ").strip()
    image_size = int(image_size) if image_size.isdigit() else 224
    
    bbox_scale = input("人脸边界框扩展比例 (默认: 1.3): ").strip()
    bbox_scale = float(bbox_scale) if bbox_scale.replace('.', '').isdigit() else 1.3
    
    max_faces = input("每视频最大人脸数量 (默认: 50): ").strip()
    max_faces = int(max_faces) if max_faces.isdigit() else 50
    
    # 创建配置
    config = {
        "base_paths": {
            "raw_datasets": raw_datasets_path,
            "processed_data": processed_data_path
        },
        "datasets": {
            "celebdf_v2": {
                "name": "CelebDF-v2",
                "base_path": "CelebDF-v2",
                "real_videos": [
                    "Celeb-real"
                ],
                "fake_videos": [
                    "Celeb-synthesis"
                ],
                "test_list_file": "List_of_testing_videos.txt",
                "supported_extensions": [".mp4", ".avi", ".mov"],
                "description": "Celebrity deepfake detection dataset"
            },
            "df40": {
                "name": "DF40",
                "base_path": "DF40",
                "real_videos": [],
                "fake_videos": [
                    "blendface",
                    "e4s",
                    "facedancer",
                    "faceswap",
                    "frames",
                    "fsgan",
                    "inswap",
                    "mobileswap",
                    "simswap"
                ],
                "supported_extensions": [".mp4", ".avi", ".mov"],
                "description": "DF40 fake video dataset"
            },
            "dfdc": {
                "name": "DFDC",
                "base_path": "DFDC",
                "metadata_file": "metadata.json",
                "folder_pattern": "*",
                "real_videos": [
                    "real"
                ],
                "fake_videos": [
                    "fake"
                ],
                "supported_extensions": [".mp4"],
                "description": "Deepfake Detection Challenge dataset (classified)"
            },
            "ffpp": {
                "name": "FaceForensics++",
                "base_path": "FF++",
                "compressions": ["c23"],
                "real_videos": {
                    "original_sequences": [
                        "actors",
                        "youtube"
                    ]
                },
                "fake_videos": {
                    "manipulated_sequences": [
                        "DeepFakeDetection",
                        "Deepfakes",
                        "Face2Face",
                        "FaceShifter",
                        "FaceSwap",
                        "NeuralTextures"
                    ]
                },
                "splits_file": "splits/train.json",
                "supported_extensions": [".mp4", ".avi", ".mov"],
                "description": "Face manipulation detection dataset"
            },
            "dfdc_classified": {
                "name": "DFDC_Classified",
                "base_path": "DFDC_classified",
                "real_videos": [
                    "real"
                ],
                "fake_videos": [
                    "fake"
                ],
                "supported_extensions": [".mp4"],
                "description": "DFDC dataset pre-classified into real/fake folders"
            }
        },
        "processing": {
            "frame_interval": frame_interval,
            "image_size": [image_size, image_size],
            "bbox_scale": bbox_scale,
            "max_faces_per_video": max_faces,
            "min_face_size": 80,
            "face_detector": {
                "name": "mtcnn",
                "min_face_size": 20,
                "thresholds": [0.6, 0.7, 0.7],
                "factor": 0.709,
                "post_process": True
            }
        },
        "output_structure": {
            "train_split_ratio": 0.7,
            "val_split_ratio": 0.15,
            "test_split_ratio": 0.15,
            "preserve_official_test_sets": True,
            "manifest_format": "csv"
        }
    }
    
    # 保存配置
    config_file = "config/dataset_paths.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 配置已保存到 {config_file}")
        
        # 显示配置摘要
        print("\n3. 配置摘要")
        print("-" * 50)
        print(f"原始数据集路径: {raw_datasets_path}")
        print(f"处理后数据路径: {processed_data_path}")
        print(f"帧采样间隔: {frame_interval}")
        print(f"图像尺寸: {image_size}x{image_size}")
        print(f"边界框扩展: {bbox_scale}")
        print(f"每视频最大人脸数: {max_faces}")
        
        return config_file
        
    except Exception as e:
        print(f"✗ 保存配置失败: {e}")
        return None

def validate_dataset_structure(config_file: str):
    """验证数据集结构"""
    print("\n4. 数据集结构验证")
    print("-" * 50)
    
    try:
        config_manager = DatasetPathConfig(config_file)
        validation_results = config_manager.validate_paths()
        
        for path_name, exists in validation_results.items():
            status = "✓" if exists else "✗"
            print(f"{status} {path_name}")
        
        # 检查具体的数据集子目录
        raw_path = config_manager.config["base_paths"]["raw_datasets"]
        
        print(f"\n检查数据集子目录:")
        for dataset_name in config_manager.config["datasets"]:
            dataset_config = config_manager.get_dataset_paths(dataset_name)
            dataset_path = dataset_config["full_base_path"]
            exists = os.path.exists(dataset_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {dataset_name}: {dataset_path}")
            
            if exists and dataset_name == "celebdf_v2":
                # 检查CelebDF-v2子目录
                for subdir in ["Celeb-real", "Celeb-synthesis"]:
                    subdir_path = os.path.join(dataset_path, subdir)
                    sub_exists = os.path.exists(subdir_path)
                    sub_status = "✓" if sub_exists else "✗"
                    print(f"    {sub_status} {subdir}")
            
            elif exists and dataset_name == "df40":
                # 检查DF40子目录（全是fake）
                fake_dirs = ["blendface", "e4s", "facedancer", "faceswap", "frames", "fsgan", "inswap", "mobileswap", "simswap"]
                for subdir in fake_dirs:
                    subdir_path = os.path.join(dataset_path, subdir)
                    sub_exists = os.path.exists(subdir_path)
                    sub_status = "✓" if sub_exists else "✗"
                    print(f"    {sub_status} {subdir} (fake)")
            
            elif exists and dataset_name == "dfdc":
                # 检查DFDC子目录
                for subdir in ["real", "fake"]:
                    subdir_path = os.path.join(dataset_path, subdir)
                    sub_exists = os.path.exists(subdir_path)
                    sub_status = "✓" if sub_exists else "✗"
                    print(f"    {sub_status} {subdir}")
            
            elif exists and dataset_name == "dfdc_classified":
                # 检查DFDC分类后的子目录
                for subdir in ["real", "fake"]:
                    subdir_path = os.path.join(dataset_path, subdir)
                    sub_exists = os.path.exists(subdir_path)
                    sub_status = "✓" if sub_exists else "✗"
                    print(f"    {sub_status} {subdir}")
            
            elif exists and dataset_name == "ffpp":
                # 检查FF++子目录
                for main_dir in ["original_sequences", "manipulated_sequences"]:
                    main_path = os.path.join(dataset_path, main_dir)
                    main_exists = os.path.exists(main_path)
                    main_status = "✓" if main_exists else "✗"
                    print(f"    {main_status} {main_dir}")
                    
                    if main_exists and main_dir == "manipulated_sequences":
                        # 检查假视频子目录
                        fake_dirs = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
                        for fake_dir in fake_dirs:
                            fake_path = os.path.join(main_path, fake_dir)
                            fake_exists = os.path.exists(fake_path)
                            fake_status = "✓" if fake_exists else "✗"
                            print(f"      {fake_status} {fake_dir} (fake)")
        
        return True
        
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        return False

def quick_setup_from_existing():
    """从现有目录结构快速设置"""
    print("=== 快速设置（从现有目录结构） ===\n")
    
    current_dir = os.getcwd()
    dataset_dir = None
    
    # 查找可能的数据集目录
    possible_dirs = ["dataset", "datasets", "data", "raw_data"]
    for dirname in possible_dirs:
        test_path = os.path.join(current_dir, dirname)
        if os.path.exists(test_path):
            print(f"发现可能的数据集目录: {test_path}")
            use_dir = input(f"使用此目录作为原始数据集路径？ (y/n): ").lower() == 'y'
            if use_dir:
                dataset_dir = test_path
                break
    
    if not dataset_dir:
        dataset_dir = input("请输入数据集目录路径: ").strip()
    
    if not os.path.exists(dataset_dir):
        print(f"✗ 目录不存在: {dataset_dir}")
        return None
    
    # 自动检测数据集结构
    detected_datasets = []
    
    # 检查CelebDF-v2
    celebdf_path = os.path.join(dataset_dir, "CelebDF-v2")
    if os.path.exists(celebdf_path):
        if os.path.exists(os.path.join(celebdf_path, "Celeb-real")):
            detected_datasets.append("CelebDF-v2")
    
    # 检查FF++
    ffpp_path = os.path.join(dataset_dir, "FF++")
    if os.path.exists(ffpp_path):
        if os.path.exists(os.path.join(ffpp_path, "original_sequences")):
            detected_datasets.append("FF++")
    
    # 检查DFDC
    dfdc_path = os.path.join(dataset_dir, "DFDC")
    if os.path.exists(dfdc_path):
        detected_datasets.append("DFDC")
    
    print(f"\n检测到的数据集: {', '.join(detected_datasets) if detected_datasets else '无'}")
    
    # 使用默认配置并更新路径
    processed_path = os.path.join(current_dir, "processed_data")
    
    config = {
        "base_paths": {
            "raw_datasets": dataset_dir,
            "processed_data": processed_path
        }
    }
    
    # 加载默认配置的其余部分
    default_config = DatasetPathConfig()._get_default_config()
    config.update({k: v for k, v in default_config.items() if k != "base_paths"})
    
    # 保存配置
    config_file = "config/dataset_paths.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 配置已保存到 {config_file}")
        print(f"✓ 原始数据集路径: {dataset_dir}")
        print(f"✓ 处理后数据路径: {processed_path}")
        
        return config_file
        
    except Exception as e:
        print(f"✗ 保存配置失败: {e}")
        return None

def main():
    """主函数"""
    print("选择配置方式:")
    print("1. 交互式配置")
    print("2. 快速设置（从现有目录结构）")
    print("3. 验证现有配置")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        config_file = interactive_setup()
        if config_file:
            validate_dataset_structure(config_file)
    
    elif choice == "2":
        config_file = quick_setup_from_existing()
        if config_file:
            validate_dataset_structure(config_file)
    
    elif choice == "3":
        config_file = "config/dataset_paths.json"
        if os.path.exists(config_file):
            validate_dataset_structure(config_file)
        else:
            print(f"✗ 配置文件 {config_file} 不存在")
    
    else:
        print("无效选择")
    
    print("\n使用方法:")
    print("python scripts/preprocess_datasets_v2.py --config config/dataset_paths.json")
    print("python scripts/preprocess_datasets_v2.py --print-config")
    print("python scripts/preprocess_datasets_v2.py --validate-only")

if __name__ == "__main__":
    main()