#!/usr/bin/env python3
"""
測試新的數據集分割邏輯
驗證 CelebDF-v2 和 FF++ 的 train/val/test 分割是否正確
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.dataset_config import DatasetPathConfig
from collections import Counter

def test_celebdf_split():
    """測試 CelebDF-v2 分割邏輯"""
    print("=== 測試 CelebDF-v2 分割邏輯 ===")
    
    # 初始化配置
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    celebdf_config = config_manager.config["datasets"]["celebdf_v2"]
    celebdf_config["full_base_path"] = os.path.join(config_manager.config["base_paths"]["raw_datasets"], celebdf_config["base_path"])
    
    # 模擬一些視頻文件名
    test_videos = [
        "YouTube-real/00001.mp4",
        "YouTube-real/00002.mp4", 
        "YouTube-real/00003.mp4",
        "YouTube-real/00170.mp4",  # 這個在官方測試集中
        "Celeb-real/id0_0001.mp4",
        "Celeb-real/id0_0002.mp4",
        "Celeb-synthesis/id0_id1_0001.mp4",
        "Celeb-synthesis/id0_id1_0002.mp4",
    ]
    
    # 測試分割結果
    splits = Counter()
    for video in test_videos:
        split = config_manager._determine_celebdf_split(video, celebdf_config)
        splits[split] += 1
        print(f"  {video:<35} -> {split}")
    
    print(f"\n分割統計: {dict(splits)}")
    print(f"預期比例: train={config_manager.config['output_structure']['train_split_ratio']}, "
          f"val={config_manager.config['output_structure']['val_split_ratio']}, "
          f"test={config_manager.config['output_structure']['test_split_ratio']}")

def test_ffpp_split():
    """測試 FF++ 分割邏輯"""
    print("\n=== 測試 FF++ 分割邏輯 ===")
    
    # 初始化配置
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    ffpp_config = config_manager.config["datasets"]["ffpp"]
    ffpp_config["full_base_path"] = os.path.join(config_manager.config["base_paths"]["raw_datasets"], ffpp_config["base_path"])
    
    # 模擬一些視頻ID
    test_video_ids = [
        "000_003",
        "001_000", 
        "002_005",
        "010_020",
        "050_100",
        "100_200",
        "200_300",
        "300_400",
    ]
    
    # 測試分割結果
    splits = Counter()
    for video_id in test_video_ids:
        split = config_manager._determine_ffpp_split(video_id, ffpp_config)
        splits[split] += 1
        print(f"  {video_id:<15} -> {split}")
    
    print(f"\n分割統計: {dict(splits)}")
    print(f"預期比例: train={config_manager.config['output_structure']['train_split_ratio']}, "
          f"val={config_manager.config['output_structure']['val_split_ratio']}, "
          f"test={config_manager.config['output_structure']['test_split_ratio']}")

def test_large_scale_split():
    """測試大規模分割的比例"""
    print("\n=== 測試大規模分割比例 ===")
    
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    
    # 生成大量模擬視頻文件名
    test_videos = [f"video_{i:05d}.mp4" for i in range(1000)]
    
    splits = Counter()
    for video in test_videos:
        # 使用 CelebDF-v2 配置進行測試
        celebdf_config = config_manager.config["datasets"]["celebdf_v2"]
        celebdf_config["full_base_path"] = os.path.join(config_manager.config["base_paths"]["raw_datasets"], celebdf_config["base_path"])
        split = config_manager._determine_celebdf_split(video, celebdf_config)
        splits[split] += 1
    
    total = sum(splits.values())
    print(f"總視頻數: {total}")
    print(f"實際分割:")
    for split_name, count in splits.items():
        percentage = count / total * 100
        print(f"  {split_name}: {count} ({percentage:.1f}%)")
    
    # 預期比例
    expected = config_manager.config['output_structure']
    print(f"\n預期分割:")
    print(f"  train: {expected['train_split_ratio']*100:.1f}%")
    print(f"  val: {expected['val_split_ratio']*100:.1f}%") 
    print(f"  test: {expected['test_split_ratio']*100:.1f}%")

if __name__ == "__main__":
    test_celebdf_split()
    test_ffpp_split()
    test_large_scale_split()
    
    print("\n=== 測試完成 ===")
    print("如果看到 train/val/test 三種分割且比例大致符合 70/15/15，則分割邏輯正確！")