#!/usr/bin/env python3
"""
测试配置文件和数据集路径
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dataset_config import DatasetPathConfig

def main():
    print("=== Testing Dataset Configuration ===\n")
    
    # 创建配置管理器
    try:
        config_manager = DatasetPathConfig("config/dataset_paths.json")
        print("OK Configuration loaded successfully")
    except Exception as e:
        print(f"ERROR Failed to load configuration: {e}")
        return
    
    # 打印配置摘要
    print("\n=== Configuration Summary ===")
    config_manager.print_config_summary()
    
    # 验证路径
    print("\n=== Path Validation ===")
    validation_results = config_manager.validate_paths()
    for path_name, exists in validation_results.items():
        status = "OK" if exists else "ERROR"
        print(f"{status} {path_name}")
    
    # 测试获取视频路径
    print("\n=== Testing Video Path Discovery ===")
    for dataset_name in config_manager.config["datasets"]:
        try:
            video_paths = config_manager.get_all_video_paths(dataset_name)
            print(f"OK {dataset_name}: {len(video_paths)} videos found")
            
            # 显示前几个视频/图像的信息
            if video_paths:
                print(f"  Sample videos:")
                for i, video_info in enumerate(video_paths[:3]):
                    # 兼容video_id和image_id
                    item_id = video_info.get('video_id') or video_info.get('image_id', 'unknown')
                    print(f"    {i+1}. {item_id} ({video_info['label']})")
                if len(video_paths) > 3:
                    print(f"    ... and {len(video_paths) - 3} more")
        except Exception as e:
            print(f"ERROR {dataset_name}: Error - {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()