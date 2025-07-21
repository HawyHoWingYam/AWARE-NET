#!/usr/bin/env python3
"""
DF40官方测试数据组织脚本 - organize_df40_test_only.py
仅处理DF40官方测试数据，使用10%并分配到val和final_test_sets
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def organize_df40_test_data():
    """组织DF40官方测试数据，只使用10%并分配到val和final_test_sets"""
    
    df40_test_source = Path("D:/work/AWARE-NET/dataset/DF40/test")
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    
    print("=== DF40官方测试数据组织 (使用10%，分配到val和final_test_sets) ===")
    
    if not df40_test_source.exists():
        print(f"⚠️ 官方测试目录不存在: {df40_test_source}")
        return 0, 0
    
    # 收集所有测试图像
    all_test_images = []
    for method_dir in df40_test_source.iterdir():
        if method_dir.is_dir():
            # 检查fake和real子目录
            for subdir in ["fake", "real"]:
                subdir_path = method_dir / subdir
                if subdir_path.exists():
                    for img_file in subdir_path.glob("*.jpg"):
                        all_test_images.append((img_file, method_dir.name, subdir))
    
    print(f"找到 {len(all_test_images)} 个官方测试图像")
    
    # 只使用10%
    random.seed(42)  # 固定种子确保可重复
    sample_size = int(len(all_test_images) * 0.1)
    selected_images = random.sample(all_test_images, sample_size)
    
    print(f"选择使用 {len(selected_images)} 个图像 (10%)")
    
    # 50/50分配到val和final_test_sets
    mid_point = len(selected_images) // 2
    val_images = selected_images[:mid_point]
    final_test_images = selected_images[mid_point:]
    
    print(f"分配: {len(val_images)}个到val, {len(final_test_images)}个到final_test_sets")
    
    # 复制到val目录
    for img_file, method_name, label in tqdm(val_images, desc="复制到val"):
        val_dir = processed_data / "val" / "df40" / label
        val_dir.mkdir(parents=True, exist_ok=True)
        filename = f"df40_test_{method_name}_{label}_{img_file.stem}.jpg"
        target_path = val_dir / filename
        if not target_path.exists():
            shutil.copy2(img_file, target_path)
    
    # 复制到final_test_sets目录
    for img_file, method_name, label in tqdm(final_test_images, desc="复制到final_test_sets"):
        final_test_dir = processed_data / "final_test_sets" / "df40" / label
        final_test_dir.mkdir(parents=True, exist_ok=True)
        filename = f"df40_test_{method_name}_{label}_{img_file.stem}.jpg"
        target_path = final_test_dir / filename
        if not target_path.exists():
            shutil.copy2(img_file, target_path)
    
    print(f"官方测试数据组织完成: {len(val_images)}个在val, {len(final_test_images)}个在final_test_sets")
    
    return len(val_images), len(final_test_images)

def main():
    print("DF40官方测试数据组织工具")
    print("="*50)
    
    # 组织官方测试数据 (10%用量，50/50分配)
    val_count, final_test_count = organize_df40_test_data()
    
    print(f"\n{'='*50}")
    print("✅ DF40官方测试数据组织完成!")
    print(f"官方测试数据: {val_count}个在val, {final_test_count}个在final_test_sets")

if __name__ == "__main__":
    main()