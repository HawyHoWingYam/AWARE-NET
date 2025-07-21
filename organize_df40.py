#!/usr/bin/env python3
"""
DF40数据集组织脚本 - organize_df40.py
将DF40数据按照70/15/15的比例分配到train/val/test中
"""

import os
import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm
import hashlib
from collections import defaultdict

def get_deterministic_split(item_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """使用MD5哈希确保分割的确定性"""
    item_hash = hashlib.md5(str(item_path).encode()).hexdigest()
    random.seed(item_hash)
    rand_val = random.random()
    
    if rand_val < train_ratio:
        return "train"
    elif rand_val < train_ratio + val_ratio:
        return "val"
    else:
        return "test"

def organize_df40_train_data():
    """组织DF40训练数据到processed_data目录结构"""
    
    # 路径配置
    df40_source = Path("D:/work/AWARE-NET/dataset/DF40")
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    
    # 训练数据源目录
    train_source_dirs = [
        df40_source / "Entire Face Synthesis",
        df40_source / "Face-reenactment", 
        df40_source / "Face-swapping"
    ]
    
    print("=== DF40训练数据组织 (70/15/15分割) ===")
    
    # 统计信息
    stats = defaultdict(int)
    split_stats = {"train": 0, "val": 0, "test": 0}
    
    for source_dir in train_source_dirs:
        if not source_dir.exists():
            print(f"⚠️ 跳过不存在的目录: {source_dir}")
            continue
            
        category_name = source_dir.name.lower().replace("-", "_").replace(" ", "_")
        print(f"\n处理类别: {category_name}")
        
        # 遍历所有方法文件夹
        for method_dir in source_dir.iterdir():
            if not method_dir.is_dir():
                continue
                
            method_name = method_dir.name
            print(f"  处理方法: {method_name}")
            
            # 查找图像文件
            image_files = []
            
            # 检查frames子目录
            frames_dir = method_dir / "frames"
            if frames_dir.exists():
                for subdir in frames_dir.iterdir():
                    if subdir.is_dir():
                        for img_file in subdir.glob("*.png"):
                            image_files.append(img_file)
            
            # 检查直接在方法目录下的子目录
            for subdir in method_dir.iterdir():
                if subdir.is_dir() and subdir.name != "frames":
                    for img_file in subdir.glob("*.png"):
                        image_files.append(img_file)
            
            if not image_files:
                print(f"    ⚠️ {method_name}中没有找到图像文件")
                continue
            
            print(f"    找到 {len(image_files)} 个图像文件")
            stats[f"{category_name}_{method_name}"] = len(image_files)
            
            # 分割图像文件
            split_counts = {"train": 0, "val": 0, "test": 0}
            
            for img_file in tqdm(image_files, desc=f"    分割{method_name}"):
                # 确定分割
                split = get_deterministic_split(img_file)
                split_counts[split] += 1
                split_stats[split] += 1
                
                # 创建目标路径
                target_dir = processed_data / split / "df40" / "fake"
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成唯一文件名
                filename = f"df40_{category_name}_{method_name}_{img_file.stem}.png"
                target_path = target_dir / filename
                
                # 复制文件
                if not target_path.exists():
                    shutil.copy2(img_file, target_path)
            
            print(f"    分割结果: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    
    print(f"\n=== DF40训练数据组织完成 ===")
    print(f"总分割统计: Train={split_stats['train']}, Val={split_stats['val']}, Test={split_stats['test']}")
    
    # 保存统计信息
    with open("df40_train_organization_stats.json", "w") as f:
        json.dump({
            "split_stats": split_stats,
            "category_stats": dict(stats),
            "split_ratios": {
                "train": split_stats['train'] / sum(split_stats.values()) * 100,
                "val": split_stats['val'] / sum(split_stats.values()) * 100,
                "test": split_stats['test'] / sum(split_stats.values()) * 100
            }
        }, f, indent=2)
    
    return split_stats

def organize_df40_test_data():
    """组织DF40官方测试数据，只使用10%并分配到val和final_test_sets"""
    
    df40_test_source = Path("D:/work/AWARE-NET/dataset/DF40/test")
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    
    print("\n=== DF40官方测试数据组织 (使用10%，分配到val和final_test_sets) ===")
    
    if not df40_test_source.exists():
        print(f"⚠️ 官方测试目录不存在: {df40_test_source}")
        return
    
    # 收集所有测试图像
    all_test_images = []
    for method_dir in df40_test_source.iterdir():
        if method_dir.is_dir():
            for img_file in method_dir.glob("*.png"):
                all_test_images.append((img_file, method_dir.name))
    
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
    val_dir = processed_data / "val" / "df40" / "fake"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file, method_name in tqdm(val_images, desc="复制到val"):
        filename = f"df40_test_{method_name}_{img_file.stem}.png"
        target_path = val_dir / filename
        if not target_path.exists():
            shutil.copy2(img_file, target_path)
    
    # 复制到final_test_sets目录
    final_test_dir = processed_data / "final_test_sets" / "df40" / "fake"
    final_test_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file, method_name in tqdm(final_test_images, desc="复制到final_test_sets"):
        filename = f"df40_test_{method_name}_{img_file.stem}.png"
        target_path = final_test_dir / filename
        if not target_path.exists():
            shutil.copy2(img_file, target_path)
    
    print(f"官方测试数据组织完成: {len(val_images)}个在val, {len(final_test_images)}个在final_test_sets")
    
    return len(val_images), len(final_test_images)

def update_manifests():
    """更新manifest文件以包含DF40数据"""
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    manifests_dir = processed_data / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    print("\n=== 更新Manifest文件 ===")
    
    splits = ["train", "val"]
    
    for split in splits:
        df40_dir = processed_data / split / "df40" / "fake"
        if not df40_dir.exists():
            continue
            
        manifest_file = manifests_dir / f"{split}_manifest.csv"
        
        # 读取现有manifest (如果存在)
        existing_data = []
        if manifest_file.exists():
            import pandas as pd
            existing_df = pd.read_csv(manifest_file)
            existing_data = existing_df.to_dict('records')
        
        # 添加DF40数据
        df40_data = []
        for img_file in df40_dir.glob("*.png"):
            df40_data.append({
                "image_path": str(img_file.relative_to(processed_data)),
                "label": "fake",
                "dataset": "df40",
                "split": split
            })
        
        # 合并数据
        all_data = existing_data + df40_data
        
        # 保存manifest
        import pandas as pd
        df = pd.DataFrame(all_data)
        df.to_csv(manifest_file, index=False)
        
        print(f"更新了 {manifest_file}: 添加了 {len(df40_data)} 条DF40记录")
    
    # 处理final_test_sets
    final_test_dir = processed_data / "final_test_sets" / "df40" / "fake"
    if final_test_dir.exists():
        manifest_file = manifests_dir / "final_test_manifest.csv"
        
        existing_data = []
        if manifest_file.exists():
            import pandas as pd
            existing_df = pd.read_csv(manifest_file)
            existing_data = existing_df.to_dict('records')
        
        df40_data = []
        for img_file in final_test_dir.glob("*.png"):
            df40_data.append({
                "image_path": str(img_file.relative_to(processed_data)),
                "label": "fake", 
                "dataset": "df40",
                "split": "test"
            })
        
        all_data = existing_data + df40_data
        import pandas as pd
        df = pd.DataFrame(all_data)
        df.to_csv(manifest_file, index=False)
        
        print(f"更新了 {manifest_file}: 添加了 {len(df40_data)} 条DF40记录")

def main():
    print("DF40数据集组织工具")
    print("="*50)
    
    # 1. 组织训练数据 (70/15/15)
    train_stats = organize_df40_train_data()
    
    # 2. 组织官方测试数据 (10%用量，50/50分配)
    val_count, final_test_count = organize_df40_test_data()
    
    # 3. 更新manifest文件
    update_manifests()
    
    print(f"\n{'='*50}")
    print("✅ DF40数据组织完成!")
    print(f"训练数据分割: Train={train_stats['train']}, Val={train_stats['val']}, Test={train_stats['test']}")
    print(f"官方测试数据: {val_count}个在val, {final_test_count}个在final_test_sets")
    print("\n建议下一步:")
    print("1. 检查processed_data目录结构")
    print("2. 验证manifest文件")
    print("3. 开始模型训练")

if __name__ == "__main__":
    main()