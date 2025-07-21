#!/usr/bin/env python3
"""
DF40数据简单分配脚本 - 按照用户需求实现
只处理Face-swapping类别，简单随机分配
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def collect_face_swapping_files(df40_base):
    """收集Face-swapping数据文件"""
    df40_path = Path(df40_base)
    
    print("=== 收集Face-swapping数据 ===")
    
    # 1. 收集训练集数据：DF40/Face-swapping/
    train_files = []
    face_swapping_dir = df40_path / 'Face-swapping'
    
    if face_swapping_dir.exists():
        print(f"处理训练集: {face_swapping_dir}")
        
        for method_dir in face_swapping_dir.iterdir():
            if method_dir.is_dir():
                frames_dir = method_dir / 'frames'
                if frames_dir.exists():
                    for video_dir in frames_dir.iterdir():
                        if video_dir.is_dir():
                            for img_file in video_dir.glob('*.png'):
                                train_files.append({
                                    'source': img_file,
                                    'method': method_dir.name,
                                    'video_id': video_dir.name,
                                    'data_type': 'train'
                                })
    
    print(f"收集到训练集文件: {len(train_files)}")
    
    # 2. 收集测试集数据：DF40/test/Face-swapping/ (先收集，后面随机选10%)
    test_files = []
    test_face_swapping_dir = df40_path / 'test' / 'Face-swapping'
    
    if test_face_swapping_dir.exists():
        print(f"处理测试集: {test_face_swapping_dir}")
        
        for method_dir in test_face_swapping_dir.iterdir():
            if method_dir.is_dir():
                # test目录结构不同，直接在method目录下查找PNG文件
                for img_file in method_dir.rglob('*.png'):
                    # 从路径中提取video_id信息
                    relative_path = img_file.relative_to(method_dir)
                    video_id = relative_path.parent.name if relative_path.parent.name != '.' else 'unknown'
                    
                    test_files.append({
                        'source': img_file,
                        'method': method_dir.name,
                        'video_id': video_id,
                        'data_type': 'test'
                    })
    
    print(f"收集到测试集文件: {len(test_files)}")
    
    return train_files, test_files

def allocate_files(train_files, test_files):
    """分配文件到不同的split"""
    
    print("=== 开始文件分配 ===")
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    allocated_files = []
    
    # 1. 处理训练集数据分配 (70%/15%/15%)
    print(f"分配训练集数据 ({len(train_files)} 个文件)...")
    
    random.shuffle(train_files)
    
    train_count = int(len(train_files) * 0.70)
    val_count = int(len(train_files) * 0.15)
    # 剩余的作为final_test_sets
    
    for i, file_info in enumerate(train_files):
        if i < train_count:
            file_info['final_split'] = 'train'
        elif i < train_count + val_count:
            file_info['final_split'] = 'val'
        else:
            file_info['final_split'] = 'final_test_sets'
        
        file_info['final_label'] = 'fake'
        allocated_files.append(file_info)
    
    print(f"  -> train: {train_count}")
    print(f"  -> val: {val_count}")
    print(f"  -> final_test_sets: {len(train_files) - train_count - val_count}")
    
    # 2. 处理测试集数据分配 (随机取10%，然后5%/5%)
    print(f"分配测试集数据 ({len(test_files)} 个文件)...")
    
    random.shuffle(test_files)
    
    # 只取10%的测试集数据
    test_10_percent = int(len(test_files) * 0.10)
    selected_test_files = test_files[:test_10_percent]
    
    print(f"  选择10%测试数据: {len(selected_test_files)} 个文件")
    
    # 5%/5%分配
    half_count = len(selected_test_files) // 2
    
    for i, file_info in enumerate(selected_test_files):
        if i < half_count:
            file_info['final_split'] = 'val'
        else:
            file_info['final_split'] = 'final_test_sets'
        
        file_info['final_label'] = 'fake'
        allocated_files.append(file_info)
    
    print(f"  -> val: {half_count}")
    print(f"  -> final_test_sets: {len(selected_test_files) - half_count}")
    
    return allocated_files

def copy_files_to_processed_data(allocated_files, processed_data_dir):
    """将分配的文件复制到processed_data目录"""
    
    processed_path = Path(processed_data_dir)
    
    print("=== 复制文件到processed_data ===")
    
    # 统计信息
    copy_stats = {'train': 0, 'val': 0, 'final_test_sets': 0}
    
    for file_info in tqdm(allocated_files, desc="复制文件"):
        final_split = file_info['final_split']
        final_label = file_info['final_label']
        
        # 创建目标目录
        target_dir = processed_path / final_split / 'df40' / final_label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        method = file_info['method']
        video_id = file_info['video_id']
        data_type = file_info['data_type']
        filename = f"df40_{method}_{data_type}_{video_id}_{file_info['source'].stem}.png"
        target_path = target_dir / filename
        
        # 复制文件
        if not target_path.exists():
            shutil.copy2(file_info['source'], target_path)
        
        copy_stats[final_split] += 1
    
    print("=== 复制统计 ===")
    for split, count in copy_stats.items():
        print(f"{split}: {count} 个文件")
    
    return copy_stats

def main():
    setup_logging()
    
    # 配置路径
    df40_base = "D:/work/AWARE-NET/dataset/DF40"
    processed_data_dir = "D:/work/AWARE-NET/processed_data"
    
    print("DF40数据简单分配脚本")
    print("="*50)
    print("数据来源:")
    print("1. 训练集: DF40/Face-swapping/ (70%/15%/15%)")
    print("2. 测试集: DF40/test/ 随机10% (5%/5%)")
    print("目标: 所有数据标记为fake")
    print("="*50)
    
    # 1. 收集文件
    train_files, test_files = collect_face_swapping_files(df40_base)
    
    if not train_files and not test_files:
        print("未找到任何文件，退出")
        return
    
    # 2. 分配文件
    allocated_files = allocate_files(train_files, test_files)
    
    print(f"总共分配 {len(allocated_files)} 个文件")
    
    # 3. 复制文件
    copy_stats = copy_files_to_processed_data(allocated_files, processed_data_dir)
    
    print("\\n✅ DF40数据分配完成！")
    print("\\n最终分布:")
    print(f"- train/df40/fake: {copy_stats['train']} 个文件")
    print(f"- val/df40/fake: {copy_stats['val']} 个文件") 
    print(f"- final_test_sets/df40/fake: {copy_stats['final_test_sets']} 个文件")

if __name__ == "__main__":
    main()