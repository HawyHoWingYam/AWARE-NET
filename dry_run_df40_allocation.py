#!/usr/bin/env python3
"""
DF40分配脚本的干运行测试 - 不实际复制文件，只测试逻辑
"""

import os
import random
from pathlib import Path

def dry_run_collect_files(df40_base, limit_per_method=100):
    """干运行：收集少量文件用于测试"""
    
    df40_path = Path(df40_base)
    
    print("=== 干运行：收集Face-swapping数据 ===")
    
    # 1. 收集训练集数据（限制每个方法的文件数）
    train_files = []
    face_swapping_dir = df40_path / 'Face-swapping'
    
    if face_swapping_dir.exists():
        for method_dir in face_swapping_dir.iterdir():
            if method_dir.is_dir():
                method_count = 0
                frames_dir = method_dir / 'frames'
                if frames_dir.exists():
                    for video_dir in frames_dir.iterdir():
                        if video_dir.is_dir():
                            for img_file in video_dir.glob('*.png'):
                                if method_count < limit_per_method:
                                    train_files.append({
                                        'source': img_file,
                                        'method': method_dir.name,
                                        'video_id': video_dir.name,
                                        'data_type': 'train'
                                    })
                                    method_count += 1
                                else:
                                    break
                            if method_count >= limit_per_method:
                                break
                
                print(f"  {method_dir.name}: {method_count} 个文件（训练集）")
    
    # 2. 收集测试集数据（限制每个方法的文件数）
    test_files = []
    test_face_swapping_dir = df40_path / 'test' / 'Face-swapping'
    
    if test_face_swapping_dir.exists():
        for method_dir in test_face_swapping_dir.iterdir():
            if method_dir.is_dir():
                method_count = 0
                for img_file in method_dir.rglob('*.png'):
                    if method_count < limit_per_method:
                        relative_path = img_file.relative_to(method_dir)
                        video_id = relative_path.parent.name if relative_path.parent.name != '.' else 'unknown'
                        
                        test_files.append({
                            'source': img_file,
                            'method': method_dir.name,
                            'video_id': video_id,
                            'data_type': 'test'
                        })
                        method_count += 1
                    else:
                        break
                
                print(f"  {method_dir.name}: {method_count} 个文件（测试集）")
    
    print(f"\\n收集到训练集文件: {len(train_files)}")
    print(f"收集到测试集文件: {len(test_files)}")
    
    return train_files, test_files

def dry_run_allocate_files(train_files, test_files):
    """干运行：测试分配逻辑"""
    
    print("\\n=== 干运行：文件分配测试 ===")
    
    random.seed(42)
    
    allocated_files = []
    
    # 训练集分配 (70%/15%/15%)
    print(f"分配训练集数据 ({len(train_files)} 个文件)...")
    
    random.shuffle(train_files)
    
    train_count = int(len(train_files) * 0.70)
    val_count = int(len(train_files) * 0.15)
    
    for i, file_info in enumerate(train_files):
        if i < train_count:
            file_info['final_split'] = 'train'
        elif i < train_count + val_count:
            file_info['final_split'] = 'val'
        else:
            file_info['final_split'] = 'final_test_sets'
        
        file_info['final_label'] = 'fake'
        allocated_files.append(file_info)
    
    train_to_train = train_count
    train_to_val = val_count
    train_to_test = len(train_files) - train_count - val_count
    
    print(f"  -> train: {train_to_train}")
    print(f"  -> val: {train_to_val}")
    print(f"  -> final_test_sets: {train_to_test}")
    
    # 测试集分配 (随机取10%，然后5%/5%)
    print(f"\\n分配测试集数据 ({len(test_files)} 个文件)...")
    
    random.shuffle(test_files)
    
    test_10_percent = int(len(test_files) * 0.10)
    selected_test_files = test_files[:test_10_percent]
    
    print(f"  选择10%测试数据: {len(selected_test_files)} 个文件")
    
    half_count = len(selected_test_files) // 2
    
    for i, file_info in enumerate(selected_test_files):
        if i < half_count:
            file_info['final_split'] = 'val'
        else:
            file_info['final_split'] = 'final_test_sets'
        
        file_info['final_label'] = 'fake'
        allocated_files.append(file_info)
    
    test_to_val = half_count
    test_to_test = len(selected_test_files) - half_count
    
    print(f"  -> val: {test_to_val}")
    print(f"  -> final_test_sets: {test_to_test}")
    
    # 最终统计
    final_stats = {'train': train_to_train, 'val': train_to_val + test_to_val, 'final_test_sets': train_to_test + test_to_test}
    
    print(f"\\n=== 最终分配统计 ===")
    for split, count in final_stats.items():
        print(f"{split}: {count} 个文件")
    
    print(f"总计: {sum(final_stats.values())} 个文件")
    
    return allocated_files, final_stats

if __name__ == "__main__":
    df40_base = "D:/work/AWARE-NET/dataset/DF40"
    
    print("DF40数据分配脚本 - 干运行测试")
    print("="*50)
    print("限制：每个方法最多100个文件用于测试")
    print("="*50)
    
    # 收集少量文件
    train_files, test_files = dry_run_collect_files(df40_base, limit_per_method=100)
    
    if train_files or test_files:
        # 测试分配逻辑
        allocated_files, stats = dry_run_allocate_files(train_files, test_files)
        
        print(f"\\n✅ 干运行测试完成！")
        print("逻辑验证通过，可以执行完整脚本")
    else:
        print("未找到测试文件")