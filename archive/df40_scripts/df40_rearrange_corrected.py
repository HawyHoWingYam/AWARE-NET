#!/usr/bin/env python3
"""
DF40数据集重新组织脚本 - 修正版
正确处理标签分类：保持原始real/fake标签不变，只重新分配train/val/test划分
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_all_json_labels(rearrangement_dir):
    """加载所有JSON文件中的标签信息"""
    labels = {}  # path -> label_info
    rearrangement_path = Path(rearrangement_dir)
    
    print("=== 加载JSON标签文件 ===")
    
    for json_file in rearrangement_path.glob("*.json"):
        print(f"处理: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历JSON结构
            for main_key, main_data in data.items():
                if isinstance(main_data, dict):
                    for label_type, label_data in main_data.items():
                        if isinstance(label_data, dict):
                            for split_type in ['train', 'test']:
                                if split_type in label_data and isinstance(label_data[split_type], dict):
                                    for video_id, video_data in label_data[split_type].items():
                                        if 'frames' in video_data:
                                            original_label = video_data.get('label', 'unknown')
                                            
                                            # 正确解析标签：保持原始含义
                                            if ('fake' in original_label.lower() or 'synthesis' in original_label.lower()):
                                                actual_label = 'fake'
                                            elif ('real' in original_label.lower()):
                                                actual_label = 'real'
                                            else:
                                                actual_label = 'unknown'
                                            
                                            for frame_path in video_data['frames']:
                                                # 转换路径格式
                                                converted_path = convert_json_path_to_actual(frame_path)
                                                if converted_path:
                                                    labels[converted_path] = {
                                                        'label': actual_label,  # 保持原始标签
                                                        'original_split': split_type,
                                                        'method': main_key,
                                                        'original_label': original_label,
                                                        'json_file': json_file.name
                                                    }
        except Exception as e:
            print(f"处理{json_file.name}时出错: {e}")
    
    print(f"总共加载了 {len(labels)} 个图像标签")
    return labels

def convert_json_path_to_actual(json_path):
    """将JSON中的路径转换为实际文件路径"""
    if 'DF40' not in json_path:
        return None
    
    parts = json_path.split('/')
    df40_index = None
    
    # 找到DF40在路径中的位置
    for i, part in enumerate(parts):
        if 'DF40' in part:
            df40_index = i
            break
    
    if df40_index is None:
        return None
    
    # 重构路径
    remaining_parts = parts[df40_index:]
    
    if len(remaining_parts) < 4:
        return None
    
    # 确定类别目录
    method_name = remaining_parts[1] if len(remaining_parts) > 1 else 'unknown'
    
    # 映射到实际目录结构
    category_mapping = {
        # Face-swapping methods
        'blendface': 'Face-swapping',
        'e4s': 'Face-swapping', 
        'facedancer': 'Face-swapping',
        'faceswap': 'Face-swapping',
        'fsgan': 'Face-swapping',
        'inswap': 'Face-swapping',
        'mobileswap': 'Face-swapping',
        'simswap': 'Face-swapping',
        'uniface': 'Face-swapping',
        
        # Face-reenactment methods  
        'facevid2vid': 'Face-reenactment',
        'fomm': 'Face-reenactment',
        'hyperreenact': 'Face-reenactment',
        'lia': 'Face-reenactment',
        'mcnet': 'Face-reenactment',
        'one_shot_free': 'Face-reenactment',
        'pirender': 'Face-reenactment',
        'sadtalker': 'Face-reenactment',
        'tpsm': 'Face-reenactment',
        'wav2lip': 'Face-reenactment',
        'MRAA': 'Face-reenactment',
        
        # Entire Face Synthesis methods
        'danet': 'Entire Face Synthesis',
        'ddim': 'Entire Face Synthesis',
        'DiT': 'Entire Face Synthesis',
        'pixart': 'Entire Face Synthesis', 
        'RDDM': 'Entire Face Synthesis',
        'sd2.1': 'Entire Face Synthesis',
        'SiT': 'Entire Face Synthesis',
        'StyleGAN2': 'Entire Face Synthesis',
        'StyleGAN3': 'Entire Face Synthesis',
        'StyleGANXL': 'Entire Face Synthesis',
        'VQGAN': 'Entire Face Synthesis',
    }
    
    category = category_mapping.get(method_name, 'unknown')
    if category == 'unknown':
        return None
    
    # 构建实际路径
    if 'train' in json_path:
        actual_path = f"DF40/{category}/{method_name}/frames/{remaining_parts[-2]}/{remaining_parts[-1]}"
    else:
        # test路径
        actual_path = f"DF40/test/{method_name}/frames/{remaining_parts[-2]}/{remaining_parts[-1]}"
        
    return actual_path

def organize_df40_data(df40_base, labels, processed_data_dir):
    """根据JSON标签组织DF40数据"""
    df40_path = Path(df40_base)
    processed_path = Path(processed_data_dir)
    
    print("=== 开始组织DF40数据 ===")
    
    # 统计信息
    stats = defaultdict(lambda: defaultdict(int))
    file_operations = []
    
    # 遍历所有实际存在的图像文件
    for category in ['Entire Face Synthesis', 'Face-reenactment', 'Face-swapping']:
        category_path = df40_path / category
        if category_path.exists():
            print(f"处理类别: {category}")
            
            for method_dir in category_path.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    print(f"  处理方法: {method_name}")
                    
                    frames_dir = method_dir / 'frames'
                    if frames_dir.exists():
                        for video_dir in frames_dir.iterdir():
                            if video_dir.is_dir():
                                for img_file in video_dir.glob('*.png'):
                                    # 构建相对路径来查找标签
                                    relative_path = f"DF40/{category}/{method_name}/frames/{video_dir.name}/{img_file.name}"
                                    
                                    if relative_path in labels:
                                        label_info = labels[relative_path]
                                        label = label_info['label']  # 保持原始标签
                                        original_split = label_info['original_split']
                                        
                                        stats[f"{category}_{method_name}"][label] += 1
                                        
                                        file_operations.append({
                                            'source': img_file,
                                            'category': category,
                                            'method': method_name,
                                            'label': label,  # 原始标签
                                            'original_split': original_split,
                                            'video_id': video_dir.name,
                                            'json_file': label_info['json_file']
                                        })
    
    # 处理测试数据
    test_path = df40_path / 'test'
    if test_path.exists():
        print("处理测试数据")
        
        for method_dir in test_path.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                print(f"  处理测试方法: {method_name}")
                
                frames_dir = method_dir / 'frames'
                if frames_dir.exists():
                    for video_dir in frames_dir.iterdir():
                        if video_dir.is_dir():
                            for img_file in video_dir.glob('*.png'):
                                relative_path = f"DF40/test/{method_name}/frames/{video_dir.name}/{img_file.name}"
                                
                                if relative_path in labels:
                                    label_info = labels[relative_path]
                                    label = label_info['label']  # 保持原始标签
                                    original_split = label_info['original_split']
                                    
                                    stats[f"test_{method_name}"][label] += 1
                                    
                                    file_operations.append({
                                        'source': img_file,
                                        'category': 'test',
                                        'method': method_name,
                                        'label': label,  # 原始标签
                                        'original_split': original_split,
                                        'video_id': video_dir.name,
                                        'json_file': label_info['json_file']
                                    })
    
    print(f"\n=== 数据统计 ===")
    for method, counts in stats.items():
        print(f"{method}: Real={counts['real']}, Fake={counts['fake']}")
    
    return file_operations, stats

def redistribute_data_splits(file_operations):
    """
    修正后的重新分配逻辑：
    - 保持原始real/fake标签不变
    - 只重新分配train/val/final_test_sets划分
    """
    print(f"\n=== 重新分配train/val/test划分 ===")
    
    # 按类别分组数据
    face_swapping_train = []
    face_swapping_test = []
    other_train = []
    other_test = []
    
    for op in file_operations:
        is_face_swapping = 'face-swapping' in op['category'].lower()
        is_train = op['original_split'] == 'train'
        
        if is_face_swapping:
            if is_train:
                face_swapping_train.append(op)
            else:
                face_swapping_test.append(op)
        else:
            if is_train:
                other_train.append(op)
            else:
                other_test.append(op)
    
    print(f"Face-swapping训练数据: {len(face_swapping_train)}")
    print(f"Face-swapping测试数据: {len(face_swapping_test)}")
    print(f"其他训练数据: {len(other_train)}")
    print(f"其他测试数据: {len(other_test)}")
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # Face-swapping数据重新分配（只改变split，不改变label）
    redistributed_operations = []
    
    # 处理Face-swapping测试数据 (25%/25%/25%/25%)
    random.shuffle(face_swapping_test)
    test_chunk_size = len(face_swapping_test) // 4
    
    for i, item in enumerate(face_swapping_test):
        chunk = i // test_chunk_size
        if chunk == 0:  # 25% -> val
            item['final_split'] = 'val'
        elif chunk == 1:  # 25% -> val  
            item['final_split'] = 'val'
        elif chunk == 2:  # 25% -> final_test_sets
            item['final_split'] = 'final_test_sets'
        else:  # 25% -> final_test_sets
            item['final_split'] = 'final_test_sets'
        
        # 保持原始标签不变
        item['final_label'] = item['label']
        redistributed_operations.append(item)
    
    # 处理Face-swapping训练数据 (35%/35%/7.5%/7.5%/7.5%/7.5%)
    random.shuffle(face_swapping_train)
    train_chunk_size = len(face_swapping_train) // 100
    
    for i, item in enumerate(face_swapping_train):
        percentage = (i // train_chunk_size)
        
        if percentage < 35:  # 35% -> train
            item['final_split'] = 'train'
        elif percentage < 70:  # 35% -> train
            item['final_split'] = 'train'
        elif percentage < 77:  # 7.5% -> final_test_sets
            item['final_split'] = 'final_test_sets'
        elif percentage < 85:  # 7.5% -> final_test_sets
            item['final_split'] = 'final_test_sets'
        elif percentage < 92:  # 7.5% -> val
            item['final_split'] = 'val'
        else:  # 7.5% -> val
            item['final_split'] = 'val'
        
        # 保持原始标签不变
        item['final_label'] = item['label']
        redistributed_operations.append(item)
    
    # 其他数据保持原始分配
    for item in other_train:
        item['final_split'] = 'train'
        item['final_label'] = item['label']  # 保持原始标签
        redistributed_operations.append(item)
        
    for item in other_test:
        item['final_split'] = 'final_test_sets'
        item['final_label'] = item['label']  # 保持原始标签
        redistributed_operations.append(item)
    
    return redistributed_operations

def copy_files_to_processed_data(file_operations, processed_data_dir):
    """将文件复制到processed_data目录"""
    processed_path = Path(processed_data_dir)
    
    print(f"\n=== 复制文件到processed_data ===")
    
    # 创建目录并复制文件
    copy_stats = defaultdict(int)
    
    for op in tqdm(file_operations, desc="复制文件"):
        final_split = op['final_split']
        final_label = op['final_label']
        
        # 创建目标目录
        target_dir = processed_path / final_split / 'df40' / final_label
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        method = op['method']
        video_id = op['video_id']
        filename = f"df40_{method}_{video_id}_{op['source'].stem}.png"
        target_path = target_dir / filename
        
        # 复制文件
        if not target_path.exists():
            shutil.copy2(op['source'], target_path)
        
        copy_stats[f"{final_split}/{final_label}"] += 1
    
    print(f"\n=== 复制统计 ===")
    for location, count in copy_stats.items():
        print(f"{location}: {count} 个文件")

def main():
    setup_logging()
    
    # 配置路径
    df40_base = "D:/work/AWARE-NET/dataset/DF40"
    rearrangement_dir = "D:/work/AWARE-NET/dataset/DF40/rearrangement" 
    processed_data_dir = "D:/work/AWARE-NET/processed_data"
    
    print("DF40数据重组工具（修正版）")
    print("="*60)
    print("修正逻辑：保持原始real/fake标签，只重新分配train/val/test划分")
    
    # 加载JSON标签
    labels = load_all_json_labels(rearrangement_dir)
    
    if not labels:
        print("未找到任何标签，退出")
        return
    
    # 组织数据
    file_operations, stats = organize_df40_data(df40_base, labels, processed_data_dir)
    
    print(f"\n找到 {len(file_operations)} 个需要处理的文件")
    
    # 重新分配train/val/test划分（保持标签不变）
    redistributed_operations = redistribute_data_splits(file_operations)
    
    # 复制文件
    copy_files_to_processed_data(redistributed_operations, processed_data_dir)
    
    print(f"\nDF40数据重组完成（修正版）！")
    print(f"\n关键修正:")
    print("1. 保持原始real/fake标签不变")
    print("2. 只重新分配train/val/final_test_sets划分")
    print("3. facedancer_Fake始终分类为fake")

if __name__ == "__main__":
    main()