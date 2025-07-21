#!/usr/bin/env python3
"""
分析DF40数据结构和JSON重排列文件
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_json_structure():
    """分析JSON重排列文件的结构"""
    rearrangement_dir = Path("D:/work/AWARE-NET/dataset/DF40/rearrangement")
    
    print("=== DF40 JSON重排列文件分析 ===")
    
    if not rearrangement_dir.exists():
        print(f"目录不存在: {rearrangement_dir}")
        return
    
    json_files = list(rearrangement_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 分析每个JSON文件
    source_datasets = defaultdict(int)
    label_types = defaultdict(int)
    
    for json_file in json_files[:3]:  # 只分析前3个文件
        print(f"\n--- {json_file.name} ---")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for main_key, main_data in data.items():
                print(f"主键: {main_key}")
                
                if isinstance(main_data, dict):
                    for label_type, label_data in main_data.items():
                        print(f"  标签类型: {label_type}")
                        label_types[label_type] += 1
                        
                        if isinstance(label_data, dict):
                            for split_type in ['train', 'test']:
                                if split_type in label_data and label_data[split_type]:
                                    print(f"    {split_type}: {len(label_data[split_type])} 个视频")
                                    
                                    # 分析第一个视频的路径来源
                                    first_video = next(iter(label_data[split_type].values()))
                                    if 'frames' in first_video and first_video['frames']:
                                        first_path = first_video['frames'][0]
                                        print(f"      示例路径: {first_path}")
                                        
                                        # 识别数据源
                                        if 'FaceForensics++' in first_path:
                                            source_datasets['FaceForensics++'] += 1
                                        elif 'Celeb-DF-v2' in first_path:
                                            source_datasets['CelebDF-v2'] += 1
                                        elif 'DF40' in first_path:
                                            source_datasets['DF40'] += 1
                
        except Exception as e:
            print(f"处理{json_file.name}时出错: {e}")
    
    print(f"\n=== 总结 ===")
    print("数据源分布:")
    for source, count in source_datasets.items():
        print(f"  {source}: {count}")
    
    print("标签类型:")
    for label, count in label_types.items():
        print(f"  {label}: {count}")

def check_df40_actual_structure():
    """检查DF40实际目录结构"""
    df40_path = Path("D:/work/AWARE-NET/dataset/DF40")
    
    print(f"\n=== DF40实际目录结构 ===")
    
    if not df40_path.exists():
        print(f"DF40目录不存在: {df40_path}")
        return
    
    print(f"DF40根目录: {df40_path}")
    
    # 列出主要子目录
    for item in df40_path.iterdir():
        if item.is_dir():
            print(f"  目录: {item.name}")
            
            # 如果是方法类别目录，进一步检查
            if item.name in ['Entire Face Synthesis', 'Face-reenactment', 'Face-swapping', 'test']:
                method_count = len([x for x in item.iterdir() if x.is_dir()])
                print(f"    包含 {method_count} 个方法子目录")
        else:
            print(f"  文件: {item.name}")

def main():
    analyze_json_structure()
    check_df40_actual_structure()
    
    print(f"\n=== 结论 ===")
    print("1. JSON重排列文件包含来自FaceForensics++和CelebDF-v2的路径")
    print("2. 这些文件定义了如何重新组织现有数据集来创建DF40测试集") 
    print("3. 需要从源数据集复制对应文件，而不是在DF40目录中查找")
    print("4. 当前的重组脚本逻辑需要完全重写")

if __name__ == "__main__":
    main()