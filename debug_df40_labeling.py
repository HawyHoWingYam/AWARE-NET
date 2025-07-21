#!/usr/bin/env python3
"""
调试DF40标签分类问题
"""

import json
from pathlib import Path

def debug_specific_file():
    """调试具体的JSON文件"""
    json_file = Path("D:/work/AWARE-NET/dataset/DF40/rearrangement/facedancer_ff.json")
    
    if not json_file.exists():
        print(f"文件不存在: {json_file}")
        return
    
    print(f"=== 调试 {json_file.name} ===")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for main_key, main_data in data.items():
            print(f"主键: {main_key}")
            
            if isinstance(main_data, dict):
                for label_type, label_data in main_data.items():
                    print(f"  标签类型: {label_type}")
                    
                    if isinstance(label_data, dict):
                        for split_type in ['train', 'test']:
                            if split_type in label_data and isinstance(label_data[split_type], dict):
                                print(f"    {split_type}数据:")
                                
                                # 找到包含993_989的条目
                                for video_id, video_data in label_data[split_type].items():
                                    if '993_989' in video_id:
                                        print(f"      找到目标视频ID: {video_id}")
                                        
                                        if 'label' in video_data:
                                            original_label = video_data['label']
                                            print(f"        原始标签: {original_label}")
                                            
                                            # 测试标签解析
                                            is_fake = 'fake' in original_label.lower() or 'synthesis' in original_label.lower()
                                            actual_label = 'fake' if is_fake else 'real'
                                            print(f"        解析结果: {actual_label}")
                                            
                                            # 检查帧路径
                                            if 'frames' in video_data:
                                                frames = video_data['frames']
                                                print(f"        帧数量: {len(frames)}")
                                                
                                                # 找到131.png
                                                target_frames = [f for f in frames if f.endswith('131.png')]
                                                if target_frames:
                                                    print(f"        找到131.png: {target_frames[0]}")
                                                    
                                                    # 检查路径转换
                                                    json_path = target_frames[0]
                                                    converted_path = convert_json_path_to_actual(json_path)
                                                    print(f"        转换路径: {converted_path}")
                                                    
                                                    # 检查实际文件是否存在
                                                    actual_file = Path(f"D:/work/AWARE-NET/dataset/{converted_path}")
                                                    print(f"        文件存在: {actual_file.exists()}")
                                                    
                                                    # 检查处理后的文件位置
                                                    processed_real = Path(f"D:/work/AWARE-NET/processed_data/train/df40/real")
                                                    processed_fake = Path(f"D:/work/AWARE-NET/processed_data/train/df40/fake")
                                                    
                                                    real_files = list(processed_real.glob("*993_989*131*")) if processed_real.exists() else []
                                                    fake_files = list(processed_fake.glob("*993_989*131*")) if processed_fake.exists() else []
                                                    
                                                    print(f"        在real目录找到: {len(real_files)} 个文件")
                                                    print(f"        在fake目录找到: {len(fake_files)} 个文件")
                                                    
                                                    if real_files:
                                                        print(f"          real文件: {real_files[0].name}")
                                                    if fake_files:
                                                        print(f"          fake文件: {fake_files[0].name}")
    
    except Exception as e:
        print(f"处理文件时出错: {e}")

def convert_json_path_to_actual(json_path):
    """路径转换函数（复制自原脚本）"""
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
        'facedancer': 'Face-swapping',
        # ... 其他映射
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

if __name__ == "__main__":
    debug_specific_file()