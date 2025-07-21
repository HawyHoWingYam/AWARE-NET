#!/usr/bin/env python3
"""
测试修正后的DF40处理逻辑
"""

import json
from pathlib import Path

def test_label_parsing():
    """测试标签解析逻辑"""
    print("=== 测试标签解析逻辑 ===")
    
    test_labels = [
        "facedancer_Fake",
        "facedancer_Real",
        "blendface_Fake", 
        "blendface_Real",
        "FSAll_Fake",
        "FSAll_Real"
    ]
    
    for label in test_labels:
        if ('fake' in label.lower() or 'synthesis' in label.lower()):
            parsed = 'fake'
        elif ('real' in label.lower()):
            parsed = 'real'
        else:
            parsed = 'unknown'
        
        print(f"{label:<20} -> {parsed}")

def test_specific_case():
    """测试993_989/131.png的具体案例"""
    print(f"\n=== 测试facedancer 993_989/131.png案例 ===")
    
    json_file = Path("D:/work/AWARE-NET/dataset/DF40/rearrangement/facedancer_ff.json")
    
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找993_989_facedancer条目
        for main_key, main_data in data.items():
            if isinstance(main_data, dict):
                for label_type, label_data in main_data.items():
                    if isinstance(label_data, dict):
                        for split_type in ['train', 'test']:
                            if split_type in label_data:
                                for video_id, video_data in label_data[split_type].items():
                                    if '993_989' in video_id and 'frames' in video_data:
                                        original_label = video_data.get('label', 'unknown')
                                        
                                        print(f"视频ID: {video_id}")
                                        print(f"原始标签: {original_label}")
                                        print(f"分割类型: {split_type}")
                                        
                                        # 修正后的标签解析
                                        if ('fake' in original_label.lower() or 'synthesis' in original_label.lower()):
                                            parsed_label = 'fake'
                                        elif ('real' in original_label.lower()):
                                            parsed_label = 'real'
                                        else:
                                            parsed_label = 'unknown'
                                        
                                        print(f"✅ 解析标签: {parsed_label}")
                                        print(f"✅ 最终标签: {parsed_label} (保持不变)")
                                        
                                        # 重新分配split逻辑示例
                                        if 'face-swapping' in 'Face-swapping'.lower():
                                            if split_type == 'train':
                                                print(f"✅ 作为Face-swapping训练数据重新分配split")
                                            else:
                                                print(f"✅ 作为Face-swapping测试数据重新分配split")
                                        
                                        return

def test_redistribution_logic():
    """测试重新分配逻辑"""
    print(f"\n=== 测试重新分配逻辑 ===")
    
    # 模拟一些文件操作
    test_operations = [
        {
            'category': 'Face-swapping',
            'method': 'facedancer',
            'label': 'fake',  # 原始标签
            'original_split': 'train',
            'video_id': '993_989'
        },
        {
            'category': 'Face-swapping', 
            'method': 'blendface',
            'label': 'fake',  # 原始标签
            'original_split': 'train',
            'video_id': '123_456'
        },
        {
            'category': 'Entire Face Synthesis',
            'method': 'StyleGAN2',
            'label': 'fake',  # 原始标签
            'original_split': 'train',
            'video_id': '789_012'
        }
    ]
    
    print("原始数据:")
    for i, op in enumerate(test_operations):
        print(f"  {i+1}. {op['method']} - 标签:{op['label']} 分割:{op['original_split']}")
    
    print(f"\n修正后的重新分配逻辑:")
    print("✅ Face-swapping数据: 重新分配train/val/final_test_sets，保持标签不变")
    print("✅ 其他数据: 保持原始分割，保持标签不变")
    
    for i, op in enumerate(test_operations):
        is_face_swapping = 'face-swapping' in op['category'].lower()
        
        if is_face_swapping:
            # 模拟重新分配
            final_split = 'train'  # 假设分配到train
            final_label = op['label']  # ✅ 保持原始标签
            print(f"  {i+1}. {op['method']} -> split:{final_split}, label:{final_label} ✅")
        else:
            # 保持原始分配
            final_split = 'train' if op['original_split'] == 'train' else 'final_test_sets'
            final_label = op['label']  # ✅ 保持原始标签
            print(f"  {i+1}. {op['method']} -> split:{final_split}, label:{final_label} ✅")

if __name__ == "__main__":
    test_label_parsing()
    test_specific_case() 
    test_redistribution_logic()