#!/usr/bin/env python3
"""
测试DF40 JSON标签解析逻辑
"""

import json
from pathlib import Path

def test_label_classification():
    """测试标签分类逻辑"""
    
    # 测试不同的标签格式
    test_labels = [
        "facedancer_fake",
        "facedancer_Real", 
        "blendface_fake",
        "blendface_Real",
        "simswap_fake",
        "simswap_Real",
        "e4s_fake",
        "e4s_Real",
        "fomm_fake",
        "fomm_Real",
        "StyleGAN2_synthesis",
        "StyleGAN3_synthesis",
        "ddim_synthesis"
    ]
    
    print("=== 测试标签分类逻辑 ===")
    print("格式: 原始标签 -> 分类结果")
    print("-" * 40)
    
    def classify_label_old(label):
        """旧的错误逻辑"""
        is_fake = 'fake' in label.lower() or 'synthesis' in label.lower()
        return 'fake' if is_fake else 'real'
    
    def classify_label_new(label):
        """新的正确逻辑"""
        label_lower = label.lower()
        
        # 检查是否为fake类型
        if ('_fake' in label_lower or 
            'fake' in label_lower or 
            'synthesis' in label_lower):
            return 'fake'
        
        # 检查是否为real类型  
        elif ('_real' in label_lower or 
              'real' in label_lower):
            return 'real'
        
        # 未知类型
        else:
            return 'unknown'
    
    for label in test_labels:
        old_result = classify_label_old(label)
        new_result = classify_label_new(label)
        
        status = "✓" if old_result == new_result else "✗"
        print(f"{status} {label:<20} -> 旧:{old_result:<4} | 新:{new_result:<4}")

def test_actual_json_file():
    """测试实际JSON文件中的标签"""
    json_file = Path("D:/work/AWARE-NET/dataset/DF40/rearrangement/facedancer_ff.json")
    
    if not json_file.exists():
        print(f"JSON文件不存在: {json_file}")
        return
        
    print(f"\n=== 测试实际JSON文件 ===")
    print(f"文件: {json_file.name}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels_found = set()
        
        # 遍历JSON找到所有标签
        for main_key, main_data in data.items():
            if isinstance(main_data, dict):
                for label_type, label_data in main_data.items():
                    if isinstance(label_data, dict):
                        for split_type in ['train', 'test']:
                            if split_type in label_data:
                                for video_id, video_data in label_data[split_type].items():
                                    if 'label' in video_data:
                                        labels_found.add(video_data['label'])
        
        print(f"找到的标签: {sorted(labels_found)}")
        
        # 测试分类
        def classify_label_corrected(label):
            """修正后的分类逻辑"""
            label_lower = label.lower()
            
            if ('fake' in label_lower or 'synthesis' in label_lower):
                return 'fake'
            elif ('real' in label_lower):
                return 'real'
            else:
                return 'unknown'
        
        print("\n标签分类结果:")
        for label in sorted(labels_found):
            result = classify_label_corrected(label)
            print(f"  {label:<25} -> {result}")
            
    except Exception as e:
        print(f"处理JSON文件时出错: {e}")

if __name__ == "__main__":
    test_label_classification()
    test_actual_json_file()