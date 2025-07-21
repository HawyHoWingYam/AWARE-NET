#!/usr/bin/env python3
"""
分析DF40路径转换和标签解析逻辑
"""

import json
from pathlib import Path

def analyze_path_conversion():
    """分析路径转换逻辑"""
    
    # 从截图中的示例路径
    test_cases = [
        {
            "json_path": "deepfakes_detection_datasets/DF40_train/facedancer/frames/993_989/131.png",
            "expected_local": "DF40/Face-swapping/facedancer/frames/993_989/131.png",
            "json_label": "facedancer_Fake"
        },
        {
            "json_path": "deepfakes_detection_datasets/DF40_train/inswap/frames/993_989/131.png", 
            "expected_local": "DF40/Face-swapping/inswap/frames/993_989/131.png",
            "json_label": "inswap_Fake"
        },
        {
            "json_path": "deepfakes_detection_datasets/DF40_train/blendface/frames/993_989/131.png",
            "expected_local": "DF40/Face-swapping/blendface/frames/993_989/131.png", 
            "json_label": "blendface_Fake"
        }
    ]
    
    print("=== 路径转换逻辑分析 ===")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"JSON路径: {case['json_path']}")
        print(f"期望本地路径: {case['expected_local']}")
        print(f"JSON标签: {case['json_label']}")
        
        # 测试我的路径转换函数
        converted = convert_json_path_to_actual(case['json_path'])
        print(f"转换结果: {converted}")
        
        match = converted == case['expected_local'] if converted else False
        print(f"转换正确: {'✓' if match else '✗'}")
        
        # 测试标签解析
        parsed_label = parse_label(case['json_label'])
        print(f"标签解析: {case['json_label']} -> {parsed_label}")
        expected_label = 'fake'
        label_match = parsed_label == expected_label
        print(f"标签正确: {'✓' if label_match else '✗'}")

def convert_json_path_to_actual(json_path):
    """路径转换函数（从原脚本复制并分析）"""
    print(f"  开始转换: {json_path}")
    
    if 'DF40' not in json_path:
        print(f"  ❌ 路径中不包含DF40")
        return None
    
    parts = json_path.split('/')
    print(f"  分割路径: {parts}")
    
    df40_index = None
    
    # 找到DF40在路径中的位置
    for i, part in enumerate(parts):
        if 'DF40' in part:
            df40_index = i
            break
    
    if df40_index is None:
        print(f"  ❌ 找不到DF40索引")
        return None
        
    print(f"  DF40索引: {df40_index}")
    
    # 重构路径
    remaining_parts = parts[df40_index:]
    print(f"  剩余部分: {remaining_parts}")
    
    if len(remaining_parts) < 4:
        print(f"  ❌ 剩余部分长度不足: {len(remaining_parts)}")
        return None
    
    # 确定方法名称
    method_name = remaining_parts[1] if len(remaining_parts) > 1 else 'unknown'
    print(f"  方法名称: {method_name}")
    
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
    print(f"  映射类别: {category}")
    
    if category == 'unknown':
        print(f"  ❌ 未知类别")
        return None
    
    # 构建实际路径
    if 'train' in json_path:
        actual_path = f"DF40/{category}/{method_name}/frames/{remaining_parts[-2]}/{remaining_parts[-1]}"
        print(f"  构建训练路径: {actual_path}")
    else:
        # test路径
        actual_path = f"DF40/test/{method_name}/frames/{remaining_parts[-2]}/{remaining_parts[-1]}"
        print(f"  构建测试路径: {actual_path}")
        
    return actual_path

def parse_label(json_label):
    """标签解析函数"""
    print(f"  开始解析标签: {json_label}")
    
    label_lower = json_label.lower()
    print(f"  转为小写: {label_lower}")
    
    # 检查是否为fake类型
    if ('fake' in label_lower or 'synthesis' in label_lower):
        result = 'fake'
        print(f"  匹配fake条件")
    elif ('real' in label_lower):
        result = 'real'
        print(f"  匹配real条件")
    else:
        result = 'unknown'
        print(f"  未匹配任何条件")
    
    return result

def analyze_file_naming():
    """分析文件命名逻辑"""
    print(f"\n=== 文件命名逻辑分析 ===")
    
    # 模拟文件操作结构
    test_file_ops = [
        {
            'source': Path('DF40/Face-swapping/facedancer/frames/993_989/131.png'),
            'method': 'facedancer',
            'video_id': '993_989',
            'label': 'fake'
        },
        {
            'source': Path('DF40/Face-swapping/inswap/frames/993_989/131.png'),
            'method': 'inswap', 
            'video_id': '993_989',
            'label': 'fake'
        }
    ]
    
    for i, op in enumerate(test_file_ops, 1):
        print(f"\n--- 文件命名案例 {i} ---")
        print(f"源文件: {op['source']}")
        print(f"方法: {op['method']}")
        print(f"视频ID: {op['video_id']}")
        print(f"标签: {op['label']}")
        
        # 生成目标文件名（按原脚本逻辑）
        filename = f"df40_{op['method']}_{op['video_id']}_{op['source'].stem}.png"
        print(f"生成文件名: {filename}")
        
        # 目标路径
        target_dir = f"processed_data/train/df40/{op['label']}"
        target_path = f"{target_dir}/{filename}"
        print(f"目标路径: {target_path}")

if __name__ == "__main__":
    analyze_path_conversion()
    analyze_file_naming()