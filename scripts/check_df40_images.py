#!/usr/bin/env python3
"""
检查DF40数据集中图像的规格
"""

import os
import cv2
import numpy as np
from pathlib import Path

def check_image_specs(image_path):
    """检查单个图像的规格"""
    try:
        # 使用OpenCV读取图像
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width, channels = img.shape
        
        # 获取文件大小
        file_size = os.path.getsize(image_path)
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size_kb': file_size / 1024,
            'dtype': img.dtype
        }
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def analyze_df40_images():
    """分析DF40数据集中的图像"""
    base_path = "D:/work/AWARE-NET/dataset/DF40"
    
    # 检查根目录frames文件夹
    frames_dir = os.path.join(base_path, "frames")
    print("=== 分析 DF40 图像规格 ===\n")
    
    # 统计信息
    image_specs = []
    total_images = 0
    
    if os.path.exists(frames_dir):
        print(f"检查根目录frames文件夹: {frames_dir}")
        
        # 遍历子文件夹
        subfolders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        print(f"找到 {len(subfolders)} 个子文件夹")
        
        # 检查前几个文件夹的图像
        for i, subfolder in enumerate(subfolders[:5]):  # 只检查前5个文件夹
            subfolder_path = os.path.join(frames_dir, subfolder)
            images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            
            print(f"\n子文件夹 {subfolder}: {len(images)} 张图像")
            
            # 检查前几张图像
            for j, img_file in enumerate(images[:3]):  # 每个文件夹检查前3张图像
                img_path = os.path.join(subfolder_path, img_file)
                specs = check_image_specs(img_path)
                
                if specs:
                    image_specs.append(specs)
                    total_images += 1
                    print(f"  {img_file}: {specs['width']}x{specs['height']}, {specs['channels']}通道, {specs['file_size_kb']:.1f}KB")
    
    # 检查method-specific文件夹
    method_folders = ['blendface', 'e4s', 'facedancer', 'faceswap', 'fsgan', 'inswap', 'mobileswap', 'simswap']
    
    for method in method_folders:
        method_frames_dir = os.path.join(base_path, method, "frames")
        if os.path.exists(method_frames_dir):
            print(f"\n检查 {method}/frames 文件夹:")
            
            # 获取前几个子文件夹
            try:
                subfolders = [f for f in os.listdir(method_frames_dir) if os.path.isdir(os.path.join(method_frames_dir, f))]
                if subfolders:
                    # 检查第一个子文件夹
                    first_subfolder = subfolders[0]
                    subfolder_path = os.path.join(method_frames_dir, first_subfolder)
                    images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
                    
                    if images:
                        # 检查第一张图像
                        img_path = os.path.join(subfolder_path, images[0])
                        specs = check_image_specs(img_path)
                        
                        if specs:
                            image_specs.append(specs)
                            print(f"  样本 {first_subfolder}/{images[0]}: {specs['width']}x{specs['height']}, {specs['channels']}通道, {specs['file_size_kb']:.1f}KB")
                        
                        total_images += len(images) * len(subfolders)  # 估算总数
                    print(f"  文件夹数量: {len(subfolders)}, 估计总图像数: {len(images) * len(subfolders)}")
                else:
                    print(f"  {method}/frames 为空")
            except Exception as e:
                print(f"  无法访问 {method}/frames: {e}")
    
    # 分析统计结果
    if image_specs:
        print(f"\n=== 统计分析 (基于 {len(image_specs)} 个样本) ===")
        
        widths = [spec['width'] for spec in image_specs]
        heights = [spec['height'] for spec in image_specs]
        channels = [spec['channels'] for spec in image_specs]
        file_sizes = [spec['file_size_kb'] for spec in image_specs]
        
        print(f"图像尺寸:")
        print(f"  宽度: 最小={min(widths)}, 最大={max(widths)}, 平均={np.mean(widths):.1f}")
        print(f"  高度: 最小={min(heights)}, 最大={max(heights)}, 平均={np.mean(heights):.1f}")
        print(f"  通道数: {set(channels)}")
        print(f"  文件大小: 最小={min(file_sizes):.1f}KB, 最大={max(file_sizes):.1f}KB, 平均={np.mean(file_sizes):.1f}KB")
        
        # 检查是否所有图像都是相同尺寸
        if len(set(widths)) == 1 and len(set(heights)) == 1:
            print(f"\n✓ 所有图像都是统一尺寸: {widths[0]}x{heights[0]}")
        else:
            print(f"\n⚠ 图像尺寸不统一")
        
        # 推荐预处理参数
        most_common_width = max(set(widths), key=widths.count)
        most_common_height = max(set(heights), key=heights.count)
        
        print(f"\n=== 推荐的预处理参数 ===")
        print(f"图像尺寸: [{most_common_width}, {most_common_height}]")
        print(f"格式: PNG")
        print(f"通道数: {channels[0]}")
        
        return {
            'width': most_common_width,
            'height': most_common_height,
            'channels': channels[0],
            'format': 'PNG'
        }
    else:
        print("未找到有效的图像文件")
        return None

if __name__ == "__main__":
    try:
        specs = analyze_df40_images()
        if specs:
            print(f"\n建议在config/dataset_paths.json中使用以下设置:")
            print(f'"image_size": [{specs["width"]}, {specs["height"]}]')
    except Exception as e:
        print(f"分析过程中出错: {e}")