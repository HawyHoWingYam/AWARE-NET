#!/usr/bin/env python3
"""
检查DF40数据集中图像的规格（不使用OpenCV）
"""

import os
from pathlib import Path

def analyze_df40_structure():
    """分析DF40数据集结构和图像信息"""
    base_path = "D:/work/AWARE-NET/dataset/DF40"
    
    print("=== 分析 DF40 数据集结构 ===\n")
    
    # 检查根目录frames文件夹
    frames_dir = os.path.join(base_path, "frames")
    
    if os.path.exists(frames_dir):
        print(f"✓ 根目录frames文件夹存在: {frames_dir}")
        
        # 遍历子文件夹
        subfolders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        print(f"  找到 {len(subfolders)} 个子文件夹")
        
        # 分析前几个文件夹
        total_images = 0
        sample_files = []
        
        for i, subfolder in enumerate(subfolders[:5]):
            subfolder_path = os.path.join(frames_dir, subfolder)
            images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            total_images += len(images)
            
            print(f"  {subfolder}: {len(images)} 张PNG图像")
            
            if images:
                # 记录样本文件信息
                sample_path = os.path.join(subfolder_path, images[0])
                file_size = os.path.getsize(sample_path)
                sample_files.append({
                    'path': sample_path,
                    'size_kb': file_size / 1024,
                    'folder': subfolder,
                    'filename': images[0]
                })
        
        print(f"  总图像数（前5个文件夹）: {total_images}")
        
        # 估算总数
        avg_images_per_folder = total_images / min(5, len(subfolders)) if subfolders else 0
        estimated_total = int(avg_images_per_folder * len(subfolders))
        print(f"  估算根目录总图像数: {estimated_total}")
    
    # 检查method-specific文件夹
    method_folders = ['blendface', 'e4s', 'facedancer', 'faceswap', 'fsgan', 'inswap', 'mobileswap', 'simswap']
    method_stats = {}
    
    for method in method_folders:
        method_path = os.path.join(base_path, method)
        if os.path.exists(method_path):
            frames_path = os.path.join(method_path, "frames")
            landmarks_path = os.path.join(method_path, "landmarks")
            
            method_info = {
                'has_frames': os.path.exists(frames_path),
                'has_landmarks': os.path.exists(landmarks_path),
                'folders': 0,
                'sample_images': 0
            }
            
            if method_info['has_frames']:
                try:
                    subfolders = [f for f in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, f))]
                    method_info['folders'] = len(subfolders)
                    
                    # 检查第一个子文件夹的图像数量
                    if subfolders:
                        first_folder_path = os.path.join(frames_path, subfolders[0])
                        images = [f for f in os.listdir(first_folder_path) if f.endswith('.png')]
                        method_info['sample_images'] = len(images)
                        
                        # 记录样本文件
                        if images:
                            sample_path = os.path.join(first_folder_path, images[0])
                            file_size = os.path.getsize(sample_path)
                            sample_files.append({
                                'path': sample_path,
                                'size_kb': file_size / 1024,
                                'folder': f"{method}/{subfolders[0]}",
                                'filename': images[0]
                            })
                except Exception as e:
                    print(f"  错误读取 {method}: {e}")
            
            method_stats[method] = method_info
    
    # 打印method统计
    print(f"\n=== Method-specific 文件夹分析 ===")
    for method, info in method_stats.items():
        print(f"{method}:")
        print(f"  frames文件夹: {'✓' if info['has_frames'] else '✗'}")
        print(f"  landmarks文件夹: {'✓' if info['has_landmarks'] else '✗'}")
        if info['has_frames']:
            print(f"  子文件夹数量: {info['folders']}")
            print(f"  样本图像数: {info['sample_images']}")
            estimated_method_total = info['folders'] * info['sample_images']
            print(f"  估算总图像数: {estimated_method_total}")
    
    # 分析样本文件
    if sample_files:
        print(f"\n=== 样本文件分析 ===")
        
        file_sizes = [f['size_kb'] for f in sample_files]
        min_size = min(file_sizes)
        max_size = max(file_sizes)
        avg_size = sum(file_sizes) / len(file_sizes)
        
        print(f"文件大小范围: {min_size:.1f}KB - {max_size:.1f}KB (平均: {avg_size:.1f}KB)")
        
        print(f"\n样本文件详情:")
        for sample in sample_files[:10]:  # 只显示前10个样本
            print(f"  {sample['folder']}/{sample['filename']}: {sample['size_kb']:.1f}KB")
    
    # 尝试使用PIL读取一张图像获取尺寸
    print(f"\n=== 尝试获取图像尺寸 ===")
    if sample_files:
        try:
            from PIL import Image
            sample_path = sample_files[0]['path']
            with Image.open(sample_path) as img:
                width, height = img.size
                mode = img.mode
                format_info = img.format
                
                print(f"✓ 成功读取样本图像: {sample_path}")
                print(f"  尺寸: {width} x {height}")
                print(f"  模式: {mode}")
                print(f"  格式: {format_info}")
                
                # 检查更多样本确认一致性
                consistent_size = True
                for sample in sample_files[1:5]:  # 检查更多样本
                    try:
                        with Image.open(sample['path']) as img2:
                            if img2.size != (width, height):
                                consistent_size = False
                                print(f"  ⚠ 尺寸不一致: {sample['path']} = {img2.size}")
                    except Exception:
                        continue
                
                if consistent_size:
                    print(f"  ✓ 样本图像尺寸一致")
                
                print(f"\n=== 推荐的预处理配置 ===")
                print(f"image_size: [{width}, {height}]")
                print(f"格式: PNG")
                print(f"颜色模式: {mode}")
                
                return {
                    'width': width,
                    'height': height,
                    'format': 'PNG',
                    'mode': mode
                }
                
        except ImportError:
            print("PIL/Pillow 不可用，无法获取图像尺寸")
        except Exception as e:
            print(f"读取图像时出错: {e}")
    
    return None

if __name__ == "__main__":
    analyze_df40_structure()