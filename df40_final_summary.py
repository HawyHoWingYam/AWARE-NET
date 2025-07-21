#!/usr/bin/env python3
"""
DF40数据重组最终总结
"""

import os
from pathlib import Path
import pandas as pd

def summarize_df40_organization():
    """总结DF40数据组织结果"""
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    manifests_dir = processed_data / "manifests"
    
    print("=" * 80)
    print("🎉 DF40数据重组完成 - 最终总结")
    print("=" * 80)
    
    # 统计文件数量
    print("\n📊 数据分布统计:")
    print("-" * 50)
    
    splits = ['train', 'val', 'final_test_sets']
    total_files = 0
    
    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")
        
        for label in ['real', 'fake']:
            df40_dir = processed_data / split / 'df40' / label
            if df40_dir.exists():
                # 统计PNG和JPG文件
                png_files = list(df40_dir.glob('*.png'))
                jpg_files = list(df40_dir.glob('*.jpg'))
                count = len(png_files) + len(jpg_files)
                split_total += count
                print(f"  {label:>4}: {count:>8,} 个文件")
        
        print(f"  {'总计':>4}: {split_total:>8,} 个文件")
        total_files += split_total
    
    print(f"\n{'DF40总计':>12}: {total_files:>8,} 个文件")
    
    # 检查manifest文件
    print(f"\n📋 Manifest文件统计:")
    print("-" * 50)
    
    manifest_files = ['train_manifest.csv', 'val_manifest.csv', 'final_test_manifest.csv']
    
    for manifest_file in manifest_files:
        manifest_path = manifests_dir / manifest_file
        if manifest_path.exists():
            try:
                df = pd.read_csv(manifest_path)
                df40_count = len(df[df['dataset'] == 'df40'])
                total_count = len(df)
                print(f"{manifest_file:>20}: {df40_count:>8,} DF40记录 / {total_count:>8,} 总记录")
            except Exception as e:
                print(f"{manifest_file:>20}: 读取错误 - {e}")
        else:
            print(f"{manifest_file:>20}: 文件不存在")
    
    # 显示数据来源分析
    print(f"\n🔍 数据来源分析:")
    print("-" * 50)
    print("训练数据来源:")
    print("• Entire Face Synthesis: 包含各种生成模型(DDIM, DiT, StyleGAN等)")
    print("• Face-reenactment: 包含各种重演技术(FOMM, Wav2lip等)")
    print("• Face-swapping: 包含各种换脸技术(E4S, SimSwap, Blendface等)")
    print("\n测试数据来源:")
    print("• 来自DF40官方测试集，基于JSON标签正确分类")
    
    print(f"\n📁 目录结构:")
    print("-" * 50)
    print("processed_data/")
    print("├── train/df40/")
    print("│   ├── real/    - 真实数据")
    print("│   └── fake/    - 合成数据")
    print("├── val/df40/")
    print("│   ├── real/    - 验证集真实数据")
    print("│   └── fake/    - 验证集合成数据")
    print("└── final_test_sets/df40/")
    print("    ├── real/    - 最终测试集真实数据")
    print("    └── fake/    - 最终测试集合成数据")
    
    print(f"\n✅ 数据重组策略:")
    print("-" * 50)
    print("Face-swapping数据特殊分配:")
    print("• 训练数据: 35% -> train/real, 35% -> train/fake")
    print("           7.5% -> val/real, 7.5% -> val/fake")
    print("           7.5% -> final_test_sets/real, 7.5% -> final_test_sets/fake")
    print("• 测试数据: 25%各分配到val和final_test_sets的real/fake中")
    print("\n其他数据保持原始标签分配")
    
    print(f"\n🚀 下一步建议:")
    print("-" * 50)
    print("1. 验证数据完整性")
    print("2. 开始AWARE-NET模型训练")
    print("3. 使用多个数据集进行cross-dataset评估")
    print("4. 对比不同方法的检测效果")

def main():
    summarize_df40_organization()

if __name__ == "__main__":
    main()