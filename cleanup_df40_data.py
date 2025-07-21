#!/usr/bin/env python3
"""
清理之前错误处理的DF40数据
"""

import shutil
from pathlib import Path
import pandas as pd

def cleanup_df40_data():
    """清理processed_data中的DF40数据和manifest文件中的DF40记录"""
    
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    manifests_dir = processed_data / "manifests"
    
    print("=== 清理错误的DF40数据 ===")
    
    # 1. 删除processed_data中的df40目录
    splits = ['train', 'val', 'final_test_sets']
    
    for split in splits:
        df40_dir = processed_data / split / 'df40'
        if df40_dir.exists():
            print(f"删除目录: {df40_dir}")
            shutil.rmtree(df40_dir)
        else:
            print(f"目录不存在: {df40_dir}")
    
    # 2. 清理manifest文件中的DF40记录
    manifest_files = [
        'train_manifest.csv',
        'val_manifest.csv', 
        'final_test_manifest.csv'
    ]
    
    print(f"\n=== 清理manifest文件中的DF40记录 ===")
    
    for manifest_file in manifest_files:
        manifest_path = manifests_dir / manifest_file
        if manifest_path.exists():
            try:
                # 读取现有manifest
                df = pd.read_csv(manifest_path)
                original_count = len(df)
                df40_count = len(df[df['dataset'] == 'df40'])
                
                # 过滤掉df40记录
                df_filtered = df[df['dataset'] != 'df40']
                new_count = len(df_filtered)
                
                # 保存过滤后的数据
                df_filtered.to_csv(manifest_path, index=False)
                
                print(f"{manifest_file}:")
                print(f"  原始记录: {original_count}")
                print(f"  DF40记录: {df40_count}")
                print(f"  清理后: {new_count}")
                
            except Exception as e:
                print(f"处理{manifest_file}时出错: {e}")
        else:
            print(f"manifest文件不存在: {manifest_file}")
    
    print(f"\n✅ DF40数据清理完成")

if __name__ == "__main__":
    cleanup_df40_data()