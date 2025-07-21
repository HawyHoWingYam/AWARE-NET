#!/usr/bin/env python3
"""
更新DF40 Manifest文件脚本
"""

import os
import pandas as pd
from pathlib import Path

def update_manifests():
    """更新manifest文件以包含DF40数据"""
    processed_data = Path("D:/work/AWARE-NET/processed_data")
    manifests_dir = processed_data / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    print("=== 更新Manifest文件 ===")
    
    splits = ["train", "val"]
    
    for split in splits:
        df40_fake_dir = processed_data / split / "df40" / "fake"
        df40_real_dir = processed_data / split / "df40" / "real"
        
        if not df40_fake_dir.exists():
            print(f"跳过不存在的目录: {df40_fake_dir}")
            continue
            
        manifest_file = manifests_dir / f"{split}_manifest.csv"
        
        # 读取现有manifest (如果存在)
        existing_data = []
        if manifest_file.exists():
            try:
                existing_df = pd.read_csv(manifest_file)
                if not existing_df.empty:
                    # 移除已存在的DF40数据
                    existing_data = existing_df[existing_df['dataset'] != 'df40'].to_dict('records')
                    print(f"从现有{split}_manifest.csv中移除旧的DF40数据")
                else:
                    print(f"{split}_manifest.csv为空文件")
            except pd.errors.EmptyDataError:
                print(f"{split}_manifest.csv为空文件，重新创建")
        
        # 添加新的DF40数据
        df40_data = []
        
        # 添加fake数据
        if df40_fake_dir.exists():
            for img_file in df40_fake_dir.glob("*"):
                if img_file.is_file():
                    df40_data.append({
                        "image_path": str(img_file.relative_to(processed_data)),
                        "label": "fake",
                        "dataset": "df40",
                        "split": split
                    })
        
        # 添加real数据（如果存在）
        if df40_real_dir.exists():
            for img_file in df40_real_dir.glob("*"):
                if img_file.is_file():
                    df40_data.append({
                        "image_path": str(img_file.relative_to(processed_data)),
                        "label": "real",
                        "dataset": "df40",
                        "split": split
                    })
        
        # 合并数据
        all_data = existing_data + df40_data
        
        # 保存manifest
        df = pd.DataFrame(all_data)
        df.to_csv(manifest_file, index=False)
        
        print(f"更新了 {manifest_file}: 添加了 {len(df40_data)} 条DF40记录 (总共{len(all_data)}条记录)")
    
    # 处理final_test_sets
    df40_final_fake_dir = processed_data / "final_test_sets" / "df40" / "fake"
    df40_final_real_dir = processed_data / "final_test_sets" / "df40" / "real"
    
    if df40_final_fake_dir.exists() or df40_final_real_dir.exists():
        manifest_file = manifests_dir / "final_test_manifest.csv"
        
        existing_data = []
        if manifest_file.exists():
            try:
                existing_df = pd.read_csv(manifest_file)
                if not existing_df.empty:
                    # 移除已存在的DF40数据
                    existing_data = existing_df[existing_df['dataset'] != 'df40'].to_dict('records')
                    print(f"从现有final_test_manifest.csv中移除旧的DF40数据")
                else:
                    print(f"final_test_manifest.csv为空文件")
            except pd.errors.EmptyDataError:
                print(f"final_test_manifest.csv为空文件，重新创建")
        
        df40_data = []
        
        # 添加fake数据
        if df40_final_fake_dir.exists():
            for img_file in df40_final_fake_dir.glob("*"):
                if img_file.is_file():
                    df40_data.append({
                        "image_path": str(img_file.relative_to(processed_data)),
                        "label": "fake", 
                        "dataset": "df40",
                        "split": "test"
                    })
        
        # 添加real数据（如果存在）
        if df40_final_real_dir.exists():
            for img_file in df40_final_real_dir.glob("*"):
                if img_file.is_file():
                    df40_data.append({
                        "image_path": str(img_file.relative_to(processed_data)),
                        "label": "real",
                        "dataset": "df40", 
                        "split": "test"
                    })
        
        all_data = existing_data + df40_data
        df = pd.DataFrame(all_data)
        df.to_csv(manifest_file, index=False)
        
        print(f"更新了 {manifest_file}: 添加了 {len(df40_data)} 条DF40记录 (总共{len(all_data)}条记录)")

def main():
    print("DF40 Manifest文件更新工具")
    print("="*50)
    
    update_manifests()
    
    print(f"\n{'='*50}")
    print("✅ DF40 Manifest文件更新完成!")
    print("\n建议下一步:")
    print("1. 检查manifest文件")
    print("2. 开始模型训练")

if __name__ == "__main__":
    main()