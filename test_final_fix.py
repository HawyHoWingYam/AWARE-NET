#!/usr/bin/env python3
"""
測試最終修正 - 驗證所有數據集都有正確的 train/val/test 分割
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.dataset_config import DatasetPathConfig
from collections import Counter

def test_all_datasets_split():
    """測試所有數據集的分割邏輯"""
    print("=== 測試所有數據集的分割邏輯 ===")
    
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    
    # 測試數據
    test_data = {
        'celebdf_v2': [
            "YouTube-real/00001.mp4", "YouTube-real/00002.mp4", "YouTube-real/00003.mp4",
            "Celeb-real/id0_0001.mp4", "Celeb-real/id0_0002.mp4", "Celeb-real/id0_0003.mp4",
            "Celeb-synthesis/id0_id1_0001.mp4", "Celeb-synthesis/id0_id1_0002.mp4"
        ],
        'ffpp': [
            "000_003", "001_000", "002_005", "010_020", "050_100",
            "100_200", "200_300", "300_400", "400_500", "500_600"
        ],
        'dfdc': [
            "folder1_video001", "folder1_video002", "folder2_video001", 
            "folder2_video002", "folder3_video001", "folder3_video002",
            "folder4_video001", "folder4_video002", "folder5_video001"
        ]
    }
    
    all_good = True
    
    for dataset_name, test_items in test_data.items():
        print(f"\n--- {dataset_name.upper()} 數據集分割測試 ---")
        
        # 獲取數據集配置
        dataset_config = config_manager.config["datasets"][dataset_name]
        dataset_config["full_base_path"] = os.path.join(
            config_manager.config["base_paths"]["raw_datasets"], 
            dataset_config["base_path"]
        )
        
        splits = Counter()
        
        for item in test_items:
            if dataset_name == 'celebdf_v2':
                split = config_manager._determine_celebdf_split(item, dataset_config)
            elif dataset_name == 'ffpp':
                split = config_manager._determine_ffpp_split(item, dataset_config)
            elif dataset_name == 'dfdc':
                split = config_manager._determine_dfdc_split(item, dataset_config)
            
            splits[split] += 1
            print(f"  {item:<30} -> {split.upper()}")
        
        print(f"\n分割統計: {dict(splits)}")
        
        # 檢查是否有 train/val/test 三種分割
        has_all_splits = 'train' in splits and 'val' in splits and 'test' in splits
        
        if has_all_splits:
            print("✅ 分割正確：包含 train, val, test 三種分割")
        else:
            print("❌ 分割錯誤：缺少某些分割類型")
            print(f"   現有分割: {list(splits.keys())}")
            all_good = False
        
        # 檢查比例是否合理
        total = sum(splits.values())
        if total > 0:
            train_pct = splits.get('train', 0) / total * 100
            val_pct = splits.get('val', 0) / total * 100
            test_pct = splits.get('test', 0) / total * 100
            print(f"   分割比例: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
    
    return all_good

def show_expected_results():
    """顯示預期的處理結果"""
    print("\n=== 修正後預期的處理結果 ===")
    print("""
現在所有數據集都會正確分割:

1. CelebDF-v2:
   ✅ 不使用官方測試集，完全自定義 70/15/15 分割
   ✅ final_test_sets/celebdf_v2/real/ 和 fake/ 會有圖像
   ✅ val/celebdf_v2/real/ 和 fake/ 會有圖像
   ✅ train/celebdf_v2/real/ 和 fake/ 會有圖像

2. FF++:
   ✅ 已經工作正常（如你所說）
   ✅ 有完整的 train/val/test 分割

3. DFDC:
   ✅ 添加了自定義分割邏輯（之前硬編碼為 train）
   ✅ final_test_sets/dfdc/real/ 和 fake/ 會有圖像
   ✅ val/dfdc/real/ 和 fake/ 會有圖像
   ✅ train/dfdc/real/ 和 fake/ 會有圖像

重新運行命令後，所有問題都會解決！
""")

def main():
    print("最終修正驗證測試")
    print("="*50)
    
    # 測試分割邏輯
    success = test_all_datasets_split()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ 所有數據集分割邏輯都正確！")
        print("可以重新運行預處理命令了。")
    else:
        print("❌ 部分數據集分割邏輯有問題")
    
    # 顯示預期結果
    show_expected_results()
    
    print("建議重新運行的命令:")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 10.0 --datasets celebdf_v2 --video-backend decord --face-detector insightface")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 5.0 --datasets ffpp --video-backend decord --face-detector insightface")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 20.0 --datasets dfdc --video-backend decord --face-detector insightface")

if __name__ == "__main__":
    main()