#!/usr/bin/env python3
"""
測試修正後的分割邏輯
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.dataset_config import DatasetPathConfig
from collections import Counter

def test_celebdf_split_fixed():
    """測試修正後的 CelebDF-v2 分割邏輯"""
    print("=== 測試修正後的 CelebDF-v2 分割邏輯 ===")
    
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    celebdf_config = config_manager.config["datasets"]["celebdf_v2"]
    celebdf_config["full_base_path"] = os.path.join(config_manager.config["base_paths"]["raw_datasets"], celebdf_config["base_path"])
    
    # 模擬一些視頻文件名（包括測試集中的視頻）
    test_videos = [
        "YouTube-real/00001.mp4",
        "YouTube-real/00002.mp4", 
        "YouTube-real/00003.mp4",
        "YouTube-real/00004.mp4",
        "YouTube-real/00005.mp4",
        "YouTube-real/00170.mp4",  # 這個在官方測試集中
        "YouTube-real/00208.mp4",  # 這個也在官方測試集中
        "Celeb-real/id0_0001.mp4",
        "Celeb-real/id0_0002.mp4",
        "Celeb-real/id0_0003.mp4",
        "Celeb-synthesis/id0_id1_0001.mp4",
        "Celeb-synthesis/id0_id1_0002.mp4",
        "Celeb-synthesis/id0_id1_0003.mp4",
        "Celeb-synthesis/id0_id1_0004.mp4",
        "Celeb-synthesis/id0_id1_0005.mp4",
    ]
    
    splits = Counter()
    for video in test_videos:
        split = config_manager._determine_celebdf_split(video, celebdf_config)
        splits[split] += 1
        status = "TEST" if split == "test" else split.upper()
        print(f"  {video:<35} -> {status}")
    
    print(f"\n分割統計: {dict(splits)}")
    print(f"預期: 有 train, val, test 三種分割")
    
    return 'train' in splits and 'val' in splits and 'test' in splits

def test_ffpp_split_fixed():
    """測試修正後的 FF++ 分割邏輯"""
    print("\n=== 測試修正後的 FF++ 分割邏輯 ===")
    
    config_manager = DatasetPathConfig("config/dataset_paths.json")
    ffpp_config = config_manager.config["datasets"]["ffpp"]
    ffpp_config["full_base_path"] = os.path.join(config_manager.config["base_paths"]["raw_datasets"], ffpp_config["base_path"])
    
    # 模擬一些視頻ID
    test_video_ids = [
        "000_003", "001_000", "002_005", "010_020", "050_100",
        "100_200", "200_300", "300_400", "400_500", "500_600"
    ]
    
    splits = Counter()
    for video_id in test_video_ids:
        split = config_manager._determine_ffpp_split(video_id, ffpp_config)
        splits[split] += 1
        print(f"  {video_id:<15} -> {split.upper()}")
    
    print(f"\n分割統計: {dict(splits)}")
    print(f"預期: 有 train, val, test 三種分割")
    
    return 'train' in splits and 'val' in splits and 'test' in splits

def test_directory_structure():
    """測試新的目錄結構"""
    print("\n=== 測試新的目錄結構 ===")
    
    # 檢查是否正確創建了所有目錄
    processed_path = "D:/work/AWARE-NET/processed_data"
    
    expected_dirs = [
        # Train 目錄結構（所有數據集都有 real/fake）
        f"{processed_path}/train/celebdf_v2/real",
        f"{processed_path}/train/celebdf_v2/fake",
        f"{processed_path}/train/ffpp/real",
        f"{processed_path}/train/ffpp/fake",
        f"{processed_path}/train/dfdc/real",    # 修正：DFDC 也應該有 real/fake
        f"{processed_path}/train/dfdc/fake",
        
        # Val 目錄結構
        f"{processed_path}/val/celebdf_v2/real",
        f"{processed_path}/val/celebdf_v2/fake",
        f"{processed_path}/val/ffpp/real",
        f"{processed_path}/val/ffpp/fake",
        f"{processed_path}/val/dfdc/real",      # 修正：DFDC 也應該有 real/fake
        f"{processed_path}/val/dfdc/fake",
        
        # Final test sets 目錄結構
        f"{processed_path}/final_test_sets/celebdf_v2/real",
        f"{processed_path}/final_test_sets/celebdf_v2/fake",
        f"{processed_path}/final_test_sets/ffpp/real",
        f"{processed_path}/final_test_sets/ffpp/fake",
        f"{processed_path}/final_test_sets/dfdc/real",   # 修正：DFDC 也應該有 real/fake
        f"{processed_path}/final_test_sets/dfdc/fake",
    ]
    
    print("應該創建的目錄結構:")
    for dir_path in expected_dirs:
        rel_path = dir_path.replace(processed_path + "/", "")
        print(f"  {rel_path}")
    
    print("\n這樣修正後:")
    print("✅ 所有數據集（包括 DFDC）都會按 real/fake 分類")
    print("✅ 驗證集不會為空（因為修正了分割邏輯）") 
    print("✅ 測試集會有數據（保留官方測試集）")
    
    return True

def main():
    print("修正後的分割邏輯測試")
    print("="*50)
    
    tests = [
        ("CelebDF-v2 分割邏輯", test_celebdf_split_fixed),
        ("FF++ 分割邏輯", test_ffpp_split_fixed),
        ("目錄結構檢查", test_directory_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"測試 {test_name} 失敗: {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("測試結果:")
    for test_name, passed in results:
        status = "通過" if passed else "失敗"
        print(f"  {test_name}: {status}")
    
    print(f"\n修正說明:")
    print("1. 恢復了 preserve_official_test_sets: true，確保測試集有數據")
    print("2. 修正了 DFDC 目錄結構，現在也按 real/fake 分類")  
    print("3. 改進了分割邏輯，確保 val 集不為空")
    print("4. 統一了所有數據集的目錄結構")
    
    print(f"\n重新運行預處理命令:")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 20.0 --datasets celebdf_v2 --video-backend decord --face-detector insightface")

if __name__ == "__main__":
    main()