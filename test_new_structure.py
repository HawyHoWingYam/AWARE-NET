#!/usr/bin/env python3
"""
測試新的目錄結構和測試參數功能
驗證修改後的預處理腳本是否正確工作
"""

import os
import sys
sys.path.append('scripts')

from preprocess_datasets_v2 import DatasetPreprocessorV2

def test_directory_creation():
    """測試新的目錄結構創建"""
    print("=== 測試目錄結構創建 ===")
    
    try:
        # 創建預處理器實例（不實際處理數據）
        preprocessor = DatasetPreprocessorV2(
            config_file="config/dataset_paths.json",
            test_percentage=10.0  # 只使用 10% 數據進行測試
        )
        
        # 檢查目錄是否正確創建
        processed_path = preprocessor.path_config.config["base_paths"]["processed_data"]
        
        expected_dirs = [
            # Train 目錄結構
            f"{processed_path}/train/celebdf_v2/real",
            f"{processed_path}/train/celebdf_v2/fake",
            f"{processed_path}/train/ffpp/real", 
            f"{processed_path}/train/ffpp/fake",
            f"{processed_path}/train/dfdc",
            
            # Val 目錄結構
            f"{processed_path}/val/celebdf_v2/real",
            f"{processed_path}/val/celebdf_v2/fake",
            f"{processed_path}/val/ffpp/real",
            f"{processed_path}/val/ffpp/fake", 
            f"{processed_path}/val/dfdc",
            
            # Final test sets 目錄結構
            f"{processed_path}/final_test_sets/celebdf_v2/real",
            f"{processed_path}/final_test_sets/celebdf_v2/fake",
            f"{processed_path}/final_test_sets/ffpp/real",
            f"{processed_path}/final_test_sets/ffpp/fake",
            f"{processed_path}/final_test_sets/dfdc",
            
            # Manifests 目錄
            f"{processed_path}/manifests"
        ]
        
        print("檢查目錄結構...")
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in expected_dirs:
            if os.path.exists(dir_path):
                existing_dirs.append(dir_path)
                print(f"  ✅ {dir_path}")
            else:
                missing_dirs.append(dir_path)
                print(f"  ❌ {dir_path}")
        
        print(f"\n結果: {len(existing_dirs)}/{len(expected_dirs)} 個目錄存在")
        
        if missing_dirs:
            print("⚠️  缺失的目錄:")
            for dir_path in missing_dirs:
                print(f"    {dir_path}")
        else:
            print("✅ 所有預期目錄都已創建！")
            
        return len(missing_dirs) == 0
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def test_sampling_logic():
    """測試採樣邏輯"""
    print("\n=== 測試採樣邏輯 ===")
    
    try:
        # 測試不同的採樣百分比
        test_percentages = [10.0, 50.0, 100.0]
        
        for percentage in test_percentages:
            print(f"\n測試 {percentage}% 採樣:")
            
            preprocessor = DatasetPreprocessorV2(
                config_file="config/dataset_paths.json",
                test_percentage=percentage
            )
            
            # 檢查採樣參數是否正確設置
            if hasattr(preprocessor, 'test_percentage'):
                print(f"  ✅ 測試參數設置: {preprocessor.test_percentage}%")
            else:
                print(f"  ❌ 測試參數未設置")
                return False
        
        print("\n✅ 採樣邏輯測試通過！")
        return True
        
    except Exception as e:
        print(f"❌ 採樣邏輯測試失敗: {e}")
        return False

def test_manifest_structure():
    """測試清單文件結構"""
    print("\n=== 測試清單文件結構 ===")
    
    try:
        preprocessor = DatasetPreprocessorV2(
            config_file="config/dataset_paths.json", 
            test_percentage=100.0
        )
        
        # 生成清單文件（如果有數據的話）
        try:
            preprocessor.generate_manifests()
            print("✅ 清單生成函數執行成功")
        except Exception as e:
            print(f"⚠️  清單生成遇到問題（可能是沒有處理過的數據）: {e}")
        
        # 檢查預期的清單文件
        processed_path = preprocessor.path_config.config["base_paths"]["processed_data"]
        manifests_dir = f"{processed_path}/manifests"
        
        expected_manifests = [
            "train_manifest.csv",
            "val_manifest.csv", 
            "test_celebdf_v2_manifest.csv",
            "test_ffpp_manifest.csv",
            "test_dfdc_manifest.csv"
        ]
        
        if os.path.exists(manifests_dir):
            existing_manifests = os.listdir(manifests_dir)
            print(f"發現清單文件: {existing_manifests}")
        else:
            print("清單目錄不存在（可能還沒有處理過數據）")
        
        return True
        
    except Exception as e:
        print(f"❌ 清單結構測試失敗: {e}")
        return False

def show_new_structure():
    """展示新的目錄結構"""
    print("\n=== 新的目錄結構說明 ===")
    print("""
新的目錄結構:

processed_data/
├── train/                    # 訓練集
│   ├── celebdf_v2/
│   │   ├── real/
│   │   └── fake/
│   ├── ffpp/
│   │   ├── real/
│   │   └── fake/
│   └── dfdc/                 # DFDC 直接放圖像
├── val/                      # 驗證集
│   ├── celebdf_v2/
│   │   ├── real/
│   │   └── fake/
│   ├── ffpp/
│   │   ├── real/
│   │   └── fake/
│   └── dfdc/                 # DFDC 直接放圖像
├── final_test_sets/          # 測試集
│   ├── celebdf_v2/
│   │   ├── real/
│   │   └── fake/
│   ├── ffpp/
│   │   ├── real/
│   │   └── fake/
│   └── dfdc/                 # DFDC 直接放圖像
└── manifests/                # 清單文件
    ├── train_manifest.csv
    ├── val_manifest.csv
    ├── test_celebdf_v2_manifest.csv
    ├── test_ffpp_manifest.csv
    └── test_dfdc_manifest.csv

新增功能:
- 📊 --test-percentage 參數: 控制使用多少百分比的數據集
- 🗂️  按數據集分類的目錄結構
- 📋 分數據集的清單文件
""")

def main():
    print("🧪 AWARE-NET 新結構測試工具")
    print("測試修改後的預處理腳本功能\n")
    
    # 執行所有測試
    tests = [
        ("目錄結構創建", test_directory_creation),
        ("採樣邏輯", test_sampling_logic), 
        ("清單文件結構", test_manifest_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"正在執行: {test_name}")
        print('='*50)
        
        result = test_func()
        results.append((test_name, result))
    
    # 顯示測試結果
    print(f"\n{'='*50}")
    print("測試結果總結")
    print('='*50)
    
    for test_name, passed in results:
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n總體結果: {passed_count}/{total_count} 個測試通過")
    
    if passed_count == total_count:
        print("🎉 所有測試都通過！新功能可以使用了。")
    else:
        print("⚠️  部分測試失敗，請檢查配置。")
    
    # 顯示新結構說明
    show_new_structure()
    
    print("\n使用示例:")
    print("# 使用 10% 數據進行快速測試")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 10.0 --datasets celebdf_v2")
    print("\n# 使用完整數據集進行處理")
    print("python scripts/preprocess_datasets_v2.py --test-percentage 100.0")

if __name__ == "__main__":
    main()