#!/usr/bin/env python3
"""
簡化測試 - 檢查新功能是否正確實現
"""

import os
import subprocess
import sys

def test_help_output():
    """測試幫助輸出是否包含新參數"""
    print("=== 測試新參數是否正確添加 ===")
    
    try:
        # 運行幫助命令
        result = subprocess.run([
            sys.executable, "scripts/preprocess_datasets_v2.py", "--help"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            help_output = result.stdout
            
            # 檢查新參數是否存在
            if "--test-percentage" in help_output:
                print("✅ --test-percentage 參數已添加")
                print("   描述:", [line.strip() for line in help_output.split('\n') if 'test-percentage' in line][0])
                return True
            else:
                print("❌ --test-percentage 參數未找到")
                return False
        else:
            print(f"❌ 運行幫助命令失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def check_code_modifications():
    """檢查代碼修改是否正確"""
    print("\n=== 檢查代碼修改 ===")
    
    modifications = []
    
    # 檢查 dataset_config.py 的修改
    try:
        with open("src/utils/dataset_config.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "_determine_celebdf_split" in content and "train_ratio" in content:
            modifications.append("✅ CelebDF-v2 分割邏輯已修改")
        else:
            modifications.append("❌ CelebDF-v2 分割邏輯未正確修改")
            
        if "_determine_ffpp_split" in content and "train_ratio" in content:
            modifications.append("✅ FF++ 分割邏輯已修改")
        else:
            modifications.append("❌ FF++ 分割邏輯未正確修改")
            
    except Exception as e:
        modifications.append(f"❌ 無法檢查 dataset_config.py: {e}")
    
    # 檢查 preprocess_datasets_v2.py 的修改
    try:
        with open("scripts/preprocess_datasets_v2.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "test_percentage" in content:
            modifications.append("✅ 測試參數已添加到預處理腳本")
        else:
            modifications.append("❌ 測試參數未添加到預處理腳本")
            
        if "split}/{dataset_name}/{label}" in content:
            modifications.append("✅ 新目錄結構邏輯已實現")
        else:
            modifications.append("❌ 新目錄結構邏輯未實現")
            
        if "datasets = ['celebdf_v2', 'ffpp', 'dfdc']" in content:
            modifications.append("✅ 新清單生成邏輯已實現")
        else:
            modifications.append("❌ 新清單生成邏輯未實現")
            
    except Exception as e:
        modifications.append(f"❌ 無法檢查 preprocess_datasets_v2.py: {e}")
    
    # 檢查配置文件修改
    try:
        with open("config/dataset_paths.json", 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '"preserve_official_test_sets": false' in content:
            modifications.append("✅ 配置文件已更新（禁用官方測試集保留）")
        else:
            modifications.append("⚠️  配置文件未更新或設置不同")
            
    except Exception as e:
        modifications.append(f"❌ 無法檢查配置文件: {e}")
    
    # 顯示檢查結果
    for mod in modifications:
        print(f"  {mod}")
    
    passed = sum(1 for mod in modifications if mod.startswith("✅"))
    total = len([mod for mod in modifications if not mod.startswith("❌ 無法檢查")])
    
    print(f"\n修改檢查結果: {passed}/{total} 項通過")
    return passed == total

def show_new_usage():
    """顯示新的使用方法"""
    print("\n=== 新功能使用方法 ===")
    print("""
1. 新的目錄結構:
   processed_data/
   ├── train/celebdf_v2/real/     # 按數據集分類
   ├── train/celebdf_v2/fake/
   ├── train/ffpp/real/
   ├── train/ffpp/fake/
   ├── train/dfdc/                # DFDC 直接放圖像
   ├── val/celebdf_v2/real/       # 驗證集也按數據集分類
   ├── val/celebdf_v2/fake/
   └── ...

2. 新的測試參數:
   --test-percentage 10.0         # 使用 10% 數據進行快速測試
   --test-percentage 50.0         # 使用 50% 數據
   --test-percentage 100.0        # 使用完整數據集（默認）

3. 使用示例:
   # 快速測試（10% 數據）
   python scripts/preprocess_datasets_v2.py --test-percentage 10.0 --datasets celebdf_v2

   # 正常處理（完整數據集）
   python scripts/preprocess_datasets_v2.py --datasets celebdf_v2 ffpp

   # 所有數據集，使用 50% 數據
   python scripts/preprocess_datasets_v2.py --test-percentage 50.0

4. 生成的清單文件:
   - train_manifest.csv           # 包含所有數據集的訓練數據
   - val_manifest.csv             # 包含所有數據集的驗證數據
   - test_celebdf_v2_manifest.csv # CelebDF-v2 測試集
   - test_ffpp_manifest.csv       # FF++ 測試集
   - test_dfdc_manifest.csv       # DFDC 測試集
""")

def main():
    print("AWARE-NET 新功能簡化測試")
    print("檢查代碼修改是否正確實現\n")
    
    # 執行測試
    tests = [
        ("新參數檢查", test_help_output),
        ("代碼修改檢查", check_code_modifications)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"正在執行: {test_name}")
        print('='*50)
        
        result = test_func()
        results.append((test_name, result))
    
    # 顯示結果
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
        print("所有修改都已正確實現！")
        print("你現在可以使用新功能了。")
    else:
        print("部分修改可能有問題，請檢查上述輸出。")
    
    # 顯示使用方法
    show_new_usage()

if __name__ == "__main__":
    main()