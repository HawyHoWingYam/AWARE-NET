#!/usr/bin/env python3
"""
测试DF40分配脚本的逻辑
"""

from pathlib import Path

def test_source_data():
    """测试数据源是否存在"""
    
    df40_base = Path("D:/work/AWARE-NET/dataset/DF40")
    
    print("=== 测试数据源 ===")
    
    # 1. 测试训练集数据源
    face_swapping_dir = df40_base / 'Face-swapping'
    print(f"训练集目录: {face_swapping_dir}")
    print(f"存在: {face_swapping_dir.exists()}")
    
    if face_swapping_dir.exists():
        methods = [d.name for d in face_swapping_dir.iterdir() if d.is_dir()]
        print(f"Face-swapping方法: {methods}")
        
        # 统计文件数量
        total_train = 0
        for method_dir in face_swapping_dir.iterdir():
            if method_dir.is_dir():
                frames_dir = method_dir / 'frames'
                if frames_dir.exists():
                    count = len(list(frames_dir.rglob('*.png')))
                    total_train += count
                    print(f"  {method_dir.name}: {count} 个文件")
        
        print(f"训练集总计: {total_train} 个文件")
    
    # 2. 测试测试集数据源 (修正：使用test/Face-swapping)
    test_fs_dir = df40_base / 'test' / 'Face-swapping'
    print(f"\\n测试集目录: {test_fs_dir}")
    print(f"存在: {test_fs_dir.exists()}")
    
    if test_fs_dir.exists():
        methods = [d.name for d in test_fs_dir.iterdir() if d.is_dir()]
        print(f"测试方法: {methods}")
        
        # 统计文件数量
        total_test = 0
        for method_dir in test_fs_dir.iterdir():
            if method_dir.is_dir():
                count = len(list(method_dir.rglob('*.png')))
                total_test += count
                print(f"  {method_dir.name}: {count} 个文件")
        
        print(f"测试集总计: {total_test} 个文件")
        print(f"测试集10%: {int(total_test * 0.1)} 个文件")

def test_allocation_logic():
    """测试分配逻辑"""
    
    print("\\n=== 测试分配逻辑 ===")
    
    # 模拟数据
    train_count = 100000  # 假设10万个训练文件
    test_count = 20000    # 假设2万个测试文件
    
    print(f"模拟训练集: {train_count:,} 个文件")
    print(f"模拟测试集: {test_count:,} 个文件")
    
    # 训练集分配 (70%/15%/15%)
    train_to_train = int(train_count * 0.70)
    train_to_val = int(train_count * 0.15)
    train_to_test = train_count - train_to_train - train_to_val
    
    print(f"\\n训练集分配:")
    print(f"  -> train: {train_to_train:,} ({train_to_train/train_count*100:.1f}%)")
    print(f"  -> val: {train_to_val:,} ({train_to_val/train_count*100:.1f}%)")
    print(f"  -> final_test_sets: {train_to_test:,} ({train_to_test/train_count*100:.1f}%)")
    
    # 测试集分配 (10%然后5%/5%)
    test_10_percent = int(test_count * 0.10)
    test_to_val = test_10_percent // 2
    test_to_test = test_10_percent - test_to_val
    
    print(f"\\n测试集分配:")
    print(f"  选择10%: {test_10_percent:,} 个文件")
    print(f"  -> val: {test_to_val:,}")
    print(f"  -> final_test_sets: {test_to_test:,}")
    
    # 最终统计
    print(f"\\n最终分布:")
    print(f"  train: {train_to_train:,}")
    print(f"  val: {train_to_val + test_to_val:,}")
    print(f"  final_test_sets: {train_to_test + test_to_test:,}")
    print(f"  总计: {train_to_train + train_to_val + train_to_test + test_to_val + test_to_test:,}")

if __name__ == "__main__":
    test_source_data()
    test_allocation_logic()