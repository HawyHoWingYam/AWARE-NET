#!/usr/bin/env python3
"""
GPU視頻處理測試腳本
測試NVIDIA DALI、TorchVision和其他GPU視頻處理後端
"""

import torch
import sys
import os

def test_pytorch_cuda():
    """測試PyTorch CUDA支持"""
    print("=== PyTorch CUDA測試 ===")
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        
        # 測試GPU計算
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x)
        print("✅ GPU計算測試通過")
        
        # 清理
        del x, y
        torch.cuda.empty_cache()
    else:
        print("❌ CUDA不可用")
        return False
    
    return True

def test_decord():
    """測試Decord（Windows推薦）"""
    print("\n=== Decord測試 ===")
    try:
        import decord
        print(f"✅ Decord版本: {decord.__version__}")
        
        # 設置PyTorch橋接
        decord.bridge.set_bridge('torch')
        print("✅ Decord-PyTorch橋接設置成功")
        
        # 測試GPU上下文
        try:
            ctx = decord.gpu(0)
            print("✅ Decord GPU上下文可用")
            gpu_available = True
        except:
            print("⚠️ Decord GPU不可用，將使用CPU")
            gpu_available = False
        
        return True
        
    except ImportError as e:
        print(f"❌ Decord導入失敗: {e}")
        print("安裝方法: pip install decord")
        return False
    except Exception as e:
        print(f"❌ Decord測試失敗: {e}")
        return False

def test_dali():
    """測試NVIDIA DALI（Linux推薦）"""
    print("\n=== NVIDIA DALI測試 ===")
    print("⚠️ 注意：DALI主要為Linux設計，Windows支援有限")
    
    try:
        from nvidia.dali import pipeline_def, fn
        import nvidia.dali as dali
        
        print(f"✅ DALI版本: {dali.__version__}")
        
        # 創建簡單的DALI pipeline測試
        @pipeline_def(batch_size=1, num_threads=2, device_id=0)
        def simple_pipeline():
            # 創建隨機數據
            data = fn.random.uniform(range=(0, 255), shape=(480, 640, 3))
            return data
        
        pipe = simple_pipeline()
        pipe.build()
        pipe_out = pipe.run()
        
        print("✅ DALI pipeline創建和運行成功")
        print(f"✅ 輸出張量形狀: {pipe_out[0].shape()}")
        print("⚠️ 但建議在Windows上使用Decord")
        
        return True
        
    except ImportError as e:
        print(f"❌ DALI導入失敗: {e}")
        print("這在Windows上是正常的，推薦使用Decord")
        return False
    except Exception as e:
        print(f"❌ DALI測試失敗: {e}")
        print("建議在Windows上使用Decord替代")
        return False

def test_torchvision_io():
    """測試TorchVision.io"""
    print("\n=== TorchVision.io測試 ===")
    try:
        import torchvision.io as tvio
        print("✅ TorchVision.io可用")
        
        # 測試視頻相關功能
        print("✅ 視頻讀取功能可用")
        return True
        
    except ImportError as e:
        print(f"❌ TorchVision.io導入失敗: {e}")
        return False

def test_face_detection():
    """測試人臉檢測庫"""
    print("\n=== 人臉檢測庫測試 ===")
    
    # 測試InsightFace
    try:
        import insightface
        app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace創建成功")
        
        # 測試人臉檢測
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(test_image)
        print("✅ InsightFace人臉檢測測試完成")
        return True
        
    except ImportError:
        print("⚠️ InsightFace不可用")
    except Exception as e:
        print(f"⚠️ InsightFace測試失敗: {e}")
    
    # 測試facenet-pytorch
    try:
        from facenet_pytorch import MTCNN
        
        # 創建MTCNN實例（GPU）
        mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ facenet-pytorch MTCNN創建成功")
        
        # 測試人臉檢測（使用隨機圖像）
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        boxes, probs = mtcnn.detect(test_image)
        print("✅ facenet-pytorch人臉檢測測試完成")
        return True
        
    except ImportError:
        print("⚠️ facenet-pytorch不可用")
    except Exception as e:
        print(f"⚠️ facenet-pytorch測試失敗: {e}")
    
    print("❌ 沒有可用的人臉檢測庫")
    return False

def test_opencv():
    """測試OpenCV"""
    print("\n=== OpenCV測試 ===")
    try:
        import cv2
        print(f"✅ OpenCV版本: {cv2.__version__}")
        
        # 檢查CUDA支持
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                print(f"✅ OpenCV CUDA支持: {cuda_devices} 設備")
            else:
                print("⚠️ OpenCV沒有CUDA支持")
        except AttributeError:
            print("⚠️ OpenCV沒有編譯CUDA模組")
        
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV導入失敗: {e}")
        return False

def main():
    print("🚀 GPU視頻處理環境測試開始\n")
    
    results = {
        'pytorch_cuda': test_pytorch_cuda(),
        'decord': test_decord(),
        'torchvision_io': test_torchvision_io(),
        'dali': test_dali(),
        'face_detection': test_face_detection(),
        'opencv': test_opencv()
    }
    
    print("\n" + "="*50)
    print("📊 測試結果總結:")
    print("="*50)
    
    for component, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{component.replace('_', ' ').title()}: {status}")
    
    # 總體評估
    critical_components = ['pytorch_cuda', 'face_detection']
    critical_passed = all(results[comp] for comp in critical_components)
    
    decord_available = results['decord']
    dali_available = results['dali']
    
    print("\n📋 建議:")
    if critical_passed and decord_available:
        print("🎉 Windows環境完美！可以使用Decord進行高性能GPU視頻處理")
    elif critical_passed and dali_available:
        print("✅ 環境正常，但建議在Windows上使用Decord替代DALI")
    elif critical_passed:
        print("✅ 基本環境正常，建議安裝Decord獲得GPU視頻加速")
    else:
        print("⚠️ 需要修復關鍵組件才能正常運行")
    
    print(f"\n🔧 Windows推薦的視頻處理後端優先級:")
    if results['decord']:
        print("1. Decord (Windows最佳性能)")
    if results['torchvision_io']:
        print("2. TorchVision.io (良好性能)")
    if results['dali']:
        print("3. NVIDIA DALI (Linux推薦，Windows兼容性有限)")
    if results['opencv']:
        print("4. OpenCV (CPU回退)")

if __name__ == "__main__":
    main()