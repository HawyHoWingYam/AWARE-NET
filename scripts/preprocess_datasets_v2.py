#!/usr/bin/env python3
"""
数据预处理脚本 V2 - preprocess_datasets_v2.py
使用灵活的配置系统处理多种数据集结构

功能:
1. 基于JSON配置文件的灵活路径管理
2. 支持任意数据集目录结构
3. 智能的数据集解析和处理
4. 自动生成数据清单文件
"""

import os
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
import concurrent.futures
import threading
import torch
warnings.filterwarnings('ignore')

# GPU視頻處理庫的可選導入
VIDEO_BACKENDS = {
    'opencv': True,  # 默認可用
    'torchvision': False,
    'decord': False,
    'dali': False
}

# 人臉檢測庫的可選導入
FACE_DETECTION_BACKENDS = {
    'facenet_pytorch': False,
    'insightface': False,
    'mtcnn': False
}

try:
    import torchvision.io as tvio
    VIDEO_BACKENDS['torchvision'] = True
    print("✅ TorchVision.io可用 - GPU視頻處理")
except ImportError:
    pass

try:
    import decord
    decord.bridge.set_bridge('torch')
    VIDEO_BACKENDS['decord'] = True  
    print("✅ Decord可用 - 高效GPU視頻處理（Windows推薦）")
except ImportError:
    pass

try:
    from nvidia.dali import pipeline_def, fn
    VIDEO_BACKENDS['dali'] = True
    print("✅ NVIDIA DALI可用 - 高性能GPU視頻處理")
except ImportError:
    pass

# 人臉檢測庫檢測
try:
    from facenet_pytorch import MTCNN
    FACE_DETECTION_BACKENDS['facenet_pytorch'] = True
    print("✅ facenet-pytorch可用")
except ImportError:
    print("⚠️ facenet-pytorch不可用（與PyTorch 2.7+不兼容）")

try:
    import insightface
    FACE_DETECTION_BACKENDS['insightface'] = True
    print("✅ InsightFace可用 - 高性能人臉檢測")
except ImportError:
    pass

try:
    import mtcnn
    FACE_DETECTION_BACKENDS['mtcnn'] = True
    print("✅ MTCNN可用")
except ImportError:
    pass

# 檢查OpenCV DNN人臉檢測
try:
    import cv2
    # 檢查是否可以載入DNN模型
    FACE_DETECTION_BACKENDS['opencv_dnn'] = True
    print("✅ OpenCV DNN人臉檢測可用（輕量級）")
except:
    pass

# 檢查YOLOv8人臉檢測
try:
    from ultralytics import YOLO
    FACE_DETECTION_BACKENDS['yolov8'] = True
    print("✅ YOLOv8人臉檢測可用（GPU加速）")
except ImportError:
    pass

# 檢查MediaPipe人臉檢測
try:
    import mediapipe as mp
    FACE_DETECTION_BACKENDS['mediapipe'] = True
    print("✅ MediaPipe人臉檢測可用（GPU加速）")
except ImportError:
    pass

# 檢查RetinaFace人臉檢測
try:
    from retinaface import RetinaFace
    FACE_DETECTION_BACKENDS['retinaface'] = True
    print("✅ RetinaFace人臉檢測可用（專業級GPU）")
except (ImportError, ValueError) as e:
    if "tf-keras" in str(e):
        print("⚠️ RetinaFace需要tf-keras: pip install tf-keras")
    else:
        print(f"⚠️ RetinaFace不可用: {e}")

# 导入配置管理器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.dataset_config import DatasetPathConfig

# =============================================================================
# 通用人臉檢測器類
# =============================================================================

class UniversalFaceDetector:
    """通用人臉檢測器，支援多種後端"""
    
    def __init__(self, backend='auto', device='cuda', **kwargs):
        self.device = device
        self.backend = self._select_backend(backend)
        self.detector = self._initialize_detector(**kwargs)
        print(f"🔍 使用人臉檢測後端: {self.backend}")
    
    def _select_backend(self, preferred):
        """選擇最佳的人臉檢測後端"""
        if preferred != 'auto' and FACE_DETECTION_BACKENDS.get(preferred, False):
            return preferred
        
        # 按性能和兼容性優先級選擇（人臉檢測專用優先）
        priority = ['insightface', 'mediapipe', 'facenet_pytorch', 'mtcnn', 'opencv_dnn', 'yolov8']
        for backend in priority:
            if FACE_DETECTION_BACKENDS.get(backend, False):
                return backend
        
        available_backends = [k for k, v in FACE_DETECTION_BACKENDS.items() if v]
        if not available_backends:
            error_msg = """
❌ 沒有可用的人臉檢測後端！

推薦安裝方案（按優先級）：
1. InsightFace（推薦，與PyTorch 2.7+完全兼容）：
   pip install insightface onnxruntime-gpu

2. MTCNN（備用選擇）：
   pip install mtcnn

3. facenet-pytorch（不推薦，與PyTorch 2.7+有衝突）：
   pip install facenet-pytorch --force-reinstall --no-deps
            """
            raise RuntimeError(error_msg)
        
        raise RuntimeError(f"人臉檢測後端 '{preferred}' 不可用。可用後端: {available_backends}")
    
    def _initialize_detector(self, **kwargs):
        """初始化檢測器"""
        if self.backend == 'yolov8':
            return self._init_yolov8(**kwargs)
        elif self.backend == 'mediapipe':
            return self._init_mediapipe(**kwargs)
        elif self.backend == 'opencv_dnn':
            return self._init_opencv_dnn(**kwargs)
        elif self.backend == 'insightface':
            return self._init_insightface(**kwargs)
        elif self.backend == 'facenet_pytorch':
            return self._init_facenet_pytorch(**kwargs)
        elif self.backend == 'mtcnn':
            return self._init_mtcnn(**kwargs)
        else:
            raise RuntimeError(f"不支援的人臉檢測後端: {self.backend}")
    
    def _init_insightface(self, **kwargs):
        """初始化InsightFace檢測器（僅檢測，不做特徵提取）"""
        import insightface
        
        # 只啟用檢測模型，禁用特徵提取以提升速度
        app = insightface.app.FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider'],
            allowed_modules=['detection']  # 只啟用檢測模塊
        )
        app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
        
        return app
    
    def _init_yolov8(self, **kwargs):
        """初始化YOLOv8檢測器（高性能GPU）"""
        from ultralytics import YOLO
        
        try:
            # 使用YOLOv8通用檢測模型（可以檢測人臉）
            model = YOLO('yolov8n.pt')  # 會自動下載通用檢測模型
            
            # 設置設備
            if self.device == 'cuda':
                model.to('cuda')
                print("✅ YOLOv8使用GPU")
            else:
                print("✅ YOLOv8使用CPU")
            
            return {
                'model': model,
                'confidence_threshold': kwargs.get('confidence_threshold', 0.5)
            }
        except Exception as e:
            print(f"YOLOv8初始化失敗: {e}")
            raise
    
    def _init_mediapipe(self, **kwargs):
        """初始化MediaPipe人臉檢測器（Google GPU優化）"""
        import mediapipe as mp
        
        # 初始化MediaPipe人臉檢測
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0: 2m內檢測, 1: 5m內檢測
            min_detection_confidence=kwargs.get('confidence_threshold', 0.5)
        )
        
        print("✅ MediaPipe人臉檢測初始化成功")
        
        return {
            'detector': face_detection,
            'mp_face_detection': mp_face_detection,
            'confidence_threshold': kwargs.get('confidence_threshold', 0.5)
        }
    
    def _init_opencv_dnn(self, **kwargs):
        """初始化OpenCV DNN人臉檢測器（輕量級，快速）"""
        import cv2
        import urllib.request
        import os
        
        # 模型文件路徑
        model_dir = "weights"
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # 如果模型不存在，下載它們
        if not os.path.exists(prototxt_path):
            print("下載OpenCV DNN人臉檢測模型...")
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        if not os.path.exists(model_path):
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            urllib.request.urlretrieve(model_url, model_path)
        
        # 載入模型
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        # 簡化設置：直接使用CPU（穩定且快速）
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✅ OpenCV DNN使用CPU（穩定模式）")
        
        return {
            'net': net,
            'confidence_threshold': kwargs.get('confidence_threshold', 0.5)
        }
    
    def _init_facenet_pytorch(self, **kwargs):
        """初始化facenet-pytorch檢測器"""
        try:
            from facenet_pytorch import MTCNN
            
            return MTCNN(
                min_face_size=kwargs.get("min_face_size", 20),
                thresholds=kwargs.get("thresholds", [0.6, 0.7, 0.7]),
                factor=kwargs.get("factor", 0.709),
                post_process=kwargs.get("post_process", True),
                device=self.device,
                select_largest=False,
                keep_all=True
            )
        except Exception as e:
            print(f"⚠️ facenet-pytorch初始化失敗: {e}")
            raise
    
    def _init_mtcnn(self, **kwargs):
        """初始化MTCNN檢測器"""
        import mtcnn
        
        return mtcnn.MTCNN(
            min_face_size=kwargs.get("min_face_size", 20),
            thresholds=kwargs.get("thresholds", [0.6, 0.7, 0.7]),
            factor=kwargs.get("factor", 0.709),
            device=self.device
        )
    
    def detect(self, image):
        """統一的人臉檢測介面"""
        if self.backend == 'yolov8':
            return self._detect_yolov8(image)
        elif self.backend == 'mediapipe':
            return self._detect_mediapipe(image)
        elif self.backend == 'opencv_dnn':
            return self._detect_opencv_dnn(image)
        elif self.backend == 'insightface':
            return self._detect_insightface(image)
        elif self.backend == 'facenet_pytorch':
            return self._detect_facenet_pytorch(image)
        elif self.backend == 'mtcnn':
            return self._detect_mtcnn(image)
    
    def _detect_yolov8(self, image):
        """YOLOv8檢測（高性能GPU）- 檢測人臉"""
        try:
            model = self.detector['model']
            confidence_threshold = self.detector['confidence_threshold']
            
            # YOLOv8推理
            results = model(image, conf=confidence_threshold, verbose=False)
            
            boxes = []
            confidences = []
            
            # 解析結果，過濾人臉類別 (class_id = 0 for person)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # 檢查類別是否為人 (class 0)
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id == 0:  # person class
                            # 獲取邊界框座標 [x1, y1, x2, y2]
                            coords = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # 將人體檢測框當作人臉區域（粗略近似）
                            # 取上半部分作為人臉區域
                            x1, y1, x2, y2 = coords.astype(int)
                            face_height = int((y2 - y1) * 0.3)  # 人臉約占人體上30%
                            face_y2 = y1 + face_height
                            
                            boxes.append([x1, y1, x2, face_y2])
                            confidences.append(float(confidence))
            
            return boxes, confidences
            
        except Exception as e:
            print(f"YOLOv8檢測錯誤: {e}")
            return None, None
    
    def _detect_mediapipe(self, image):
        """MediaPipe檢測（Google GPU優化）"""
        try:
            detector = self.detector['detector']
            mp_face_detection = self.detector['mp_face_detection']
            
            # 轉換BGR到RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 檢測人臉
            results = detector.process(rgb_image)
            
            boxes = []
            confidences = []
            
            if results.detections:
                h, w, _ = image.shape
                
                for detection in results.detections:
                    # 獲取邊界框
                    bbox = detection.location_data.relative_bounding_box
                    
                    # 轉換為絕對座標
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    # 獲取置信度
                    confidence = detection.score[0]
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
            
            return boxes, confidences
            
        except Exception as e:
            print(f"MediaPipe檢測錯誤: {e}")
            return None, None
    
    def _detect_opencv_dnn(self, image):
        """OpenCV DNN檢測（輕量級，快速）"""
        try:
            net = self.detector['net']
            confidence_threshold = self.detector['confidence_threshold']
            
            (h, w) = image.shape[:2]
            
            # 創建blob
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))
            
            # 設置輸入並執行前向傳播
            net.setInput(blob)
            detections = net.forward()
            
            boxes = []
            confidences = []
            
            # 處理檢測結果
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # 計算邊界框座標
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
            
            return boxes, confidences
            
        except Exception as e:
            print(f"OpenCV DNN檢測錯誤: {e}")
            return None, None
    
    def _detect_insightface(self, image):
        """InsightFace檢測"""
        try:
            faces = self.detector.get(image)
            
            if not faces:
                return None, None
            
            # 轉換為與MTCNN兼容的格式
            boxes = []
            confidences = []
            
            for face in faces:
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                confidence = face.det_score
                
                boxes.append(bbox)
                confidences.append(confidence)
            
            return boxes, confidences
            
        except Exception as e:
            print(f"InsightFace檢測錯誤: {e}")
            return None, None
    
    def _detect_facenet_pytorch(self, image):
        """facenet-pytorch檢測"""
        try:
            boxes, probs = self.detector.detect(image)
            return boxes, probs
        except Exception as e:
            print(f"facenet-pytorch檢測錯誤: {e}")
            return None, None
    
    def _detect_mtcnn(self, image):
        """MTCNN檢測"""
        try:
            result = self.detector.detect_faces(image)
            
            if not result:
                return None, None
            
            boxes = []
            confidences = []
            
            for face in result:
                bbox = face['box']  # [x, y, w, h]
                confidence = face['confidence']
                
                # 轉換為[x1, y1, x2, y2]格式
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
            
            return boxes, confidences
            
        except Exception as e:
            print(f"MTCNN檢測錯誤: {e}")
            return None, None

# =============================================================================
# GPU視頻處理器類
# =============================================================================

class GPUVideoProcessor:
    """GPU加速的視頻處理器"""
    
    def __init__(self, backend='auto', device='cuda'):
        self.device = device
        self.backend = self._select_backend(backend)
        logging.info(f"🎬 使用視頻處理後端: {self.backend}")
        
    def _select_backend(self, preferred):
        """選擇最佳的視頻處理後端"""
        if preferred != 'auto' and VIDEO_BACKENDS.get(preferred, False):
            return preferred
        
        # 按Windows性能優先級選擇（DALI在Windows上支援有限）
        priority = ['decord', 'torchvision', 'dali', 'opencv']
        for backend in priority:
            if VIDEO_BACKENDS.get(backend, False):
                return backend
        return 'opencv'  # 默認回退
    
    def read_video_frames(self, video_path: str, frame_interval: int = 10, max_frames: int = None):
        """GPU加速的視頻幀讀取"""
        if self.backend == 'torchvision':
            return self._read_with_torchvision(video_path, frame_interval, max_frames)
        elif self.backend == 'decord':
            return self._read_with_decord(video_path, frame_interval, max_frames)
        elif self.backend == 'dali':
            return self._read_with_dali(video_path, frame_interval, max_frames)
        else:
            return self._read_with_opencv(video_path, frame_interval, max_frames)
    
    def _read_with_torchvision(self, video_path, frame_interval, max_frames):
        """使用TorchVision.io進行GPU視頻處理"""
        try:
            # 讀取視頻元數據
            video_info = tvio.read_video_timestamps(video_path)
            total_frames = len(video_info[0])
            
            # 計算需要的幀索引
            frame_indices = list(range(0, total_frames, frame_interval))
            if max_frames:
                frame_indices = frame_indices[:max_frames]
            
            frames = []
            for frame_idx in frame_indices:
                # 讀取單幀 (GPU直接處理)
                frame_tensor, _, _ = tvio.read_video(
                    video_path, 
                    start_pts=frame_idx, 
                    end_pts=frame_idx + 1,
                    pts_unit='frame'
                )
                
                if frame_tensor.size(0) > 0:
                    # 轉換為numpy格式 (MTCNN需要)
                    frame = frame_tensor[0].permute(1, 2, 0).cpu().numpy()
                    frames.append((frame_idx, frame))
            
            logging.info(f"✅ TorchVision讀取 {len(frames)} 幀 (GPU加速)")
            return frames
            
        except Exception as e:
            logging.warning(f"TorchVision讀取失敗: {e}，回退到OpenCV")
            return self._read_with_opencv(video_path, frame_interval, max_frames)
    
    def _read_with_decord(self, video_path, frame_interval, max_frames):
        """使用Decord進行視頻處理（智能GPU/CPU選擇）"""
        try:
            # 先嘗試GPU，如果失敗則使用CPU
            try:
                vr = decord.VideoReader(video_path, ctx=decord.gpu(0))
                gpu_used = True
            except:
                vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
                gpu_used = False
                
            total_frames = len(vr)
            
            # 計算需要的幀索引
            frame_indices = list(range(0, total_frames, frame_interval))
            if max_frames:
                frame_indices = frame_indices[:max_frames]
            
            # 批量讀取幀
            frames_tensor = vr.get_batch(frame_indices)
            frames = []
            
            for i, frame_idx in enumerate(frame_indices):
                # 轉換為numpy - 處理不同的tensor格式
                try:
                    if hasattr(frames_tensor, '__getitem__'):
                        # 標準tensor索引
                        if gpu_used:
                            frame = frames_tensor[i].cpu().numpy()
                        else:
                            frame = frames_tensor[i].numpy()
                    else:
                        # 如果不支持索引，嘗試其他方法
                        frame = frames_tensor.asnumpy()[i] if hasattr(frames_tensor, 'asnumpy') else frames_tensor[i]
                    
                    # Decord返回RGB格式，需要轉換為BGR以保持一致性
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append((frame_idx, frame_bgr))
                    continue  # 跳過異常處理部分
                    
                except Exception as e:
                    logging.warning(f"Frame {i} conversion failed: {e}, trying alternative method")
                    # 備用方法：轉換整個batch然後索引
                    if gpu_used and hasattr(frames_tensor, 'cpu'):
                        numpy_frames = frames_tensor.cpu().numpy()
                    else:
                        numpy_frames = frames_tensor.numpy() if hasattr(frames_tensor, 'numpy') else frames_tensor.asnumpy()
                    frame = numpy_frames[i]
                
                # Decord返回RGB格式，需要轉換為BGR以保持一致性
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append((frame_idx, frame_bgr))
            
            gpu_status = "GPU" if gpu_used else "CPU"
            logging.info(f"✅ Decord讀取 {len(frames)} 幀 ({gpu_status})")
            return frames
            
        except Exception as e:
            logging.warning(f"Decord讀取失敗: {e}，回退到OpenCV")
            return self._read_with_opencv(video_path, frame_interval, max_frames)
    
    def _read_with_dali(self, video_path, frame_interval, max_frames):
        """使用NVIDIA DALI進行高性能GPU視頻處理"""
        try:
            # DALI pipeline定義
            @pipeline_def(batch_size=1, num_threads=2, device_id=0)
            def video_pipeline():
                video = fn.readers.video(
                    device="gpu",
                    file_root="",
                    filenames=[video_path],
                    sequence_length=max_frames or 100,
                    step=frame_interval,
                    normalized=False
                )
                return video
            
            pipe = video_pipeline()
            pipe.build()
            
            pipe_out = pipe.run()
            video_batch = pipe_out[0].as_cpu()  # 移到CPU進行人臉檢測
            
            frames = []
            for i in range(video_batch.shape[0]):
                frame = np.array(video_batch[i])
                frames.append((i * frame_interval, frame))
            
            logging.info(f"✅ DALI讀取 {len(frames)} 幀 (高性能GPU)")
            return frames
            
        except Exception as e:
            logging.warning(f"DALI讀取失敗: {e}，回退到OpenCV")
            return self._read_with_opencv(video_path, frame_interval, max_frames)
    
    def _read_with_opencv(self, video_path, frame_interval, max_frames):
        """OpenCV回退方案 (CPU)"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            frames = []
            frame_idx = 0
            count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frames.append((frame_idx, frame))
                    count += 1
                    
                    if max_frames and count >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            logging.info(f"⚠️ OpenCV讀取 {len(frames)} 幀 (CPU)")
            return frames
            
        except Exception as e:
            logging.error(f"OpenCV讀取失敗: {e}")
            return []

# =============================================================================
# 核心预处理类 V2
# =============================================================================

class DatasetPreprocessorV2:
    """数据集预处理器 V2 - 使用配置系统"""
    
    def __init__(self, config_file: Optional[str] = None, video_backend: str = 'auto', face_detector: str = 'auto'):
        """
        初始化预处理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认的dataset_paths.json
            video_backend: 視頻處理後端偏好
            face_detector: 人臉檢測器後端偏好
        """
        self.path_config = DatasetPathConfig(config_file)
        self.processing_config = self.path_config.get_processing_config()
        
        # 設置後端偏好
        self.processing_config['video_backend'] = video_backend
        if 'face_detector' not in self.processing_config:
            self.processing_config['face_detector'] = {}
        self.processing_config['face_detector']['backend'] = face_detector
        
        # 線程本地存儲，用於存儲每個線程的人臉檢測器實例
        self._thread_local = threading.local()
        
        self.setup_directories()
        self.setup_video_processor()
        self.setup_face_detector()
        self.setup_logging()
    
    def get_thread_local_face_detector(self):
        """獲取線程本地的人臉檢測器實例"""
        if not hasattr(self._thread_local, 'face_detector'):
            # 為每個線程創建獨立的人臉檢測器實例
            detector_config = self.processing_config.get("face_detector", {})
            preferred_detector = detector_config.get("backend", "auto")
            
            # 如果在處理配置中有指定，使用它
            if hasattr(self, 'processing_config') and 'face_detector' in self.processing_config:
                face_detector_config = self.processing_config.get('face_detector', {})
                if isinstance(face_detector_config, dict) and 'backend' in face_detector_config:
                    preferred_detector = face_detector_config['backend']
            
            device = 'cuda'
            
            logging.info(f"🔧 線程 {threading.current_thread().name} 創建人臉檢測器: {preferred_detector}")
            self._thread_local.face_detector = UniversalFaceDetector(
                backend=preferred_detector,
                device=device,
                min_face_size=detector_config.get("min_face_size", 20),
                thresholds=detector_config.get("thresholds", [0.6, 0.7, 0.7]),
                factor=detector_config.get("factor", 0.709),
                post_process=detector_config.get("post_process", True)
            )
            
        return self._thread_local.face_detector
    
    def set_workers(self, workers: int):
        """設置並行工作線程數"""
        self._workers = workers
        
    def setup_directories(self):
        """创建必要的目录结构"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        
        dirs_to_create = [
            f"{processed_path}/train/real",
            f"{processed_path}/train/fake", 
            f"{processed_path}/val/real",
            f"{processed_path}/val/fake",
            f"{processed_path}/final_test_sets/celebdf_v2/real",
            f"{processed_path}/final_test_sets/celebdf_v2/fake",
            f"{processed_path}/final_test_sets/ffpp/real",
            f"{processed_path}/final_test_sets/ffpp/fake",
            f"{processed_path}/final_test_sets/dfdc",
            f"{processed_path}/manifests"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logging.info(f"Created directory structure at {processed_path}")
    
    def setup_video_processor(self):
        """初始化GPU視頻處理器"""
        try:
            preferred_backend = self.processing_config.get("video_backend", "auto")
            self.video_processor = GPUVideoProcessor(backend=preferred_backend, device='cuda')
            
            # 打印可用的視頻處理後端
            available_backends = [k for k, v in VIDEO_BACKENDS.items() if v]
            logging.info(f"🎬 可用視頻後端: {', '.join(available_backends)}")
            
        except Exception as e:
            logging.error(f"視頻處理器初始化失敗: {e}")
            # 創建默認的OpenCV處理器
            self.video_processor = GPUVideoProcessor(backend='opencv', device='cuda')
    
    def setup_face_detector(self):
        """初始化通用人脸检测器 - 支持多種後端"""
        try:
            # 检查CUDA可用性
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用！此预处理脚本需要GPU支持。请安装支持CUDA的PyTorch。")
            
            # 检查OpenCV CUDA支持 (安全检查)
            try:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                self.opencv_cuda_available = cuda_devices > 0
                
                if self.opencv_cuda_available:
                    logging.info(f"✅ OpenCV CUDA支持可用: {cuda_devices} 个设备")
                    # 设置OpenCV使用GPU内存池
                    cv2.cuda.setGlDevice(0)
                else:
                    logging.warning("⚠️ OpenCV没有CUDA支持 - 视频读取和图像处理将使用CPU")
                    logging.warning("注意：主要的GPU加速(人脸检测)仍然可用")
            except AttributeError:
                # OpenCV没有编译CUDA支持
                self.opencv_cuda_available = False
                logging.warning("⚠️ OpenCV版本没有CUDA模块 - 使用CPU版本")
                logging.info("✅ 人脸检测仍将使用GPU加速")
            
            detector_config = self.processing_config.get("face_detector", {})
            preferred_detector = detector_config.get("backend", "auto")
            
            # 如果在處理配置中有指定，使用它
            if hasattr(self, 'processing_config') and 'face_detector' in self.processing_config:
                face_detector_config = self.processing_config.get('face_detector', {})
                if isinstance(face_detector_config, dict) and 'backend' in face_detector_config:
                    preferred_detector = face_detector_config['backend']
            
            # 强制使用CUDA设备
            device = 'cuda'
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            
            # 創建通用人臉檢測器
            self.face_detector = UniversalFaceDetector(
                backend=preferred_detector,
                device=device,
                min_face_size=detector_config.get("min_face_size", 20),
                thresholds=detector_config.get("thresholds", [0.6, 0.7, 0.7]),
                factor=detector_config.get("factor", 0.709),
                post_process=detector_config.get("post_process", True)
            )
            
            logging.info(f"✅ 人脸检测器已初始化 - 使用GPU: {gpu_name} (设备ID: {gpu_id})")
            
            # 打印GPU内存信息
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            logging.info(f"📊 GPU内存: {allocated_memory:.2f}GB / {total_memory:.2f}GB")
            
            # 預熱GPU
            self._warmup_gpu()
            
        except Exception as e:
            logging.error(f"❌ 人臉檢測器初始化失败: {e}")
            logging.error("可用解決方案:")
            logging.error("1. 安裝InsightFace: pip install insightface onnxruntime-gpu")
            logging.error("2. 或強制安裝facenet-pytorch: pip install facenet-pytorch --force-reinstall")
            logging.error("3. 或安裝MTCNN: pip install mtcnn")
            raise
    
    def _warmup_gpu(self):
        """预热GPU以获得更好的性能"""
        try:
            import torch
            logging.info("🔥 预热GPU...")
            
            # 创建一个dummy图像进行预热
            dummy_image = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
            dummy_image_np = dummy_image.numpy()
            
            # 预热人脸检测器
            self.face_detector.detect(dummy_image_np)
            
            logging.info("✅ GPU预热完成")
            
        except Exception as e:
            logging.warning(f"⚠️ GPU预热失败: {e}")
    
    def setup_logging(self):
        """设置日志记录"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        log_file = f"{processed_path}/preprocessing_v2.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def process_video(self, video_info: Dict) -> int:
        """
        处理单个视频文件，提取人脸图像
        
        Args:
            video_info: 包含视频路径、标签、划分等信息的字典
            
        Returns:
            提取的人脸数量
        """
        video_path = video_info["video_path"]
        label = video_info["label"]
        split = video_info["split"]
        dataset_name = video_info["dataset"]
        video_id = video_info["video_id"]
        
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found: {video_path}")
            return 0
        
        # 确定输出目录
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        
        if split == 'test' and dataset_name in ['celebdf_v2', 'ffpp']:
            output_dir = f"{processed_path}/final_test_sets/{dataset_name}/{label}"
        elif dataset_name == 'dfdc':
            output_dir = f"{processed_path}/final_test_sets/dfdc"
        else:
            output_dir = f"{processed_path}/{split}/{label}"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        face_count = 0
        frame_interval = self.processing_config.get("frame_interval", 10)
        max_faces = self.processing_config.get("max_faces_per_video", 50)
        
        try:
            # 使用GPU視頻處理器讀取幀
            max_video_frames = max_faces * 2  # 估算需要的視頻幀數
            frames = self.video_processor.read_video_frames(
                video_path, 
                frame_interval=frame_interval,
                max_frames=max_video_frames
            )
            
            if not frames:
                logging.error(f"無法從視頻讀取幀: {video_path}")
                return 0
            
            logging.info(f"📹 成功讀取 {len(frames)} 幀，開始人臉檢測")
            
            # 處理每一幀
            for frame_idx, frame in frames:
                if face_count >= max_faces:
                    break
                
                faces_extracted = self._extract_faces_from_frame(
                    frame, output_dir, video_id, frame_idx, face_count
                )
                face_count += faces_extracted
                
                # 定期清理GPU記憶體
                if frame_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {e}")
            return 0
        
        return face_count
    
    def _extract_faces_from_frame(self, frame: np.ndarray, output_dir: str, 
                                 video_id: str, frame_idx: int, face_count: int) -> int:
        """从单帧中提取人脸 - GPU加速版本"""
        try:
            # 使用GPU加速的图像预处理 (如果OpenCV支持CUDA)
            if self.opencv_cuda_available:
                # 上传到GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # GPU上进行颜色转换
                gpu_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                
                # 下载到CPU进行人脸检测 (MTCNN需要numpy array)
                rgb_frame = gpu_rgb.download()
            else:
                # CPU颜色转换
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # GPU人脸检测 - 使用線程本地檢測器
            thread_face_detector = self.get_thread_local_face_detector()
            boxes, _ = thread_face_detector.detect(rgb_frame)
            
            if boxes is None:
                return 0
            
            extracted_count = 0
            max_faces = self.processing_config.get("max_faces_per_video", 50)
            image_size = tuple(self.processing_config.get("image_size", [256, 256]))  # DF40 format
            bbox_scale = self.processing_config.get("bbox_scale", 1.3)
            min_face_size = self.processing_config.get("min_face_size", 80)
            
            # 批量处理人脸裁剪和缩放
            faces_to_save = []
            
            for i, box in enumerate(boxes):
                if face_count + extracted_count >= max_faces:
                    break
                
                # 處理不同檢測器的邊界框格式
                if isinstance(box, (list, tuple)):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                else:
                    x1, y1, x2, y2 = box.astype(int)
                
                # 计算扩展后的边界框
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                
                new_width = int(width * bbox_scale)
                new_height = int(height * bbox_scale)
                
                x1_new = max(0, center_x - new_width // 2)
                y1_new = max(0, center_y - new_height // 2)
                x2_new = min(frame.shape[1], center_x + new_width // 2)
                y2_new = min(frame.shape[0], center_y + new_height // 2)
                
                # 检查人脸尺寸
                if (x2_new - x1_new) < min_face_size or (y2_new - y1_new) < min_face_size:
                    continue
                
                # GPU加速的人脸裁剪和缩放 - 修復顏色通道問題
                if self.opencv_cuda_available:
                    # 在GPU上进行裁剪和缩放 (使用BGR格式的原始幀)
                    gpu_face_crop = gpu_frame[y1_new:y2_new, x1_new:x2_new]
                    gpu_face_resized = cv2.cuda.resize(gpu_face_crop, image_size)
                    face_resized = gpu_face_resized.download()  # BGR格式
                else:
                    # CPU处理 (使用BGR格式的原始幀)
                    face_crop = frame[y1_new:y2_new, x1_new:x2_new]  # BGR格式
                    face_resized = cv2.resize(face_crop, image_size)  # BGR格式
                
                # 准备文件名和路径
                filename = f"{video_id}_frame_{frame_idx:06d}_face_{i:02d}.png"
                output_path = os.path.join(output_dir, filename)
                
                faces_to_save.append((face_resized, output_path))
                extracted_count += 1
            
            # 批量保存 (减少I/O开销) - 使用異步I/O
            saved_count = 0
            for face_data, output_path in faces_to_save:
                try:
                    success = cv2.imwrite(output_path, face_data, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if success:
                        saved_count += 1
                    else:
                        logging.warning(f"Failed to save image: {output_path}")
                except Exception as e:
                    logging.warning(f"Error saving image {output_path}: {e}")
            
            # 返回實際保存的數量
            extracted_count = saved_count
            
            return extracted_count
            
        except Exception as e:
            logging.error(f"Error extracting faces from frame: {e}")
            return 0
    
    def process_dataset(self, dataset_name: str) -> Dict[str, int]:
        """处理指定数据集 - 多線程並行版本"""
        logging.info(f"Starting processing of {dataset_name} dataset...")
        
        # 跳过DF40，因为它已经是预处理的图像
        if dataset_name == "df40":
            logging.info(f"Skipping {dataset_name} - already preprocessed images")
            return {"processed": 0, "total_faces": 0, "errors": 0, "skipped": True}
        
        try:
            video_paths = self.path_config.get_all_video_paths(dataset_name)
            logging.info(f"Found {len(video_paths)} videos in {dataset_name}")
            
            # 計算最優線程數（基於CPU核心數，但限制以避免過度競爭GPU）
            import multiprocessing
            if hasattr(self, '_workers') and self._workers > 0:
                max_workers = min(self._workers, 4)  # 用戶指定，但仍限制最大值
            else:
                max_workers = min(multiprocessing.cpu_count(), 4)  # 限制為4個線程以平衡GPU使用
            logging.info(f"🚀 使用多線程並行處理: {max_workers} workers")
            
            stats = {"processed": 0, "total_faces": 0, "errors": 0}
            stats_lock = threading.Lock()
            
            def process_video_worker(video_info):
                """線程工作函數"""
                try:
                    face_count = self.process_video(video_info)
                    with stats_lock:
                        stats["total_faces"] += face_count
                        stats["processed"] += 1
                        
                        if stats["processed"] % 50 == 0:
                            # 監控GPU內存使用
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                                    cached_memory = torch.cuda.memory_reserved() / (1024**3)
                                    logging.info(f"Processed {stats['processed']} videos, extracted {stats['total_faces']} faces")
                                    logging.info(f"GPU Memory: {allocated_memory:.2f}GB allocated, {cached_memory:.2f}GB cached")
                            except Exception:
                                pass  # 忽略內存監控錯誤
                    
                    return face_count
                except Exception as e:
                    with stats_lock:
                        stats["errors"] += 1
                    logging.error(f"Error processing video {video_info.get('video_path', 'unknown')}: {e}")
                    return 0
            
            # 使用ThreadPoolExecutor進行並行處理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任務
                future_to_video = {executor.submit(process_video_worker, video_info): video_info for video_info in video_paths}
                
                # 使用tqdm顯示進度
                for future in tqdm(concurrent.futures.as_completed(future_to_video), 
                                 total=len(video_paths), desc=f"Processing {dataset_name}"):
                    video_info = future_to_video[future]
                    try:
                        face_count = future.result()
                    except Exception as exc:
                        logging.error(f"Video {video_info.get('video_path', 'unknown')} generated an exception: {exc}")
                        with stats_lock:
                            stats["errors"] += 1
            
            logging.info(f"Completed {dataset_name}: {stats['processed']} videos, {stats['total_faces']} faces, {stats['errors']} errors")
            return stats
            
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {e}")
            return {"processed": 0, "total_faces": 0, "errors": 1}
    
    def generate_manifests(self):
        """生成数据清单文件"""
        processed_path = self.path_config.config["base_paths"]["processed_data"]
        manifests_dir = f"{processed_path}/manifests"
        
        # 为每个数据集划分生成清单
        splits_to_process = [
            ('train', f"{processed_path}/train"),
            ('val', f"{processed_path}/val"),
            ('test_celebdf_v2', f"{processed_path}/final_test_sets/celebdf_v2"),
            ('test_ffpp', f"{processed_path}/final_test_sets/ffpp"),
            ('test_dfdc', f"{processed_path}/final_test_sets/dfdc")
        ]
        
        for split_name, split_path in splits_to_process:
            if not os.path.exists(split_path):
                continue
            
            manifest_data = []
            
            # 处理有real/fake子目录的情况
            if split_name.startswith('test_') and split_name != 'test_dfdc':
                for label in ['real', 'fake']:
                    label_path = os.path.join(split_path, label)
                    
                    if not os.path.exists(label_path):
                        continue
                    
                    for image_file in os.listdir(label_path):
                        if image_file.endswith('.png'):
                            relative_path = os.path.join(split_name.replace('test_', ''), label, image_file)
                            label_numeric = 0 if label == 'real' else 1
                            
                            manifest_data.append({
                                'filepath': relative_path,
                                'label': label_numeric,
                                'label_name': label,
                                'dataset': split_name.replace('test_', '')
                            })
            
            # 处理DFDC等混合目录的情况
            elif split_name == 'test_dfdc':
                for image_file in os.listdir(split_path):
                    if image_file.endswith('.png'):
                        # 从文件名推断标签 (需要在处理时保存这个信息)
                        relative_path = os.path.join('dfdc', image_file)
                        
                        manifest_data.append({
                            'filepath': relative_path,
                            'label': -1,  # 需要后续处理
                            'label_name': 'unknown',
                            'dataset': 'dfdc'
                        })
            
            # 处理train/val目录
            else:
                for label in ['real', 'fake']:
                    label_path = os.path.join(split_path, label)
                    
                    if not os.path.exists(label_path):
                        continue
                    
                    for image_file in os.listdir(label_path):
                        if image_file.endswith('.png'):
                            relative_path = os.path.join(split_name, label, image_file)
                            label_numeric = 0 if label == 'real' else 1
                            
                            manifest_data.append({
                                'filepath': relative_path,
                                'label': label_numeric,
                                'label_name': label,
                                'dataset': 'mixed'
                            })
            
            # 保存清单文件
            if manifest_data:
                manifest_df = pd.DataFrame(manifest_data)
                manifest_file = os.path.join(manifests_dir, f"{split_name}_manifest.csv")
                manifest_df.to_csv(manifest_file, index=False)
                logging.info(f"Generated manifest: {manifest_file} with {len(manifest_data)} samples")
    
    def run_full_preprocessing(self, datasets: Optional[List[str]] = None):
        """运行完整的数据预处理流程"""
        logging.info("Starting full dataset preprocessing...")
        logging.info(f"Target image format: {self.processing_config.get('image_size', [256, 256])} PNG (matching DF40 specifications)")
        
        # 验证配置路径
        validation_results = self.path_config.validate_paths()
        for path_name, exists in validation_results.items():
            status = "OK" if exists else "ERROR"
            logging.info(f"{status} {path_name}")
        
        # 确定要处理的数据集 (过滤掉DF40)
        if datasets is None:
            datasets = [name for name in self.path_config.config["datasets"].keys() if name != "df40"]
        else:
            datasets = [name for name in datasets if name != "df40"]
        
        logging.info(f"Processing datasets: {datasets} (DF40 skipped - already preprocessed)")
        
        total_stats = {"processed": 0, "total_faces": 0, "errors": 0}
        
        # 逐个处理数据集
        for dataset_name in datasets:
            if dataset_name in self.path_config.config["datasets"]:
                stats = self.process_dataset(dataset_name)
                for key in total_stats:
                    if key in stats:
                        total_stats[key] += stats[key]
            else:
                logging.warning(f"Dataset {dataset_name} not found in configuration")
        
        logging.info(f"All datasets processed: {total_stats['processed']} videos, {total_stats['total_faces']} faces, {total_stats['errors']} errors")
        
        # 生成清单文件
        self.generate_manifests()
        logging.info("All manifest files generated successfully")
        
        return total_stats

# =============================================================================
# 主程序入口
# =============================================================================

def check_gpu_requirements():
    """检查GPU要求"""
    print("=== GPU要求检查 ===")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ 错误：未检测到CUDA支持的PyTorch")
            print("请安装支持CUDA的PyTorch:")
            print("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            return False
        
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        
        print(f"✅ 检测到 {gpu_count} 个GPU设备")
        print(f"✅ 当前使用GPU: {gpu_name} (设备ID: {current_gpu})")
        
        # 检查GPU内存
        total_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
        print(f"✅ GPU内存: {total_memory:.2f}GB")
        
        if total_memory < 4.0:
            print("⚠️  警告：GPU内存可能不足（建议至少4GB）")
        
        # 检查OpenCV CUDA支持 (安全检查)
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                print(f"✅ OpenCV CUDA支持: {cuda_devices} 个设备 - 视频处理将使用GPU加速")
                # 测试OpenCV CUDA功能
                try:
                    test_mat = cv2.cuda_GpuMat()
                    print("✅ OpenCV GPU内存分配测试通过")
                except Exception as e:
                    print(f"⚠️  OpenCV CUDA初始化问题: {e}")
            else:
                print("⚠️  OpenCV没有CUDA支持 - 视频处理将使用CPU")
                print("   (人脸检测仍将使用GPU加速)")
        except AttributeError:
            print("⚠️  OpenCV版本没有CUDA模块 - 视频处理将使用CPU")
            print("   (这不影响主要的GPU加速功能：人脸检测)")
            print("   可選安裝CUDA版本: conda install opencv-cuda -c conda-forge")
        
        # 檢查視頻處理後端
        print("\n=== 視頻處理後端檢查 ===")
        available_backends = []
        for backend, available in VIDEO_BACKENDS.items():
            if available:
                available_backends.append(backend)
                if backend == 'opencv':
                    print(f"✅ {backend.upper()}: 可用 (CPU視頻處理)")
                else:
                    print(f"✅ {backend.upper()}: 可用 (GPU視頻處理)")
            else:
                print(f"❌ {backend.upper()}: 不可用")
        
        if len(available_backends) > 1:
            # Windows平台推薦順序
            windows_priority = ['decord', 'torchvision', 'dali']
            best_backend = next((b for b in windows_priority if b in available_backends), 'opencv')
            print(f"🚀 Windows推薦使用: {best_backend.upper()}")
            
            if 'dali' in available_backends:
                print("⚠️ 注意：DALI在Windows上可能有兼容性問題，建議使用Decord")
        
        print("\n✅ GPU检查通过 - 可以开始GPU加速预处理")
        print("")
        return True
        
    except ImportError:
        print("❌ 错误：PyTorch未安装")
        print("请安装PyTorch: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing V2 for Deepfake Detection - REQUIRES GPU")
    parser.add_argument("--config", default="config/dataset_paths.json", help="Path to configuration file")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to process (celebdf_v2, ffpp, dfdc)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate paths without processing")
    parser.add_argument("--print-config", action="store_true", help="Print configuration summary and exit")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU requirement check (NOT RECOMMENDED)")
    parser.add_argument("--video-backend", choices=['auto', 'torchvision', 'decord', 'dali', 'opencv'], 
                        default='auto', help="Choose video processing backend for GPU acceleration")
    parser.add_argument("--face-detector", choices=['auto', 'yolov8', 'mediapipe', 'insightface', 'opencv_dnn', 'facenet_pytorch', 'mtcnn'], 
                        default='auto', help="Choose face detection backend for GPU acceleration")
    parser.add_argument("--workers", type=int, default=0, 
                        help="Number of parallel workers (0=auto, max 4 to balance GPU usage)")
    
    args = parser.parse_args()
    
    print("=== AWARE-NET Dataset Preprocessing V2 (GPU Required) ===")
    print("Targeting DF40-compatible format: 256x256 PNG images")
    print("Processing: CelebDF-V2, FF++, DFDC (DF40 skipped - already preprocessed)")
    print("")
    
    # GPU要求检查
    if not args.skip_gpu_check:
        if not check_gpu_requirements():
            print("GPU检查失败！预处理需要GPU支持。")
            print("如果要强制跳过此检查，请使用 --skip-gpu-check 参数")
            return 1
    
    # 创建预处理器
    try:
        preprocessor = DatasetPreprocessorV2(
            config_file=args.config,
            video_backend=args.video_backend,
            face_detector=args.face_detector
        )
        preprocessor.set_workers(args.workers)
    except Exception as e:
        print(f"Error initializing preprocessor: {e}")
        return 1
    
    # 打印配置摘要
    if args.print_config:
        preprocessor.path_config.print_config_summary()
        return 0
    
    # 验证路径
    if args.validate_only:
        validation_results = preprocessor.path_config.validate_paths()
        print("Path validation results:")
        for path_name, exists in validation_results.items():
            status = "OK" if exists else "ERROR"
            print(f"  {status} {path_name}")
        return 0
    
    # 运行预处理
    try:
        stats = preprocessor.run_full_preprocessing(args.datasets)
        print(f"\n=== Preprocessing completed successfully! ===")
        print(f"Total videos processed: {stats['processed']}")
        print(f"Total faces extracted: {stats['total_faces']}")
        print(f"Errors encountered: {stats['errors']}")
        print(f"Output format: 256x256 PNG images (DF40-compatible)")
        print(f"Output directory: {preprocessor.path_config.config['base_paths']['processed_data']}")
        return 0
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())