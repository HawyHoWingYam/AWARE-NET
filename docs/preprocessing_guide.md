# 数据预处理指南 - preprocess_datasets.py

## 概述

`preprocess_datasets.py` 是一个功能强大的数据预处理脚本，专为 AWARE-NET 深度伪造检测项目设计。该脚本实现了从原始视频数据集到结构化人脸图像数据集的完整自动化处理流程。

## 核心功能

### 1. 支持的数据集
- **FaceForensics++ (FF++)**: 支持所有压缩级别 (c0, c23, c40) 和伪造方法
- **Celeb-DF-v2**: 自动解析真实和合成视频
- **DFDC**: 支持官方训练集格式
- **Deepfake-Eval-2024**: 作为独立测试集

### 2. 主要特性
- **智能人脸检测**: 使用 MTCNN 进行高精度人脸检测
- **自适应边界框扩展**: 可配置的边界框扩展比例
- **严格数据分离**: 确保训练/验证/测试集完全隔离
- **自动清单生成**: 生成便于模型使用的 CSV 清单文件
- **完整错误处理**: 详细的日志记录和异常处理

## 使用方法

### 基本用法

```bash
python preprocess_datasets.py \
    --raw_data_path /path/to/raw_datasets \
    --output_path /path/to/processed_data \
    --frame_interval 10 \
    --image_size 224 \
    --bbox_scale 1.3
```

### 参数说明

- `--raw_data_path`: 原始数据集根目录路径
- `--output_path`: 处理后数据的输出目录
- `--frame_interval`: 帧采样间隔（默认：10，即每10帧处理一次）
- `--image_size`: 输出图像尺寸（默认：224x224）
- `--bbox_scale`: 人脸边界框扩展因子（默认：1.3）

### 输入数据结构要求

脚本期望以下输入目录结构：

```
/raw_datasets/
├── FF++/
│   ├── original/c0/videos/
│   ├── Deepfakes/c0/videos/
│   ├── Face2Face/c0/videos/
│   ├── FaceSwap/c0/videos/
│   ├── NeuralTextures/c0/videos/
│   └── splits/train.json
├── Celeb-DF-v2/
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   └── List_of_testing_videos.txt
├── DFDC/
│   ├── dfdc_train_part_0/
│   │   ├── metadata.json
│   │   └── *.mp4
│   └── dfdc_train_part_1/
└── Deepfake-Eval-2024/
```

### 输出数据结构

处理后将生成以下目录结构：

```
/processed_data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
├── final_test_sets/
│   ├── celebdf_v2/
│   │   ├── real/
│   │   └── fake/
│   ├── ffpp/
│   │   ├── real/
│   │   └── fake/
│   └── deepfake_eval_2024/
├── manifests/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── test_celebdf_v2_manifest.csv
│   └── test_ffpp_manifest.csv
└── preprocessing.log
```

## 配置选项

### PreprocessConfig 类配置

脚本提供了丰富的配置选项，可在 `PreprocessConfig` 类中修改：

```python
class PreprocessConfig:
    # 路径配置
    RAW_DATA_PATH = "/path/to/raw_datasets"
    PROCESSED_DATA_PATH = "/path/to/processed_data"
    
    # 人脸检测参数
    FACE_DETECTOR_SETTINGS = {
        'min_face_size': 20,
        'thresholds': [0.6, 0.7, 0.7],
        'factor': 0.709,
        'post_process': True
    }
    
    # 处理参数
    FRAME_INTERVAL = 10          # 帧采样间隔
    IMAGE_SIZE = (224, 224)      # 输出图像尺寸
    BBOX_SCALE = 1.3             # 边界框扩展比例
    MAX_FACES_PER_VIDEO = 50     # 每视频最大人脸数
    MIN_FACE_SIZE_PIXELS = 80    # 最小人脸尺寸
```

## 处理流程详解

### 1. 视频处理 (`process_video`)
- 使用 OpenCV 逐帧读取视频
- 按设定间隔进行人脸检测
- 应用边界框扩展策略
- 保存标准化尺寸的人脸图像

### 2. 数据集解析
- **FF++**: 解析官方划分文件，支持多种压缩级别
- **Celeb-DF-v2**: 基于官方测试列表进行数据分离
- **DFDC**: 读取 metadata.json 获取标签信息

### 3. 清单文件生成
每个清单文件包含以下列：
- `filepath`: 图像相对路径
- `label`: 数值标签 (0=real, 1=fake)
- `label_name`: 文本标签

## 性能优化建议

### 1. 硬件配置
- **GPU**: 启用 CUDA 加速人脸检测
- **存储**: 使用 SSD 提高 I/O 性能
- **内存**: 建议 16GB+ RAM

### 2. 处理参数调优
- **Frame Interval**: 增大间隔可减少处理时间，但可能降低数据多样性
- **Image Size**: 较大尺寸保留更多细节，但增加存储需求
- **Max Faces**: 限制每视频人脸数量以控制数据集大小

### 3. 批量处理
对于大规模数据集，可以：
- 分批处理不同数据集
- 使用多进程并行处理
- 监控磁盘空间使用

## 故障排除

### 常见问题

1. **CUDA 不可用**
   - 检查 GPU 驱动和 CUDA 安装
   - 脚本会自动回退到 CPU 模式

2. **内存不足**
   - 减少 `MAX_FACES_PER_VIDEO`
   - 增大 `FRAME_INTERVAL`

3. **视频格式不支持**
   - 确保视频文件格式在 `SUPPORTED_VIDEO_FORMATS` 中
   - 使用 FFmpeg 转换不支持的格式

4. **人脸检测失败**
   - 调整 `FACE_DETECTOR_SETTINGS` 中的阈值
   - 检查视频质量和人脸清晰度

### 日志分析

脚本生成详细的日志文件 `preprocessing.log`，包含：
- 处理进度统计
- 错误和警告信息
- 性能指标

## 扩展性

### 添加新数据集

要支持新的数据集格式，需要：

1. 在 `DATASET_CONFIGS` 中添加配置
2. 实现对应的 `parse_xxx_dataset()` 方法
3. 在 `run_preprocessing()` 中调用新的解析方法

### 自定义人脸检测器

可以替换 MTCNN 为其他检测器：

1. 修改 `setup_face_detector()` 方法
2. 更新 `_extract_faces_from_frame()` 中的检测逻辑

## 质量保证

脚本包含多层质量检查：
- 人脸尺寸验证
- 文件存在性检查
- 数据完整性验证
- 异常处理和恢复

处理完成后，建议验证：
- 清单文件的完整性
- 图像质量和数量
- 数据集划分的正确性