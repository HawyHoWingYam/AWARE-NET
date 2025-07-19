# 数据集配置指南 - 灵活路径管理系统

## 概述

针对您显示的数据集目录结构，我们创建了一个灵活的配置系统来处理各种不同的数据集布局。这个系统通过JSON配置文件管理所有路径和处理参数，无需修改代码即可适应不同的目录结构。

## 您的数据集结构

根据您提供的截图，您的数据集结构如下：

```
dataset/
├── CelebDF-v2/
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   ├── YouTube-real/
│   └── List_of_testing_videos.txt
└── FF++/
    ├── manipulated_sequences/
    │   ├── DeepFakeDetection/c23/videos/
    │   ├── Deepfakes/c23/videos/
    │   ├── Face2Face/c23/videos/
    │   ├── FaceShifter/c23/videos/
    │   ├── FaceSwap/c23/videos/
    │   └── NeuralTextures/c23/videos/
    └── original_sequences/
        ├── actors/c23/videos/
        └── youtube/c23/videos/
```

## 快速开始

### 1. 自动配置设置

运行配置设置助手来创建适合您数据集结构的配置文件：

```bash
python setup_dataset_config.py
```

选择选项 2 "快速设置（从现有目录结构）"，脚本会自动检测您的数据集并生成配置。

### 2. 手动配置

或者直接编辑 `dataset_paths.json` 文件：

```json
{
  "base_paths": {
    "raw_datasets": "/Users/hawyho/Documents/GitHub/AWARE-NET/dataset",
    "processed_data": "/Users/hawyho/Documents/GitHub/AWARE-NET/processed_data"
  },
  "datasets": {
    "celebdf_v2": {
      "name": "CelebDF-v2",
      "base_path": "CelebDF-v2",
      "real_videos": ["Celeb-real", "YouTube-real"],
      "fake_videos": ["Celeb-synthesis"],
      "test_list_file": "List_of_testing_videos.txt",
      "supported_extensions": [".mp4", ".avi", ".mov"]
    },
    "ffpp": {
      "name": "FaceForensics++",
      "base_path": "FF++",
      "compressions": ["c23"],
      "real_videos": {
        "original_sequences": ["actors", "youtube"]
      },
      "fake_videos": {
        "manipulated_sequences": [
          "Deepfakes", "Face2Face", "FaceSwap", 
          "NeuralTextures", "DeepFakeDetection", "FaceShifter"
        ]
      }
    }
  },
  "processing": {
    "frame_interval": 10,
    "image_size": [224, 224],
    "bbox_scale": 1.3,
    "max_faces_per_video": 50
  }
}
```

### 3. 验证配置

验证配置是否正确：

```bash
python preprocess_datasets_v2.py --validate-only
```

### 4. 运行预处理

```bash
# 处理所有数据集
python preprocess_datasets_v2.py

# 只处理特定数据集
python preprocess_datasets_v2.py --datasets celebdf_v2 ffpp

# 查看配置摘要
python preprocess_datasets_v2.py --print-config
```

## 配置文件详解

### 基础路径配置

```json
"base_paths": {
  "raw_datasets": "/path/to/your/dataset",      // 原始数据集根目录
  "processed_data": "/path/to/processed_data"   // 处理后数据输出目录
}
```

### CelebDF-v2 配置

```json
"celebdf_v2": {
  "name": "CelebDF-v2",
  "base_path": "CelebDF-v2",                    // 相对于raw_datasets的路径
  "real_videos": ["Celeb-real", "YouTube-real"], // 真实视频目录列表
  "fake_videos": ["Celeb-synthesis"],           // 伪造视频目录列表
  "test_list_file": "List_of_testing_videos.txt", // 官方测试集列表文件
  "supported_extensions": [".mp4", ".avi", ".mov"]
}
```

### FaceForensics++ 配置

```json
"ffpp": {
  "name": "FaceForensics++",
  "base_path": "FF++",
  "compressions": ["c23"],                      // 支持的压缩级别
  "real_videos": {
    "original_sequences": ["actors", "youtube"] // 真实视频类别
  },
  "fake_videos": {
    "manipulated_sequences": [                  // 伪造方法列表
      "Deepfakes", "Face2Face", "FaceSwap",
      "NeuralTextures", "DeepFakeDetection", "FaceShifter"
    ]
  },
  "splits_file": "splits/train.json"           // 官方数据集划分文件
}
```

### 处理参数配置

```json
"processing": {
  "frame_interval": 10,          // 每隔多少帧提取一次
  "image_size": [224, 224],      // 输出图像尺寸
  "bbox_scale": 1.3,             // 人脸边界框扩展比例
  "max_faces_per_video": 50,     // 每视频最大人脸数量
  "min_face_size": 80,           // 最小人脸尺寸（像素）
  "face_detector": {
    "name": "mtcnn",
    "min_face_size": 20,
    "thresholds": [0.6, 0.7, 0.7],
    "factor": 0.709,
    "post_process": true
  }
}
```

## 文件结构说明

### 核心文件

- **`dataset_config.py`**: 配置管理核心类
- **`dataset_paths.json`**: 您的具体配置文件
- **`preprocess_datasets_v2.py`**: 使用配置系统的预处理脚本
- **`setup_dataset_config.py`**: 交互式配置设置助手

### 输出结构

处理后的数据将按照以下结构组织：

```
processed_data/
├── train/
│   ├── real/        # 来自各数据集训练部分的真实人脸
│   └── fake/        # 来自各数据集训练部分的伪造人脸
├── val/
│   ├── real/        # 验证集真实人脸
│   └── fake/        # 验证集伪造人脸
├── final_test_sets/
│   ├── celebdf_v2/
│   │   ├── real/    # CelebDF-v2官方测试集真实人脸
│   │   └── fake/    # CelebDF-v2官方测试集伪造人脸
│   ├── ffpp/
│   │   ├── real/    # FF++官方测试集真实人脸
│   │   └── fake/    # FF++官方测试集伪造人脸
│   └── dfdc/        # DFDC数据集（如果有）
├── manifests/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── test_celebdf_v2_manifest.csv
│   └── test_ffpp_manifest.csv
└── preprocessing_v2.log
```

## 使用命令

### 基本命令

```bash
# 1. 设置配置
python setup_dataset_config.py

# 2. 验证配置
python preprocess_datasets_v2.py --validate-only

# 3. 查看配置摘要
python preprocess_datasets_v2.py --print-config

# 4. 运行预处理
python preprocess_datasets_v2.py
```

### 高级用法

```bash
# 使用自定义配置文件
python preprocess_datasets_v2.py --config my_custom_config.json

# 只处理特定数据集
python preprocess_datasets_v2.py --datasets celebdf_v2

# 处理多个特定数据集
python preprocess_datasets_v2.py --datasets celebdf_v2 ffpp
```

## 配置系统的优势

### 1. 灵活性
- 支持任意数据集目录结构
- 无需修改代码即可适应新的数据集布局
- 易于添加新的数据集类型

### 2. 可维护性
- 所有路径配置集中管理
- JSON格式易于阅读和修改
- 配置验证和错误检查

### 3. 可扩展性
- 支持多种压缩级别（c0, c23, c40等）
- 支持新的伪造方法
- 灵活的处理参数配置

## 常见问题解决

### 1. 路径不存在
```bash
# 运行验证命令检查所有路径
python preprocess_datasets_v2.py --validate-only
```

### 2. 数据集结构不匹配
编辑 `dataset_paths.json` 中对应数据集的配置，确保目录名称和结构匹配您的实际情况。

### 3. 添加新的压缩级别
在FF++配置中添加新的压缩级别：
```json
"compressions": ["c0", "c23", "c40"]
```

### 4. 添加新的伪造方法
在FF++的fake_videos配置中添加：
```json
"manipulated_sequences": [
  "Deepfakes", "Face2Face", "FaceSwap",
  "NeuralTextures", "DeepFakeDetection", "FaceShifter",
  "YourNewMethod"
]
```

## 配置示例

### 完整的配置示例（适用于您的结构）

```json
{
  "base_paths": {
    "raw_datasets": "/Users/hawyho/Documents/GitHub/AWARE-NET/dataset",
    "processed_data": "/Users/hawyho/Documents/GitHub/AWARE-NET/processed_data"
  },
  "datasets": {
    "celebdf_v2": {
      "name": "CelebDF-v2",
      "base_path": "CelebDF-v2",
      "real_videos": ["Celeb-real", "YouTube-real"],
      "fake_videos": ["Celeb-synthesis"],
      "test_list_file": "List_of_testing_videos.txt",
      "supported_extensions": [".mp4", ".avi", ".mov"],
      "description": "Celebrity deepfake detection dataset"
    },
    "ffpp": {
      "name": "FaceForensics++",
      "base_path": "FF++",
      "compressions": ["c23"],
      "real_videos": {
        "original_sequences": ["actors", "youtube"]
      },
      "fake_videos": {
        "manipulated_sequences": [
          "Deepfakes", "Face2Face", "FaceSwap",
          "NeuralTextures", "DeepFakeDetection", "FaceShifter"
        ]
      },
      "splits_file": "splits/train.json",
      "supported_extensions": [".mp4", ".avi", ".mov"],
      "description": "Face manipulation detection dataset"
    }
  },
  "processing": {
    "frame_interval": 10,
    "image_size": [224, 224],
    "bbox_scale": 1.3,
    "max_faces_per_video": 50,
    "min_face_size": 80,
    "face_detector": {
      "name": "mtcnn",
      "min_face_size": 20,
      "thresholds": [0.6, 0.7, 0.7],
      "factor": 0.709,
      "post_process": true
    }
  },
  "output_structure": {
    "train_split_ratio": 0.7,
    "val_split_ratio": 0.15,
    "test_split_ratio": 0.15,
    "preserve_official_test_sets": true,
    "manifest_format": "csv"
  }
}
```

这个配置系统完全适配您当前的数据集结构，让您可以直接开始数据预处理，而无需重新组织目录或修改代码。