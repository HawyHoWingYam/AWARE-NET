# AWARE-NET

PyTorch implementation of our paper "**AWARE-NET: Adaptive Weighted Averaging for Robust Ensemble Network in Deepfake Detection**". [ICEPECC2025](https://digital-library.theiet.org/doi/abs/10.1049/icp.2025.1162), [IEEE](https://ieeexplore.ieee.org/abstract/document/10969682), [arxiv](https://arxiv.org/abs/2505.00312)

## Architecture Overview

1. **Tier 1**: Averages predictions within each architecture (Xception, Res2Net101, EfficientNet-B7) to reduce model variance.
2. **Tier 2**: Learns optimal weights for each architecture’s contribution through backpropagation, improving overall ensemble performance.

![image](https://github.com/user-attachments/assets/8bd64d5e-fab3-4e94-98f7-4e0fc44ed81c)
<p align="center">
  <img src="https://github.com/user-attachments/assets/e5622fa7-993e-4605-adf0-012a6bff854c" alt="Image Description"/>
</p>


## Configuration Options

* **Dataset Fraction**: Control the fraction of data used (default: 50%).
* **Train/Val/Test Split**: Default 70/15/15 split.
* **Annotation Management**: Options to force new splits or use cached annotations.

## Training Pipeline

1. **Train Individual Models**: Start by training each model (Xception, Res2Net101, EfficientNet-B7) with/without augmentation.
2. **Train Ensemble**: Fine-tune the ensemble with pre-trained individual models.
3. **Cross-Dataset Evaluation**: Test model generalization across datasets.

## Results

* **FF++**:

  * AUC: 99.22% (no aug.), 99.47% (aug.)
  * F1: 98.06% (no aug.), 98.43% (aug.)
* **CelebDF-v2**:

  * AUC: 100% (both)
  * F1: 99.94% (no aug.), 99.95% (aug.)

**Cross-Dataset**:

* **AUC**: 88.20% (FF++ → CelebDF-v2), 72.52% (CelebDF-v2 → FF++)
* **F1**: 93.16% (FF++ → CelebDF-v2), 80.62% (CelebDF-v2 → FF++)

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/recluzegeek/aware-net.git
   cd aware-net
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets (FF++ & CelebDF-v2), extract faces from videos and configure the respective paths in the `config.py`

## Usage

To start the training and evaluation:

```bash
python main.py
```

## Citation

```bibtex
@article{doi:10.1049/icp.2025.1162,
author = {Muhammad Salman  and Iqra Tariq  and Mishal Zulfiqar  and Muqadas Jalal  and Sami Aujla  and Sumbal Fatima },
title = {AWARE-NET: adaptive weighted averaging for robust ensemble network in deepfake detection},
journal = {IET Conference Proceedings},
volume = {2025},
issue = {3},
pages = {526-533},
year = {2025},
doi = {10.1049/icp.2025.1162},

URL = {https://digital-library.theiet.org/doi/abs/10.1049/icp.2025.1162},
eprint = {https://digital-library.theiet.org/doi/pdf/10.1049/icp.2025.1162},
abstract = { Deepfake detection has become increasingly important due to the rise of synthetic media, which poses significant risks to digital identity and cyber presence for security and trust. While multiple approaches have improved detection accuracy, challenges remain in achieving consistent performance across diverse datasets and manipulation types. In response, we propose a novel two-tier ensemble framework for deepfake detection based on deep learning that hierarchically combines multiple instances of three state-of-the-art architectures: Xception, Res2Net101, and EfficientNet-B7. Our framework employs a unique approach where each architecture is instantiated three times with different initializations to enhance model diversity, followed by a learnable weighting mechanism that dynamically combines their predictions.Unlike traditional fixed-weight ensembles, our first-tier averages predictions within each architecture family to reduce model variance, while the second tier learns optimal contribution weights through backpropagation, automatically adjusting each architecture's influence based on their detection reliability.Our experiments achieved state-of-the-art intra-dataset performance with AUC scores of 99.22\% (FF++) and 100.00\% (CelebDF-v2), and F1 scores of 98.06\% (FF++) and 99.94\% (CelebDF-v2) without augmentation. With augmentation, we achieve AUC scores of 99.47\% (FF++) and 100.00\% (CelebDF-v2), and F1 scores of 98.43\% (FF++) and 99.95\% (CelebDF-v2). The framework demonstrates robust cross-dataset generalization, achieving AUC scores of 88.20\% and 72.52\%, and F1 scores of 93.16\% and 80.62\% in cross-dataset evaluations. }
}
```
以下是AWARE-NET项目中各个Python文件的主要用途：

1. `config.py` - 配置文件，包含所有项目配置参数，如数据路径、模型配置、训练参数等
   
2. `cross_evaluation.py` - 用于在不同数据集间进行交叉评估的模块，测试模型的泛化能力

3. `dataset.py` - 定义数据集类和数据处理函数，包括DeepfakeDataset类和数据分割功能

4. `ensemble.py` - 实现集成模型的核心代码，包括不同模型的组合方式

5. `experiments.py` - 实验管理器，用于执行实验、评估模型和生成结果

6. `main.py` - 主入口文件，协调训练流程和实验执行

7. `model.py` - 模型定义文件，包括单模型检测器和集成检测器的实现

8. `train.py` - 训练过程管理器，实现模型训练循环、验证和检查点保存

9. `visualization.py` - 可视化工具，用于生成训练曲线、ROC曲线、混淆矩阵等可视化结果

整体来说，这个项目实现了论文中的AWARE-NET（自适应加权平均鲁棒集成网络）用于深度伪造检测，使用了三种主要架构的模型（Xception、Res2Net101和EfficientNet-B7）通过两级集成方式来提高检测性能。

ENV1:
conda activate retinaface
cd AWARE-NET
python preprocessing.py

ENV2:
python AWARE-NET/src/main.py

https://github.com/huggingface/pytorch-image-models/releases

res2net101_26w_4s-02a759a1.pth : https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-res2net/res2net101_26w_4s-02a759a1.pth

tf_efficientnet_b7_ns.pth : 
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth


faces
- celebdf
- ff++

videos
- CelebDF-v2
- FF++

src