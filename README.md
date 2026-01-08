# 基于 STN-CNN 的德国交通标志识别 (GTSRB)

> **课程名称**：人工智能 (Artificial Intelligence)  
> **项目性质**：期末课程项目 (Course Project)  
> **测试集准确率**：**98.23%** (Top-1 Accuracy)

---

## 1. 项目简介 (Introduction)

本项目是《人工智能》课程的期末大作业，旨在解决交通标志识别（Traffic Sign Recognition）中的几何形变难题。

本项目复现并改进了一个结合 **空间变换网络 (Spatial Transformer Networks, STN)** 与卷积神经网络 (CNN) 的深度学习模型。针对 GTSRB 数据集中的图像倾斜、旋转和缩放问题，STN 模块能够自动学习仿射变换参数并对图像进行空间矫正，从而显著提升分类准确率。

### 核心特性 (Key Features)

* **STN 创新架构**：引入 Localization Network 实现输入图像的自适应空间对齐。
* **高鲁棒性**：在独立测试集上达到了 **97.81%** 的准确率，优于传统 CNN 模型。
* **科学的训练策略**：集成了 **早停策略 (Early Stopping)** 防止过拟合，并使用 **TensorBoard** 记录训练全过程。
* **完全可复现**：通过固定随机种子 (Random Seed) 确保实验结果的可复现性。

---

## 2. 环境依赖 (Requirements)

本项目基于 Python 3.10+ 和 PyTorch 框架开发。

### 快速安装

建议使用 Conda 创建独立虚拟环境：

```bash
# 1. 创建并激活环境
conda create -n gtsrb_env python=3.8
conda activate gtsrb_env

# 2. 安装项目依赖
pip install -r requirements.txt
```

### 核心依赖库 (Dependencies)

* `torch >= 1.8.0`（推荐使用 CUDA 版本以加速训练）
* `torchvision >= 0.9.0`
* `pandas`（用于处理 CSV 标签文件）
* `scikit-learn`（用于计算混淆矩阵）
* `tensorboard`（用于可视化训练日志）
* `matplotlib`（用于绘图）

---

## 3. 项目结构 (Project Structure)

项目文件组织结构如下：

```text
GTSRB-Project/
├── data.py          # [核心] 自定义 Dataset 类，负责图像读取、预处理与增强
├── model.py         # [核心] 模型定义文件，包含 STN 模块与 CNN 主干网络
├── train.py         # [核心] 训练脚本，包含 Early Stopping 与 TensorBoard 记录
├── evaluate.py      # [评估] 评估脚本，生成准确率报告与混淆矩阵
├── predict.py       # [演示] 单张图片推理脚本，用于 Demo 展示
├── requirements.txt # 依赖库列表
├── README.md        # 项目说明文档
├── gtsrb-german-traffic-sign/  # 数据集目录 (Git 已忽略)
│   ├── Train/       # 训练集图片
│   ├── Test/        # 测试集图片
│   ├── Train.csv    # 训练标签
│   └── Test.csv     # 测试标签
├── model/           # 模型权重保存目录
│   └── best_stn_model.pth # 训练好的最佳模型权重
└── runs/            # TensorBoard 训练日志目录
```
