# GNN-Vis 学习项目

这是一个图神经网络 (Graph Neural Network) 的学习项目，包含多个不同主题的学习模块。

## 项目结构

```
gnn-vis/
├── GraphCast/                    # GraphCast 天气预报 GNN 学习
├── AdaptiveBlueNoiseSampling/    # 自适应蓝噪声采样学习
├── TIES/                         # TIES 图采样算法学习
├── data/                         # 公共数据集
├── utils/                        # 通用工具函数
├── load_dataset.py               # 数据集加载脚本
└── requirements.txt              # Python 依赖
```

## 学习模块

### 1. GraphCast - 天气预报图神经网络

学习 Google DeepMind 的 GraphCast 模型，用于中期天气预报。

```
GraphCast/
├── graphcast/                    # GraphCast 核心模块源码
├── graphcast-test/               # 测试和实验代码
├── graphcast-main/               # 主程序
├── graphcast-data/               # 气象数据
├── GraphCast-from-Ground-Zero/   # 从零开始的教程
└── gnn-observable/               # Observable 可视化前端
```

**学习要点**：
- 正二十面体 (Icosahedral) 网格结构
- 类型化图神经网络
- 自回归预测
- 气象数据处理 (ERA5)

### 2. AdaptiveBlueNoiseSampling - 蓝噪声采样

学习自适应蓝噪声采样算法，用于图数据的密度感知采样。

**学习要点**：
- 蓝噪声采样原理
- 核密度估计 (KDE)
- Node2Vec 图嵌入
- t-SNE 降维可视化

### 3. TIES - 图采样算法

学习 TIES (Totally Induced Edge Sampling) 图采样方法。

**学习要点**：
- 随机节点采样
- 诱导子图构建
- 采样可视化

## 数据集

项目使用的公共数据集：
- **Cora**: 论文引用网络数据集
- **Amazon**: 商品共购网络数据集 (Computers)

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.4.1
- PyTorch Geometric 2.6.1
- NetworkX 3.3
- NumPy / SciPy / Pandas
- Matplotlib

## 快速开始

```python
# 加载数据集
from load_dataset import load_dataset

G, data = load_dataset('Cora')  # 或 'Amazon'
print(f"节点数: {len(G.nodes())}, 边数: {len(G.edges())}")
```

## 参考资料

- [GraphCast Paper](https://arxiv.org/abs/2212.12794) - Learning skillful medium-range global weather forecasting
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
- [Observable Framework](https://observablehq.com/framework/) - 数据可视化框架
