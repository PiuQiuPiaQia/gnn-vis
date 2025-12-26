# GraphCast 预测结果可视化

GraphCast 气象预测模型结果的可视化与分析工具。

## 📂 文件说明

```
graphcast-main/
├── main.ipynb        # 嵌入分析与聚类可视化
└── plt-graph.ipynb   # 预测结果绘图
```

### main.ipynb
- **t-SNE 降维**：将 471 维潜在表示降至 2 维
- **K-Means 聚类**：对 mesh 节点聚类并映射到地球坐标
- **预测可视化**：绘制全球温度分布

### plt-graph.ipynb
- **预测对比**：Target vs Prediction 差异分析
- **Mesh 节点叠加**：显示 GraphCast 图结构
- **交互式控件**：选择变量、层级、时间步等

## 🚀 使用

```bash
# 安装依赖
pip install jupyter xarray numpy matplotlib scikit-learn ipywidgets

# 运行
jupyter notebook
```

数据依赖 `../graphcast-data/` 目录下的预测文件。

## 📖 参考
- [GraphCast 论文](https://arxiv.org/abs/2212.12794)
