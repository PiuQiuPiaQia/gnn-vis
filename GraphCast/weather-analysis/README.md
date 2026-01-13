# 天气分析模块

## 主要分析脚本

| 文件 | 说明 |
|------|------|
| `cyclone_saliency_analysis.py` | **台风梯度显著性分析主脚本**，计算输入气象场对台风中心预测的敏感度，生成梯度热力图 |
| `cyclone_track_prediction.py` | 台风路径预测模块（被主脚本引用），从预测数据中提取台风中心位置并对比真实路径 |
| `sliding_window_saliency.py` | 滑动窗口梯度分析（被主脚本引用），使用连续真实观测数据计算每个时间窗口的梯度 |

### 主脚本使用方法
```bash
python cyclone_saliency_analysis.py
```

输出：
- 多张梯度热力图（显示哪些区域对台风预测最敏感）
- 台风路径对比图（AI预测 vs 真实观测）

---

## 辅助验证脚本

| 文件 | 说明 |
|------|------|
| `verify_cyclone_data.py` | 验证 ERA5 数据集中的台风位置是否与真实观测一致（使用区域搜索 ±10° 避免找到错误的低压系统） |
| `simple_track_comparison.py` | 简单台风路径对比，只绘制台风局部区域的预测路径 vs 真实路径（无复杂分析） |
| `region_utils.py` | 区域工具函数模块 |
| `check_levels.py` | 检查数据集中的气压层级配置 |
| `diagnose_layer_winds.py` | 诊断不同气压层的风场数据 |
| `test_steering_parameters.py` | 测试台风引导气流参数 |


