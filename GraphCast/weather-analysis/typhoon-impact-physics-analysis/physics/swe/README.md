# SWE 目录说明

本目录包含当前 SWE 物理分析主线及其配套模块。

核心模块：
- `comparison_core.py`：SWE 与 IG 对齐流程主编排
- `swe_model.py`：可微分 SWE 前向模型
- `swe_sensitivity.py`：基于 JAX 梯度的敏感度计算
- `alignment.py`：Spearman 与 Top-K IoU 对齐指标
- `steering.py`：深层环境引导气流（DLMSF）辅助计算
