# -*- coding: utf-8 -*-
"""台风影响分析的重要性模式分发模块。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from analysis_pipeline import AnalysisConfig, AnalysisContext

MapByVar = Dict[str, Any]
CompareMaps = Dict[str, MapByVar]
ImportanceFn = Callable[[AnalysisContext, AnalysisConfig], MapByVar]


def run_importance_methods(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    *,
    compute_perturbation: ImportanceFn,
    compute_gradient: ImportanceFn,
    compute_erf: ImportanceFn,
) -> Tuple[MapByVar, Optional[CompareMaps]]:
    """根据配置的重要性模式进行分发并返回用于报告的图谱。

    返回：
        importance_maps: 下游单模式逻辑使用的主要图谱输出。
        compare_maps: 仅在 IMPORTANCE_MODE='compare' 时填充的完整多方法图谱。
    """
    compare_maps: Optional[CompareMaps] = None

    if runtime_cfg.importance_mode == "perturbation":
        importance_maps = compute_perturbation(context, runtime_cfg)
    elif runtime_cfg.importance_mode == "input_gradient":
        importance_maps = compute_gradient(context, runtime_cfg)
    elif runtime_cfg.importance_mode == "erf":
        importance_maps = compute_erf(context, runtime_cfg)
    elif runtime_cfg.importance_mode == "compare":
        # 比较模式要求恰好一个目标变量，使每个面板可直接对比。
        if len(context.target_vars) != 1:
            raise ValueError("IMPORTANCE_MODE='compare' requires exactly one target variable")

        # 运行所有方法并保留带键的映射包，用于绘图/Top-N 报告。
        perturb_maps = compute_perturbation(context, runtime_cfg)
        gradient_maps = compute_gradient(context, runtime_cfg)
        erf_maps = compute_erf(context, runtime_cfg)
        compare_maps = {
            "perturbation": perturb_maps,
            "input_gradient": gradient_maps,
            "erf": erf_maps,
        }
        importance_maps = perturb_maps
    else:
        raise ValueError(f"Unknown IMPORTANCE_MODE: {runtime_cfg.importance_mode}")

    return importance_maps, compare_maps
