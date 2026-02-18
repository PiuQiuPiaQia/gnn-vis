#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""台风影响物理分析的入口模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import config as cfg

from analysis_pipeline import AnalysisConfig, build_analysis_context
from analysis_reporting import (
    build_importance_dataarrays,
    build_importance_map_specs,
    print_top_n,
    save_importance_plots_with_center,
)


ROOT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def run_typhoon_impact_analysis() -> Dict[str, Any]:
    """端到端地运行重要性分析并返回可复用的结果数据。"""
    # 局部导入：确保模块在无重依赖的环境中仍可导入。
    from importance_methods import (
        compute_erf_importance,
        compute_gradient_importance,
        compute_perturbation_importance,
    )
    from importance_mode_runner import run_importance_methods

    # 1) 构建运行时配置并初始化模型/数据上下文。
    runtime_cfg = AnalysisConfig.from_module(cfg)
    context = build_analysis_context(runtime_cfg)

    # 2) 根据所选模式计算重要性图。
    importance_maps, compare_maps = run_importance_methods(
        context,
        runtime_cfg,
        compute_perturbation=compute_perturbation_importance,
        compute_gradient=compute_gradient_importance,
        compute_erf=compute_erf_importance,
    )
    # `importance_maps` 为主要输出；`compare_maps` 仅在比较模式下填充。

    # 3) 将数值图转换为结构化输出，用于绘图/报告。
    lat_sel_vals = context.lat_vals[context.lat_indices]
    lon_sel_vals = context.lon_vals[context.lon_indices]

    map_specs = build_importance_map_specs(
        importance_mode=runtime_cfg.importance_mode,
        target_vars=context.target_vars,
        importance_maps=importance_maps,
        compare_maps=compare_maps,
    )
    importance_das = build_importance_dataarrays(
        map_specs=map_specs,
        lat_sel_vals=lat_sel_vals,
        lon_sel_vals=lon_sel_vals,
    )

    # 4) 保存图像并打印按排名的影响力网格点。
    save_importance_plots_with_center(
        runtime_cfg=runtime_cfg,
        root_dir=ROOT_DIR,
        target_vars=context.target_vars,
        importance_das=importance_das,
        center_lat=context.center_lat,
        center_lon=context.center_lon,
    )

    print_top_n(
        runtime_cfg=runtime_cfg,
        target_vars=context.target_vars,
        lat_sel_vals=lat_sel_vals,
        lon_sel_vals=lon_sel_vals,
        importance_maps=importance_maps,
        compare_maps=compare_maps,
    )

    # 5) 返回结果数据供 notebook/调试复用。
    return {
        "runtime_cfg": runtime_cfg,
        "context": context,
        "importance_maps": importance_maps,
        "compare_maps": compare_maps,
        "importance_das": importance_das,
    }


def main() -> int:
    run_typhoon_impact_analysis()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
