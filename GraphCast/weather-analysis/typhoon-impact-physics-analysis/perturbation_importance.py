# -*- coding: utf-8 -*-
"""台风分析的基于遮蔽的扰动重要性计算模块。"""

from __future__ import annotations

from typing import Dict

import jax
import numpy as np

from analysis_pipeline import AnalysisConfig, AnalysisContext, select_target_data
from impact_analysis_utils import (
    build_baseline_indexer,
    build_indexer,
    compute_baseline,
    resolve_level_sel,
)


def _match_shape(base_vals: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """将基线切片广播至扰动张量的形状。"""
    if base_vals.shape == target_shape:
        return base_vals
    if base_vals.ndim < len(target_shape):
        base_vals = base_vals.reshape(
            base_vals.shape + (1,) * (len(target_shape) - base_vals.ndim)
        )
    elif base_vals.ndim > len(target_shape):
        squeeze_axes = tuple(i for i, size in enumerate(base_vals.shape) if size == 1)
        if squeeze_axes:
            base_vals = np.squeeze(base_vals, axis=squeeze_axes)
        if base_vals.ndim < len(target_shape):
            base_vals = base_vals.reshape(
                base_vals.shape + (1,) * (len(target_shape) - base_vals.ndim)
            )
    return base_vals


def compute_perturbation_importance(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
) -> Dict[str, np.ndarray]:
    """通过局部替换和输出增量计算遮蔽式重要性。"""
    maps = {
        var: np.zeros((len(context.lat_indices), len(context.lon_indices)), dtype=np.float32)
        for var in context.target_vars
    }

    # 一次性构建基线数据集，然后在每个扫描格点上复用。
    baseline_ds = compute_baseline(
        context.eval_inputs,
        context.vars_to_perturb,
        runtime_cfg.baseline_mode,
        center_lat=context.center_lat,
        center_lon=context.center_lon,
        inner_deg=runtime_cfg.local_baseline_inner_deg,
        outer_deg=runtime_cfg.local_baseline_outer_deg,
        min_points=runtime_cfg.local_baseline_min_points,
    )
    if runtime_cfg.baseline_mode.startswith("local_annulus"):
        print(
            "Local baseline annulus: "
            f"inner={runtime_cfg.local_baseline_inner_deg:.2f}deg, "
            f"outer={runtime_cfg.local_baseline_outer_deg:.2f}deg, "
            f"min_points={runtime_cfg.local_baseline_min_points}"
        )

    time_sel = (
        slice(None) if runtime_cfg.perturb_time == "all" else int(runtime_cfg.perturb_time)
    )

    print("Scanning perturbations...")
    for i, lat_idx in enumerate(context.lat_indices):
        for j, lon_idx in enumerate(context.lon_indices):
            # 补丁定义了当前扫描位置处被基线替换的局部区域。
            lat_start = max(int(lat_idx) - runtime_cfg.patch_radius, 0)
            lat_end = min(int(lat_idx) + runtime_cfg.patch_radius + 1, len(context.lat_vals))
            lon_start = max(int(lon_idx) - runtime_cfg.patch_radius, 0)
            lon_end = min(int(lon_idx) + runtime_cfg.patch_radius + 1, len(context.lon_vals))
            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

            # 保存原始值，以便在一次扰动前向传播后恢复输入。
            saved_values = {}
            for var in context.vars_to_perturb:
                data_array = context.eval_inputs[var]
                level_sel = resolve_level_sel(data_array, runtime_cfg.perturb_levels)
                idx = build_indexer(data_array, lat_slice, lon_slice, time_sel, level_sel)
                base_idx = build_baseline_indexer(data_array, time_sel, level_sel)
                arr = data_array.values
                base_arr = baseline_ds[var].values
                saved_values[var] = (idx, arr[idx].copy())
                base_vals = base_arr[base_idx]
                base_vals = _match_shape(base_vals, arr[idx].shape)
                arr[idx] = np.broadcast_to(base_vals, arr[idx].shape)

            # 运行一次扰动前向传播并记录目标中心的差值。
            outputs = context.run_forward_jitted(
                rng=jax.random.PRNGKey(0),
                inputs=context.eval_inputs,
                targets_template=context.targets_template,
                forcings=context.eval_forcings,
            )
            for var in context.target_vars:
                out_var = select_target_data(
                    outputs,
                    var,
                    target_levels=runtime_cfg.target_levels,
                    target_level=runtime_cfg.target_level,
                )
                new_value = out_var.isel(time=runtime_cfg.target_time_idx).sel(
                    lat=context.center_lat,
                    lon=context.center_lon,
                    method="nearest",
                ).values.item()
                maps[var][i, j] = new_value - context.base_values[var]

            # 在进入下一个扫描点之前恢复原地编辑的输入数据。
            for var, (idx, old_vals) in saved_values.items():
                context.eval_inputs[var].values[idx] = old_vals

        if (i + 1) % 5 == 0:
            print(f"progress: {i + 1}/{len(context.lat_indices)} rows")

    return maps
