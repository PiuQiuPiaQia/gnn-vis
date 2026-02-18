# -*- coding: utf-8 -*-
"""台风分析的积分梯度重要性计算模块。"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import numpy as np
import xarray

from analysis_pipeline import AnalysisConfig, AnalysisContext
from importance_common import reduce_input_attribution_to_latlon, target_scalar


def compute_gradient_importance(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
) -> Dict[str, np.ndarray]:
    """在选定的经纬度区域上计算积分梯度（IG）图。"""
    maps = {
        var: np.zeros((len(context.lat_indices), len(context.lon_indices)), dtype=np.float32)
        for var in context.target_vars
    }

    if runtime_cfg.importance_mode == "compare":
        vars_to_grad = [runtime_cfg.target_variable]
    elif runtime_cfg.gradient_variables is None:
        vars_to_grad = context.vars_to_perturb
    else:
        vars_to_grad = [
            var_name
            for var_name in runtime_cfg.gradient_variables
            if var_name in context.eval_inputs.data_vars
        ]

    if not vars_to_grad:
        raise ValueError("No gradient variables found")

    ig_steps = runtime_cfg.gradient_steps

    print("\n=== Integrated Gradients Computation ===")
    print(f"IG variables: {vars_to_grad}")
    print(f"IG steps: {ig_steps}")
    print("Baseline: global median")

    # 构建每个变量的常量基线，用于 IG 路径积分。
    baseline_inputs = context.eval_inputs.copy(deep=False)
    for var_name in vars_to_grad:
        original_da = context.eval_inputs[var_name]
        median_val = float(np.median(original_da.values))
        baseline_da = xarray.DataArray(
            np.full_like(original_da.values, median_val),
            dims=original_da.dims,
            coords=original_da.coords,
            attrs=original_da.attrs,
        )
        baseline_inputs[var_name] = baseline_da
        print(f"  {var_name}: baseline = {median_val:.2f}")

    for target_var in context.target_vars:
        print(f"\nComputing IG for target: {target_var}")

        rhs = float(
            np.array(
                target_scalar(context, runtime_cfg, context.eval_inputs, target_var)
                - target_scalar(context, runtime_cfg, baseline_inputs, target_var)
            )
        )
        lhs_total = 0.0

        def _loss(inputs_data):
            # 用于梯度反向传播的标量目标函数。
            return target_scalar(context, runtime_cfg, inputs_data, target_var)

        grad_fn = jax.grad(_loss)

        for var_name in vars_to_grad:
            print(f"  Variable: {var_name}")
            accumulated_grads: Optional[np.ndarray] = None

            for step in range(ig_steps):
                alpha = (step + 1) / ig_steps
                interpolated_inputs = baseline_inputs.copy(deep=False)
                original_da = context.eval_inputs[var_name]
                baseline_da = baseline_inputs[var_name]
                interp_array = np.array(baseline_da.values) + alpha * (
                    np.array(original_da.values) - np.array(baseline_da.values)
                )
                interp_da = xarray.DataArray(
                    interp_array,
                    dims=original_da.dims,
                    coords=original_da.coords,
                    attrs=original_da.attrs,
                )
                interpolated_inputs[var_name] = interp_da

                grads = grad_fn(interpolated_inputs)
                grad_da = grads[var_name]

                if accumulated_grads is None:
                    accumulated_grads = np.array(grad_da.values)
                else:
                    accumulated_grads += np.array(grad_da.values)

                if (step + 1) % 10 == 0:
                    print(f"    Step: {step + 1}/{ig_steps}")

            if accumulated_grads is None:
                raise RuntimeError("No gradients accumulated for IG path integral")

            # IG 近似：(x - x0) × 插值路径上的平均梯度。
            avg_grads = accumulated_grads / ig_steps
            diff = np.array(context.eval_inputs[var_name].values) - np.array(
                baseline_inputs[var_name].values
            )
            ig_attr = diff * avg_grads

            original_da = context.eval_inputs[var_name]
            ig_da_full = xarray.DataArray(
                ig_attr,
                dims=original_da.dims,
                coords=original_da.coords,
                attrs=original_da.attrs,
            )
            lhs_total += float(np.array(ig_da_full.sum().values))
            ig_da = reduce_input_attribution_to_latlon(ig_attr, original_da, runtime_cfg)

            ig_region = ig_da.isel(
                lat=context.lat_indices,
                lon=context.lon_indices,
            ).values
            maps[target_var] += ig_region

            q50 = float(np.percentile(np.abs(ig_region.ravel()), 50))
            q90 = float(np.percentile(np.abs(ig_region.ravel()), 90))
            q99 = float(np.percentile(np.abs(ig_region.ravel()), 99))
            print(
                "    IG |attribution| percentiles -> "
                f"50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}"
            )

        rel_err = abs(lhs_total - rhs) / (abs(rhs) + 1e-8)
        print(
            "  IG completeness: "
            f"lhs={lhs_total:.6e}, rhs={rhs:.6e}, rel_err={rel_err:.6%}"
        )

    maps = {
        var: (values / float(len(vars_to_grad)))
        for var, values in maps.items()
    }

    print("\n=== Final IG Attribution Maps ===")
    for var, val_map in maps.items():
        val_flat = val_map.ravel()
        q50 = float(np.percentile(np.abs(val_flat), 50))
        q90 = float(np.percentile(np.abs(val_flat), 90))
        q99 = float(np.percentile(np.abs(val_flat), 99))
        print(
            f"  {var}: final |IG| percentiles -> "
            f"50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}"
        )

    return maps
