# -*- coding: utf-8 -*-
"""台风分析的有效感受野（ERF）敏感性计算模块。"""

from __future__ import annotations

from typing import Dict

import jax
import numpy as np

from analysis_pipeline import AnalysisConfig, AnalysisContext
from importance_common import reduce_input_attribution_to_latlon, target_scalar


def compute_erf_importance(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
) -> Dict[str, np.ndarray]:
    """从输入输出敏感性梯度计算 ERF 图。"""
    maps = {
        var: np.zeros((len(context.lat_indices), len(context.lon_indices)), dtype=np.float32)
        for var in context.target_vars
    }

    if runtime_cfg.importance_mode == "compare":
        vars_to_erf = [runtime_cfg.target_variable]
    elif runtime_cfg.erf_variables is None:
        vars_to_erf = context.vars_to_perturb
    else:
        vars_to_erf = [
            var_name
            for var_name in runtime_cfg.erf_variables
            if var_name in context.eval_inputs.data_vars
        ]

    if not vars_to_erf:
        raise ValueError("No ERF variables found")

    print("\n=== ERF Computation ===")
    print(f"ERF variables: {vars_to_erf}")
    print(f"ERF absolute amplitude: {runtime_cfg.erf_abs}")

    for target_var in context.target_vars:
        print(f"\nComputing ERF for target: {target_var}")

        def _loss(inputs_data):
            # 与 IG 使用相同的标量目标，但 ERF 仅在当前输入状态下进行一次直接梯度计算。
            return target_scalar(context, runtime_cfg, inputs_data, target_var)

        grad_fn = jax.grad(_loss)
        # 在未修改的输入状态上执行一次反向传播。
        grads = grad_fn(context.eval_inputs)

        for var_name in vars_to_erf:
            grad_values = np.array(grads[var_name].values)
            if runtime_cfg.erf_abs:
                # 典型的 ERF 可视化仅使用幅值型敏感性。
                grad_values = np.abs(grad_values)

            erf_da = reduce_input_attribution_to_latlon(
                grad_values,
                context.eval_inputs[var_name],
                runtime_cfg,
            )
            erf_region = erf_da.isel(
                lat=context.lat_indices,
                lon=context.lon_indices,
            ).values

            maps[target_var] += erf_region

            q50 = float(np.percentile(np.abs(erf_region.ravel()), 50))
            q90 = float(np.percentile(np.abs(erf_region.ravel()), 90))
            q99 = float(np.percentile(np.abs(erf_region.ravel()), 99))
            print(
                "    ERF |sensitivity| percentiles -> "
                f"50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}"
            )

        # 保持不同变量数量选择下的尺度可比性。
        maps[target_var] = maps[target_var] / float(len(vars_to_erf))

    print("\n=== Final ERF Maps ===")
    for var, val_map in maps.items():
        val_flat = val_map.ravel()
        q50 = float(np.percentile(np.abs(val_flat), 50))
        q90 = float(np.percentile(np.abs(val_flat), 90))
        q99 = float(np.percentile(np.abs(val_flat), 99))
        print(
            f"  {var}: final |ERF| percentiles -> "
            f"50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}"
        )

    return maps
