# -*- coding: utf-8 -*-
"""梯度型重要性方法的共享辅助函数。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast import xarray_jax

from analysis_pipeline import AnalysisConfig, AnalysisContext, select_target_data
from impact_analysis_utils import resolve_level_sel


def target_scalar(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    inputs_data: xarray.Dataset,
    target_var: str,
) -> jax.Array:
    """返回用于归因梯度的台风中心处的标量目标值。"""
    # 使用提供的输入张量和固定强迫场执行一次前向传播。
    outputs = context.run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs_data,
        targets_template=context.targets_template,
        forcings=context.eval_forcings,
    )
    out_var = select_target_data(
        outputs,
        target_var,
        target_levels=runtime_cfg.target_levels,
        target_level=runtime_cfg.target_level,
    )
    value = out_var.isel(time=runtime_cfg.target_time_idx).sel(
        lat=context.center_lat,
        lon=context.center_lon,
        method="nearest",
    )
    # 去除批次维度以获得 JAX 标量梯度目标。
    if "batch" in value.dims:
        value = value.isel(batch=0)
    scalar = xarray_jax.unwrap_data(value, require_jax=True)
    return jnp.squeeze(scalar)


def reduce_input_attribution_to_latlon(
    attribution: np.ndarray,
    original_da: xarray.DataArray,
    runtime_cfg: AnalysisConfig,
) -> xarray.DataArray:
    """将原始归因张量降维为二维（lat, lon）图。"""
    attr_da = xarray.DataArray(
        attribution,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    if "batch" in attr_da.dims:
        attr_da = attr_da.isel(batch=0)

    if "time" in attr_da.dims:
        # 遵循 PERTURB_TIME 规则以与扰动模式保持一致。
        if runtime_cfg.perturb_time == "all":
            if runtime_cfg.gradient_time_agg == "mean":
                attr_da = attr_da.mean(dim="time")
            else:
                attr_da = attr_da.isel(time=0)
        else:
            attr_da = attr_da.isel(time=int(runtime_cfg.perturb_time))

    if "level" in attr_da.dims:
        # 将选定气压层合并为一张图，用于绘图/top-k 排名。
        level_sel = resolve_level_sel(original_da, runtime_cfg.perturb_levels)
        attr_da = attr_da.isel(level=level_sel)
        if "level" in attr_da.dims:
            attr_da = attr_da.mean(dim="level")

    return attr_da.transpose("lat", "lon")
