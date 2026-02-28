from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast import xarray_jax

from shared.analysis_pipeline import AnalysisConfig, AnalysisContext, select_target_data
from shared.impact_analysis_utils import resolve_level_sel


def target_scalar(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    inputs_data: xarray.Dataset,
    target_var: str,
) -> jax.Array:
    outputs = context.run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs_data,
        targets_template=context.targets_template,
        forcings=context.eval_forcings,
    )
    out_var = select_target_data(
        outputs,
        target_var,
        target_level=runtime_cfg.target_level,
    )
    value = out_var.isel(time=runtime_cfg.target_time_idx).sel(
        lat=context.center_lat,
        lon=context.center_lon,
        method="nearest",
    )
    if "batch" in value.dims:
        value = value.isel(batch=0)
    scalar = xarray_jax.unwrap_data(value, require_jax=True)
    return jnp.squeeze(scalar)


def reduce_input_attribution_to_latlon(
    attribution: np.ndarray,
    original_da: xarray.DataArray,
    runtime_cfg: AnalysisConfig,
) -> xarray.DataArray:
    attr_da = xarray.DataArray(
        attribution,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    if "batch" in attr_da.dims:
        attr_da = attr_da.isel(batch=0)

    if "time" in attr_da.dims:
        if runtime_cfg.perturb_time == "all":
            if runtime_cfg.gradient_time_agg == "mean":
                attr_da = attr_da.mean(dim="time")
            else:
                attr_da = attr_da.isel(time=0)
        else:
            attr_da = attr_da.isel(time=int(runtime_cfg.perturb_time))

    if "level" in attr_da.dims:
        level_sel = resolve_level_sel(original_da, runtime_cfg.perturb_levels)
        attr_da = attr_da.isel(level=level_sel)
        if "level" in attr_da.dims:
            attr_da = attr_da.mean(dim="level")

    return attr_da.transpose("lat", "lon")
