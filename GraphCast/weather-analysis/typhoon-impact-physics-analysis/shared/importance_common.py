from __future__ import annotations

import numpy as np
import xarray
from types import SimpleNamespace

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - lightweight tests import helpers without jax
    def _missing_jax(*args, **kwargs):
        raise ModuleNotFoundError("jax is required for this code path")

    jax = SimpleNamespace(Array=object, random=SimpleNamespace(PRNGKey=lambda seed: seed))
    jnp = np

try:
    from graphcast import xarray_jax
except ModuleNotFoundError:  # pragma: no cover - lightweight tests do not need GraphCast bindings
    class _XarrayJaxStub:
        @staticmethod
        def unwrap_data(value, require_jax: bool = False):
            return np.asarray(getattr(value, "values", value))

    xarray_jax = _XarrayJaxStub()

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


def _combined_target_scalar(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    inputs_data: xarray.Dataset,
) -> jax.Array:
    """Return a scalar objective compatible with legacy IG runner usage.

    Prefer multi-target mean when available; otherwise fallback to single target.
    """
    if hasattr(context, "target_vars") and getattr(context, "target_vars"):
        target_vars = list(getattr(context, "target_vars"))
        vals = [target_scalar(context, runtime_cfg, inputs_data, tv) for tv in target_vars]
        return jnp.mean(jnp.stack(vals))
    return target_scalar(context, runtime_cfg, inputs_data, context.target_var)


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


def collapse_input_attribution_to_latlon(
    attribution: np.ndarray,
    original_da: xarray.DataArray,
    *,
    abs_before_sum: bool = False,
) -> xarray.DataArray:
    """Collapse attribution to lat/lon by summing every non-spatial channel.

    This matches patch-level formulas that sum over all channels c before
    aggregating over spatial cells g in a patch.
    """
    attr_da = xarray.DataArray(
        attribution,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    if "batch" in attr_da.dims:
        attr_da = attr_da.isel(batch=0)
    if abs_before_sum:
        attr_da = np.abs(attr_da)

    reduce_dims = [dim for dim in attr_da.dims if dim not in {"lat", "lon"}]
    if reduce_dims:
        attr_da = attr_da.sum(dim=reduce_dims)
    return attr_da.transpose("lat", "lon")
