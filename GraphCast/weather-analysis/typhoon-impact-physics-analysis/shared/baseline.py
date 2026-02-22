from __future__ import annotations

import numpy as np
import xarray
from typing import List


def _build_climatology_baseline_da(
    original_da: xarray.DataArray,
    climatology_da: xarray.DataArray,
) -> xarray.DataArray:
    original_vals = np.asarray(original_da.values)

    if "level" in original_da.dims and "level" in climatology_da.dims:
        clim_levels = np.asarray(climatology_da.coords["level"].values)
        clim_values = np.asarray(climatology_da.values)
        level_lookup = {
            float(level): float(value)
            for level, value in zip(clim_levels, clim_values)
        }
        fallback = float(np.mean(clim_values))
        current_levels = np.asarray(original_da.coords["level"].values)
        selected_levels = np.array(
            [level_lookup.get(float(level), fallback) for level in current_levels],
            dtype=original_vals.dtype,
        )
        shape = [1] * original_vals.ndim
        shape[original_da.dims.index("level")] = selected_levels.shape[0]
        baseline_vals = np.broadcast_to(selected_levels.reshape(shape), original_vals.shape)
    else:
        scalar = float(np.asarray(climatology_da.values).mean())
        baseline_vals = np.full(original_vals.shape, scalar, dtype=original_vals.dtype)

    return xarray.DataArray(
        baseline_vals,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )


def _build_climatology_baseline_inputs(
    eval_inputs: xarray.Dataset,
    vars_to_use: List[str],
    mean_by_level: xarray.Dataset,
) -> xarray.Dataset:
    baseline_inputs = eval_inputs.copy(deep=False)
    for var_name in vars_to_use:
        original_da = eval_inputs[var_name]
        if var_name in mean_by_level:
            baseline_da = _build_climatology_baseline_da(original_da, mean_by_level[var_name])
        else:
            fallback = float(np.median(np.asarray(original_da.values)))
            baseline_vals = np.full_like(np.asarray(original_da.values), fallback)
            baseline_da = xarray.DataArray(
                baseline_vals,
                dims=original_da.dims,
                coords=original_da.coords,
                attrs=original_da.attrs,
            )
            print(f"[warn] {var_name} not found in climatology stats, fallback=sample median")
        baseline_inputs[var_name] = baseline_da
    return baseline_inputs


def _to_data_array_with_same_meta(
    original_da: xarray.DataArray,
    values: np.ndarray,
) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )
