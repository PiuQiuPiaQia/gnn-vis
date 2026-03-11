from __future__ import annotations

from typing import Any, Dict, List
from types import SimpleNamespace

import numpy as np
import xarray

from shared.analysis_pipeline import AnalysisConfig, AnalysisContext
from shared.importance_common import _combined_target_scalar, reduce_input_attribution_to_latlon
from shared.patch_scoring_utils import window_reduce_2d

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - lightweight tests may monkeypatch jax.grad
    def _missing_jax(*args, **kwargs):
        raise ModuleNotFoundError("jax is required for this code path")

    jax = SimpleNamespace(grad=_missing_jax)


def _build_patch_candidate_maps(
    ig_maps_by_var: Dict[str, np.ndarray],
    patch_radius: int,
    patch_score_agg: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> Dict[str, Any]:
    patch_maps_by_var: Dict[str, np.ndarray] = {}
    score_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)
    for var_name, var_map in ig_maps_by_var.items():
        patch_map = window_reduce_2d(
            np.abs(np.asarray(var_map, dtype=np.float64)),
            patch_radius,
            patch_score_agg,
        )
        patch_maps_by_var[var_name] = patch_map
        score_map += patch_map

    score_da = xarray.DataArray(
        score_map,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="patch_ig_candidate_score",
    )
    return {
        "score_da": score_da,
        "maps_by_var": patch_maps_by_var,
    }


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


def _compute_ig_candidate_score_map(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
    baseline_inputs: xarray.Dataset,
    ig_steps: int,
) -> Dict[str, Any]:
    def _loss(inputs_data):
        return _combined_target_scalar(context, runtime_cfg, inputs_data)

    grad_fn = jax.grad(_loss)

    diff_arrays = {
        var_name: np.asarray(context.eval_inputs[var_name].values)
        - np.asarray(baseline_inputs[var_name].values)
        for var_name in vars_to_use
    }
    ig_full_sum = {
        var_name: np.zeros_like(diff_arrays[var_name], dtype=np.float64)
        for var_name in vars_to_use
    }

    print("\n[IG] Computing global attribution maps...")
    for step in range(ig_steps):
        alpha = (step + 0.5) / float(ig_steps)
        interpolated_inputs = context.eval_inputs.copy(deep=False)
        for var_name in vars_to_use:
            interp_values = np.asarray(baseline_inputs[var_name].values) + alpha * diff_arrays[var_name]
            interpolated_inputs[var_name] = _to_data_array_with_same_meta(
                context.eval_inputs[var_name],
                interp_values,
            )

        grads = grad_fn(interpolated_inputs)
        for var_name in vars_to_use:
            ig_full_sum[var_name] += diff_arrays[var_name] * np.asarray(grads[var_name].values)

        if (step + 1) % 10 == 0 or (step + 1) == ig_steps:
            print(f"  IG steps: {step + 1}/{ig_steps}")

    ig_maps_by_var: Dict[str, np.ndarray] = {}
    for var_name in vars_to_use:
        ig_full = ig_full_sum[var_name] / float(ig_steps)
        ig_map_da = reduce_input_attribution_to_latlon(
            attribution=ig_full,
            original_da=context.eval_inputs[var_name],
            runtime_cfg=runtime_cfg,
        )
        ig_maps_by_var[var_name] = np.asarray(ig_map_da.values, dtype=np.float64)

    lat_vals = np.asarray(context.eval_inputs.coords["lat"].values)
    lon_vals = np.asarray(context.eval_inputs.coords["lon"].values)
    score_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)
    for var_name in vars_to_use:
        score_map += np.abs(ig_maps_by_var[var_name])

    score_da = xarray.DataArray(
        score_map,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="ig_candidate_score",
    )
    return {
        "score_da": score_da,
        "maps_by_var": ig_maps_by_var,
    }
