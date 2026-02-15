# -*- coding: utf-8 -*-
"""Importance computation methods for typhoon impact analysis."""

from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast import xarray_jax

from analysis_pipeline import AnalysisConfig, AnalysisContext, select_target_data
from impact_analysis_utils import (
    build_baseline_indexer,
    build_indexer,
    compute_baseline,
    resolve_level_sel,
)


def _match_shape(base_vals: np.ndarray, target_shape) -> np.ndarray:
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
    maps = {
        var: np.zeros((len(context.lat_indices), len(context.lon_indices)), dtype=np.float32)
        for var in context.target_vars
    }

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
            lat_start = max(int(lat_idx) - runtime_cfg.patch_radius, 0)
            lat_end = min(int(lat_idx) + runtime_cfg.patch_radius + 1, len(context.lat_vals))
            lon_start = max(int(lon_idx) - runtime_cfg.patch_radius, 0)
            lon_end = min(int(lon_idx) + runtime_cfg.patch_radius + 1, len(context.lon_vals))
            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

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

            for var, (idx, old_vals) in saved_values.items():
                context.eval_inputs[var].values[idx] = old_vals

        if (i + 1) % 5 == 0:
            print(f"progress: {i + 1}/{len(context.lat_indices)} rows")

    return maps


def _target_scalar(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    inputs_data,
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
        target_levels=runtime_cfg.target_levels,
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


def _reduce_ig_to_latlon(
    ig_attr: np.ndarray,
    original_da: xarray.DataArray,
    runtime_cfg: AnalysisConfig,
) -> xarray.DataArray:
    ig_da = xarray.DataArray(
        ig_attr,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    if "batch" in ig_da.dims:
        ig_da = ig_da.isel(batch=0)

    if "time" in ig_da.dims:
        if runtime_cfg.perturb_time == "all":
            if runtime_cfg.gradient_time_agg == "mean":
                ig_da = ig_da.mean(dim="time")
            else:
                ig_da = ig_da.isel(time=0)
        else:
            ig_da = ig_da.isel(time=int(runtime_cfg.perturb_time))

    if "level" in ig_da.dims:
        level_sel = resolve_level_sel(original_da, runtime_cfg.perturb_levels)
        ig_da = ig_da.isel(level=level_sel)
        if "level" in ig_da.dims:
            ig_da = ig_da.mean(dim="level")

    return ig_da.transpose("lat", "lon")


def compute_gradient_importance(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
) -> Dict[str, np.ndarray]:
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
                _target_scalar(context, runtime_cfg, context.eval_inputs, target_var)
                - _target_scalar(context, runtime_cfg, baseline_inputs, target_var)
            )
        )
        lhs_total = 0.0

        def _loss(inputs_data):
            return _target_scalar(context, runtime_cfg, inputs_data, target_var)

        grad_fn = jax.grad(_loss)

        for var_name in vars_to_grad:
            print(f"  Variable: {var_name}")
            accumulated_grads = None

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
            ig_da = _reduce_ig_to_latlon(ig_attr, original_da, runtime_cfg)

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
