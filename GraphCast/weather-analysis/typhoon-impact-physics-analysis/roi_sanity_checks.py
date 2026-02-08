#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal ROI checks for IG sign + rank consistency.

Implements two checks (center ROI only):
1) Sign sanity: pick argmax(IG) and argmin(IG), verify the perturbation delta
   at those points matches the expected sign when using -IG as a proxy for Δy.
2) Rank consistency: Spearman rho between |IG| and |Δy|, plus one shuffle
   control (shuffle IG within ROI once; rho should drop).

Notes:
- Δy here is defined as (perturbed - original), matching the perturbation code.
- To make signs comparable, IG uses the same baseline mode/value as perturbation.
- If cfg.PERTURB_TIME == 'all', both IG and perturbation aggregate over all
  perturbed input times by summing contributions.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import xarray
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr

import config as cfg
from cyclone_points import pick_target_cyclone
from model_utils import (
    load_checkpoint,
    load_dataset,
    extract_eval_data,
    load_normalization_stats,
    build_run_forward,
)
from impact_analysis_utils import (
    compute_baseline,
    select_region_indices,
    build_indexer,
    build_baseline_indexer,
    resolve_level_sel,
)
from graphcast import xarray_jax


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (rho, p) as plain floats across SciPy versions."""
    res = spearmanr(x, y, nan_policy="omit")
    # SciPy may return either a SignificanceResult (preferred) or a tuple-like.
    rho = float(getattr(res, "statistic", res[0]))  # type: ignore[index]
    pval = float(getattr(res, "pvalue", res[1]))  # type: ignore[index]
    return rho, pval


def _match_shape(base_vals: np.ndarray, target_shape) -> np.ndarray:
    """Broadcast baseline values to match a slice shape."""
    if base_vals.shape == target_shape:
        return base_vals
    if base_vals.ndim < len(target_shape):
        base_vals = base_vals.reshape(base_vals.shape + (1,) * (len(target_shape) - base_vals.ndim))
    elif base_vals.ndim > len(target_shape):
        squeeze_axes = tuple(i for i, s in enumerate(base_vals.shape) if s == 1)
        if squeeze_axes:
            base_vals = np.squeeze(base_vals, axis=squeeze_axes)
        if base_vals.ndim < len(target_shape):
            base_vals = base_vals.reshape(base_vals.shape + (1,) * (len(target_shape) - base_vals.ndim))
    return base_vals


def _select_target_data(outputs: xarray.Dataset, var: str) -> xarray.DataArray:
    data = outputs[var]
    if "level" in data.dims:
        level = getattr(cfg, "TARGET_LEVELS", {}).get(var, getattr(cfg, "TARGET_LEVEL", None))
        if level is not None:
            data = data.sel(level=level)
    return data


def _target_scalar(
    run_forward_jitted,
    *,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    center_lat: float,
    center_lon: float,
    target_var: str,
) -> jax.Array:
    outputs = run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    out_var = _select_target_data(outputs, target_var)
    value = out_var.isel(time=cfg.TARGET_TIME_IDX).sel(lat=center_lat, lon=center_lon, method="nearest")
    if "batch" in value.dims:
        value = value.isel(batch=0)
    scalar = xarray_jax.unwrap_data(value, require_jax=True)
    return jnp.squeeze(scalar)


def _reduce_ig_to_latlon(ig_attr: np.ndarray, original_da: xarray.DataArray) -> xarray.DataArray:
    """Reduce raw IG attribution to a 2D (lat, lon) map consistent with perturbation."""
    ig_da = xarray.DataArray(
        ig_attr,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    if "batch" in ig_da.dims:
        ig_da = ig_da.isel(batch=0)

    if "time" in ig_da.dims:
        if cfg.PERTURB_TIME == "all":
            ig_da = ig_da.sum(dim="time")
        else:
            ig_da = ig_da.isel(time=int(cfg.PERTURB_TIME))

    if "level" in ig_da.dims:
        level_sel = resolve_level_sel(original_da, cfg.PERTURB_LEVELS)
        ig_da = ig_da.isel(level=level_sel)
        if "level" in ig_da.dims:
            ig_da = ig_da.sum(dim="level")

    ig_da = ig_da.transpose("lat", "lon")
    return ig_da


def _center_roi_mask(lat_vals: np.ndarray, lon_vals: np.ndarray, center_lat: float, center_lon: float, window_deg: float):
    lat_mask = (lat_vals >= (center_lat - window_deg)) & (lat_vals <= (center_lat + window_deg))
    dlon = ((lon_vals - center_lon + 180.0) % 360.0) - 180.0
    lon_mask = np.abs(dlon) <= window_deg
    return lat_mask, lon_mask


def main() -> int:
    script_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    target_var = getattr(cfg, "TARGET_VARIABLE", "mean_sea_level_pressure")
    if target_var is None:
        raise ValueError("cfg.TARGET_VARIABLE is required")

    config = cfg.DATASET_CONFIGS[cfg.DATASET_TYPE]
    center_window_deg = float(getattr(cfg, "GRADIENT_CENTER_WINDOW_DEG", 10.0))
    ig_steps = int(getattr(cfg, "IG_STEPS", 50))

    print(f"=== ROI sanity checks ===")
    print(f"dataset: {config['name']}")
    print(f"target_var: {target_var}")
    print(f"target_time_idx: {cfg.TARGET_TIME_IDX}")
    print(f"perturb_time: {cfg.PERTURB_TIME}")
    print(f"baseline_mode: {cfg.BASELINE_MODE}")
    print(f"center ROI window: +/- {center_window_deg:.2f} deg")
    print(f"IG steps: {ig_steps}")

    ckpt = load_checkpoint(f"{cfg.DIR_PATH_PARAMS}/{config['params_file']}")
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    example_batch = load_dataset(f"{cfg.DIR_PATH_DATASET}/{config['dataset_file']}")
    eval_inputs, eval_targets, eval_forcings = extract_eval_data(example_batch, task_config)
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_stats(cfg.DIR_PATH_STATS)

    print("JIT compiling model...")
    run_forward_jitted = build_run_forward(
        model_config,
        task_config,
        params,
        state,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
    )
    print("Model ready!")

    target_cyclone = pick_target_cyclone(cfg.TARGET_TIME_IDX)
    center_lat = float(target_cyclone["lat"])
    center_lon = float(target_cyclone["lon"])
    print(f"center: lat={center_lat:.2f}, lon={center_lon:.2f}")

    targets_template = eval_targets * np.nan
    base_y = float(
        np.array(
            _target_scalar(
                run_forward_jitted,
                inputs=eval_inputs,
                targets_template=targets_template,
                forcings=eval_forcings,
                center_lat=center_lat,
                center_lon=center_lon,
                target_var=target_var,
            )
        )
    )
    print(f"baseline output y0: {base_y:.6f}")

    if target_var not in eval_inputs.data_vars:
        raise ValueError(f"TARGET_VARIABLE '{target_var}' not found in eval_inputs")

    # Baseline values for perturbation/IG (same baseline mode).
    baseline_ds = compute_baseline(
        eval_inputs,
        [target_var],
        cfg.BASELINE_MODE,
        center_lat=center_lat,
        center_lon=center_lon,
        inner_deg=float(getattr(cfg, "LOCAL_BASELINE_INNER_DEG", 5.0)),
        outer_deg=float(getattr(cfg, "LOCAL_BASELINE_OUTER_DEG", 12.0)),
        min_points=int(getattr(cfg, "LOCAL_BASELINE_MIN_POINTS", 120)),
    )
    original_da = eval_inputs[target_var]
    base_vals = baseline_ds[target_var].values
    base_vals = _match_shape(base_vals, original_da.values.shape)
    baseline_da_full = xarray.DataArray(
        np.broadcast_to(base_vals, original_da.values.shape),
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )

    # Build IG baseline inputs.
    baseline_inputs = eval_inputs.copy(deep=False)
    baseline_inputs[target_var] = baseline_da_full

    def _loss(inputs_data: xarray.Dataset) -> jax.Array:
        return _target_scalar(
            run_forward_jitted,
            inputs=inputs_data,
            targets_template=targets_template,
            forcings=eval_forcings,
            center_lat=center_lat,
            center_lon=center_lon,
            target_var=target_var,
        )

    grad_fn = jax.grad(_loss)

    print("\nComputing IG...")
    t0 = time.time()
    accumulated_grads = None
    for step in range(ig_steps):
        alpha = (step + 1) / ig_steps
        interpolated_inputs = baseline_inputs.copy(deep=False)
        interpolated_inputs[target_var] = baseline_inputs[target_var] + alpha * (
            eval_inputs[target_var] - baseline_inputs[target_var]
        )
        grads = grad_fn(interpolated_inputs)
        g = np.array(grads[target_var].values)
        accumulated_grads = g if accumulated_grads is None else (accumulated_grads + g)
        if (step + 1) % max(1, ig_steps // 5) == 0:
            print(f"  step {step + 1:>3}/{ig_steps}")

    if accumulated_grads is None:
        raise RuntimeError("No gradients accumulated for IG path integral")
    avg_grads = accumulated_grads / float(ig_steps)
    diff = np.array(eval_inputs[target_var].values) - np.array(baseline_inputs[target_var].values)
    ig_attr = diff * avg_grads
    ig_map_full = _reduce_ig_to_latlon(ig_attr, original_da)
    print(f"IG done in {time.time() - t0:.1f}s")

    # Region selection (outer) then center ROI selection.
    lat_vals = eval_inputs.coords["lat"].values
    lon_vals = eval_inputs.coords["lon"].values
    lat_indices, lon_indices = select_region_indices(lat_vals, lon_vals, center_lat, center_lon, cfg.REGION_RADIUS_DEG)
    lat_sel_vals = lat_vals[lat_indices]
    lon_sel_vals = lon_vals[lon_indices]

    ig_region = ig_map_full.isel(lat=lat_indices, lon=lon_indices).values
    lat_mask_center, lon_mask_center = _center_roi_mask(lat_sel_vals, lon_sel_vals, center_lat, center_lon, center_window_deg)
    if not np.any(lat_mask_center) or not np.any(lon_mask_center):
        raise ValueError("Center ROI mask is empty; check GRADIENT_CENTER_WINDOW_DEG")

    ig_roi = ig_region[np.ix_(lat_mask_center, lon_mask_center)]
    roi_lat_vals = lat_sel_vals[lat_mask_center]
    roi_lon_vals = lon_sel_vals[lon_mask_center]
    print(f"center ROI shape: {ig_roi.shape} (n={ig_roi.size})")

    # Find argmax/min IG within center ROI.
    flat = ig_roi.ravel()
    argmax_idx = int(np.nanargmax(flat))
    argmin_idx = int(np.nanargmin(flat))
    r_max, c_max = np.unravel_index(argmax_idx, ig_roi.shape)
    r_min, c_min = np.unravel_index(argmin_idx, ig_roi.shape)
    p_max = (float(roi_lat_vals[r_max]), float(roi_lon_vals[c_max]))
    p_min = (float(roi_lat_vals[r_min]), float(roi_lon_vals[c_min]))
    print("\n[Sign sanity points]")
    print(f"  argmax(IG): lat={p_max[0]:.2f}, lon={p_max[1]:.2f}, IG={float(ig_roi[r_max, c_max]):+.6e}")
    print(f"  argmin(IG): lat={p_min[0]:.2f}, lon={p_min[1]:.2f}, IG={float(ig_roi[r_min, c_min]):+.6e}")

    # Compute Δy for all points in center ROI (single variable).
    # We occlude inputs at (lat, lon) to baseline values, matching perturbation Δy.
    time_sel = slice(None) if cfg.PERTURB_TIME == "all" else int(cfg.PERTURB_TIME)
    level_sel = resolve_level_sel(original_da, cfg.PERTURB_LEVELS)
    base_idx = build_baseline_indexer(original_da, time_sel, level_sel)
    base_arr = baseline_ds[target_var].values
    arr = original_da.values

    # Absolute indices in the full grid for center ROI.
    lat_indices_roi = lat_indices[lat_mask_center]
    lon_indices_roi = lon_indices[lon_mask_center]
    delta_roi = np.zeros((len(lat_indices_roi), len(lon_indices_roi)), dtype=np.float32)

    print("\nComputing Δy map on center ROI...")
    t1 = time.time()
    for ii, lat_idx in enumerate(lat_indices_roi):
        for jj, lon_idx in enumerate(lon_indices_roi):
            lat_start = max(int(lat_idx) - int(cfg.PATCH_RADIUS), 0)
            lat_end = min(int(lat_idx) + int(cfg.PATCH_RADIUS) + 1, len(lat_vals))
            lon_start = max(int(lon_idx) - int(cfg.PATCH_RADIUS), 0)
            lon_end = min(int(lon_idx) + int(cfg.PATCH_RADIUS) + 1, len(lon_vals))

            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)
            idx = build_indexer(original_da, lat_slice, lon_slice, time_sel, level_sel)
            saved = arr[idx].copy()

            base_vals = base_arr[base_idx]
            base_vals = _match_shape(base_vals, arr[idx].shape)
            arr[idx] = np.broadcast_to(base_vals, arr[idx].shape)

            y_new = float(
                np.array(
                    _target_scalar(
                        run_forward_jitted,
                        inputs=eval_inputs,
                        targets_template=targets_template,
                        forcings=eval_forcings,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        target_var=target_var,
                    )
                )
            )
            delta_roi[ii, jj] = y_new - base_y
            arr[idx] = saved

        if (ii + 1) % max(1, len(lat_indices_roi) // 5) == 0:
            print(f"  rows {ii + 1}/{len(lat_indices_roi)}")

    print(f"Δy done in {time.time() - t1:.1f}s")

    # Check sign sanity.
    dy_argmax = float(delta_roi[r_max, c_max])
    dy_argmin = float(delta_roi[r_min, c_min])
    print("\n[Sign sanity results] (Δy = perturbed - original)")
    print(f"  Δy(argmax IG): {dy_argmax:+.6e}  (expect < 0 if using -IG -> Δy)")
    print(f"  Δy(argmin IG): {dy_argmin:+.6e}  (expect > 0 if using -IG -> Δy)")
    print(f"  pass1 Δy(argmax IG) < 0: {dy_argmax < 0}")
    print(f"  pass2 Δy(argmin IG) > 0: {dy_argmin > 0}")

    # Rank consistency.
    ig_flat = np.abs(ig_roi.ravel())
    dy_flat = np.abs(delta_roi.ravel())
    rho, pval = _spearman(ig_flat, dy_flat)
    rng = np.random.default_rng(0)
    ig_shuf = ig_flat.copy()
    rng.shuffle(ig_shuf)
    rho_shuf, pval_shuf = _spearman(ig_shuf, dy_flat)

    print("\n[Rank consistency] (center ROI)")
    print(f"  Spearman rho(|IG|, |Δy|): {rho:+.6f} (p={pval:.3e})")
    print(f"  Shuffled control rho:      {rho_shuf:+.6f} (p={pval_shuf:.3e})")
    print(f"  |rho| drop: {abs(rho) - abs(rho_shuf):+.6f}")

    out_dir = script_dir / "validation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "roi_ig_delta_sanity.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("ROI sanity checks (center ROI)\n")
        f.write(f"target_var: {target_var}\n")
        f.write(f"center: lat={center_lat:.2f}, lon={center_lon:.2f}\n")
        f.write(f"center_window_deg: {center_window_deg:.2f}\n")
        f.write(f"baseline_mode: {cfg.BASELINE_MODE}\n")
        f.write(f"perturb_time: {cfg.PERTURB_TIME}\n")
        f.write(f"patch_radius: {cfg.PATCH_RADIUS}\n")
        f.write(f"IG steps: {ig_steps}\n")
        f.write("\n[Sign sanity]\n")
        f.write(f"argmax_ig: lat={p_max[0]:.2f}, lon={p_max[1]:.2f}, IG={float(ig_roi[r_max, c_max]):+.6e}, dy={dy_argmax:+.6e}\n")
        f.write(f"argmin_ig: lat={p_min[0]:.2f}, lon={p_min[1]:.2f}, IG={float(ig_roi[r_min, c_min]):+.6e}, dy={dy_argmin:+.6e}\n")
        f.write(f"check dy(argmax IG) < 0: {dy_argmax < 0}\n")
        f.write(f"check dy(argmin IG) > 0: {dy_argmin > 0}\n")
        f.write("\n[Rank consistency]\n")
        f.write(f"rho: {rho:+.6f} (p={pval:.3e})\n")
        f.write(f"rho_shuffled: {rho_shuf:+.6f} (p={pval_shuf:.3e})\n")
        f.write(f"abs_drop: {abs(rho) - abs(rho_shuf):+.6f}\n")

    print(f"\nSaved: {out_txt}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
