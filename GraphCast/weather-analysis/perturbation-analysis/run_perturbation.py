#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for cyclone point perturbation analysis."""

from pathlib import Path
import numpy as np
import xarray
import jax

from config import (
    DATASET_CONFIGS,
    DATASET_TYPE,
    TARGET_TIME_IDX,
    TARGET_VARIABLE,
    TARGET_LEVEL,
    REGION_RADIUS_DEG,
    PATCH_RADIUS,
    PERTURB_TIME,
    PERTURB_VARIABLES,
    PERTURB_LEVELS,
    BASELINE_MODE,
    TOP_N,
    OUTPUT_NC,
    OUTPUT_PNG,
    OUTPUT_PNG_CARTOPY,
    HEATMAP_DPI,
    HEATMAP_CMAP,
    HEATMAP_VMAX_QUANTILE,
    DIR_PATH_PARAMS,
    DIR_PATH_DATASET,
    DIR_PATH_STATS,
)
from cyclone_points import pick_target_cyclone
from model_utils import (
    load_checkpoint,
    load_dataset,
    extract_eval_data,
    load_normalization_stats,
    build_run_forward,
)
from perturbation_utils import (
    compute_baseline,
    select_region_indices,
    build_indexer,
    build_baseline_indexer,
    resolve_level_sel,
)
from heatmap_utils import plot_importance_heatmap
from heatmap_utils import plot_importance_heatmap_cartopy


def main():
    if DATASET_TYPE not in DATASET_CONFIGS:
        raise ValueError(f"invalid DATASET_TYPE: {DATASET_TYPE}")

    config = DATASET_CONFIGS[DATASET_TYPE]

    print(f"\n=== Config: {config['name']} ===")
    print(f"target_time_idx: {TARGET_TIME_IDX}")
    print(f"target_variable: {TARGET_VARIABLE}")
    print(f"region_radius_deg: {REGION_RADIUS_DEG} | patch_radius: {PATCH_RADIUS}")

    # load model
    ckpt = load_checkpoint(f"{DIR_PATH_PARAMS}/{config['params_file']}")
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    # load dataset
    example_batch = load_dataset(f"{DIR_PATH_DATASET}/{config['dataset_file']}")
    eval_inputs, eval_targets, eval_forcings = extract_eval_data(example_batch, task_config)

    # load normalization stats
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_stats(DIR_PATH_STATS)

    # build model runner
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

    # cyclone target point
    target_cyclone = pick_target_cyclone(TARGET_TIME_IDX)
    center_lat = target_cyclone["lat"]
    center_lon = target_cyclone["lon"]

    # baseline output
    targets_template = eval_targets * np.nan
    base_outputs = run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=targets_template,
        forcings=eval_forcings,
    )

    target_data = base_outputs[TARGET_VARIABLE]
    if "level" in target_data.dims:
        target_data = target_data.sel(level=TARGET_LEVEL)
    base_value = float(
        target_data.isel(time=TARGET_TIME_IDX).sel(lat=center_lat, lon=center_lon, method="nearest").values
    )
    print(f"Baseline {TARGET_VARIABLE}: {base_value:.4f} at ({center_lat:.2f}, {center_lon:.2f})")

    # select scan region
    lat_vals = eval_inputs.coords["lat"].values
    lon_vals = eval_inputs.coords["lon"].values
    lat_indices, lon_indices = select_region_indices(lat_vals, lon_vals, center_lat, center_lon, REGION_RADIUS_DEG)

    # variables to perturb
    if PERTURB_VARIABLES is None:
        vars_to_perturb = [
            v for v, da in eval_inputs.data_vars.items()
            if ("lat" in da.dims and "lon" in da.dims)
        ]
    else:
        vars_to_perturb = [v for v in PERTURB_VARIABLES if v in eval_inputs.data_vars]

    if not vars_to_perturb:
        raise ValueError("no perturbation variables found")

    print(f"Perturb variables: {len(vars_to_perturb)}")

    baseline_ds = compute_baseline(eval_inputs, vars_to_perturb, BASELINE_MODE)

    # time selection
    time_sel = slice(None) if PERTURB_TIME == "all" else int(PERTURB_TIME)

    # importance map
    importance = np.zeros((len(lat_indices), len(lon_indices)), dtype=np.float32)

    print("Scanning perturbations...")
    for i, lat_idx in enumerate(lat_indices):
        for j, lon_idx in enumerate(lon_indices):
            lat_start = max(lat_idx - PATCH_RADIUS, 0)
            lat_end = min(lat_idx + PATCH_RADIUS + 1, len(lat_vals))
            lon_start = max(lon_idx - PATCH_RADIUS, 0)
            lon_end = min(lon_idx + PATCH_RADIUS + 1, len(lon_vals))

            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

            saved_values = {}
            for var in vars_to_perturb:
                da = eval_inputs[var]
                level_sel = resolve_level_sel(da, PERTURB_LEVELS)
                idx = build_indexer(da, lat_slice, lon_slice, time_sel, level_sel)
                base_idx = build_baseline_indexer(da, time_sel, level_sel)

                arr = da.values
                base_arr = baseline_ds[var].values

                saved_values[var] = (idx, arr[idx].copy())
                base_vals = base_arr[base_idx]
                arr[idx] = np.broadcast_to(base_vals, arr[idx].shape)

            outputs = run_forward_jitted(
                rng=jax.random.PRNGKey(0),
                inputs=eval_inputs,
                targets_template=targets_template,
                forcings=eval_forcings,
            )
            out_var = outputs[TARGET_VARIABLE]
            if "level" in out_var.dims:
                out_var = out_var.sel(level=TARGET_LEVEL)
            new_value = float(
                out_var.isel(time=TARGET_TIME_IDX).sel(lat=center_lat, lon=center_lon, method="nearest").values
            )

            importance[i, j] = abs(new_value - base_value)

            # restore
            for var, (idx, old_vals) in saved_values.items():
                eval_inputs[var].values[idx] = old_vals

        if (i + 1) % 5 == 0:
            print(f"progress: {i + 1}/{len(lat_indices)} rows")

    # save results
    lat_sel_vals = lat_vals[lat_indices]
    lon_sel_vals = lon_vals[lon_indices]
    importance_da = xarray.DataArray(
        importance,
        dims=("lat", "lon"),
        coords={"lat": lat_sel_vals, "lon": lon_sel_vals},
        name="importance",
    )
    importance_ds = xarray.Dataset(
        {"importance": importance_da},
        attrs={
            "target_variable": TARGET_VARIABLE,
            "target_level": TARGET_LEVEL if "level" in base_outputs[TARGET_VARIABLE].dims else "none",
            "target_time_idx": TARGET_TIME_IDX,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "baseline_mode": BASELINE_MODE,
            "patch_radius": PATCH_RADIUS,
            "region_radius_deg": REGION_RADIUS_DEG,
        },
    )

    output_path = Path(__file__).parent / OUTPUT_NC
    importance_ds.to_netcdf(output_path)
    print(f"Saved importance map: {output_path}")

    if OUTPUT_PNG:
        png_path = Path(__file__).parent / OUTPUT_PNG
        title = f"Perturbation Importance (t={TARGET_TIME_IDX}, var={TARGET_VARIABLE})"
        plot_importance_heatmap(
            importance_da,
            center_lat,
            center_lon,
            png_path,
            title,
            cmap=HEATMAP_CMAP,
            dpi=HEATMAP_DPI,
            vmax_quantile=HEATMAP_VMAX_QUANTILE,
        )
        print(f"Saved heatmap: {png_path}")

    if OUTPUT_PNG_CARTOPY:
        map_path = Path(__file__).parent / OUTPUT_PNG_CARTOPY
        title = f"Perturbation Importance Map (t={TARGET_TIME_IDX}, var={TARGET_VARIABLE})"
        try:
            plot_importance_heatmap_cartopy(
                importance_da,
                center_lat,
                center_lon,
                map_path,
                title,
                cmap=HEATMAP_CMAP,
                dpi=HEATMAP_DPI,
                vmax_quantile=HEATMAP_VMAX_QUANTILE,
            )
            print(f"Saved map heatmap: {map_path}")
        except RuntimeError as exc:
            print(f"Skip cartopy map: {exc}")
    # top-N
    flat_idx = np.argsort(importance.ravel())[::-1][:TOP_N]
    print(f"\nTop-{TOP_N} influential grid points:")
    for rank, idx in enumerate(flat_idx, start=1):
        r = idx // importance.shape[1]
        c = idx % importance.shape[1]
        lat = float(lat_sel_vals[r])
        lon = float(lon_sel_vals[c])
        score = float(importance[r, c])
        print(f"{rank:02d}. lat={lat:.2f}, lon={lon:.2f}, score={score:.6f}")


if __name__ == "__main__":
    main()
