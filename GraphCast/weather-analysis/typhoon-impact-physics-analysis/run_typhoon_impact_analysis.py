#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Notebook-style entry for typhoon impact physics analysis."""

# %% Imports
from pathlib import Path
import numpy as np
import xarray
import jax
import jax.numpy as jnp

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
from heatmap_utils import (
    plot_importance_heatmap,
    plot_importance_heatmap_cartopy,
    plot_importance_heatmap_dual,
)
from graphcast import xarray_jax

# %% Notebook-friendly root
ROOT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
HEATMAP_DIVERGING = getattr(cfg, "HEATMAP_DIVERGING", False)
OUTPUT_PNG_CARTOPY = getattr(cfg, "OUTPUT_PNG_CARTOPY", None)
OUTPUT_PNG_COMBINED = getattr(cfg, "OUTPUT_PNG_COMBINED", None)
OUTPUT_PNG_METHOD_COMPARE = getattr(cfg, "OUTPUT_PNG_METHOD_COMPARE", None)
TARGET_VARIABLES = getattr(cfg, "TARGET_VARIABLES", None)
TARGET_LEVELS = getattr(cfg, "TARGET_LEVELS", {})
IMPORTANCE_MODE = getattr(cfg, "IMPORTANCE_MODE", "perturbation")
# 默认改为 signed 模式，便于可视化正负贡献
GRADIENT_MODE = getattr(cfg, "GRADIENT_MODE", "signed")
GRADIENT_X_INPUT = getattr(cfg, "GRADIENT_X_INPUT", False)
GRADIENT_VARIABLES = getattr(cfg, "GRADIENT_VARIABLES", None)
GRADIENT_VMAX_QUANTILE = getattr(cfg, "GRADIENT_VMAX_QUANTILE", cfg.HEATMAP_VMAX_QUANTILE)
GRADIENT_CMAP = getattr(cfg, "GRADIENT_CMAP", "RdBu_r")
GRADIENT_DIVERGING = getattr(cfg, "GRADIENT_DIVERGING", True)
GRADIENT_CENTER_WINDOW_DEG = getattr(cfg, "GRADIENT_CENTER_WINDOW_DEG", 10.0)
GRADIENT_CENTER_SCALE_QUANTILE = getattr(cfg, "GRADIENT_CENTER_SCALE_QUANTILE", 0.99)
GRADIENT_ALPHA_QUANTILE = getattr(cfg, "GRADIENT_ALPHA_QUANTILE", 0.90)
# 新增配置：梯度时间聚合方式，默认 "single" 表示取单个时刻（time=0），可选 "mean"
GRADIENT_TIME_AGG = getattr(cfg, "GRADIENT_TIME_AGG", "single")
LOCAL_BASELINE_INNER_DEG = getattr(cfg, "LOCAL_BASELINE_INNER_DEG", 5.0)
LOCAL_BASELINE_OUTER_DEG = getattr(cfg, "LOCAL_BASELINE_OUTER_DEG", 12.0)
LOCAL_BASELINE_MIN_POINTS = getattr(cfg, "LOCAL_BASELINE_MIN_POINTS", 120)


def _match_shape(base_vals: np.ndarray, target_shape) -> np.ndarray:
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

# %% Optional overrides (edit as needed)
# cfg.TARGET_TIME_IDX = 0
# cfg.TARGET_VARIABLE = "mean_sea_level_pressure"
# cfg.REGION_RADIUS_DEG = 15
# cfg.PATCH_RADIUS = 0
# cfg.PERTURB_TIME = "all"

# %% Config and model/data loading
if cfg.DATASET_TYPE not in cfg.DATASET_CONFIGS:
    raise ValueError(f"Invalid DATASET_TYPE: {cfg.DATASET_TYPE}")

config = cfg.DATASET_CONFIGS[cfg.DATASET_TYPE]
print(f"\n=== Config: {config['name']} ===")
print(f"target_time_idx: {cfg.TARGET_TIME_IDX}")
print(f"region_radius_deg: {cfg.REGION_RADIUS_DEG} | patch_radius: {cfg.PATCH_RADIUS}")

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

# %% Baseline output at cyclone point
target_cyclone = pick_target_cyclone(cfg.TARGET_TIME_IDX)
center_lat = target_cyclone["lat"]
center_lon = target_cyclone["lon"]

targets_template = eval_targets * np.nan
base_outputs = run_forward_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=targets_template,
    forcings=eval_forcings,
)

target_vars = TARGET_VARIABLES if TARGET_VARIABLES else [cfg.TARGET_VARIABLE]
print(f"target_variables: {', '.join(target_vars)}")


def _select_target_data(outputs, var):
    data = outputs[var]
    if "level" in data.dims:
        level = TARGET_LEVELS.get(var, cfg.TARGET_LEVEL)
        data = data.sel(level=level)
    return data


base_values = {}
for var in target_vars:
    target_data = _select_target_data(base_outputs, var)
    base_values[var] = target_data.isel(time=cfg.TARGET_TIME_IDX).sel(
        lat=center_lat, lon=center_lon, method="nearest"
    ).values.item()
    print(f"Baseline {var}: {base_values[var]:.4f} at ({center_lat:.2f}, {center_lon:.2f})")

# %% Region selection and variable list
lat_vals = eval_inputs.coords["lat"].values
lon_vals = eval_inputs.coords["lon"].values
lat_indices, lon_indices = select_region_indices(lat_vals, lon_vals, center_lat, center_lon, cfg.REGION_RADIUS_DEG)

# 需求1：在 compare 模式下强制对齐输入变量集合
if IMPORTANCE_MODE == "compare":
    # compare 模式下，必须只针对 cfg.TARGET_VARIABLE 做归因/扰动
    if cfg.TARGET_VARIABLE not in eval_inputs.data_vars:
        raise ValueError(f"TARGET_VARIABLE '{cfg.TARGET_VARIABLE}' not found in eval_inputs for compare mode")
    vars_to_perturb = [cfg.TARGET_VARIABLE]
    print(f"[COMPARE MODE] Aligning perturbation and gradient to single variable: {cfg.TARGET_VARIABLE}")
else:
    # 非 compare 模式保留原逻辑
    if cfg.PERTURB_VARIABLES is None:
        vars_to_perturb = [
            v for v, da in eval_inputs.data_vars.items()
            if ("lat" in da.dims and "lon" in da.dims)
        ]
    else:
        vars_to_perturb = [v for v in cfg.PERTURB_VARIABLES if v in eval_inputs.data_vars]

if not vars_to_perturb:
    raise ValueError("No perturbation variables found")
print(f"Spatial vars: {len(vars_to_perturb)}")

# %% Importance scan
def _compute_perturbation_importance(target_vars_local):
    maps = {var: np.zeros((len(lat_indices), len(lon_indices)), dtype=np.float32) for var in target_vars_local}
    baseline_ds = compute_baseline(
        eval_inputs,
        vars_to_perturb,
        cfg.BASELINE_MODE,
        center_lat=center_lat,
        center_lon=center_lon,
        inner_deg=LOCAL_BASELINE_INNER_DEG,
        outer_deg=LOCAL_BASELINE_OUTER_DEG,
        min_points=LOCAL_BASELINE_MIN_POINTS,
    )
    if cfg.BASELINE_MODE.startswith("local_annulus"):
        print(
            "Local baseline annulus: "
            f"inner={LOCAL_BASELINE_INNER_DEG:.2f}deg, "
            f"outer={LOCAL_BASELINE_OUTER_DEG:.2f}deg, "
            f"min_points={LOCAL_BASELINE_MIN_POINTS}"
        )
    time_sel = slice(None) if cfg.PERTURB_TIME == "all" else int(cfg.PERTURB_TIME)

    print("Scanning perturbations...")
    for i, lat_idx in enumerate(lat_indices):
        for j, lon_idx in enumerate(lon_indices):
            lat_start = max(lat_idx - cfg.PATCH_RADIUS, 0)
            lat_end = min(lat_idx + cfg.PATCH_RADIUS + 1, len(lat_vals))
            lon_start = max(lon_idx - cfg.PATCH_RADIUS, 0)
            lon_end = min(lon_idx + cfg.PATCH_RADIUS + 1, len(lon_vals))
            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

            saved_values = {}
            for var in vars_to_perturb:
                da = eval_inputs[var]
                level_sel = resolve_level_sel(da, cfg.PERTURB_LEVELS)
                idx = build_indexer(da, lat_slice, lon_slice, time_sel, level_sel)
                base_idx = build_baseline_indexer(da, time_sel, level_sel)
                arr = da.values
                base_arr = baseline_ds[var].values
                saved_values[var] = (idx, arr[idx].copy())
                base_vals = base_arr[base_idx]
                base_vals = _match_shape(base_vals, arr[idx].shape)
                arr[idx] = np.broadcast_to(base_vals, arr[idx].shape)

            outputs = run_forward_jitted(
                rng=jax.random.PRNGKey(0),
                inputs=eval_inputs,
                targets_template=targets_template,
                forcings=eval_forcings,
            )
            for var in target_vars_local:
                out_var = _select_target_data(outputs, var)
                new_value = out_var.isel(time=cfg.TARGET_TIME_IDX).sel(
                    lat=center_lat, lon=center_lon, method="nearest"
                ).values.item()
                maps[var][i, j] = new_value - base_values[var]

            for var, (idx, old_vals) in saved_values.items():
                eval_inputs[var].values[idx] = old_vals

        if (i + 1) % 5 == 0:
            print(f"progress: {i + 1}/{len(lat_indices)} rows")
    return maps


def _compute_gradient_importance(target_vars_local):
    """
    使用 Integrated Gradients 计算重要性
    IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂f/∂x_i(x' + α(x - x')) dα
    """
    maps = {var: np.zeros((len(lat_indices), len(lon_indices)), dtype=np.float32) for var in target_vars_local}
    
    # 确定要计算IG的输入变量
    if IMPORTANCE_MODE == "compare":
        vars_to_grad = [cfg.TARGET_VARIABLE]
    elif GRADIENT_VARIABLES is None:
        vars_to_grad = vars_to_perturb
    else:
        vars_to_grad = [v for v in GRADIENT_VARIABLES if v in eval_inputs.data_vars]

    if not vars_to_grad:
        raise ValueError("No gradient variables found")

    # IG配置
    IG_STEPS = 50  # 积分步数
    
    print(f"\n=== Integrated Gradients Computation ===")
    print(f"IG variables: {vars_to_grad}")
    print(f"IG steps: {IG_STEPS}")
    print(f"Baseline: global median")

    # 为每个输入变量创建baseline
    baseline_inputs = eval_inputs.copy(deep=False)
    for var_name in vars_to_grad:
        original_da = eval_inputs[var_name]
        median_val = float(np.median(original_da.values))
        baseline_array = np.full_like(original_da.values, median_val)
        baseline_da = xarray.DataArray(
            baseline_array,
            dims=original_da.dims,
            coords=original_da.coords,
            attrs=original_da.attrs
        )
        baseline_inputs[var_name] = baseline_da
        print(f"  {var_name}: baseline = {median_val:.2f}")

    def _target_scalar(inputs_data, target_var):
        outputs = run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs_data,
            targets_template=targets_template,
            forcings=eval_forcings,
        )
        out_var = _select_target_data(outputs, target_var)
        value = out_var.isel(time=cfg.TARGET_TIME_IDX).sel(
            lat=center_lat, lon=center_lon, method="nearest"
        )
        if "batch" in value.dims:
            value = value.isel(batch=0)
        scalar = xarray_jax.unwrap_data(value, require_jax=True)
        return jnp.squeeze(scalar)

    # 对每个目标变量计算IG
    for target_var in target_vars_local:
        print(f"\nComputing IG for target: {target_var}")
        rhs = float(np.array(_target_scalar(eval_inputs, target_var) - _target_scalar(baseline_inputs, target_var)))
        lhs_total = 0.0
        
        # 创建梯度函数
        def _loss(inputs_data):
            return _target_scalar(inputs_data, target_var)
        grad_fn = jax.grad(_loss)
        
        # 对每个输入变量计算IG
        for var_name in vars_to_grad:
            print(f"  Variable: {var_name}")
            accumulated_grads = None
            
            # 沿路径积分
            for step in range(IG_STEPS):
                alpha = (step + 1) / IG_STEPS
                
                # 创建插值输入
                interpolated_inputs = baseline_inputs.copy(deep=False)
                original_da = eval_inputs[var_name]
                baseline_da = baseline_inputs[var_name]
                
                interp_array = np.array(baseline_da.values) + alpha * (
                    np.array(original_da.values) - np.array(baseline_da.values)
                )
                
                interp_da = xarray.DataArray(
                    interp_array,
                    dims=original_da.dims,
                    coords=original_da.coords,
                    attrs=original_da.attrs
                )
                interpolated_inputs[var_name] = interp_da
                
                # 计算梯度
                grads = grad_fn(interpolated_inputs)
                grad_da = grads[var_name]
                
                if accumulated_grads is None:
                    accumulated_grads = np.array(grad_da.values)
                else:
                    accumulated_grads += np.array(grad_da.values)
                
                if (step + 1) % 10 == 0:
                    print(f"    Step: {step + 1}/{IG_STEPS}")
            
            # 计算IG归因：(x - baseline) * avg_gradient
            avg_grads = accumulated_grads / IG_STEPS
            diff = np.array(eval_inputs[var_name].values) - np.array(baseline_inputs[var_name].values)
            ig_attr = diff * avg_grads

            # --- Reduce non-spatial dims by name (robust to dim order) ---
            # The raw numpy array follows the input DataArray's dim order.
            # Using xarray avoids accidentally indexing the wrong axis
            # (e.g. selecting batch/level instead of lat).
            original_da = eval_inputs[var_name]
            ig_da_full = xarray.DataArray(
                ig_attr,
                dims=original_da.dims,
                coords=original_da.coords,
                attrs=original_da.attrs,
            )
            lhs_total += float(np.array(ig_da_full.sum().values))
            ig_da = ig_da_full

            if "batch" in ig_da.dims:
                ig_da = ig_da.isel(batch=0)

            if "time" in ig_da.dims:
                if cfg.PERTURB_TIME == "all":
                    if GRADIENT_TIME_AGG == "mean":
                        ig_da = ig_da.mean(dim="time")
                    else:
                        ig_da = ig_da.isel(time=0)
                else:
                    ig_da = ig_da.isel(time=int(cfg.PERTURB_TIME))

            if "level" in ig_da.dims:
                level_sel = resolve_level_sel(original_da, cfg.PERTURB_LEVELS)
                ig_da = ig_da.isel(level=level_sel)
                # Always collapse level to produce a 2D (lat, lon) map.
                if "level" in ig_da.dims:
                    ig_da = ig_da.mean(dim="level")

            # 提取区域
            ig_da = ig_da.transpose("lat", "lon")
            ig_region = ig_da.isel(lat=lat_indices, lon=lon_indices).values
            maps[target_var] += ig_region
            
            # 统计信息
            q50 = float(np.percentile(np.abs(ig_region.ravel()), 50))
            q90 = float(np.percentile(np.abs(ig_region.ravel()), 90))
            q99 = float(np.percentile(np.abs(ig_region.ravel()), 99))
            print(f"    IG |attribution| percentiles -> 50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}")

        rel_err = abs(lhs_total - rhs) / (abs(rhs) + 1e-8)
        print(
            "  IG completeness: "
            f"lhs={lhs_total:.6e}, rhs={rhs:.6e}, rel_err={rel_err:.6%}"
        )

    # 多变量平均
    maps = {var: (vals / float(len(vars_to_grad))) for var, vals in maps.items()}
    
    print(f"\n=== Final IG Attribution Maps ===")
    for var, val_map in maps.items():
        val_flat = val_map.ravel()
        q50 = float(np.percentile(np.abs(val_flat), 50))
        q90 = float(np.percentile(np.abs(val_flat), 90))
        q99 = float(np.percentile(np.abs(val_flat), 99))
        print(f"  {var}: final |IG| percentiles -> 50%={q50:.6e}, 90%={q90:.6e}, 99%={q99:.6e}")
    
    return maps


compare_maps = None
if IMPORTANCE_MODE == "perturbation":
    importance_maps = _compute_perturbation_importance(target_vars)
elif IMPORTANCE_MODE == "input_gradient":
    importance_maps = _compute_gradient_importance(target_vars)
elif IMPORTANCE_MODE == "compare":
    if len(target_vars) != 1:
        raise ValueError("IMPORTANCE_MODE='compare' requires exactly one target variable")
    perturb_maps = _compute_perturbation_importance(target_vars)
    gradient_maps = _compute_gradient_importance(target_vars)
    compare_maps = {
        "perturbation": perturb_maps,
        "input_gradient": gradient_maps,
    }
    importance_maps = perturb_maps
else:
    raise ValueError(f"Unknown IMPORTANCE_MODE: {IMPORTANCE_MODE}")

# %% Save results and plots
lat_sel_vals = lat_vals[lat_indices]
lon_sel_vals = lon_vals[lon_indices]

importance_das = {}
if IMPORTANCE_MODE == "compare":
    var = target_vars[0]
    importance_das["perturbation"] = xarray.DataArray(
        compare_maps["perturbation"][var],
        dims=("lat", "lon"),
        coords={"lat": lat_sel_vals, "lon": lon_sel_vals},
        name="perturbation_importance",
    )
    importance_das["input_gradient"] = xarray.DataArray(
        compare_maps["input_gradient"][var],
        dims=("lat", "lon"),
        coords={"lat": lat_sel_vals, "lon": lon_sel_vals},
        name="gradient_importance",
    )
else:
    for var in target_vars:
        name = "importance" if len(target_vars) == 1 else f"importance_{var}"
        importance_das[name] = xarray.DataArray(
            importance_maps[var],
            dims=("lat", "lon"),
            coords={"lat": lat_sel_vals, "lon": lon_sel_vals},
            name=name,
        )

if IMPORTANCE_MODE == "compare":
    var = target_vars[0]
    if OUTPUT_PNG_METHOD_COMPARE:
        compare_path = ROOT_DIR / OUTPUT_PNG_METHOD_COMPARE
        time_label = f"out_t={cfg.TARGET_TIME_IDX}, in_t={cfg.PERTURB_TIME}"
        titles = [
            f"Perturbation ({var}, {time_label})",
            f"IG ({var}, {time_label})",
        ]
        gradient_label = "IG attribution"
        plot_importance_heatmap_dual(
            [importance_das["perturbation"], importance_das["input_gradient"]],
            titles,
            center_lat,
            center_lon,
            compare_path,
            cmap=[cfg.HEATMAP_CMAP, GRADIENT_CMAP],
            dpi=cfg.HEATMAP_DPI,
            vmax_quantile=[cfg.HEATMAP_VMAX_QUANTILE, GRADIENT_VMAX_QUANTILE],
            diverging=[HEATMAP_DIVERGING, GRADIENT_DIVERGING],
            cbar_label=["Δoutput (perturbed - baseline)", gradient_label],
            center_window_deg=[0.0, GRADIENT_CENTER_WINDOW_DEG],
            center_s_quantile=[cfg.HEATMAP_VMAX_QUANTILE, GRADIENT_CENTER_SCALE_QUANTILE],
            alpha_quantile=[None, GRADIENT_ALPHA_QUANTILE],
        )
        print(f"Saved method-compare heatmap: {compare_path}")
    else:
        print("Skip method-compare heatmap: OUTPUT_PNG_METHOD_COMPARE is None")
elif len(target_vars) == 1 and cfg.OUTPUT_PNG:
    var = target_vars[0]
    da_name = "importance"
    png_path = ROOT_DIR / cfg.OUTPUT_PNG
    if IMPORTANCE_MODE == "input_gradient":
        title = f"IG Importance (t={cfg.TARGET_TIME_IDX}, var={var})"
        cbar_label = "IG attribution"
        vmax_quantile = GRADIENT_VMAX_QUANTILE
    else:
        title = f"Perturbation Importance (t={cfg.TARGET_TIME_IDX}, var={var})"
        cbar_label = None
        vmax_quantile = cfg.HEATMAP_VMAX_QUANTILE

    plot_importance_heatmap(
        importance_das[da_name],
        center_lat,
        center_lon,
        png_path,
        title,
        cmap=GRADIENT_CMAP if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_CMAP,
        dpi=cfg.HEATMAP_DPI,
        vmax_quantile=vmax_quantile,
        diverging=GRADIENT_DIVERGING if IMPORTANCE_MODE == "input_gradient" else HEATMAP_DIVERGING,
        cbar_label=cbar_label,
        center_window_deg=GRADIENT_CENTER_WINDOW_DEG if IMPORTANCE_MODE == "input_gradient" else 0.0,
        center_s_quantile=GRADIENT_CENTER_SCALE_QUANTILE if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_VMAX_QUANTILE,
        alpha_quantile=GRADIENT_ALPHA_QUANTILE if IMPORTANCE_MODE == "input_gradient" else None,
    )
    print(f"Saved heatmap: {png_path}")

if IMPORTANCE_MODE != "compare" and len(target_vars) == 2 and OUTPUT_PNG_COMBINED:
    da_list = [importance_das[f"importance_{target_vars[0]}"], importance_das[f"importance_{target_vars[1]}"]]
    titles = [
        f"{target_vars[0]} (t={cfg.TARGET_TIME_IDX})",
        f"{target_vars[1]} (t={cfg.TARGET_TIME_IDX})",
    ]
    if IMPORTANCE_MODE == "input_gradient":
        cbar_label = "IG attribution"
        vmax_quantile = GRADIENT_VMAX_QUANTILE
    else:
        cbar_label = None
        vmax_quantile = cfg.HEATMAP_VMAX_QUANTILE

    combined_path = ROOT_DIR / OUTPUT_PNG_COMBINED
    plot_importance_heatmap_dual(
        da_list,
        titles,
        center_lat,
        center_lon,
        combined_path,
        cmap=GRADIENT_CMAP if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_CMAP,
        dpi=cfg.HEATMAP_DPI,
        vmax_quantile=vmax_quantile,
        diverging=GRADIENT_DIVERGING if IMPORTANCE_MODE == "input_gradient" else HEATMAP_DIVERGING,
        cbar_label=cbar_label,
        center_window_deg=GRADIENT_CENTER_WINDOW_DEG if IMPORTANCE_MODE == "input_gradient" else 0.0,
        center_s_quantile=GRADIENT_CENTER_SCALE_QUANTILE if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_VMAX_QUANTILE,
        alpha_quantile=GRADIENT_ALPHA_QUANTILE if IMPORTANCE_MODE == "input_gradient" else None,
    )
    print(f"Saved combined heatmap: {combined_path}")
if IMPORTANCE_MODE != "compare" and OUTPUT_PNG_CARTOPY and len(target_vars) == 1:
    map_path = ROOT_DIR / OUTPUT_PNG_CARTOPY
    if IMPORTANCE_MODE == "input_gradient":
        title = f"IG Importance Map (t={cfg.TARGET_TIME_IDX}, var={target_vars[0]})"
        cbar_label = "IG attribution"
        vmax_quantile = GRADIENT_VMAX_QUANTILE
    else:
        title = f"Perturbation Importance Map (t={cfg.TARGET_TIME_IDX}, var={target_vars[0]})"
        cbar_label = None
        vmax_quantile = cfg.HEATMAP_VMAX_QUANTILE
    try:
        plot_importance_heatmap_cartopy(
            importance_das["importance"],
            center_lat,
            center_lon,
            map_path,
            title,
            cmap=GRADIENT_CMAP if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_CMAP,
            dpi=cfg.HEATMAP_DPI,
            vmax_quantile=vmax_quantile,
            diverging=GRADIENT_DIVERGING if IMPORTANCE_MODE == "input_gradient" else HEATMAP_DIVERGING,
            cbar_label=cbar_label,
            center_window_deg=GRADIENT_CENTER_WINDOW_DEG if IMPORTANCE_MODE == "input_gradient" else 0.0,
            center_s_quantile=GRADIENT_CENTER_SCALE_QUANTILE if IMPORTANCE_MODE == "input_gradient" else cfg.HEATMAP_VMAX_QUANTILE,
            alpha_quantile=GRADIENT_ALPHA_QUANTILE if IMPORTANCE_MODE == "input_gradient" else None,
        )
        print(f"Saved map heatmap: {map_path}")
    except RuntimeError as exc:
        print(f"Skip cartopy map: {exc}")

# %% Top-N
if IMPORTANCE_MODE == "compare":
    var = target_vars[0]
    for mode_name, mode_maps in (("Perturbation", compare_maps["perturbation"]), ("IG", compare_maps["input_gradient"])):
        flat_idx = np.argsort(np.abs(mode_maps[var]).ravel())[::-1][:cfg.TOP_N]
        print(f"\nTop-{cfg.TOP_N} influential grid points for {var} ({mode_name}):")
        for rank, idx in enumerate(flat_idx, start=1):
            r = idx // mode_maps[var].shape[1]
            c = idx % mode_maps[var].shape[1]
            lat = float(lat_sel_vals[r])
            lon = float(lon_sel_vals[c])
            score = float(mode_maps[var][r, c])
            print(f"{rank:02d}. lat={lat:.2f}, lon={lon:.2f}, score={score:.6f}")
else:
    for var in target_vars:
        flat_idx = np.argsort(np.abs(importance_maps[var]).ravel())[::-1][:cfg.TOP_N]
        print(f"\nTop-{cfg.TOP_N} influential grid points for {var}:")
        for rank, idx in enumerate(flat_idx, start=1):
            r = idx // importance_maps[var].shape[1]
            c = idx % importance_maps[var].shape[1]
            lat = float(lat_sel_vals[r])
            lon = float(lon_sel_vals[c])
            score = float(importance_maps[var][r, c])
            print(f"{rank:02d}. lat={lat:.2f}, lon={lon:.2f}, score={score:.6f}")


# %% Top-N
print("done")
