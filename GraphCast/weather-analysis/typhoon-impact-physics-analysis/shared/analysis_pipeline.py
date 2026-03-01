# -*- coding: utf-8 -*-
"""台风影响分析的核心编排辅助模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AnalysisConfig:
    """当前 IG 候选筛选 + 扰动验证流程所需的最小配置集合。"""

    dataset_configs: Dict[str, Dict[str, str]]
    dataset_type: str
    target_time_idx: int
    target_variable: str
    target_level: Any
    patch_radius: int
    patch_score_agg: str
    perturb_time: Any
    perturb_variables: Optional[List[str]]
    perturb_levels: Any
    baseline_mode: str
    local_baseline_inner_deg: float
    local_baseline_outer_deg: float
    local_baseline_min_points: int
    heatmap_dpi: int
    gradient_steps: int
    include_target_inputs: bool
    gradient_vmax_quantile: float
    gradient_cmap: str
    gradient_center_window_deg: float
    gradient_center_scale_quantile: float
    gradient_alpha_quantile: Optional[float]
    gradient_time_agg: str
    dir_path_params: str
    dir_path_dataset: str
    dir_path_stats: str
    ig_sanity_enable: bool
    ig_sanity_topk: int
    ig_sanity_random_k: int
    ig_sanity_seed: int
    ig_sanity_min_lift_ratio: float

    @classmethod
    def from_module(cls, cfg_module) -> "AnalysisConfig":
        return cls(
            dataset_configs=cfg_module.DATASET_CONFIGS,
            dataset_type=cfg_module.DATASET_TYPE,
            target_time_idx=int(cfg_module.TARGET_TIME_IDX),
            target_variable=cfg_module.TARGET_VARIABLE,
            target_level=getattr(cfg_module, "TARGET_LEVEL", None),
            patch_radius=int(cfg_module.PATCH_RADIUS),
            patch_score_agg=str(getattr(cfg_module, "PATCH_SCORE_AGG", "mean")),
            perturb_time=cfg_module.PERTURB_TIME,
            perturb_variables=getattr(cfg_module, "PERTURB_VARIABLES", None),
            perturb_levels=getattr(cfg_module, "PERTURB_LEVELS", None),
            baseline_mode=cfg_module.BASELINE_MODE,
            local_baseline_inner_deg=float(getattr(cfg_module, "LOCAL_BASELINE_INNER_DEG", 5.0)),
            local_baseline_outer_deg=float(getattr(cfg_module, "LOCAL_BASELINE_OUTER_DEG", 12.0)),
            local_baseline_min_points=int(getattr(cfg_module, "LOCAL_BASELINE_MIN_POINTS", 120)),
            heatmap_dpi=int(cfg_module.HEATMAP_DPI),
            gradient_steps=int(getattr(cfg_module, "IG_STEPS", 50)),
            include_target_inputs=bool(getattr(cfg_module, "INCLUDE_TARGET_INPUTS", False)),
            gradient_vmax_quantile=float(getattr(cfg_module, "GRADIENT_VMAX_QUANTILE", 0.995)),
            gradient_cmap=getattr(cfg_module, "GRADIENT_CMAP", "RdBu_r"),
            gradient_center_window_deg=float(getattr(cfg_module, "GRADIENT_CENTER_WINDOW_DEG", 10.0)),
            gradient_center_scale_quantile=float(getattr(cfg_module, "GRADIENT_CENTER_SCALE_QUANTILE", 0.99)),
            gradient_alpha_quantile=getattr(cfg_module, "GRADIENT_ALPHA_QUANTILE", 0.90),
            gradient_time_agg=getattr(cfg_module, "GRADIENT_TIME_AGG", "single"),
            dir_path_params=cfg_module.DIR_PATH_PARAMS,
            dir_path_dataset=cfg_module.DIR_PATH_DATASET,
            dir_path_stats=cfg_module.DIR_PATH_STATS,
            ig_sanity_enable=bool(getattr(cfg_module, "IG_SANITY_ENABLE", True)),
            ig_sanity_topk=int(getattr(cfg_module, "IG_SANITY_TOPK", 10)),
            ig_sanity_random_k=int(getattr(cfg_module, "IG_SANITY_RANDOM_K", 10)),
            ig_sanity_seed=int(getattr(cfg_module, "IG_SANITY_SEED", 42)),
            ig_sanity_min_lift_ratio=float(getattr(cfg_module, "IG_SANITY_MIN_LIFT_RATIO", 1.1)),
        )


@dataclass
class AnalysisContext:
    dataset_config: Dict[str, str]
    run_forward_jitted: Any
    eval_inputs: Any
    eval_targets: Any
    eval_forcings: Any
    targets_template: Any
    center_lat: float
    center_lon: float
    target_var: str
    base_value: float
    lat_vals: Any
    lon_vals: Any


def resolve_spatial_variables(
    perturb_variables: Optional[List[str]],
    eval_inputs,
) -> List[str]:
    """解析用于归因/扰动的输入变量集合。
    
    Only includes variables that:
    - Are present in eval_inputs
    - Have both 'lat' and 'lon' dimensions
    """
    # Get all variables with lat/lon dims
    spatial_vars = {
        var_name
        for var_name, data_array in eval_inputs.data_vars.items()
        if ("lat" in data_array.dims and "lon" in data_array.dims)
    }
    
    if perturb_variables is None:
        # Return all spatial vars that exist in eval_inputs
        return [var_name for var_name in eval_inputs.data_vars if var_name in spatial_vars]
    
    # Return only specified vars that are both in perturb_variables AND have lat/lon dims in eval_inputs
    return [var_name for var_name in perturb_variables if var_name in eval_inputs.data_vars and var_name in spatial_vars]


def select_target_data(outputs, var: str, target_level: Any):
    data = outputs[var]
    if "level" in data.dims:
        if target_level is not None:
            data = data.sel(level=target_level)
    return data


def build_analysis_context(runtime_cfg: AnalysisConfig) -> AnalysisContext:
    """一次性加载模型/数据，为后续归因方法准备上下文。"""
    import jax
    import numpy as np

    from cyclone_points import pick_target_cyclone
    from shared.model_utils import (
        build_run_forward,
        extract_eval_data,
        load_checkpoint,
        load_dataset,
        load_normalization_stats,
    )

    if runtime_cfg.dataset_type not in runtime_cfg.dataset_configs:
        raise ValueError(f"Invalid DATASET_TYPE: {runtime_cfg.dataset_type}")

    dataset_config = runtime_cfg.dataset_configs[runtime_cfg.dataset_type]
    print(f"\n=== Config: {dataset_config['name']} ===")
    print(f"target_time_idx: {runtime_cfg.target_time_idx}")
    print(f"patch_radius: {runtime_cfg.patch_radius}")

    ckpt = load_checkpoint(f"{runtime_cfg.dir_path_params}/{dataset_config['params_file']}")
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    example_batch = load_dataset(f"{runtime_cfg.dir_path_dataset}/{dataset_config['dataset_file']}")
    eval_inputs, eval_targets, eval_forcings = extract_eval_data(example_batch, task_config)
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_stats(
        runtime_cfg.dir_path_stats
    )

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

    target_cyclone = pick_target_cyclone(runtime_cfg.target_time_idx)
    center_lat = float(target_cyclone["lat"])
    center_lon = float(target_cyclone["lon"])

    targets_template = eval_targets * np.nan
    base_outputs = run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=targets_template,
        forcings=eval_forcings,
    )

    target_var = runtime_cfg.target_variable
    print(f"target_variable: {target_var}")

    spatial_vars = resolve_spatial_variables(runtime_cfg.perturb_variables, eval_inputs)
    if not spatial_vars:
        raise ValueError("No spatial input variables found")
    print(f"spatial input vars: {len(spatial_vars)}")

    target_data = select_target_data(
        base_outputs,
        target_var,
        target_level=runtime_cfg.target_level,
    )
    base_value = target_data.isel(time=runtime_cfg.target_time_idx).sel(
        lat=center_lat,
        lon=center_lon,
        method="nearest",
    ).values.item()
    print(f"Baseline {target_var}: {base_value:.4f} at ({center_lat:.2f}, {center_lon:.2f})")

    lat_vals = eval_inputs.coords["lat"].values
    lon_vals = eval_inputs.coords["lon"].values

    return AnalysisContext(
        dataset_config=dataset_config,
        run_forward_jitted=run_forward_jitted,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        eval_forcings=eval_forcings,
        targets_template=targets_template,
        center_lat=center_lat,
        center_lon=center_lon,
        target_var=target_var,
        base_value=base_value,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
    )
