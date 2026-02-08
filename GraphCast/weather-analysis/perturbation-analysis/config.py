# -*- coding: utf-8 -*-
"""Configuration for cyclone perturbation analysis."""

DATASET_CONFIGS = {
    "low_res": {
        "name": "Low-res (1.0deg, 13 levels)",
        "params_file": "params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
        "dataset_file": "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
    },
    "high_res": {
        "name": "High-res (0.25deg, 37 levels)",
        "params_file": "params-GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
        "dataset_file": "dataset-source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc",
    },
}

# Experiment config
DATASET_TYPE = "low_res"  # "low_res" | "high_res"
TARGET_TIME_IDX = 0  # 0(+6h),1(+12h),2(+18h),3(+24h)
TARGET_VARIABLE = "mean_sea_level_pressure"  # default target for single-variable runs
TARGET_VARIABLES = None  # set None to use TARGET_VARIABLE; for compare mode this must resolve to exactly one variable
REGION_RADIUS_DEG = 15
PATCH_RADIUS = 0  # 0=single grid, 1=3x3 patch
PERTURB_TIME = "all"  # "all" or 0/1
PERTURB_VARIABLES = None  # None = all vars with lat/lon
PERTURB_LEVELS = None  # None = all levels
BASELINE_MODE = "local_annulus_median"  # "spatial_mean" | "spatial_median" | "local_annulus_mean" | "local_annulus_median"
LOCAL_BASELINE_INNER_DEG = 5.0
LOCAL_BASELINE_OUTER_DEG = 12.0
LOCAL_BASELINE_MIN_POINTS = 120
TOP_N = 20
OUTPUT_PNG_METHOD_COMPARE = "importance_method_compare.png"
HEATMAP_DPI = 200
HEATMAP_CMAP = "coolwarm"
HEATMAP_VMAX_QUANTILE = 0.995
HEATMAP_DIVERGING = True

# IG/gradient visualization (diverging, 0-centered)
GRADIENT_CMAP = "RdBu_r"
GRADIENT_DIVERGING = True
GRADIENT_CENTER_WINDOW_DEG = 10.0
GRADIENT_CENTER_SCALE_QUANTILE = 0.99
GRADIENT_ALPHA_QUANTILE = 0.90

# Importance computation mode
# - "perturbation": occlusion-based delta output (original behavior)
# - "input_gradient": input saliency |d output / d input|
# - "compare": run both methods and draw a side-by-side comparison figure (single target variable only)
IMPORTANCE_MODE = "compare"

# Input-gradient options
# - "abs": magnitude-only, non-negative
# - "signed": signed gradient (use diverging colormap)
GRADIENT_MODE = "abs"

# When True, uses gradient * input instead of raw gradients
GRADIENT_X_INPUT = False

# If set, only accumulate gradients from these variables (None = all vars with lat/lon)
GRADIENT_VARIABLES = None

# Gradient scaling for visualization (quantile for vmax)
# 降低分位数使色标范围更紧凑，让小值也能显示出差异
GRADIENT_VMAX_QUANTILE = 0.90

# Paths
DIR_PATH_PARAMS = "/root/data/params"
DIR_PATH_DATASET = "/root/data/dataset"
DIR_PATH_STATS = "/root/data/stats"
