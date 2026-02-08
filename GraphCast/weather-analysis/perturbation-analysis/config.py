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
TARGET_LEVEL = 500  # used only if TARGET_VARIABLE has a level dimension
TARGET_VARIABLES = ["mean_sea_level_pressure", "geopotential"]
TARGET_LEVELS = {"geopotential": 500}
REGION_RADIUS_DEG = 15
PATCH_RADIUS = 0  # 0=single grid, 1=3x3 patch
PERTURB_TIME = "all"  # "all" or 0/1
PERTURB_VARIABLES = None  # None = all vars with lat/lon
PERTURB_LEVELS = None  # None = all levels
BASELINE_MODE = "spatial_mean"  # "spatial_mean" | "spatial_median"
TOP_N = 20
OUTPUT_NC = "perturbation_importance.nc"
OUTPUT_PNG = "perturbation_importance.png"
OUTPUT_PNG_CARTOPY = "perturbation_importance_map.png"
OUTPUT_PNG_COMBINED = "perturbation_importance_dual.png"
HEATMAP_DPI = 200
HEATMAP_CMAP = "coolwarm"
HEATMAP_VMAX_QUANTILE = 0.995
HEATMAP_DIVERGING = True

# Importance computation mode
# - "perturbation": occlusion-based delta output (original behavior)
# - "input_gradient": input saliency |d output / d input|
IMPORTANCE_MODE = "perturbation"

# Input-gradient options
# - "abs": magnitude-only, non-negative
# - "signed": signed gradient (use diverging colormap)
GRADIENT_MODE = "abs"

# When True, uses gradient * input instead of raw gradients
GRADIENT_X_INPUT = False

# If set, only accumulate gradients from these variables (None = all vars with lat/lon)
GRADIENT_VARIABLES = None

# Gradient scaling for visualization (quantile for vmax)
GRADIENT_VMAX_QUANTILE = 0.995

# Paths
DIR_PATH_PARAMS = "/root/data/params"
DIR_PATH_DATASET = "/root/data/dataset"
DIR_PATH_STATS = "/root/data/stats"
