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
TARGET_VARIABLE = "mean_sea_level_pressure"  # or "geopotential"
TARGET_LEVEL = 500  # used only if TARGET_VARIABLE has a level dimension
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
HEATMAP_DPI = 200
HEATMAP_CMAP = "magma"
HEATMAP_VMAX_QUANTILE = 0.995

# Paths
DIR_PATH_PARAMS = "/root/data/params"
DIR_PATH_DATASET = "/root/data/dataset"
DIR_PATH_STATS = "/root/data/stats"
