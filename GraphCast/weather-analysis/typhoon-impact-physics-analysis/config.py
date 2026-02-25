# -*- coding: utf-8 -*-
"""台风关键网格点（IG 候选 + 扰动验证）分析配置。"""

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

# 数据与目标配置
DATASET_TYPE = "low_res"  # "low_res" | "high_res"
TARGET_TIME_IDX = 0  # 0(+6h),1(+12h),2(+18h),3(+24h)
TARGET_VARIABLE = "mean_sea_level_pressure"
TARGET_VARIABLES = None  # None = 使用 TARGET_VARIABLE；可设为多个目标变量做平均目标

# 若目标变量有 level 维，可在此指定；若无 level 维则忽略。
TARGET_LEVEL = None
TARGET_LEVELS = {}

# 扰动设置（用于 Top-K 候选点的干预验证）
PATCH_RADIUS = 2
PATCH_SCORE_AGG = "mean"
PERTURB_TIME = "all"  # "all" 或 0/1
PERTURB_VARIABLES = None  # None = 所有含经纬度的变量
PERTURB_LEVELS = None  # None = 所有气压层

INCLUDE_TARGET_INPUTS = False

# 基线设置：IG 与扰动验证统一使用
BASELINE_MODE = "local_annulus_median"  # 可选: "spatial_mean" | "spatial_median" | "local_annulus_mean" | "local_annulus_median"
LOCAL_BASELINE_INNER_DEG = 5.0
LOCAL_BASELINE_OUTER_DEG = 12.0
LOCAL_BASELINE_MIN_POINTS = 120

# 可视化基础参数
HEATMAP_DPI = 200

# IG 参数
IG_STEPS = 50
GRADIENT_CMAP = "RdBu_r"
GRADIENT_CENTER_WINDOW_DEG = 10.0
GRADIENT_CENTER_SCALE_QUANTILE = 0.99
GRADIENT_ALPHA_QUANTILE = 0.90
GRADIENT_VMAX_QUANTILE = 0.90
GRADIENT_TIME_AGG = "single"  # "single" 或 "mean"，仅在 PERTURB_TIME="all" 时生效

# 路径
DIR_PATH_PARAMS = "/root/data/params"
DIR_PATH_DATASET = "/root/data/dataset"
DIR_PATH_STATS = "/root/data/stats"

# SWE 物理敏感性 & 对齐分析
SWE_DOMAIN_HALF_DEG = 20.0      # 台风中心 ±20°（40°×40° 子域）
SWE_SIGMA_DEG = 3.0             # 高斯目标函数 J 的标准差（度）
SWE_DT = 300.0                  # 时间步长（秒），满足 CFL 条件
SWE_CORE_RADIUS_DEG = 3.0       # 引导气流计算时挖掉台风内核的半径（度）
# "none" (original) | "geostrophic_hard"
SWE_CONSTRAINT_MODE = "geostrophic_hard"

PHYSICS_TOPK_VALUES = [20, 50, 100, 200]
PHYSICS_PATCH_RADIUS = 2        # 与 PATCH_RADIUS 保持一致
PHYSICS_PATCH_SCORE_AGG = "mean"

PHYSICS_HEATMAP_DPI = 200
