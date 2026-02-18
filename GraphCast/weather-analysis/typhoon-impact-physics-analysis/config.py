# -*- coding: utf-8 -*-
"""台风影响物理分析的配置文件。"""

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

# 实验配置
DATASET_TYPE = "low_res"  # "low_res" | "high_res"
TARGET_TIME_IDX = 0  # 0(+6h),1(+12h),2(+18h),3(+24h)
TARGET_VARIABLE = "mean_sea_level_pressure"  # 单变量运行的默认目标变量
TARGET_VARIABLES = None  # 设为 None 则使用 TARGET_VARIABLE；比较模式下必须解析为恰好一个变量
REGION_RADIUS_DEG = 15
PATCH_RADIUS = 0  # 0=单格, 1=3x3 区域
PERTURB_TIME = "all"  # "all" 或 0/1
PERTURB_VARIABLES = None  # None = 所有含经纬度的变量
PERTURB_LEVELS = None  # None = 所有气压层
BASELINE_MODE = "local_annulus_median"  # 可选: "spatial_mean" | "spatial_median" | "local_annulus_mean" | "local_annulus_median"
LOCAL_BASELINE_INNER_DEG = 5.0
LOCAL_BASELINE_OUTER_DEG = 12.0
LOCAL_BASELINE_MIN_POINTS = 120
TOP_N = 20
OUTPUT_PNG_METHOD_COMPARE = "importance_method_compare.png"
HEATMAP_DPI = 200
HEATMAP_CMAP = "coolwarm"
HEATMAP_VMAX_QUANTILE = 0.995
HEATMAP_DIVERGING = True

# IG/梯度可视化参数（发散型，以0为中心）
GRADIENT_CMAP = "RdBu_r"
GRADIENT_DIVERGING = True
GRADIENT_CENTER_WINDOW_DEG = 10.0
GRADIENT_CENTER_SCALE_QUANTILE = 0.99
GRADIENT_ALPHA_QUANTILE = 0.90

# 重要性计算模式
# - "perturbation": 基于遮蔽的输出增量（原始行为）
# - "input_gradient": 积分梯度（IG）输入归因
# - "erf": 有效感受野（ERF），即 |d output / d input|
# - "compare": 同时运行 perturbation + IG + ERF 并绘制并排比较图（仅支持单目标变量）
# 提示：独立调试 ERF 图时切换为 "erf"。
IMPORTANCE_MODE = "compare"

# 若设置，则仅从这些变量累积梯度（None = 所有含经纬度的变量）
GRADIENT_VARIABLES = None

# 梯度可视化缩放参数（vmax 的分位数）
# 降低分位数使色标范围更紧凑，让小值也能显示出差异
GRADIENT_VMAX_QUANTILE = 0.90

# ERF 参数
ERF_VARIABLES = None  # None = 所有含经纬度的变量（比较模式下为 TARGET_VARIABLE）
ERF_ABS = True  # 使用 |d output / d input| 幅值
ERF_CMAP = "Blues"
ERF_DIVERGING = False
ERF_CENTER_WINDOW_DEG = 10.0
ERF_CENTER_SCALE_QUANTILE = 0.99
ERF_ALPHA_QUANTILE = 0.90
ERF_VMAX_QUANTILE = 0.995

# 路径
DIR_PATH_PARAMS = "/root/data/params"
DIR_PATH_DATASET = "/root/data/dataset"
DIR_PATH_STATS = "/root/data/stats"
