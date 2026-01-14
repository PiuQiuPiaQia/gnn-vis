# %%
"""
台风物理-AI对齐分析脚本

使用 Matplotlib + Cartopy 绘制台风的物理-AI对齐分析图:
1. 以台风眼为中心截取 ±15度 范围
2. 背景绘制 mean_sea_level_pressure 等压线
3. 叠加梯度热力图 (透明度 0.6)
4. 在台风眼位置绘制逆风向量箭头 (-u, -v)
5. 标注台风眼位置 'X'

目标: 验证梯度热力图高亮区域是否与逆风箭头指向一致

路径: GraphCast/weather-analysis/cyclone_saliency_analysis.py
"""

# %%
# ==================== 导入库 ====================

import sys
from pathlib import Path

# 添加 graphcast 源码路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

# 添加 graphcast-preprocess 路径
PREPROCESS_DIR = SCRIPT_DIR.parent / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

import dataclasses
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray

# 配置中文字体（macOS 使用 PingFang SC）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Cartopy 用于地图绘制
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk

# 导入经纬度转换工具
from latlon_utils import latlon_to_index, index_to_latlon
# 导入区域数据提取工具
from region_utils import extract_region_data, extract_annulus_mean
# 导入滑动窗口梯度分析模块
from sliding_window_saliency import (
    SlidingWindowSaliencyAnalyzer,
    SlidingWindowConfig,
    GradientResult,
    compute_sliding_gradients
)
# 导入台风路径预测模块
from cyclone_track_prediction import (
    extract_cyclone_center_from_prediction,
    predict_cyclone_track
)

print("JAX devices:", jax.devices())

# %%
# ==================== 路径配置 ====================
dir_path_params = "/root/data/params"
dir_path_dataset = "/root/data/dataset"
dir_path_stats = "/root/data/stats"

# 使用小模型以避免内存溢出
params_file = "params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

# 如果内存充足，可以使用高分辨率模型（需要 >32GB GPU 内存）
# params_file = "params-GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
# dataset_file = "dataset-source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc"

# %%
# ==================== 台风眼坐标配置 ====================

# 数据集时间点说明 (dataset-source-era5_date-2022-01-01_...):
# -------------------------------------------------------------------------------
# 原始数据集包含: 2022-01-01 00:00 至 2022-01-02 06:00 (共6个时间点，每6小时)
# 
# GraphCast 输入-预测划分:
#   【输入数据】train_inputs.time = [-6h, 0h]:
#     - 01/01 00:00 (转换后: -6h)  ← 输入时间点1
#     - 01/01 06:00 (转换后:  0h)  ← 输入时间点2（预测参考点，"现在"）
# 
#   【预测目标】train_targets.time = [+6h, +12h, +18h, +24h]:
#     - 01/01 12:00 (转换后: +6h)   ← 预测6小时后
#     - 01/01 18:00 (转换后: +12h)  ← 预测12小时后
#     - 01/02 00:00 (转换后: +18h)  ← 预测18小时后
#     - 01/02 06:00 (转换后: +24h)  ← 预测24小时后
# -------------------------------------------------------------------------------
# 
# 台风 Cyclone Seth 各时间点坐标（从ERA5数据集提取）:
# Date (UTC)  |  Lat      |  Lon      | Pressure (mb) | Wind (kt) | Category | 数据类型
# -----------------------------------------------------------------------------------
# 01/01 00Z   | -21.3138  | 156.6947  |    997.0      |    40     |   TS     | 输入
# 01/01 06Z   | -21.7054  | 157.5024  |    996.0      |    40     |   TS     | 输入（参考点）
# 01/01 12Z   | -22.5048  | 158.2994  |   1000.0      |    35     |   TS     | 预测目标
# 01/01 18Z   | -23.9030  | 158.8031  |    998.0      |    35     |   TS     | 预测目标
# 01/02 00Z   | -25.8032  | 159.0031  |    992.0      |    40     |   TS     | 预测目标
# 01/02 06Z   | (未提供)  | (未提供)  |      -        |     -     |   -      | 预测目标
# 
# 当前配置: 使用输入数据的2个时间点 + 预测目标时间点进行梯度分析
CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.3138, "lon": 156.6947, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "输入(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7054, "lon": 157.5024, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "输入(0h)", "is_input": True, "input_time_idx": 1},
    # 以下是预测目标时间点
    {"time": "2022-01-01 12Z", "lat": -22.5048, "lon": 158.2994, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9030, "lon": 158.8031, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8032, "lon": 159.0031, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "预测(+18h)", "is_input": False, "target_time_idx": 2},
]

# 数据网格分辨率
# GRID_RESOLUTION = 0.25  # 度 (与数据集分辨率一致: res-1.0)
GRID_RESOLUTION = 1
# 注意：如果切换到高分辨率数据，需要改为 GRID_RESOLUTION = 0.25

# 可视化配置
REGION_RADIUS = 15  # 裁剪半径 (度)

# ==================== 梯度计算目标变量选择说明 ====================
# TARGET_VARIABLE: 选择 'geopotential' (位势高度) 的原因：
#
# 1. **经典台风分析指标**: 500hPa 位势高度是气象学中分析中高层大气环流的标准层次
#    - 该层位于对流层中层，对台风的引导气流和发展环境至关重要
#
# 2. **台风特征显著**: 台风在 500hPa 层表现为明显的低值系统
#    - 台风中心处位势高度降低，形成"冷心"结构的一部分
#    - 位势高度梯度反映台风的强度和结构
#
# 3. **物理意义明确**: 
#    - 计算位势高度的梯度可识别哪些输入区域对台风预测影响最大
#    - 负梯度 (NEGATIVE_GRADIENT=True) 表示关注导致位势高度降低的因素
#    - 即：哪些上游区域的输入会导致台风中心位势高度下降（台风加强）
#
# 4. **与台风强度相关**: 500hPa 位势高度降低程度与台风强度正相关
#
# 5. **可解释性强**: 通过梯度热力图可以直观看到模型认为哪些区域的气象条件
#    对台风中心的位势场预测贡献最大
#
# 其他可选变量:
# - 'mean_sea_level_pressure': 海平面气压（台风的直接强度指标）
# - 'temperature': 温度场（反映热力结构）
# - '2m_temperature': 近地面温度（海温影响）
# ============================================================================

TARGET_VARIABLE = 'geopotential'  # 梯度计算的目标变量
TARGET_LEVEL = 500  # 气压层 (hPa) - 对流层中层，台风引导气流的关键层次
NEGATIVE_GRADIENT = True  # True: 关注导致位势高度降低的因素（台风加强相关）

# %%
# ==================== 加载模型 ====================

print("正在加载模型...")
with open(f"{dir_path_params}/{params_file}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params
state = {}
model_config = ckpt.model_config
task_config = ckpt.task_config

print("模型配置:", model_config)

# %%
# ==================== 加载数据集 ====================

print("正在加载数据集...")
with open(f"{dir_path_dataset}/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

print("数据维度:", example_batch.dims.mapping)

# %%
# ==================== 提取评估数据 ====================

# 增加预测步数以获取更多时间点的数据
# 原数据集包含 4 个预测步: +6h, +12h, +18h, +24h
eval_steps = 4

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config)
)

print("Eval Inputs:", eval_inputs.dims.mapping)

# %%
# ==================== 加载归一化统计数据 ====================

print("正在加载归一化统计数据...")

with open(f"{dir_path_stats}/stats-diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(f"{dir_path_stats}/stats-mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open(f"{dir_path_stats}/stats-stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

# %%
# ==================== 构建模型 ====================

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
    """构建并包装 GraphCast 预测器"""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level)
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


def with_params(fn):
    return functools.partial(fn, params=params, state=state)


def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


# JIT 编译
print("正在 JIT 编译模型...")
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
print("模型编译完成!")

# %%
# ==================== 梯度计算函数 ====================

def compute_saliency_map(
    inputs,
    targets,
    forcings,
    target_idx: Tuple[int, int],
    target_variable: str = 'geopotential',
    target_level: int = 500,
    target_time_idx: int = 0,
    negative: bool = True
):
    """
    计算 GraphCast 输入梯度 (Saliency Map)
    Args:
        inputs: 输入数据
        targets: 目标模板
        forcings: 强迫项数据
        target_idx: 目标点索引 (lat_idx, lon_idx)
        target_variable: 目标变量名
        target_level: 目标气压层 (hPa)
        target_time_idx: 预测时间步索引
        negative: True则返回负梯度
    Returns:
        grads: 输入梯度
    """
    lat_idx, lon_idx = target_idx

    def target_loss(inputs_data):
        outputs = run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs_data,
            targets_template=targets * np.nan,
            forcings=forcings
        )

        target_data = outputs[target_variable]

        if 'level' in target_data.dims:
            value = target_data.sel(level=target_level).isel(
                time=target_time_idx, lat=lat_idx, lon=lon_idx
            )
        else:
            value = target_data.isel(
                time=target_time_idx, lat=lat_idx, lon=lon_idx
            )

        if 'batch' in value.dims:
            value = value.isel(batch=0)

        scalar = xarray_jax.unwrap_data(value, require_jax=True)
        scalar = jnp.squeeze(scalar)

        return -scalar if negative else scalar

    grads = jax.grad(target_loss)(inputs)
    return grads


# %%
# ==================== 辅助函数：计算环形区域平均引导风 ====================

def compute_annulus_steering_flow(
    wind_data: xarray.Dataset,
    center_lat: float,
    center_lon: float,
    steering_levels: list,
    inner_radius: float = 2.0,
    outer_radius: float = 5.0,
    time_idx: int = 0,
    use_weighted: bool = False,
    verbose: bool = True
) -> tuple:
    """
    计算台风中心周围环形区域的深层平均引导风场
    
    Args:
        wind_data: 风场数据 (xarray Dataset)，包含 'u_component_of_wind' 和 'v_component_of_wind'
        center_lat: 台风中心纬度
        center_lon: 台风中心经度
        steering_levels: 引导气流的气压层列表 (hPa)
        inner_radius: 环形区域内半径 (度)
        outer_radius: 环形区域外半径 (度)
        time_idx: 时间索引
        use_weighted: 是否使用质量加权平均
        verbose: 是否打印调试信息
    
    Returns:
        (u_mean, v_mean): 环形区域平均的u和v风速分量 (m/s)
    """
    # 提取风场数据
    u_wind_3d = wind_data['u_component_of_wind']
    v_wind_3d = wind_data['v_component_of_wind']
    
    # 处理 Batch 和 Time 维度
    if 'batch' in u_wind_3d.dims:
        u_wind_3d = u_wind_3d.isel(batch=0)
        v_wind_3d = v_wind_3d.isel(batch=0)
    if 'time' in u_wind_3d.dims:
        actual_time_idx = min(time_idx, len(u_wind_3d.time) - 1)
        u_wind_3d = u_wind_3d.isel(time=actual_time_idx)
        v_wind_3d = v_wind_3d.isel(time=actual_time_idx)
    
    # 计算多层环形区域平均
    u_layers = []
    v_layers = []
    weights = []
    
    for level in steering_levels:
        try:
            # 选择气压层
            u_layer = u_wind_3d.sel(level=level, method='nearest')
            v_layer = v_wind_3d.sel(level=level, method='nearest')
            
            # 计算环形区域平均
            u_mean = extract_annulus_mean(u_layer, center_lat, center_lon,
                                         inner_radius, outer_radius)
            v_mean = extract_annulus_mean(v_layer, center_lat, center_lon,
                                         inner_radius, outer_radius)
            
            u_layers.append(u_mean)
            v_layers.append(v_mean)
            
            # 质量权重（可选）
            weight = level / 1000.0 if use_weighted else 1.0
            weights.append(weight)
            
            if verbose:
                print(f"    - {level:4d} hPa: u={u_mean:6.2f}, v={v_mean:6.2f} m/s")
        except Exception as e:
            if verbose:
                print(f"    - {level:4d} hPa: 跳过 (数据不存在)")
    
    # 计算加权平均
    if len(u_layers) > 0:
        u_center = sum(u * w for u, w in zip(u_layers, weights)) / sum(weights)
        v_center = sum(v * w for v, w in zip(v_layers, weights)) / sum(weights)
        
        if verbose:
            import math
            wind_speed = math.sqrt(u_center**2 + v_center**2)
            wind_angle = math.atan2(v_center, u_center) * 180 / math.pi
            if len(steering_levels) == 1:
                print(f"  环形区域 {steering_levels[0]} hPa 引导风: u={u_center:.2f}, v={v_center:.2f} m/s")
            else:
                print(f"  环形区域深层平均引导风 (DLM): u={u_center:.2f}, v={v_center:.2f} m/s")
            print(f"  风速={wind_speed:.1f} m/s, 风向={wind_angle:.0f}°")
        
        return u_center, v_center
    else:
        if verbose:
            print(f"  警告: 无法提取任何气压层数据")
        return None, None


# %%
# ==================== 物理-AI对齐可视化函数 ====================

def plot_physics_ai_alignment(
    cyclone_info: dict,
    gradients,
    era5_data,
    gradient_var: str = '2m_temperature',
    gradient_level: int = 500,
    time_idx: int = 0,
    all_cyclone_centers: Optional[list] = None,
    departure_cyclone_info: Optional[dict] = None,
    predicted_cyclone_centers: Optional[list] = None,
    predicted_wind_data: Optional[xarray.Dataset] = None,
    save_path: Optional[str] = None
):
    """
    绘制物理-AI对齐分析图

    Args:
        cyclone_info: 台风信息字典 {time, lat, lon, intensity} - 预测目标时刻的台风位置
        gradients: 梯度数据 (xarray Dataset)
        era5_data: ERA5 气象数据
        gradient_var: 用于可视化的梯度变量
        time_idx: 时间索引，用于选择引导气流的时间步（即"出发时刻"）
                  - 在滑动窗口中，time_idx=1 表示选择窗口的第二个时间步
                  - 例如：窗口[00Z, 06Z]预测12Z时，time_idx=1 选择06Z的风场
                  - 含义：黄色箭头显示的是台风"出发时刻"的环境引导气流
        all_cyclone_centers: 所有台风中心点列表，用于绘制真实台风路径
        departure_cyclone_info: "出发时刻"的台风信息字典 - 黄色箭头将从这个位置出发
                                如果为None，则使用 cyclone_info（向后兼容）
        predicted_cyclone_centers: 预测的台风中心点列表，用于绘制预测台风路径
        predicted_wind_data: 预测的风场数据 (xarray Dataset)，用于绘制预测点的风向箭头
        save_path: 保存路径
    """
    target_lat = cyclone_info['lat']
    target_lon = cyclone_info['lon']
    time_label = cyclone_info['time']
    
    print(f"\n绘制 {time_label} 的物理-AI对齐分析图...")
    
    # 1. 提取梯度数据
    grad_data = gradients[gradient_var]
    if 'batch' in grad_data.dims:
        grad_data = grad_data.isel(batch=0)
    # 梯度数据的 time 维度是输入历史时间步
    # 重要修复: 使用 time_idx 选择对应的时间步，保证梯度与物理量时间一致
    # - time_idx=0: 使用 00Z 时间步的梯度
    # - time_idx=1: 使用 06Z 时间步的梯度
    if 'time' in grad_data.dims:
        actual_grad_time_idx = min(time_idx, len(grad_data.time) - 1)
        grad_data = grad_data.isel(time=actual_grad_time_idx)
        print(f"  梯度时间步: {actual_grad_time_idx} (共 {len(gradients[gradient_var].time)} 个时间步)")
    
    # 如果有 level 维度,选择与目标变量一致的层次
    # 这样可视化展示的梯度与计算目标物理意义一致
    if 'level' in grad_data.dims:
        # 选择 TARGET_LEVEL (500hPa) 层的梯度,与梯度计算目标一致
        grad_data = grad_data.sel(level=TARGET_LEVEL)
        print(f"  显示 {TARGET_LEVEL}hPa 层的梯度（与目标变量层次一致）")
        # 备选方法: 对所有层求和,显示综合影响
        # grad_data = grad_data.sum(dim='level')
    
    # 保存坐标信息（在 unwrap 之前）
    lat_coords = grad_data.lat
    lon_coords = grad_data.lon
    
    grad_data = xarray_jax.unwrap_data(grad_data)
    if hasattr(grad_data, 'block_until_ready'):
        grad_data.block_until_ready()
    grad_np = np.array(grad_data)
    
    # 2. 提取气压场数据 (使用对应的时间步)
    if 'mean_sea_level_pressure' in era5_data.data_vars:
        pressure_data = era5_data['mean_sea_level_pressure']
        # 降维处理,确保是2D数据
        if 'batch' in pressure_data.dims:
            pressure_data = pressure_data.isel(batch=0)
        if 'time' in pressure_data.dims:
            # 使用最后一个可用时间步（如果time_idx超出范围）
            actual_time_idx = min(time_idx, len(pressure_data.time) - 1)
            pressure_data = pressure_data.isel(time=actual_time_idx)
            if actual_time_idx != time_idx:
                print(f"  警告: time_idx={time_idx} 超出范围，使用 time={actual_time_idx}")
    else:
        # 如果没有海平面气压,使用其他变量替代
        pressure_data = None
    
    # 3. 提取深层平均引导风场数据 (Deep Layer Mean Steering Flow)
    # ============================================================================
    # 气象学标准：JTWC 和 CMA 并非只看单层，而是计算深层平均气流
    # 标准做法：计算 850hPa 到 200hPa 的质量加权平均
    #
    # 为什么需要深层平均？
    # - 850hPa: 低层引导，反映边界层影响
    # - 700hPa: 中低层引导，对弱台风/热带风暴重要
    # - 500hPa: 中层引导，经典分析层
    # - 300hPa: 高层引导，受西风带影响，对强台风路径关键
    # - 200hPa: 高空急流，影响台风移速和强度
    #
    # 公式：V_steering = Σ(V_i × weight_i) / Σ(weight_i)
    # 简化版：V_steering = (V_850 + V_700 + V_500 + V_300 + V_200) / 5
    # ============================================================================

    # 定义引导气流的气压层（单位：hPa）
    # GraphCast 的 13 个气压层: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    #
    # 诊断结果：对于热带风暴（TS），700 hPa 单层最准确（平均误差 14.9°）
    # - 850 hPa: 误差 > 60°（风向完全错误，向西而非向南）
    # - 500-200 hPa: 偏东，与低层风向冲突
    # - 深层平均: 误差 29.5°（几乎是 700 hPa 的 2 倍）
    #
    # 气象学解释：
    # - 弱台风/热带风暴的引导层在 700 hPa（对流层中低层）
    # - 强台风才需要深层平均（200-850 hPa）
    STEERING_LEVELS = [700]  # hPa - 只使用 700 hPa，对热带风暴最准确

    # 可选：质量加权（气压越高，质量越大）
    # 这里使用简单平均，你也可以改为质量加权
    USE_WEIGHTED = False  # True: 质量加权平均, False: 简单平均

    if len(STEERING_LEVELS) == 1:
        print(f"  使用 {STEERING_LEVELS[0]} hPa 引导气流（对热带风暴最准确）")
    else:
        print(f"  计算深层平均引导气流 (DLM): {STEERING_LEVELS} hPa")

    # 注意：这里使用的是带 level 维度的 u/v，而不是 10m_u/v
    u_wind_3d = era5_data['u_component_of_wind']
    v_wind_3d = era5_data['v_component_of_wind']

    # 处理 Batch 和 Time 维度
    if 'batch' in u_wind_3d.dims:
        u_wind_3d = u_wind_3d.isel(batch=0)
        v_wind_3d = v_wind_3d.isel(batch=0)
    if 'time' in u_wind_3d.dims:
        actual_time_idx = min(time_idx, len(u_wind_3d.time) - 1)
        u_wind_3d = u_wind_3d.isel(time=actual_time_idx)
        v_wind_3d = v_wind_3d.isel(time=actual_time_idx)

    # 计算深层平均风场
    if 'level' in u_wind_3d.dims:
        u_layers = []
        v_layers = []
        weights = []

        for level in STEERING_LEVELS:
            try:
                # 选择气压层
                u_layer = u_wind_3d.sel(level=level, method='nearest')
                v_layer = v_wind_3d.sel(level=level, method='nearest')

                u_layers.append(u_layer)
                v_layers.append(v_layer)

                # 质量权重（可选）：气压越大，权重越高
                weight = level / 1000.0 if USE_WEIGHTED else 1.0
                weights.append(weight)

                print(f"    - {level} hPa: 已提取")
            except Exception as e:
                print(f"    - {level} hPa: 跳过 (数据不存在)")

        # 计算加权平均
        if len(u_layers) > 0:
            u_wind = sum(u * w for u, w in zip(u_layers, weights)) / sum(weights)
            v_wind = sum(v * w for v, w in zip(v_layers, weights)) / sum(weights)
            print(f"  ✓ 深层平均完成，使用 {len(u_layers)} 个气压层")
        else:
            # 回退：如果没有数据，使用 500hPa
            print(f"  警告: 无法提取多层数据，回退到 500hPa 单层")
            u_wind = u_wind_3d.sel(level=500, method='nearest')
            v_wind = v_wind_3d.sel(level=500, method='nearest')
    else:
        # Fallback (防守性编程)
        print(f"  警告: 数据无 level 维度")
        u_wind = u_wind_3d
        v_wind = v_wind_3d
    
    # 4. 裁剪到目标区域
    # 修复: 使用保存的坐标信息创建 DataArray
    # 将 JAX 数组转换为 numpy 数组以避免兼容性问题
    grad_np_cpu = np.array(grad_np)
    lat_coords_np = np.array(lat_coords)
    lon_coords_np = np.array(lon_coords)
    
    grad_region, lat_range, lon_range = extract_region_data(
        xarray.DataArray(grad_np_cpu, coords={'lat': lat_coords_np, 'lon': lon_coords_np}, dims=['lat', 'lon']),
        target_lat, target_lon, REGION_RADIUS, GRID_RESOLUTION
    )
    
    if pressure_data is not None:
        pressure_region, _, _ = extract_region_data(
            pressure_data, target_lat, target_lon, REGION_RADIUS, GRID_RESOLUTION
        )
    
    u_region, _, _ = extract_region_data(u_wind, target_lat, target_lon, REGION_RADIUS, GRID_RESOLUTION)
    v_region, _, _ = extract_region_data(v_wind, target_lat, target_lon, REGION_RADIUS, GRID_RESOLUTION)

    # 5. 计算台风中心周围环形区域的深层平均引导风
    # ============================================================================
    # 气象学标准（JTWC/CMA，Holland 1984）：
    # 引导气流 = 台风中心外围环形区域内的深层平均风场
    #
    # 参数设置：
    # - 内半径 3°：排除台风环流核心（避免台风自身环流影响）
    # - 外半径 7°：捕获环境引导气流
    # - 850-200hPa：深层平均（强度越强，层次越深）
    #
    # 物理意义：
    # - 环境风场代表"推动"台风移动的大尺度气流
    # - 台风中心点的风速受台风自身环流影响，不能代表引导气流
    # - 环形区域平均有效排除台风环流，反映真实的环境引导
    # ============================================================================

    INNER_RADIUS = 2.0  # 度（优化后参数，原为 3.0°）
    OUTER_RADIUS = 5.0  # 度（优化后参数，原为 7.0°）

    # 确定环形区域的中心：使用"出发时刻"的台风位置（如果提供）
    if departure_cyclone_info is not None:
        annulus_center_lat = departure_cyclone_info['lat']
        annulus_center_lon = departure_cyclone_info['lon']
        print(f"  环形区域中心: 出发时刻台风位置 ({annulus_center_lat:.2f}°, {annulus_center_lon:.2f}°)")
    else:
        # 向后兼容：如果没有提供，使用预测目标位置
        annulus_center_lat = target_lat
        annulus_center_lon = target_lon
        print(f"  环形区域中心: 预测目标位置 ({annulus_center_lat:.2f}°, {annulus_center_lon:.2f}°)")

    print(f"  计算环形区域平均引导风 (半径 {INNER_RADIUS}°-{OUTER_RADIUS}°, DLM)...")

    # 使用提取的公共函数计算环形区域平均引导风
    u_center, v_center = compute_annulus_steering_flow(
        wind_data=era5_data,
        center_lat=annulus_center_lat,
        center_lon=annulus_center_lon,
        steering_levels=STEERING_LEVELS,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        time_idx=time_idx,
        use_weighted=USE_WEIGHTED,
        verbose=True
    )

    # 调试：计算风向角度
    import math
    wind_angle = math.atan2(v_center, u_center) * 180 / math.pi
    upwind_angle = math.atan2(-v_center, -u_center) * 180 / math.pi
    wind_speed = math.sqrt(u_center**2 + v_center**2)
    print(f"  环境风速大小: {wind_speed:.2f} m/s")
    print(f"  环境风向角度: {wind_angle:.1f}° (0°=正东, 90°=正北)")
    print(f"  引导气流上游方向: {upwind_angle:.1f}° (逆风方向，气流来源)")
    
    # 6. 创建地图
    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 设置地图范围
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    
    # 添加地理要素
    ax.coastlines(resolution='50m', linewidth=1.2, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # 添加经纬度网格
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # 7. 绘制等压线 (如果有)
    if pressure_data is not None:
        lats = pressure_region.lat.values
        lons = pressure_region.lon.values
        pressure_vals = pressure_region.values / 100  # 转换为 hPa
        
        contour_levels = np.arange(np.floor(pressure_vals.min()), np.ceil(pressure_vals.max()), 2)
        cs = ax.contour(lons, lats, pressure_vals, levels=contour_levels, 
                       colors='blue', linewidths=1.5, alpha=0.7, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=9, fmt='%d hPa')
    
    # 8. 叠加梯度热力图
    lats_grad = grad_region.lat.values
    lons_grad = grad_region.lon.values
    grad_vals = grad_region.values

    # 轻微高斯平滑（sigma=0.5，保留细节）
    smooth_grad = gaussian_filter(grad_vals, sigma=0.5)

    # 色标范围：保留层次细节，避免饱和
    vmin, vmax = np.percentile(smooth_grad, [5, 95])
    limit = max(abs(vmin), abs(vmax))

    extent = [lons_grad.min(), lons_grad.max(), lats_grad.min(), lats_grad.max()]
    im = ax.imshow(smooth_grad, extent=extent,
                   origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit,
                   interpolation='bilinear',
                   alpha=0.7, transform=ccrs.PlateCarree(), zorder=2)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(f'Saliency (Gradient of {gradient_level}hPa {gradient_var})', fontsize=11)

    # 9. 绘制台风路径线（真实观测路径）
    if all_cyclone_centers is not None and len(all_cyclone_centers) > 1:
        # 提取所有台风中心点的经纬度（真实观测）
        track_lons = [c['lon'] for c in all_cyclone_centers]
        track_lats = [c['lat'] for c in all_cyclone_centers]

        # 绘制真实台风路径线（紫色实线）
        ax.plot(track_lons, track_lats, color='purple', linewidth=2.5,
               linestyle='-', marker='o', markersize=6, markerfacecolor='white',
               markeredgecolor='purple', markeredgewidth=2,
               transform=ccrs.PlateCarree(), zorder=4, alpha=0.8,
               label='真实台风路径')

        # 在每个点旁边标注时间
        for i, c in enumerate(all_cyclone_centers):
            # 提取时间标签 (例如 "00Z", "06Z")
            time_str = c['time'].split()[-1] if ' ' in c['time'] else c['time']
            ax.text(c['lon'] + 0.5, c['lat'] + 0.5, time_str,
                   fontsize=9, color='purple', fontweight='bold',
                   transform=ccrs.PlateCarree(), zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='purple', alpha=0.7))

    # 9.5 绘制预测台风路径线
    if predicted_cyclone_centers is not None and len(predicted_cyclone_centers) > 0:
        # 提取所有预测台风中心点的经纬度
        pred_track_lons = [c['lon'] for c in predicted_cyclone_centers]
        pred_track_lats = [c['lat'] for c in predicted_cyclone_centers]

        # 绘制预测台风路径线（绿色虚线）
        ax.plot(pred_track_lons, pred_track_lats, color='green', linewidth=2.5,
               linestyle='--', marker='s', markersize=6, markerfacecolor='lightgreen',
               markeredgecolor='green', markeredgewidth=2,
               transform=ccrs.PlateCarree(), zorder=4, alpha=0.8,
               label='AI预测路径')

        # 在每个点旁边标注时间（绿色）
        for i, c in enumerate(predicted_cyclone_centers):
            # 提取时间标签 (例如 "00Z", "06Z")
            time_str = c['time'].split()[-1] if ' ' in c['time'] else c['time']
            ax.text(c['lon'] - 0.8, c['lat'] - 0.8, f"预测\n{time_str}",
                   fontsize=8, color='green', fontweight='bold',
                   transform=ccrs.PlateCarree(), zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='green', alpha=0.7))

    # 10. 绘制当前台风眼标记（预测目标位置）
    ax.scatter(target_lon, target_lat, marker='x', s=400, c='red',
              linewidths=4, transform=ccrs.PlateCarree(), zorder=5,
              label='预测目标台风位置')

    # 10.1 绘制"出发时刻"台风位置标记（如果提供）
    if departure_cyclone_info is not None:
        ax.scatter(departure_cyclone_info['lon'], departure_cyclone_info['lat'],
                  marker='o', s=300, facecolors='none', edgecolors='orange',
                  linewidths=3, transform=ccrs.PlateCarree(), zorder=5,
                  label='出发时刻台风位置')

    # 11. 绘制出发时刻的环境引导气流箭头
    # ============================================================================
    # 【重要定义】黄色箭头的物理含义：
    # - 例如：12Z 台风位置的黄色箭头 = 06Z 时刻的环境场引导气流方向
    # - 含义：在出发的那一刻，物理定律想把它推向哪里
    #
    # 时间对应关系（滑动窗口）：
    # - 窗口1: [00Z, 06Z] → 预测12Z → 黄色箭头显示06Z的风场
    # - 窗口2: [06Z, 12Z] → 预测18Z → 黄色箭头显示12Z的风场
    # - 窗口3: [12Z, 18Z] → 预测次日00Z → 黄色箭头显示18Z的风场
    #
    # 技术实现：
    # - time_idx 参数控制选择哪个时间步的风场
    # - 当前代码使用 physics_time_idx=1，即窗口中第二个时间步（参考点，"现在"）
    # - 环形区域（2-5°）平均风场 = 环境引导气流（排除台风自身环流）
    # - 700 hPa 单层 = 热带风暴的主导引导层（诊断优化结果）
    #
    # 气象学依据：
    # - Holland (1984): 引导气流 = 环形区域平均风场
    # - 诊断结果：700 hPa 平均误差 14.9°（深层平均 29.5°）
    # - 弱台风/热带风暴的引导层在中低层（700 hPa），强台风才需深层平均
    # ============================================================================

    # 确定箭头起点：使用"出发时刻"的台风位置（如果提供）
    if departure_cyclone_info is not None:
        arrow_start_lon = departure_cyclone_info['lon']
        arrow_start_lat = departure_cyclone_info['lat']
        departure_time = departure_cyclone_info.get('time', '?')
        print(f"  箭头起点: 出发时刻 {departure_time} 的台风位置 ({arrow_start_lat:.2f}°, {arrow_start_lon:.2f}°)")
    else:
        # 向后兼容：如果没有提供，使用预测目标位置
        arrow_start_lon = target_lon
        arrow_start_lat = target_lat
        print(f"  箭头起点: 预测目标位置 ({arrow_start_lat:.2f}°, {arrow_start_lon:.2f}°)")

    # 方法：归一化风向 + 固定箭头长度（方向准确，长度统一）
    arrow_length_deg = 5.0  # 箭头固定长度（度）
    wind_magnitude = math.sqrt(u_center**2 + v_center**2)
    if wind_magnitude > 0:
        # 归一化风向，然后缩放到固定长度
        u_norm = u_center / wind_magnitude * arrow_length_deg  # 顺风方向
        v_norm = v_center / wind_magnitude * arrow_length_deg  # 顺风方向
    else:
        u_norm, v_norm = 0, 0

    ax.arrow(arrow_start_lon, arrow_start_lat, u_norm, v_norm,
             head_width=1.0, head_length=1.4, fc='yellow', ec='black',
             linewidth=3.0, transform=ccrs.PlateCarree(), zorder=6,
             label=f'出发时刻引导气流 ({wind_speed:.1f} m/s)')

    print(f"  箭头绘制: 从 ({arrow_start_lon:.1f}, {arrow_start_lat:.1f}) 指向 ({arrow_start_lon+u_norm:.1f}, {arrow_start_lat+v_norm:.1f})")

    # 11.5 绘制当前预测点的风向箭头（蓝色）
    # 使用与黄色箭头相同的方法：环形区域平均（排除台风自身环流）
    # 只绘制当前图对应的预测点，而不是所有预测点
    if predicted_cyclone_centers is not None and predicted_wind_data is not None:
        print(f"\n  绘制当前预测点的风向箭头（使用环形区域平均）...")
        
        # 找到当前图对应的预测点索引
        # 通过匹配经纬度来找到对应的预测点
        current_pred_idx = None
        for i, pred_center in enumerate(predicted_cyclone_centers):
            # 比较时间或位置来匹配（使用时间更准确）
            if pred_center['time'] == time_label:
                current_pred_idx = i
                break
        
        # 如果没有找到匹配的预测点，跳过绘制
        if current_pred_idx is None:
            print(f"    警告: 未找到匹配的预测点（时间={time_label}），跳过蓝色箭头绘制")
        else:
            pred_center = predicted_cyclone_centers[current_pred_idx]
            pred_lat = pred_center['lat']
            pred_lon = pred_center['lon']
            pred_time = pred_center['time']
            
            print(f"    当前预测点: {pred_time} (索引={current_pred_idx})")
            
            # 从预测数据中提取该点的风场
            try:
                # 使用深层平均风场（与引导气流计算方法一致）
                u_wind_pred_full = predicted_wind_data['u_component_of_wind']
                v_wind_pred_full = predicted_wind_data['v_component_of_wind']
                
                # 处理 Batch 维度
                if 'batch' in u_wind_pred_full.dims:
                    u_wind_pred_full = u_wind_pred_full.isel(batch=0)
                    v_wind_pred_full = v_wind_pred_full.isel(batch=0)
                
                # 选择对应的时间步
                if 'time' in u_wind_pred_full.dims and current_pred_idx < len(u_wind_pred_full.time):
                    pred_wind_data_timeselected = predicted_wind_data.isel(time=current_pred_idx) if 'time' in predicted_wind_data.dims else predicted_wind_data
                else:
                    pred_wind_data_timeselected = predicted_wind_data
                
                # 使用提取的公共函数计算环形区域平均引导风（与黄色箭头方法一致）
                u_pred_val, v_pred_val = compute_annulus_steering_flow(
                    wind_data=pred_wind_data_timeselected,
                    center_lat=pred_lat,
                    center_lon=pred_lon,
                    steering_levels=STEERING_LEVELS,
                    inner_radius=INNER_RADIUS,
                    outer_radius=OUTER_RADIUS,
                    time_idx=0,  # 已经选择了时间步，所以这里用0
                    use_weighted=USE_WEIGHTED,
                    verbose=False  # 避免打印过多信息
                )
                
                # 如果成功计算了风场，绘制箭头
                if u_pred_val is not None and v_pred_val is not None:
                    # 计算风速和风向
                    wind_speed_pred = math.sqrt(u_pred_val**2 + v_pred_val**2)
                    wind_angle_pred = math.atan2(v_pred_val, u_pred_val) * 180 / math.pi
                    
                    # 归一化并绘制箭头
                    arrow_length_pred = 4.0  # 预测箭头略短一些
                    if wind_speed_pred > 0:
                        u_norm_pred = u_pred_val / wind_speed_pred * arrow_length_pred
                        v_norm_pred = v_pred_val / wind_speed_pred * arrow_length_pred
                    else:
                        u_norm_pred, v_norm_pred = 0, 0
                    
                    # 绘制蓝色箭头（预测风向）
                    ax.arrow(pred_lon, pred_lat, u_norm_pred, v_norm_pred,
                            head_width=0.8, head_length=1.2, fc='blue', ec='black',
                            linewidth=2.5, transform=ccrs.PlateCarree(), zorder=6,
                            alpha=0.8)
                    
                    print(f"    ✓ 蓝色箭头已绘制:")
                    print(f"      位置: ({pred_lat:.1f}°, {pred_lon:.1f}°)")
                    print(f"      环形区域平均引导风: u={u_pred_val:.2f}, v={v_pred_val:.2f} m/s")
                    print(f"      风速={wind_speed_pred:.1f} m/s, 风向={wind_angle_pred:.0f}°")
                
            except Exception as e:
                print(f"    警告: 无法提取当前预测点的风场数据: {e}")
                import traceback
                traceback.print_exc()

    # 12. 添加标题和图例
    pressure_info = f"{cyclone_info['pressure']} mb" if 'pressure' in cyclone_info else ""
    wind_info = f"{cyclone_info['wind_speed']} kt" if 'wind_speed' in cyclone_info else cyclone_info.get('intensity', '')
    category_info = cyclone_info.get('category', '')
    
    title_parts = [time_label]
    if pressure_info:
        title_parts.append(pressure_info)
    if wind_info:
        title_parts.append(wind_info)
    if category_info:
        title_parts.append(category_info)
    
    ax.set_title(f'物理-AI对齐分析图\n{" | ".join(title_parts)}\n'
                f'位置: ({target_lat:.2f}°, {target_lon:.2f}°)',
                fontsize=14, fontweight='bold', pad=15)
    
    # 创建自定义图例（英文）
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm
    
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=1.5, linestyle='-',
                  label='Isobars (MSLP, Blue Lines)'),
        mpatches.Patch(facecolor='red', edgecolor='blue', alpha=0.6, 
                      hatch='///', label='Saliency Map (Red=+, Blue=-)'),
    ]

    # 如果绘制了台风路径，添加到图例
    if all_cyclone_centers is not None and len(all_cyclone_centers) > 1:
        legend_elements.append(
            plt.Line2D([0], [0], color='purple', linewidth=2.5, marker='o',
                      markersize=6, markerfacecolor='white', markeredgecolor='purple',
                      markeredgewidth=2, label='Actual Track')
        )

    # 如果绘制了预测路径，添加到图例
    if predicted_cyclone_centers is not None and len(predicted_cyclone_centers) > 0:
        legend_elements.append(
            plt.Line2D([0], [0], color='green', linewidth=2.5, linestyle='--', marker='s',
                      markersize=6, markerfacecolor='lightgreen', markeredgecolor='green',
                      markeredgewidth=2, label='AI Predicted Track')
        )

    legend_elements.extend([
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                  markersize=15, markeredgewidth=3, label='Target Position'),
    ])

    # 如果提供了"出发时刻"台风位置，添加到图例
    if departure_cyclone_info is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='orange', markersize=12, markeredgewidth=2,
                      label='Departure Position')
        )

    legend_elements.append(
        plt.Line2D([0], [0], marker='>', color='yellow', markerfacecolor='yellow',
                  markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                  label=f'Steering Flow ({wind_speed:.1f} m/s, {wind_angle:.0f}°)')
    )
    
    # 如果绘制了预测点风向箭头，添加到图例
    if predicted_cyclone_centers is not None and predicted_wind_data is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='>', color='blue', markerfacecolor='blue',
                      markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                      label='Predicted Wind Direction')
        )
    ax.legend(handles=legend_elements,
              loc='upper right',
              bbox_to_anchor=(1.0, 1.0),
              fontsize=9,
              framealpha=0.95,
              fancybox=True,
              shadow=True,
              prop={'family': ['PingFang SC', 'Arial Unicode MS', 'sans-serif'], 'size': 9})
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")
    
    plt.show()


# %%
# ==================== Cell 0: 固定初始输入的完整路径预测 ====================
# ============================================================================
# 使用固定的初始输入 (00Z + 06Z) 一次性预测所有未来时刻的台风位置
# 这是真正的"AI预测路径"，不使用任何未来真实数据
# ============================================================================

predicted_cyclone_centers = predict_cyclone_track(
    model_forward_fn=run_forward_jitted,
    eval_inputs=eval_inputs,
    eval_targets=eval_targets,
    eval_forcings=eval_forcings,
    cyclone_centers=CYCLONE_CENTERS,
    method='mslp',
    verbose=True
)

# 同时获取完整的预测数据（包含风场），用于绘制预测点的风向箭头
print("\n正在获取完整预测数据（包含风场）...")
full_prediction_data = run_forward_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings
)
print("✓ 预测数据获取完成!")


# %%
# ==================== Cell 1: 使用滑动窗口计算梯度 (反向传播) ====================
# ============================================================================
# 滑动窗口梯度分析原理:
#
# 【原始方法】固定输入梯度分析:
#     固定输入(00Z+06Z) → 12Z预测 → 18Z预测 → 次日00Z预测
#                          ↑所有梯度都回溯到这里
#     问题: 所有预测的梯度都回溯到初始输入，无法分析时间局部的因果关系
#
# 【滑动窗口方法】:
#     窗口1: 00Z+06Z → 12Z预测 (梯度: 00Z/06Z 如何影响 12Z)
#     窗口2: 06Z+12Z → 18Z预测 (梯度: 06Z/12Z 如何影响 18Z)
#     窗口3: 12Z+18Z → 次日00Z (梯度: 12Z/18Z 如何影响 次日00Z)
#            ↑每次用前两个真实时间点作为新输入
#
# 优点:
#     1. 时间局部性: 分析相邻时间点的因果影响
#     2. 动态追踪: 跟随台风移动路径分析每步的驱动因素
#     3. 物理解释性: 更接近"当前状态如何影响下一状态"的因果关系
# ============================================================================

print("\n" + "=" * 70)
print("【步骤 1/3】开始滑动窗口梯度计算（反向传播）")
print("=" * 70)

# 使用滑动窗口梯度分析模块
sliding_window_results = compute_sliding_gradients(
    model_forward_fn=run_forward_jitted,
    task_config=task_config,
    eval_inputs=eval_inputs,
    eval_targets=eval_targets,
    eval_forcings=eval_forcings,
    cyclone_centers=CYCLONE_CENTERS,
    target_variable=TARGET_VARIABLE,
    target_level=TARGET_LEVEL,
    negative_gradient=NEGATIVE_GRADIENT,
    grid_resolution=GRID_RESOLUTION,
    verbose=True
)

# ==================== 转换梯度结果为可视化格式 ====================
gradient_results = []

for result in sliding_window_results:
    # 获取对应的台风信息（只处理预测时间点）
    prediction_centers = [c for c in CYCLONE_CENTERS if not c.get('is_input', True)]
    cyclone = prediction_centers[result.window_idx]

    # 确定物理场数据源
    physics_data = result.input_data
    # physics_time_idx = 1: 选择滑动窗口的第二个时间步（"出发时刻"）
    # 例如：窗口[00Z, 06Z]预测12Z时，选择06Z的风场作为引导气流
    # 含义：黄色箭头显示的是台风从"出发时刻"被推向哪里
    physics_time_idx = 1

    gradient_results.append({
        'idx': result.window_idx,
        'cyclone_info': cyclone,
        'gradients': result.gradients,
        'physics_data': physics_data,
        'physics_time_idx': physics_time_idx,
        'input_times': result.input_times,
        'target_time': result.target_time,
    })

print("\n" + "=" * 70)
print(f"✓ 滑动窗口梯度计算完成！共 {len(gradient_results)} 个窗口")
print("=" * 70)


# %%
# ==================== Cell 2: 批量生成可视化图像 ====================

print("\n" + "=" * 70)
print("【步骤 2/3】开始批量生成可视化图像")
print("=" * 70)

for result in gradient_results:
    idx = result['idx']
    cyclone = result['cyclone_info']
    saliency_grads = result['gradients']
    physics_data = result['physics_data']
    physics_time_idx = result['physics_time_idx']

    # 获取滑动窗口的时间信息
    input_times = result.get('input_times', ['?', '?'])
    target_time = result.get('target_time', cyclone['time'])

    print(f"\n【窗口 {idx + 1}/{len(gradient_results)}】")
    print(f"  输入时间窗口: {input_times}")
    print(f"  预测目标时间: {target_time}")
    print(f"  预测目标台风位置: ({cyclone['lat']:.2f}°, {cyclone['lon']:.2f}°)")

    # 找到"出发时刻"的台风位置（滑动窗口第二个时间步）
    # 窗口0: [00Z(idx=0), 06Z(idx=1)] → 12Z(idx=2), 出发时刻=CYCLONE_CENTERS[1]
    # 窗口1: [06Z(idx=1), 12Z(idx=2)] → 18Z(idx=3), 出发时刻=CYCLONE_CENTERS[2]
    # 窗口2: [12Z(idx=2), 18Z(idx=3)] → 次日00Z(idx=4), 出发时刻=CYCLONE_CENTERS[3]
    departure_cyclone_idx = idx + 1  # physics_time_idx=1 对应窗口第二个时间步
    departure_cyclone = CYCLONE_CENTERS[departure_cyclone_idx]
    print(f"  出发时刻台风位置: {departure_cyclone['time']} ({departure_cyclone['lat']:.2f}°, {departure_cyclone['lon']:.2f}°)")

    # 生成可视化文件名（包含滑动窗口信息）
    save_filename = f"sliding_window_{idx:02d}_{target_time.replace(' ', '_').replace(':', '')}.png"

    # 更新 cyclone_info 以包含滑动窗口信息
    cyclone_info_extended = cyclone.copy()
    cyclone_info_extended['data_type'] = f"窗口{idx+1}: {' + '.join(input_times)} → {target_time}"

    plot_physics_ai_alignment(
        cyclone_info=cyclone_info_extended,
        gradients=saliency_grads,
        era5_data=physics_data,
        gradient_var='geopotential',  # 与 TARGET_VARIABLE 保持一致,物理逻辑自洽
        time_idx=physics_time_idx,  # 使用正确的物理场时间索引
        all_cyclone_centers=CYCLONE_CENTERS,  # 传入所有台风中心点用于绘制真实路径
        departure_cyclone_info=departure_cyclone,  # 传入"出发时刻"的台风位置
        predicted_cyclone_centers=predicted_cyclone_centers,  # 传入预测的台风路径
        predicted_wind_data=full_prediction_data,  # 传入预测的风场数据
        save_path=save_filename
    )

print("\n" + "=" * 70)
print("✓ 所有可视化图像生成完成!")
print("=" * 70)


print("\n" + "=" * 70)
print("✓ 所有分析完成!")
print("=" * 70)


# %%
