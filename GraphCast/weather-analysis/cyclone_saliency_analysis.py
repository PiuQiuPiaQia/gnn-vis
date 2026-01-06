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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray

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

print("JAX devices:", jax.devices())

# %%
# ==================== 路径配置 ====================

dir_path_params = "/root/data/params"
dir_path_dataset = "/root/data/dataset"
dir_path_stats = "/root/data/stats"

params_file = "params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

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
# 台风 Cyclone Seth 各时间点坐标:
# Date (UTC)  |  Lat      |  Lon      | Pressure (mb) | Wind (kt) | Category | 数据类型
# -----------------------------------------------------------------------------------
# 01/01 00Z   | -21.2215  | 156.7095  |    997.0      |    40     |   TS     | 输入
# 01/01 06Z   | -21.7810  | 157.4565  |    996.0      |    40     |   TS     | 输入（参考点）
# 01/01 12Z   | -22.5571  | 158.2946  |   1000.0      |    35     |   TS     | 预测目标
# 01/01 18Z   | -23.9132  | 158.8048  |    998.0      |    35     |   TS     | 预测目标
# 01/02 00Z   | -25.8306  | 159.0052  |    992.0      |    40     |   TS     | 预测目标
# 01/02 06Z   | (未提供)  | (未提供)  |      -        |     -     |   -      | 预测目标
# 
# 当前配置: 使用输入数据的2个时间点 + 预测目标时间点进行梯度分析
CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.2215, "lon": 156.7095, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "输入(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7810, "lon": 157.4565, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "输入(0h)", "is_input": True, "input_time_idx": 1},
    # 以下是预测目标时间点
    {"time": "2022-01-01 12Z", "lat": -22.5571, "lon": 158.2946, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9132, "lon": 158.8048, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8306, "lon": 159.0052, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "预测(+18h)", "is_input": False, "target_time_idx": 2},
]

# 数据网格分辨率
GRID_RESOLUTION = 1.0  # 度

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
# ==================== 提取训练数据 ====================

# 增加预测步数以获取更多时间点的数据
# 原数据集包含 4 个预测步: +6h, +12h, +18h, +24h
train_steps = 4

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config)
)

print("Train Inputs:", train_inputs.dims.mapping)

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
# ==================== 物理-AI对齐可视化函数 ====================

def plot_physics_ai_alignment(
    cyclone_info: dict,
    gradients,
    era5_data,
    gradient_var: str = '2m_temperature',
    time_idx: int = 0,
    all_cyclone_centers: Optional[list] = None,
    save_path: Optional[str] = None
):
    """
    绘制物理-AI对齐分析图

    Args:
        cyclone_info: 台风信息字典 {time, lat, lon, intensity}
        gradients: 梯度数据 (xarray Dataset)
        era5_data: ERA5 气象数据
        gradient_var: 用于可视化的梯度变量
        time_idx: 时间索引，用于从多时间步数据中选择对应的时间
                  - 0: 对应输入数据的第一个时间步 (-6h = 00Z)
                  - 1: 对应输入数据的第二个时间步 (0h = 06Z)
        all_cyclone_centers: 所有台风中心点列表，用于绘制台风路径
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

    print(f"  计算环形区域平均引导风 (半径 {INNER_RADIUS}°-{OUTER_RADIUS}°, DLM)...")

    # 对每个气压层的风场，计算环形区域平均
    u_annulus_layers = []
    v_annulus_layers = []

    for level in STEERING_LEVELS:
        u_layer = u_wind_3d.sel(level=level, method='nearest')
        v_layer = v_wind_3d.sel(level=level, method='nearest')

        # 计算环形区域平均
        u_mean = extract_annulus_mean(u_layer, target_lat, target_lon,
                                       INNER_RADIUS, OUTER_RADIUS)
        v_mean = extract_annulus_mean(v_layer, target_lat, target_lon,
                                       INNER_RADIUS, OUTER_RADIUS)

        u_annulus_layers.append(u_mean)
        v_annulus_layers.append(v_mean)

        print(f"    - {level:4d} hPa: u={u_mean:6.2f}, v={v_mean:6.2f} m/s")

    # 计算多层加权平均（简单平均，可改为压力加权）
    weights = [1.0] * len(STEERING_LEVELS)
    u_center = sum(u * w for u, w in zip(u_annulus_layers, weights)) / sum(weights)
    v_center = sum(v * w for v, w in zip(v_annulus_layers, weights)) / sum(weights)

    if len(STEERING_LEVELS) == 1:
        print(f"  环形区域 {STEERING_LEVELS[0]} hPa 引导风: u={u_center:.2f}, v={v_center:.2f} m/s")
    else:
        print(f"  环形区域深层平均引导风 (DLM): u={u_center:.2f}, v={v_center:.2f} m/s")

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
    
    # 使用 robust 百分位数设置颜色范围
    vmin_pct = np.percentile(grad_vals, 2)
    vmax_pct = np.percentile(grad_vals, 98)
    vabs = max(abs(vmin_pct), abs(vmax_pct))
    
    im = ax.imshow(grad_vals, extent=[lons_grad.min(), lons_grad.max(), 
                                      lats_grad.min(), lats_grad.max()],
                   origin='lower', cmap='RdBu_r', vmin=-vabs, vmax=vabs,
                   alpha=0.6, transform=ccrs.PlateCarree(), zorder=2)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(f'{gradient_var} Gradient (AI Saliency)', fontsize=11)

    # 9. 绘制台风路径线
    if all_cyclone_centers is not None and len(all_cyclone_centers) > 1:
        # 提取所有台风中心点的经纬度
        track_lons = [c['lon'] for c in all_cyclone_centers]
        track_lats = [c['lat'] for c in all_cyclone_centers]

        # 绘制台风路径线
        ax.plot(track_lons, track_lats, color='purple', linewidth=2.5,
               linestyle='-', marker='o', markersize=6, markerfacecolor='white',
               markeredgecolor='purple', markeredgewidth=2,
               transform=ccrs.PlateCarree(), zorder=4, alpha=0.8,
               label='台风路径')

        # 在每个点旁边标注时间
        for i, c in enumerate(all_cyclone_centers):
            # 提取时间标签 (例如 "00Z", "06Z")
            time_str = c['time'].split()[-1] if ' ' in c['time'] else c['time']
            ax.text(c['lon'] + 0.5, c['lat'] + 0.5, time_str,
                   fontsize=9, color='purple', fontweight='bold',
                   transform=ccrs.PlateCarree(), zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='purple', alpha=0.7))

    # 10. 绘制当前台风眼标记
    ax.scatter(target_lon, target_lat, marker='x', s=400, c='red',
              linewidths=4, transform=ccrs.PlateCarree(), zorder=5,
              label='当前台风中心')

    # 11. 绘制环境引导气流箭头（顺风方向 = 台风移动趋势）
    # ============================================================================
    # 箭头指向 (u, v)，表示台风移动趋势（气流推动方向）
    # 物理意义：
    # - 环形区域（2-5°）平均风场 = 环境引导气流（排除台风自身环流）
    # - 700 hPa 单层 = 热带风暴的主导引导层（诊断优化结果）
    # - 顺风方向（u, v）= 台风被推动的方向
    # - 黄色箭头指向台风下一步可能移动的方向
    #
    # 气象学依据：
    # - Holland (1984): 引导气流 = 环形区域平均风场
    # - 诊断结果：700 hPa 平均误差 14.9°（深层平均 29.5°）
    # - 弱台风/热带风暴的引导层在中低层（700 hPa），强台风才需深层平均
    # ============================================================================

    # 方法：归一化风向 + 固定箭头长度（方向准确，长度统一）
    arrow_length_deg = 5.0  # 箭头固定长度（度）
    wind_magnitude = math.sqrt(u_center**2 + v_center**2)
    if wind_magnitude > 0:
        # 归一化风向，然后缩放到固定长度
        u_norm = u_center / wind_magnitude * arrow_length_deg  # 顺风方向
        v_norm = v_center / wind_magnitude * arrow_length_deg  # 顺风方向
    else:
        u_norm, v_norm = 0, 0

    ax.arrow(target_lon, target_lat, u_norm, v_norm,
             head_width=0.8, head_length=1.2, fc='yellow', ec='black',
             linewidth=2.5, transform=ccrs.PlateCarree(), zorder=6,
             label=f'台风移动趋势 (引导气流 {wind_speed:.1f} m/s)')

    print(f"  箭头绘制: 从 ({target_lon:.1f}, {target_lat:.1f}) 指向 ({target_lon+u_norm:.1f}, {target_lat+v_norm:.1f})")

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
    
    # 创建自定义图例
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=1.5,
                      label='等压线 (MSLP)'),
        mpatches.Patch(facecolor='red', alpha=0.6, label='梯度热力图 (AI)'),
    ]

    # 如果绘制了台风路径，添加到图例
    if all_cyclone_centers is not None and len(all_cyclone_centers) > 1:
        legend_elements.append(
            plt.Line2D([0], [0], color='purple', linewidth=2.5, marker='o',
                      markersize=6, markerfacecolor='white', markeredgecolor='purple',
                      markeredgewidth=2, label='台风路径')
        )

    legend_elements.extend([
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                  markersize=15, markeredgewidth=3, label='当前台风中心'),
        plt.Line2D([0], [0], marker='>', color='yellow', markerfacecolor='yellow',
                  markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                  label=f'台风移动趋势 ({wind_speed:.1f} m/s, {wind_angle:.0f}°)')
    ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")
    
    plt.show()


# %%
# ==================== 主流程: 为5个时间点生成对齐分析图 ====================

print("\n" + "=" * 70)
print("开始生成物理-AI对齐分析图")
print("=" * 70)

for idx, cyclone in enumerate(CYCLONE_CENTERS):
    print(f"\n【{idx + 1}/{len(CYCLONE_CENTERS)}】处理时间点: {cyclone['time']} ({cyclone['data_type']})")

    # 计算目标点索引
    target_lat_idx, target_lon_idx = latlon_to_index(
        lat=cyclone['lat'],
        lon=cyclone['lon'],
        resolution=GRID_RESOLUTION,
        lat_min=-90.0,
        lon_min=0.0
    )

    print(f"  台风眼坐标: ({cyclone['lat']:.4f}°, {cyclone['lon']:.4f}°)")
    print(f"  网格索引: (lat_idx={target_lat_idx}, lon_idx={target_lon_idx})")

    # 根据是输入还是预测时间点，确定梯度计算的目标时间索引
    if cyclone.get('is_input', True):
        # 输入时间点：梯度目标是第一个预测步 (+6h = 12Z)
        grad_target_time_idx = 0
        # 物理场使用输入数据
        physics_data = train_inputs
        physics_time_idx = cyclone['input_time_idx']
    else:
        # 预测时间点：梯度目标是对应的预测步
        grad_target_time_idx = cyclone['target_time_idx']
        # 物理场使用目标数据 (ERA5 真实值)
        physics_data = train_targets
        physics_time_idx = cyclone['target_time_idx']

    # 计算梯度
    print(f"  计算梯度 (target_time_idx={grad_target_time_idx})...")
    saliency_grads = compute_saliency_map(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings,
        target_idx=(target_lat_idx, target_lon_idx),
        target_variable=TARGET_VARIABLE,
        target_level=TARGET_LEVEL,
        target_time_idx=grad_target_time_idx,
        negative=NEGATIVE_GRADIENT
    )

    # 生成可视化
    save_filename = f"physics_ai_alignment_{idx:02d}_{cyclone['time'].replace(' ', '_').replace(':', '')}.png"

    plot_physics_ai_alignment(
        cyclone_info=cyclone,
        gradients=saliency_grads,
        era5_data=physics_data,
        gradient_var='geopotential',  # 与 TARGET_VARIABLE 保持一致,物理逻辑自洽
        time_idx=physics_time_idx,  # 使用正确的物理场时间索引
        all_cyclone_centers=CYCLONE_CENTERS,  # 传入所有台风中心点用于绘制路径
        save_path=save_filename
    )

print("\n" + "=" * 70)
print("✓ 所有物理-AI对齐分析图生成完成!")
print("=" * 70)


# %%
