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
from region_utils import extract_region_data

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
# 当前配置: 使用输入数据的2个时间点进行梯度分析
CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.2215, "lon": 156.7095, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "输入(-6h)"},
    {"time": "2022-01-01 06Z", "lat": -21.7810, "lon": 157.4565, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "输入(0h)"},
    # 以下是预测目标时间点（可选添加，用于对比分析）
    # {"time": "2022-01-01 12Z", "lat": -22.5571, "lon": 158.2946, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+6h)"},
    # {"time": "2022-01-01 18Z", "lat": -23.9132, "lon": 158.8048, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+12h)"},
    # {"time": "2022-01-02 00Z", "lat": -25.8306, "lon": 159.0052, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "预测(+18h)"},
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

train_steps = 1

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
    # 梯度数据的 time 维度是输入历史时间步，通常选择最后一个（最新的）
    if 'time' in grad_data.dims:
        grad_data = grad_data.isel(time=-1)  # 使用最后一个时间步
    
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
    
    # 3. 提取高空引导风场数据 (Steering Flow)
    # 气压层选择:
    #   index 7 = 500hPa (对流层中层标准引导层，与TARGET_LEVEL一致)
    #   index 9 = 700hPa (中低层引导,对热带风暴更适用)
    #   index 10 = 850hPa (边界层顶部)
    STEERING_LEVEL_IDX = 7  # 500 hPa - 与梯度计算的目标层一致
    
    # 注意：这里使用的是带 level 维度的 u/v，而不是 10m_u/v
    u_wind_3d = era5_data['u_component_of_wind']
    v_wind_3d = era5_data['v_component_of_wind']
    
    # 处理维度：取特定层 (Steering Level)
    if 'level' in u_wind_3d.dims:
        u_wind = u_wind_3d.isel(level=STEERING_LEVEL_IDX)
        v_wind = v_wind_3d.isel(level=STEERING_LEVEL_IDX)
    else:
        # Fallback (防守性编程)
        u_wind = u_wind_3d
        v_wind = v_wind_3d
    
    # 处理 Batch 和 Time (使用对应的时间步)
    if 'batch' in u_wind.dims:
        u_wind = u_wind.isel(batch=0)
        v_wind = v_wind.isel(batch=0)
    if 'time' in u_wind.dims:
        # 使用最后一个可用时间步（如果time_idx超出范围）
        actual_time_idx = min(time_idx, len(u_wind.time) - 1)
        u_wind = u_wind.isel(time=actual_time_idx)
        v_wind = v_wind.isel(time=actual_time_idx)
    
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
    
    # 5. 计算台风周围环境的平均引导风 (区域平均,而不是单点)
    STEERING_RADIUS = 3.0  # 引导风计算半径(度)
    
    # 直接用最近邻方法提取台风中心点的风速 (避免slice方向问题)
    # 这比区域平均更直接,且避免了南半球坐标系统的slice问题
    u_center = float(u_wind.sel(lat=target_lat, lon=target_lon, method='nearest').values)
    v_center = float(v_wind.sel(lat=target_lat, lon=target_lon, method='nearest').values)
    
    print(f"  500hPa引导风速(单点,time_idx={time_idx}): u={u_center:.2f}, v={v_center:.2f} m/s")
    print(f"  逆风方向: ({-u_center:.2f}, {-v_center:.2f}) m/s")
    
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
    
    # 9. 绘制台风眼标记
    ax.scatter(target_lon, target_lat, marker='x', s=400, c='red', 
              linewidths=4, transform=ccrs.PlateCarree(), zorder=5,
              label='Cyclone Center')
    
    # 10. 绘制逆风向量箭头 (指向上游)
    # 箭头指向 (-u, -v)
    # 注意: 高空风通常比地面风强,使用较小的 scale 避免箭头过长
    # TODO: 箭头方向还是不对，需要进一步检查坐标系统和风向计算逻辑
    arrow_scale = 1.0  # 调整箭头长度 (高空风建议 0.5-1.0,地面风建议 2.0)
    ax.arrow(target_lon, target_lat, -u_center * arrow_scale, -v_center * arrow_scale,
             head_width=0.8, head_length=1.2, fc='yellow', ec='black', 
             linewidth=2.5, transform=ccrs.PlateCarree(), zorder=6,
             label=f'Upwind Direction (−u,−v)')
    
    # 11. 添加标题和图例
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
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                  markersize=15, markeredgewidth=3, label='台风中心'),
        plt.Line2D([0], [0], marker='>', color='yellow', markerfacecolor='yellow',
                  markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                  label=f'逆风方向 ({-u_center:.1f}, {-v_center:.1f} m/s)')
    ]
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
    print(f"\n【{idx + 1}/{len(CYCLONE_CENTERS)}】处理时间点: {cyclone['time']}")
    
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
    
    # 计算梯度
    print("  计算梯度...")
    saliency_grads = compute_saliency_map(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings,
        target_idx=(target_lat_idx, target_lon_idx),
        target_variable=TARGET_VARIABLE,
        target_level=TARGET_LEVEL,
        target_time_idx=0,
        negative=NEGATIVE_GRADIENT
    )
    
    # 生成可视化
    save_filename = f"physics_ai_alignment_{idx:02d}_{cyclone['time'].replace(' ', '_').replace(':', '')}.png"
    
    plot_physics_ai_alignment(
        cyclone_info=cyclone,
        gradients=saliency_grads,
        era5_data=train_inputs,
        gradient_var='geopotential',  # 与 TARGET_VARIABLE 保持一致,物理逻辑自洽
        time_idx=idx,  # ✓ 修复: 传入对应的时间索引
        save_path=save_filename
    )

print("\n" + "=" * 70)
print("✓ 所有物理-AI对齐分析图生成完成!")
print("=" * 70)


# %%
