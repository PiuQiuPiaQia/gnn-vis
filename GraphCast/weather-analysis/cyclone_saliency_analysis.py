# %%
"""
台风 Saliency Map 梯度分析脚本

分析 GraphCast 模型中哪些输入区域对台风中心预测贡献最大。
通过计算模型输出对输入的梯度，了解：
- 哪些地理位置的输入对台风预测影响最大
- 哪些气象变量对台风中心预测最敏感

路径: GraphCast/weather-analysis/cyclone_saliency_analysis.py
"""

# %%
# ==================== 导入库 ====================

import sys
from pathlib import Path

# 添加 graphcast 源码路径（相对于当前脚本）
# weather-analysis 的父目录是 GraphCast，其中包含 graphcast 源码
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

import dataclasses
import functools
import glob
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import xarray

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk

print("JAX devices:", jax.devices())

# %%
# ==================== 路径配置 ====================
# 请根据你的实际路径修改

dir_path_params = "/root/data/params"
dir_path_dataset = "/root/data/dataset"
dir_path_stats = "/root/data/stats"

# 选择模型和数据集
params_file = "params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

# %%
# ==================== 辅助函数 ====================

def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

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

# 加载气象数据
with open(f"{dir_path_dataset}/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

print("数据维度:", example_batch.dims.mapping)

# %%
# ==================== 提取训练/评估数据 ====================

train_steps = 1
eval_steps = 1

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config)
)

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config)
)

print("Train Inputs:", train_inputs.dims.mapping)
print("Train Targets:", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)

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


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))


def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


def with_params(fn):
    return functools.partial(fn, params=params, state=state)


def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


# JIT 编译
print("正在 JIT 编译模型（首次运行可能需要几分钟）...")
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
print("模型编译完成!")

# %%
# ==================== Saliency Map 配置 ====================

# 目标点索引 (lat_idx, lon_idx)
# 对于 1.0° 分辨率: lat_idx = (lat + 90) / 1.0, lon_idx = lon / 1.0
# 例如: 台风中心 (-21.7°S, 157.5°E) -> (68, 158)
TARGET_LAT_IDX = 68
TARGET_LON_IDX = 158

# 目标变量配置
TARGET_VARIABLE = 'geopotential'  # 位势高度
TARGET_LEVEL = 500                 # 目标气压层 (hPa)
TARGET_TIME_IDX = 0                # 预测的第几个时间步
NEGATIVE_GRADIENT = True           # True: 关注导致值降低的因素

print(f"\nSaliency Map 配置:")
print(f"  目标点索引: (lat={TARGET_LAT_IDX}, lon={TARGET_LON_IDX})")
print(f"  目标变量: {TARGET_VARIABLE} @ {TARGET_LEVEL} hPa")
print(f"  负梯度模式: {NEGATIVE_GRADIENT}")

# %%
# ==================== 计算 Saliency Map ====================

def compute_saliency_map(
    inputs,
    targets,
    forcings,
    target_idx,
    target_variable='geopotential',
    target_level=500,
    target_time_idx=0,
    negative=True
):
    """
    计算 GraphCast 输入梯度 (Saliency Map)

    Args:
        inputs: 输入数据 (xarray.Dataset)
        targets: 目标模板
        forcings: 强迫项数据
        target_idx: 目标点索引 (lat_idx, lon_idx)
        target_variable: 目标变量名
        target_level: 目标气压层 (hPa)
        target_time_idx: 预测时间步索引
        negative: True则返回负梯度

    Returns:
        grads: 输入梯度 (xarray.Dataset)
    """
    lat_idx, lon_idx = target_idx

    def target_loss(inputs_data):
        # 运行模型前向传播
        outputs = run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs_data,
            targets_template=targets * np.nan,
            forcings=forcings
        )

        # 提取目标变量
        target_data = outputs[target_variable]

        # 索引到目标点
        if 'level' in target_data.dims:
            value = target_data.sel(level=target_level).isel(
                time=target_time_idx, lat=lat_idx, lon=lon_idx
            )
        else:
            value = target_data.isel(
                time=target_time_idx, lat=lat_idx, lon=lon_idx
            )

        # 处理 batch 维度
        if 'batch' in value.dims:
            value = value.isel(batch=0)

        # 提取 JAX 数组并返回标量
        scalar = xarray_jax.unwrap_data(value, require_jax=True)
        scalar = jnp.squeeze(scalar)

        return -scalar if negative else scalar

    # 计算梯度
    grads = jax.grad(target_loss)(inputs)
    return grads


print("\n开始计算 Saliency Map...")
print("（首次运行需要 JIT 编译，请耐心等待）")

saliency_grads = compute_saliency_map(
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings,
    target_idx=(TARGET_LAT_IDX, TARGET_LON_IDX),
    target_variable=TARGET_VARIABLE,
    target_level=TARGET_LEVEL,
    target_time_idx=TARGET_TIME_IDX,
    negative=NEGATIVE_GRADIENT
)

print("\n✓ Saliency Map 计算完成!")

# %%
# ==================== 梯度统计分析 ====================

print("=" * 60)
print("各变量梯度统计信息")
print("=" * 60)

for var_name in saliency_grads.data_vars:
    grad_data = xarray_jax.unwrap_data(saliency_grads[var_name])
    if hasattr(grad_data, 'block_until_ready'):
        grad_data.block_until_ready()
    grad_data = np.array(grad_data)

    print(f"\n{var_name}:")
    print(f"  形状: {grad_data.shape}")
    print(f"  最大值: {grad_data.max():.6e}")
    print(f"  最小值: {grad_data.min():.6e}")
    print(f"  绝对值均值: {np.abs(grad_data).mean():.6e}")

# %%
# ==================== 坐标转换辅助函数 ====================

def idx_to_lat(lat_idx, resolution=1.0):
    """将纬度索引转换为真实纬度"""
    return lat_idx * resolution - 90.0

def idx_to_lon(lon_idx, resolution=1.0):
    """将经度索引转换为真实经度"""
    return lon_idx * resolution

def format_lat(lat):
    """格式化纬度显示"""
    if lat >= 0:
        return f"{lat:.0f}°N"
    else:
        return f"{-lat:.0f}°S"

def format_lon(lon):
    """格式化经度显示"""
    if lon >= 0:
        return f"{lon:.0f}°E"
    else:
        return f"{-lon:.0f}°W"

# %%
# ==================== 可视化函数 ====================

def visualize_saliency(
    grads,
    var_name,
    level_idx=None,
    time_idx=0,
    robust=True,
    target_lat_idx=TARGET_LAT_IDX,
    target_lon_idx=TARGET_LON_IDX,
    save_path=None,
    resolution=1.0
):
    """
    可视化指定变量的 Saliency Map

    Args:
        grads: 梯度数据
        var_name: 变量名
        level_idx: 气压层索引（对于3D变量）
        time_idx: 时间步索引
        robust: 是否使用百分位数设置颜色范围
        target_lat_idx: 目标点纬度索引
        target_lon_idx: 目标点经度索引
        save_path: 保存路径（可选）
        resolution: 网格分辨率（度），默认1.0
    """
    grad_var = grads[var_name]

    if 'time' in grad_var.dims:
        grad_var = grad_var.isel(time=time_idx)

    if 'level' in grad_var.dims:
        if level_idx is not None:
            grad_var = grad_var.isel(level=level_idx)
            level_info = f" (level={level_idx})"
        else:
            grad_var = grad_var.isel(level=0)
            level_info = " (level=0)"
    else:
        level_info = ""

    if 'batch' in grad_var.dims:
        grad_var = grad_var.isel(batch=0)

    # 获取数据
    data = xarray_jax.unwrap_data(grad_var)
    if hasattr(data, 'block_until_ready'):
        data.block_until_ready()
    data = np.array(data)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 使用百分位数设置颜色范围（解决极端值问题）
    if robust:
        vmin_pct = np.percentile(data, 2)
        vmax_pct = np.percentile(data, 98)
        vabs = max(abs(vmin_pct), abs(vmax_pct))
        vmin, vmax = -vabs, vabs
    else:
        vmax = np.abs(data).max()
        vmin = -vmax

    if vmax == 0:
        vmax = 1
        vmin = -1

    im = ax.imshow(data, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Gradient', shrink=0.8)

    # 标记目标点
    ax.scatter(target_lon_idx, target_lat_idx, c='lime', s=300, marker='*',
               edgecolors='black', linewidths=2, zorder=5,
               label=f'Target: ({target_lat_idx}, {target_lon_idx})')

    ax.set_title(f'Saliency Map: {var_name}{level_info} (robust={robust})', fontsize=14)
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    plt.show()
    return data


def visualize_saliency_log(
    grads,
    var_name,
    level_idx=None,
    time_idx=0,
    scale_factor=1e6,
    target_lat_idx=TARGET_LAT_IDX,
    target_lon_idx=TARGET_LON_IDX,
    save_path=None
):
    """
    使用对数缩放可视化 Saliency Map（适用于跨多个数量级的梯度）
    """
    grad_var = grads[var_name]

    if 'time' in grad_var.dims:
        grad_var = grad_var.isel(time=time_idx)

    if 'level' in grad_var.dims:
        if level_idx is not None:
            grad_var = grad_var.isel(level=level_idx)
            level_info = f" (level={level_idx})"
        else:
            grad_var = grad_var.isel(level=0)
            level_info = " (level=0)"
    else:
        level_info = ""

    if 'batch' in grad_var.dims:
        grad_var = grad_var.isel(batch=0)

    # 获取数据
    data = xarray_jax.unwrap_data(grad_var)
    if hasattr(data, 'block_until_ready'):
        data.block_until_ready()
    data = np.array(data)

    # 对数缩放
    data_sign = np.sign(data)
    data_log = data_sign * np.log1p(np.abs(data) * scale_factor)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    vmax = np.abs(data_log).max()
    if vmax == 0:
        vmax = 1

    im = ax.imshow(data_log, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Log Gradient', shrink=0.8)

    # 标记目标点
    ax.scatter(target_lon_idx, target_lat_idx, c='lime', s=300, marker='*',
               edgecolors='black', linewidths=2, zorder=5,
               label=f'Target: ({target_lat_idx}, {target_lon_idx})')

    ax.set_title(f'Saliency Map (Log Scale): {var_name}{level_info}', fontsize=14)
    ax.set_xlabel('Longitude Index')
    ax.set_ylabel('Latitude Index')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    plt.show()
    return data

# %%
# ==================== 局部区域裁剪与可视化 ====================

def visualize_local_saliency(
    grads,
    var_name,
    target_lat_idx,
    target_lon_idx,
    radius=50,
    level_idx=None,
    time_idx=0,
    robust=True,
    save_path=None
):
    """
    可视化目标点周围局部区域的 Saliency Map（裁剪后的热力图）

    Args:
        grads: 梯度数据
        var_name: 变量名
        target_lat_idx: 目标点纬度索引
        target_lon_idx: 目标点经度索引
        radius: 裁剪半径（网格数），默认 50
        level_idx: 气压层索引（对于3D变量）
        time_idx: 时间步索引
        robust: 是否使用百分位数设置颜色范围
        save_path: 保存路径（可选）
    """
    grad_var = grads[var_name]

    if 'time' in grad_var.dims:
        grad_var = grad_var.isel(time=time_idx)

    if 'level' in grad_var.dims:
        if level_idx is not None:
            grad_var = grad_var.isel(level=level_idx)
            level_info = f" @ level={level_idx}"
        else:
            grad_var = grad_var.isel(level=0)
            level_info = " @ level=0"
    else:
        level_info = ""

    if 'batch' in grad_var.dims:
        grad_var = grad_var.isel(batch=0)

    # 获取数据
    data = xarray_jax.unwrap_data(grad_var)
    if hasattr(data, 'block_until_ready'):
        data.block_until_ready()
    data = np.array(data)

    # 获取数据形状
    lat_size, lon_size = data.shape

    # 计算裁剪范围
    lat_min = max(0, target_lat_idx - radius)
    lat_max = min(lat_size, target_lat_idx + radius + 1)
    lon_min = max(0, target_lon_idx - radius)
    lon_max = min(lon_size, target_lon_idx + radius + 1)

    # 裁剪数据
    data_cropped = data[lat_min:lat_max, lon_min:lon_max]

    # 计算目标点在裁剪后数据中的位置
    target_lat_cropped = target_lat_idx - lat_min
    target_lon_cropped = target_lon_idx - lon_min

    print(f"\n裁剪信息:")
    print(f"  变量: {var_name}{level_info}")
    print(f"  原始数据形状: {data.shape}")
    print(f"  裁剪范围: lat=[{lat_min}:{lat_max}], lon=[{lon_min}:{lon_max}]")
    print(f"  裁剪后形状: {data_cropped.shape}")
    print(f"  目标点在裁剪数据中的位置: ({target_lat_cropped}, {target_lon_cropped})")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用百分位数设置颜色范围（解决极端值问题）
    if robust:
        vmin_pct = np.percentile(data_cropped, 2)
        vmax_pct = np.percentile(data_cropped, 98)
        vabs = max(abs(vmin_pct), abs(vmax_pct))
        vmin, vmax = -vabs, vabs
    else:
        vmax = np.abs(data_cropped).max()
        vmin = -vmax

    if vmax == 0:
        vmax = 1
        vmin = -1

    im = ax.imshow(data_cropped, origin='lower', cmap='RdBu_r', 
                   vmin=vmin, vmax=vmax, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, label='Gradient', shrink=0.8)

    # 标记目标点（台风中心）
    ax.scatter(target_lon_cropped, target_lat_cropped, 
               c='lime', s=400, marker='*',
               edgecolors='black', linewidths=2.5, zorder=5,
               label=f'Target Center')

    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 设置刻度标签（显示索引和真实经纬度）
    n_ticks = 5
    lat_tick_indices = np.linspace(0, data_cropped.shape[0]-1, n_ticks, dtype=int)
    lon_tick_indices = np.linspace(0, data_cropped.shape[1]-1, n_ticks, dtype=int)
    
    # 计算对应的真实经纬度
    resolution = 1.0
    lat_labels = []
    for i in lat_tick_indices:
        idx = lat_min + i
        lat = idx_to_lat(idx, resolution)
        lat_labels.append(f'{idx}\n{format_lat(lat)}')
    
    lon_labels = []
    for i in lon_tick_indices:
        idx = lon_min + i
        lon = idx_to_lon(idx, resolution)
        lon_labels.append(f'{idx}\n{format_lon(lon)}')
    
    ax.set_yticks(lat_tick_indices)
    ax.set_yticklabels(lat_labels)
    ax.set_xticks(lon_tick_indices)
    ax.set_xticklabels(lon_labels)

    ax.set_title(f'局部 Saliency Map: {var_name}{level_info}\n'
                 f'(目标点: lat={target_lat_idx}, lon={target_lon_idx}, 半径=±{radius} 网格)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude Index (经度)', fontsize=11)
    ax.set_ylabel('Latitude Index (纬度)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")

    plt.show()
    
    return data_cropped


# %%
# ==================== 可视化 Saliency Map ====================

print("\n【温度场梯度 - Robust 缩放】")
if 'temperature' in saliency_grads.data_vars:
    visualize_saliency(saliency_grads, 'temperature', level_idx=5, robust=True)

print("\n【位势场梯度 - Robust 缩放】")
if 'geopotential' in saliency_grads.data_vars:
    visualize_saliency(saliency_grads, 'geopotential', level_idx=5, robust=True)

print("\n【2米温度梯度 - Robust 缩放】")
if '2m_temperature' in saliency_grads.data_vars:
    visualize_saliency(saliency_grads, '2m_temperature', robust=True)

# %%
# ==================== 局部区域热力图可视化 ====================

print("\n" + "=" * 60)
print("局部区域热力图可视化 (±50 网格)")
print("=" * 60)

# 可视化温度场局部梯度
print("\n【温度场 - 局部区域热力图】")
if 'temperature' in saliency_grads.data_vars:
    local_temp_grads = visualize_local_saliency(
        grads=saliency_grads,
        var_name='temperature',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        level_idx=5,
        robust=True,
        save_path='saliency_local_temperature.png'
    )

# 可视化位势场局部梯度
print("\n【位势场 - 局部区域热力图】")
if 'geopotential' in saliency_grads.data_vars:
    local_geo_grads = visualize_local_saliency(
        grads=saliency_grads,
        var_name='geopotential',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        level_idx=5,
        robust=True,
        save_path='saliency_local_geopotential.png'
    )

# 可视化2米温度局部梯度
print("\n【2米温度 - 局部区域热力图】")
if '2m_temperature' in saliency_grads.data_vars:
    local_2m_temp_grads = visualize_local_saliency(
        grads=saliency_grads,
        var_name='2m_temperature',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        robust=True,
        save_path='saliency_local_2m_temperature.png'
    )

# %%
# ==================== 局部区域对数缩放可视化函数 ====================

def visualize_local_saliency_log(
    grads,
    var_name,
    target_lat_idx,
    target_lon_idx,
    radius=50,
    level_idx=None,
    time_idx=0,
    scale_factor=1e6,
    save_path=None
):
    """
    使用对数缩放可视化目标点周围局部区域的 Saliency Map
    
    Args:
        grads: 梯度数据
        var_name: 变量名
        target_lat_idx: 目标点纬度索引
        target_lon_idx: 目标点经度索引
        radius: 裁剪半径（网格数），默认 50
        level_idx: 气压层索引（对于3D变量）
        time_idx: 时间步索引
        scale_factor: 对数缩放因子，默认 1e6
        save_path: 保存路径（可选）
    """
    grad_var = grads[var_name]

    if 'time' in grad_var.dims:
        grad_var = grad_var.isel(time=time_idx)

    if 'level' in grad_var.dims:
        if level_idx is not None:
            grad_var = grad_var.isel(level=level_idx)
            level_info = f" @ level={level_idx}"
        else:
            grad_var = grad_var.isel(level=0)
            level_info = " @ level=0"
    else:
        level_info = ""

    if 'batch' in grad_var.dims:
        grad_var = grad_var.isel(batch=0)

    # 获取数据
    data = xarray_jax.unwrap_data(grad_var)
    if hasattr(data, 'block_until_ready'):
        data.block_until_ready()
    data = np.array(data)

    # 获取数据形状
    lat_size, lon_size = data.shape

    # 计算裁剪范围
    lat_min = max(0, target_lat_idx - radius)
    lat_max = min(lat_size, target_lat_idx + radius + 1)
    lon_min = max(0, target_lon_idx - radius)
    lon_max = min(lon_size, target_lon_idx + radius + 1)

    # 裁剪数据
    data_cropped = data[lat_min:lat_max, lon_min:lon_max]

    # 计算目标点在裁剪后数据中的位置
    target_lat_cropped = target_lat_idx - lat_min
    target_lon_cropped = target_lon_idx - lon_min

    # 对数缩放
    data_sign = np.sign(data_cropped)
    data_log = data_sign * np.log1p(np.abs(data_cropped) * scale_factor)

    print(f"\n裁剪信息 (对数缩放):")
    print(f"  变量: {var_name}{level_info}")
    print(f"  原始数据形状: {data.shape}")
    print(f"  裁剪范围: lat=[{lat_min}:{lat_max}], lon=[{lon_min}:{lon_max}]")
    print(f"  裁剪后形状: {data_cropped.shape}")
    print(f"  目标点在裁剪数据中的位置: ({target_lat_cropped}, {target_lon_cropped})")
    print(f"  对数缩放因子: {scale_factor}")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 10))

    vmax = np.abs(data_log).max()
    if vmax == 0:
        vmax = 1

    im = ax.imshow(data_log, origin='lower', cmap='RdBu_r', 
                   vmin=-vmax, vmax=vmax, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, label='Log Gradient', shrink=0.8)

    # 标记目标点（台风中心）
    ax.scatter(target_lon_cropped, target_lat_cropped, 
               c='lime', s=400, marker='*',
               edgecolors='black', linewidths=2.5, zorder=5,
               label=f'Target Center')

    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 设置刻度标签（显示索引和真实经纬度）
    n_ticks = 5
    lat_tick_indices = np.linspace(0, data_cropped.shape[0]-1, n_ticks, dtype=int)
    lon_tick_indices = np.linspace(0, data_cropped.shape[1]-1, n_ticks, dtype=int)
    
    # 计算对应的真实经纬度
    resolution = 1.0
    lat_labels = []
    for i in lat_tick_indices:
        idx = lat_min + i
        lat = idx_to_lat(idx, resolution)
        lat_labels.append(f'{idx}\n{format_lat(lat)}')
    
    lon_labels = []
    for i in lon_tick_indices:
        idx = lon_min + i
        lon = idx_to_lon(idx, resolution)
        lon_labels.append(f'{idx}\n{format_lon(lon)}')
    
    ax.set_yticks(lat_tick_indices)
    ax.set_yticklabels(lat_labels)
    ax.set_xticks(lon_tick_indices)
    ax.set_xticklabels(lon_labels)

    ax.set_title(f'局部 Saliency Map (Log Scale): {var_name}{level_info}\n'
                 f'(目标点: lat={target_lat_idx}, lon={target_lon_idx}, 半径=±{radius} 网格)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude Index (经度)', fontsize=11)
    ax.set_ylabel('Latitude Index (纬度)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ 图像已保存: {save_path}")

    plt.show()
    
    return data_cropped


# %%
# ==================== 对数缩放可视化（可选）====================

print("\n【温度场梯度 - 对数缩放】")
if 'temperature' in saliency_grads.data_vars:
    visualize_saliency_log(saliency_grads, 'temperature', level_idx=5)

# %%
# ==================== 局部区域对数缩放热力图可视化 ====================

print("\n" + "=" * 60)
print("局部区域对数缩放热力图可视化 (±50 网格)")
print("=" * 60)

# 可视化温度场局部梯度（对数缩放）
print("\n【温度场 - 局部对数缩放热力图】")
if 'temperature' in saliency_grads.data_vars:
    local_temp_grads_log = visualize_local_saliency_log(
        grads=saliency_grads,
        var_name='temperature',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        level_idx=5,
        scale_factor=1e6,
        save_path='saliency_local_temperature_log.png'
    )

# 可视化位势场局部梯度（对数缩放）
print("\n【位势场 - 局部对数缩放热力图】")
if 'geopotential' in saliency_grads.data_vars:
    local_geo_grads_log = visualize_local_saliency_log(
        grads=saliency_grads,
        var_name='geopotential',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        level_idx=5,
        scale_factor=1e6,
        save_path='saliency_local_geopotential_log.png'
    )

# 可视化2米温度局部梯度（对数缩放）
print("\n【2米温度 - 局部对数缩放热力图】")
if '2m_temperature' in saliency_grads.data_vars:
    local_2m_temp_grads_log = visualize_local_saliency_log(
        grads=saliency_grads,
        var_name='2m_temperature',
        target_lat_idx=TARGET_LAT_IDX,
        target_lon_idx=TARGET_LON_IDX,
        radius=50,
        scale_factor=1e6,
        save_path='saliency_local_2m_temperature_log.png'
    )

# %%
# ==================== 保存结果（可选）====================

import pickle

# 保存梯度数据
# with open('saliency_grads.pkl', 'wb') as f:
#     pickle.dump(saliency_grads, f)
# print("梯度数据已保存到 saliency_grads.pkl")

print("\n脚本执行完成!")
print("\n提示: 局部区域热力图已生成，展示了目标点周围 ±50 个网格的梯度分布。")
