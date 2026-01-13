#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
台风路径数据验证脚本 (Cyclone Data Verification)
============================================================================

功能简介:
---------
验证 ERA5 数据集（eval_inputs + eval_targets）中的台风位置是否与
真实观测路径（CYCLONE_CENTERS）一致。

通过从气象场数据中提取最低海平面气压点来定位台风中心，
并与标注的真实观测位置进行对比可视化。

使用方法:
---------
```bash
cd /Users/zhangao/Documents/gnn-vis/GraphCast/weather-analysis
python verify_cyclone_data.py
```

输出:
---------
- 图像: era5_vs_observation.png（对比图）
- 终端: 误差统计信息

作者: AI Assistant
创建日期: 2026-01-13
版本: 1.0.0
============================================================================
"""

import sys
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
from typing import List, Dict, Any

# 添加 graphcast 源码路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
PREPROCESS_DIR = SCRIPT_DIR.parent / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

import dataclasses
import numpy as np
import xarray
import matplotlib.pyplot as plt

from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import xarray_jax

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 配置参数（从 cyclone_saliency_analysis.py 复用）
# ============================================================================

# 路径配置
dir_path_params = "/root/data/params"
dir_path_dataset = "/root/data/dataset"
dir_path_stats = "/root/data/stats"

# 数据文件
params_file = "params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

# 台风真实观测路径（Cyclone Seth，2022年1月）
CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.3138, "lon": 156.6947, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "输入(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7054, "lon": 157.5024, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "输入(0h)", "is_input": True, "input_time_idx": 1},
    # 以下是预测目标时间点
    {"time": "2022-01-01 12Z", "lat": -22.5048, "lon": 158.2994, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9030, "lon": 158.8031, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "预测(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8032, "lon": 159.0031, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "预测(+18h)", "is_input": False, "target_time_idx": 2},
]

eval_steps = 4


# ============================================================================
# 辅助函数
# ============================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离（Haversine公式）

    Args:
        lat1, lon1: 第一个点的纬度和经度（度）
        lat2, lon2: 第二个点的纬度和经度（度）

    Returns:
        两点间的距离（km）
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径(km)
    return c * r


# ============================================================================
# 台风位置提取函数
# ============================================================================

def find_local_minimum_pressure(
    pressure_data: xarray.DataArray,
    ref_lat: float,
    ref_lon: float,
    search_radius: float = 10.0
) -> Dict[str, float]:
    """
    在参考位置附近搜索最低气压点

    避免全球搜索找到其他更强的低压系统（如其他台风、温带气旋等），
    通过限定搜索范围到台风附近区域来确保找到目标台风中心。

    Args:
        pressure_data: 气压数据（2D，已去除 batch 和 time 维度）
        ref_lat: 参考纬度（从 CYCLONE_CENTERS 获取）
        ref_lon: 参考经度（从 CYCLONE_CENTERS 获取）
        search_radius: 搜索半径（度），默认 10° ≈ 1110 km

    Returns:
        {'lat': 纬度, 'lon': 经度, 'pressure': 气压值(Pa)}
    """
    # 获取经纬度数组
    lats = pressure_data.lat.values
    lons = pressure_data.lon.values

    # 找到参考位置附近的索引范围
    lat_mask = (lats >= ref_lat - search_radius) & (lats <= ref_lat + search_radius)
    lon_mask = (lons >= ref_lon - search_radius) & (lons <= ref_lon + search_radius)

    # 提取区域数据
    region_data = pressure_data.sel(
        lat=pressure_data.lat[lat_mask],
        lon=pressure_data.lon[lon_mask]
    )

    # 转换为 numpy 数组
    region_np = xarray_jax.unwrap_data(region_data)
    if hasattr(region_np, 'block_until_ready'):
        region_np.block_until_ready()
    region_np = np.array(region_np)

    # 找最小值位置
    min_idx = np.unravel_index(np.argmin(region_np), region_np.shape)

    # 转换回全局坐标
    lat = float(region_data.lat.values[min_idx[0]])
    lon = float(region_data.lon.values[min_idx[1]])
    pressure_value = float(region_np[min_idx[0], min_idx[1]])

    return {'lat': lat, 'lon': lon, 'pressure': pressure_value}

def extract_era5_cyclone_positions(
    eval_inputs: xarray.Dataset,
    eval_targets: xarray.Dataset,
    verbose: bool = True,
    search_radius: float = 10.0
) -> List[Dict[str, Any]]:
    """
    从 ERA5 数据集中提取所有时间点的台风中心位置

    使用 CYCLONE_CENTERS 作为参考位置，在附近区域搜索最低气压点，
    避免全球搜索找到其他低压系统。

    Args:
        eval_inputs: 输入数据 (通常包含2个时间点: 00Z, 06Z)
        eval_targets: 目标数据 (通常包含3-4个时间点: 12Z, 18Z, ...)
        verbose: 是否打印详细信息
        search_radius: 搜索半径（度），默认 10° ≈ 1110 km

    Returns:
        台风位置列表，每个元素包含:
            - 'time': 时间标签
            - 'lat': 纬度
            - 'lon': 经度
            - 'pressure': 气压 (hPa)
            - 'source': 'ERA5'
    """
    if verbose:
        print("\n" + "=" * 70)
        print("【从 ERA5 数据集提取台风位置】")
        print("=" * 70)
        print(f"  搜索方式: 区域搜索 (半径 ±{search_radius}°)")

    positions = []

    # 处理 eval_inputs (00Z, 06Z, ...)
    if verbose:
        print(f"\n处理 eval_inputs（{len(eval_inputs.time)} 个时间点）...")

    for t_idx in range(len(eval_inputs.time)):
        # 检查是否有对应的 CYCLONE_CENTERS
        center_idx = t_idx
        if center_idx >= len(CYCLONE_CENTERS):
            if verbose:
                print(f"  ⚠️ 警告: 时间点 {t_idx + 1} 超出 CYCLONE_CENTERS 范围，跳过")
            break

        # 获取参考位置
        ref_lat = CYCLONE_CENTERS[center_idx]['lat']
        ref_lon = CYCLONE_CENTERS[center_idx]['lon']
        time_label = CYCLONE_CENTERS[center_idx]['time']

        # 提取海平面气压数据
        pressure = eval_inputs['mean_sea_level_pressure']

        # 降维处理
        if 'batch' in pressure.dims:
            pressure = pressure.isel(batch=0)
        if 'time' in pressure.dims:
            pressure = pressure.isel(time=t_idx)

        # 在参考位置附近搜索最低气压点
        result = find_local_minimum_pressure(pressure, ref_lat, ref_lon, search_radius)

        # 计算偏差距离
        distance = haversine(result['lat'], result['lon'], ref_lat, ref_lon)

        positions.append({
            'time': time_label,
            'lat': result['lat'],
            'lon': result['lon'],
            'pressure': result['pressure'] / 100,  # Pa → hPa
            'source': 'ERA5'
        })

        if verbose:
            print(f"  时间点 {center_idx + 1}: {time_label}")
            print(f"    参考位置: ({ref_lat:.4f}°, {ref_lon:.4f}°)")
            print(f"    提取位置: ({result['lat']:.4f}°, {result['lon']:.4f}°), 气压: {result['pressure']/100:.1f} hPa")
            print(f"    偏差距离: {distance:.2f} km", end="")
            if distance > 500:
                print(" ⚠️ 偏差过大！")
            elif distance > 200:
                print(" ⚠️ 偏差较大")
            else:
                print(" ✓")

    # 处理 eval_targets (12Z, 18Z, 次日00Z, ...)
    if verbose:
        print(f"\n处理 eval_targets（{len(eval_targets.time)} 个时间点）...")

    # 动态计算要处理的时间点数量，避免索引越界
    num_input_times = len(eval_inputs.time)
    num_available_centers = len(CYCLONE_CENTERS)
    num_targets_to_process = min(len(eval_targets.time),
                                  num_available_centers - num_input_times)

    if verbose and num_targets_to_process < len(eval_targets.time):
        print(f"  ⚠️ 注意: eval_targets 有 {len(eval_targets.time)} 个时间点，")
        print(f"          但 CYCLONE_CENTERS 只有 {num_available_centers} 个位置，")
        print(f"          只处理前 {num_targets_to_process} 个时间点")

    for t_idx in range(num_targets_to_process):
        # 计算对应的 CYCLONE_CENTERS 索引
        center_idx = num_input_times + t_idx

        # 获取参考位置
        ref_lat = CYCLONE_CENTERS[center_idx]['lat']
        ref_lon = CYCLONE_CENTERS[center_idx]['lon']
        time_label = CYCLONE_CENTERS[center_idx]['time']

        # 提取海平面气压数据
        pressure = eval_targets['mean_sea_level_pressure']

        # 降维处理
        if 'batch' in pressure.dims:
            pressure = pressure.isel(batch=0)
        if 'time' in pressure.dims:
            pressure = pressure.isel(time=t_idx)

        # 在参考位置附近搜索最低气压点
        result = find_local_minimum_pressure(pressure, ref_lat, ref_lon, search_radius)

        # 计算偏差距离
        distance = haversine(result['lat'], result['lon'], ref_lat, ref_lon)

        positions.append({
            'time': time_label,
            'lat': result['lat'],
            'lon': result['lon'],
            'pressure': result['pressure'] / 100,  # Pa → hPa
            'source': 'ERA5'
        })

        if verbose:
            print(f"  时间点 {center_idx + 1}: {time_label}")
            print(f"    参考位置: ({ref_lat:.4f}°, {ref_lon:.4f}°)")
            print(f"    提取位置: ({result['lat']:.4f}°, {result['lon']:.4f}°), 气压: {result['pressure']/100:.1f} hPa")
            print(f"    偏差距离: {distance:.2f} km", end="")
            if distance > 500:
                print(" ⚠️ 偏差过大！")
            elif distance > 200:
                print(" ⚠️ 偏差较大")
            else:
                print(" ✓")

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ 提取完成！共 {len(positions)} 个时间点")
        print("=" * 70)

    return positions


# ============================================================================
# 对比可视化函数
# ============================================================================

def plot_comparison(
    era5_positions: List[Dict[str, Any]],
    cyclone_centers: List[Dict[str, Any]],
    save_path: str = "era5_vs_observation.png",
    figsize: tuple = (14, 10),
    dpi: int = 300
) -> None:
    """
    绘制 ERA5 数据集路径 vs 真实观测路径的对比图

    Args:
        era5_positions: ERA5 提取的台风位置列表
        cyclone_centers: 真实观测的台风位置列表
        save_path: 保存路径
        figsize: 图像尺寸
        dpi: 图像分辨率
    """
    print("\n" + "=" * 70)
    print("【绘制对比图】")
    print("=" * 70)

    # 创建图形（不使用地图背景）
    fig, ax = plt.subplots(figsize=figsize)

    # 计算显示范围
    all_lats = [p['lat'] for p in era5_positions] + [c['lat'] for c in cyclone_centers]
    all_lons = [p['lon'] for p in era5_positions] + [c['lon'] for c in cyclone_centers]

    lat_min, lat_max = min(all_lats) - 2, max(all_lats) + 2
    lon_min, lon_max = min(all_lons) - 2, max(all_lons) + 2

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度 (°E)', fontsize=12)
    ax.set_ylabel('纬度 (°N)', fontsize=12)
    ax.grid(True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # 提取坐标
    era5_lats = [p['lat'] for p in era5_positions]
    era5_lons = [p['lon'] for p in era5_positions]
    obs_lats = [c['lat'] for c in cyclone_centers]
    obs_lons = [c['lon'] for c in cyclone_centers]

    # 绘制 ERA5 路径（蓝色菱形）
    ax.plot(era5_lons, era5_lats, color='blue', linestyle='-', marker='D',
            markersize=10, linewidth=2.5, markerfacecolor='lightblue',
            markeredgecolor='blue', markeredgewidth=2,
            label='ERA5 数据集', zorder=4)

    # 绘制真实观测路径（红色圆形）
    ax.plot(obs_lons, obs_lats, color='red', linestyle='--', marker='o',
            markersize=10, linewidth=2.5, markerfacecolor='pink',
            markeredgecolor='red', markeredgewidth=2,
            label='真实观测', zorder=4)

    # 绘制误差连线（灰色虚线）
    for era5, obs in zip(era5_positions, cyclone_centers):
        ax.plot([era5['lon'], obs['lon']], [era5['lat'], obs['lat']],
                color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=3)

    # 标注时间点
    for i, (era5, obs) in enumerate(zip(era5_positions, cyclone_centers)):
        time_str = era5['time'].split()[-1]  # 提取 "00Z", "06Z" 等

        # 标注 ERA5 点
        ax.text(era5['lon'] + 0.3, era5['lat'] + 0.5, f"{time_str}(E)",
                fontsize=9, color='blue', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='blue', alpha=0.7))

        # 标注观测点
        ax.text(obs['lon'] - 0.8, obs['lat'] - 0.5, f"{time_str}(O)",
                fontsize=9, color='red', fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='red', alpha=0.7))

    # 计算误差统计
    errors = [haversine(era5['lat'], era5['lon'], obs['lat'], obs['lon'])
              for era5, obs in zip(era5_positions, cyclone_centers)]

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    # 添加统计文本框
    stats_text = f"误差统计:\n" \
                 f"平均: {mean_error:.1f} km\n" \
                 f"最大: {max_error:.1f} km\n" \
                 f"最小: {min_error:.1f} km"

    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.85))

    # 添加标题和图例
    ax.set_title('台风路径数据验证: ERA5 数据集 vs 真实观测\n'
                 f'台风 Seth (2022年1月) | 平均误差: {mean_error:.1f} km',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, fancybox=True, shadow=True)

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {save_path}")

    plt.show()

    # 打印详细统计
    print("\n" + "=" * 70)
    print("【误差统计】")
    print("=" * 70)
    for i, (era5, obs, error) in enumerate(zip(era5_positions, cyclone_centers, errors)):
        print(f"\n时间点 {i + 1}: {era5['time']}")
        print(f"  ERA5:  ({era5['lat']:.4f}°, {era5['lon']:.4f}°), 气压: {era5['pressure']:.1f} hPa")
        print(f"  观测:  ({obs['lat']:.4f}°, {obs['lon']:.4f}°), 气压: {obs['pressure']:.1f} hPa")
        print(f"  误差:  {error:.2f} km")

    print("\n" + "=" * 70)
    print("统计摘要:")
    print(f"  平均误差: {mean_error:.2f} km")
    print(f"  最大误差: {max_error:.2f} km")
    print(f"  最小误差: {min_error:.2f} km")

    # 一致性判断
    if mean_error < 50:
        status = "✓ 高度一致（优秀）"
    elif mean_error < 100:
        status = "✓ 基本一致（良好）"
    elif mean_error < 200:
        status = "⚠ 存在偏差（可接受）"
    else:
        status = "✗ 数据不一致（需检查）"

    print(f"  总体评估: {status}")
    print("=" * 70)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    print("\n" + "=" * 70)
    print("台风路径数据验证脚本")
    print("验证 ERA5 数据集中的台风位置是否与真实观测一致")
    print("=" * 70)

    # 1. 加载模型配置
    print("\n正在加载模型配置...")
    with open(f"{dir_path_params}/{params_file}", "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    task_config = ckpt.task_config
    print("✓ 模型配置加载完成")

    # 2. 加载数据集
    print("\n正在加载数据集...")
    with open(f"{dir_path_dataset}/{dataset_file}", "rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    print("✓ 数据集加载完成")

    # 3. 提取输入和目标数据
    print("\n正在提取输入和目标数据...")
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config)
    )
    print(f"✓ 数据提取完成")
    print(f"  eval_inputs.time: {len(eval_inputs.time)} 个时间点")
    print(f"  eval_targets.time: {len(eval_targets.time)} 个时间点")

    # 4. 从 ERA5 数据集提取台风位置
    era5_positions = extract_era5_cyclone_positions(
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        verbose=True
    )

    # 5. 绘制对比图
    plot_comparison(
        era5_positions=era5_positions,
        cyclone_centers=CYCLONE_CENTERS,
        save_path="era5_vs_observation.png"
    )

    print("\n" + "=" * 70)
    print("✓ 验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
