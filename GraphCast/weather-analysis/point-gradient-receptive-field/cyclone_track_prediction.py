#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
台风路径预测与对比分析模块 (Cyclone Track Prediction & Comparison)
============================================================================

模块简介:
---------
本模块实现基于 GraphCast 模型的台风路径预测和可视化对比功能。
使用固定的初始输入（例如 00Z + 06Z）一次性预测未来多个时刻的台风位置，
并与真实观测路径进行对比分析。

主要功能:
---------
1. 从模型预测结果中自动提取台风中心位置（基于最低海平面气压或位势高度）
2. 使用固定初始输入进行完整的多步预测
3. 计算预测误差（大圆距离，单位：km）
4. 可视化对比真实路径与预测路径
5. 生成误差统计报告

使用方法:
---------
```python
from cyclone_track_prediction import (
    extract_cyclone_center_from_prediction,
    predict_cyclone_track,
    plot_track_comparison
)

# 1. 预测完整台风路径
predicted_centers = predict_cyclone_track(
    model_forward_fn=run_forward_jitted,
    eval_inputs=eval_inputs,
    eval_targets=eval_targets,
    eval_forcings=eval_forcings,
    cyclone_centers=CYCLONE_CENTERS
)

# 2. 绘制路径对比图
plot_track_comparison(
    true_centers=CYCLONE_CENTERS,
    predicted_centers=predicted_centers,
    save_path="track_comparison.png"
)
```

作者: AI Assistant
创建日期: 2026-01-13
版本: 1.0.0
依赖: jax, xarray, numpy, matplotlib, cartopy

参考文献:
---------
[1] Lam et al. (2023). Learning skillful medium-range global weather forecasting.
    Science, 382(6677), 1416-1421.
============================================================================
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
from math import radians, cos, sin, asin, sqrt

import numpy as np
import xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 添加 graphcast 源码路径（用于独立运行时）
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
PREPROCESS_DIR = SCRIPT_DIR.parent / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

try:
    from graphcast import xarray_jax
except ImportError:
    print("警告: 无法导入 graphcast 模块，请确保路径配置正确")


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
# 台风中心提取
# ============================================================================

def extract_cyclone_center_from_prediction(
    prediction_data: xarray.Dataset,
    time_idx: int = 0,
    method: str = 'mslp',
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    search_radius: float = 10.0
) -> Dict[str, float]:
    """
    从预测数据中提取台风中心位置

    Args:
        prediction_data: 预测数据 (xarray Dataset)
        time_idx: 时间索引
        method: 提取方法
            - 'mslp': 最低海平面气压（推荐，最准确）
            - 'geopotential': 最低位势高度（500hPa）
        ref_lat: 参考纬度（如果提供，则在附近区域搜索；否则全球搜索）
        ref_lon: 参考经度（如果提供，则在附近区域搜索；否则全球搜索）
        search_radius: 搜索半径（度），默认10度

    Returns:
        dict: {'lat': 纬度, 'lon': 经度, 'value': 变量值}
    """
    if method == 'mslp':
        # 使用海平面气压最低点
        var_data = prediction_data['mean_sea_level_pressure']
    elif method == 'geopotential':
        # 使用500hPa位势高度最低点
        var_data = prediction_data['geopotential'].sel(level=500)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mslp' or 'geopotential'.")

    # 降维
    if 'batch' in var_data.dims:
        var_data = var_data.isel(batch=0)
    if 'time' in var_data.dims:
        var_data = var_data.isel(time=time_idx)

    # 如果提供了参考位置，进行区域搜索
    if ref_lat is not None and ref_lon is not None:
        # 获取经纬度数组
        lats = var_data.lat.values
        lons = var_data.lon.values

        # 找到参考位置附近的索引范围
        lat_mask = (lats >= ref_lat - search_radius) & (lats <= ref_lat + search_radius)
        lon_mask = (lons >= ref_lon - search_radius) & (lons <= ref_lon + search_radius)

        # 提取区域数据
        var_data = var_data.sel(
            lat=var_data.lat[lat_mask],
            lon=var_data.lon[lon_mask]
        )

    # 解包为numpy数组
    var_np = xarray_jax.unwrap_data(var_data)
    if hasattr(var_np, 'block_until_ready'):
        var_np.block_until_ready()
    var_np = np.array(var_np)

    # 找到最小值位置
    min_idx = np.unravel_index(np.argmin(var_np), var_np.shape)

    # 正确地通过xarray坐标系统获取经纬度（避免维度顺序假设）
    lat = float(var_data.lat.values[min_idx[0]])
    lon = float(var_data.lon.values[min_idx[1]])
    value = float(var_np[min_idx[0], min_idx[1]])

    return {'lat': lat, 'lon': lon, 'value': value}


# ============================================================================
# 台风路径预测
# ============================================================================

def predict_cyclone_track(
    model_forward_fn: Callable,
    eval_inputs: xarray.Dataset,
    eval_targets: xarray.Dataset,
    eval_forcings: xarray.Dataset,
    cyclone_centers: List[Dict[str, Any]],
    method: str = 'mslp',
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    使用固定初始输入预测完整台风路径

    Args:
        model_forward_fn: JIT编译后的模型前向传播函数
        eval_inputs: 初始输入数据 (例如: 00Z + 06Z)
        eval_targets: 目标数据模板 (用于获取时间步数)
        eval_forcings: 强迫项数据
        cyclone_centers: 真实台风中心点列表（用于对比）
        method: 台风中心提取方法 ('mslp' 或 'geopotential')
        verbose: 是否打印详细信息

    Returns:
        预测的台风中心点列表，每个元素包含:
            - 'time': 时间标签
            - 'lat': 预测的纬度
            - 'lon': 预测的经度
            - 'pressure': 预测的中心气压（hPa）
            - 'is_prediction': True
            - 'error_km': 与真实位置的误差（km）
    """
    if verbose:
        print("\n" + "=" * 70)
        print("【台风路径预测】使用固定初始输入进行完整路径预测")
        print("=" * 70)
        print(f"初始输入时间点: {len(eval_inputs.time)} 个")
        print(f"预测目标时间点: {len(eval_targets.time)} 个")
        print(f"台风中心提取方法: {method}")

    # 运行完整的多步预测
    if verbose:
        print("\n正在运行模型预测...")

    import jax

    full_prediction = model_forward_fn(
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings
    )

    if verbose:
        print(f"✓ 预测完成! 预测时间步数: {len(full_prediction.time)}")

    # 提取每个预测时间步的台风中心位置
    predicted_cyclone_centers = []
    prediction_centers = [c for c in cyclone_centers if not c.get('is_input', True)]

    if verbose:
        print("\n正在提取预测的台风中心位置...")
        print(f"  使用区域搜索（半径 ±10°）避免找到其他低压系统")

    for idx, cyclone in enumerate(prediction_centers):
        if verbose:
            print(f"\n时间步 {idx + 1}/{len(prediction_centers)}: {cyclone['time']}")
            print(f"  参考位置: ({cyclone['lat']:.2f}°, {cyclone['lon']:.2f}°)")

        # 从预测结果中提取台风中心位置（使用参考位置进行区域搜索）
        predicted_center = extract_cyclone_center_from_prediction(
            prediction_data=full_prediction,
            time_idx=idx,
            method=method,
            ref_lat=cyclone['lat'],
            ref_lon=cyclone['lon'],
            search_radius=10.0
        )

        # 计算预测误差
        error_km = haversine(
            predicted_center['lat'], predicted_center['lon'],
            cyclone['lat'], cyclone['lon']
        )

        predicted_cyclone_centers.append({
            'time': cyclone['time'],
            'lat': predicted_center['lat'],
            'lon': predicted_center['lon'],
            'pressure': predicted_center['value'] / 100,  # Pa转换为hPa
            'is_prediction': True,
            'error_km': error_km
        })

        if verbose:
            print(f"  预测位置: ({predicted_center['lat']:.2f}°, {predicted_center['lon']:.2f}°)")
            print(f"  真实位置: ({cyclone['lat']:.2f}°, {cyclone['lon']:.2f}°)")
            print(f"  预测误差: {error_km:.1f} km")

    if verbose:
        mean_error = np.mean([c['error_km'] for c in predicted_cyclone_centers])
        max_error = np.max([c['error_km'] for c in predicted_cyclone_centers])
        print("\n" + "=" * 70)
        print(f"✓ 路径预测完成！共 {len(predicted_cyclone_centers)} 个预测点")
        print(f"  平均误差: {mean_error:.1f} km")
        print(f"  最大误差: {max_error:.1f} km")
        print("=" * 70)

    return predicted_cyclone_centers


# ============================================================================
# 路径对比可视化
# ============================================================================

def plot_track_comparison(
    true_centers: List[Dict[str, Any]],
    predicted_centers: List[Dict[str, Any]],
    title: str = "台风路径对比: 真实观测 vs AI预测",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 300,
    region_radius: float = 15.0
) -> None:
    """
    绘制台风路径对比图

    Args:
        true_centers: 真实台风中心点列表
        predicted_centers: 预测台风中心点列表
        title: 图表标题
        save_path: 保存路径（如果为None则不保存）
        figsize: 图像尺寸
        dpi: 图像分辨率
        region_radius: 区域裁剪半径（度），默认15度（与物理-AI对齐图保持一致）
    """
    print("\n" + "=" * 70)
    print("【台风路径对比可视化】生成路径对比图")
    print("=" * 70)

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建地图
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 计算台风活动区域的中心点（所有台风点的平均位置）
    all_lats = [c['lat'] for c in true_centers] + [c['lat'] for c in predicted_centers]
    all_lons = [c['lon'] for c in true_centers] + [c['lon'] for c in predicted_centers]

    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    print(f"台风活动区域中心: ({center_lat:.2f}°, {center_lon:.2f}°)")
    print(f"使用固定裁剪半径: ±{region_radius}°")

    # 使用固定半径裁剪区域（与物理-AI对齐图保持一致）
    lat_min = center_lat - region_radius
    lat_max = center_lat + region_radius
    lon_min = center_lon - region_radius
    lon_max = center_lon + region_radius

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 添加地理要素
    ax.coastlines(resolution='50m', linewidth=1.2, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    # 添加经纬度网格
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 绘制真实台风路径（紫色实线）
    track_lons = [c['lon'] for c in true_centers]
    track_lats = [c['lat'] for c in true_centers]

    ax.plot(track_lons, track_lats, color='purple', linewidth=3.0,
           linestyle='-', marker='o', markersize=10, markerfacecolor='white',
           markeredgecolor='purple', markeredgewidth=2.5,
           transform=ccrs.PlateCarree(), zorder=4, alpha=0.9,
           label='真实台风路径 (观测)')

    # 标注真实路径的时间点
    for i, c in enumerate(true_centers):
        time_str = c['time'].split()[-1] if ' ' in c['time'] else c['time']
        data_type = '输入' if c.get('is_input', False) else '观测'
        ax.text(c['lon'] + 0.5, c['lat'] + 0.8, f"{time_str}\n({data_type})",
               fontsize=10, color='purple', fontweight='bold',
               transform=ccrs.PlateCarree(), zorder=5,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='purple', alpha=0.85))

    # 绘制AI预测路径（绿色虚线）
    pred_track_lons = [c['lon'] for c in predicted_centers]
    pred_track_lats = [c['lat'] for c in predicted_centers]

    ax.plot(pred_track_lons, pred_track_lats, color='green', linewidth=3.0,
           linestyle='--', marker='s', markersize=10, markerfacecolor='lightgreen',
           markeredgecolor='green', markeredgewidth=2.5,
           transform=ccrs.PlateCarree(), zorder=4, alpha=0.9,
           label='AI预测路径 (GraphCast)')

    # 标注预测路径的时间点
    for i, c in enumerate(predicted_centers):
        time_str = c['time'].split()[-1] if ' ' in c['time'] else c['time']
        ax.text(c['lon'] - 1.2, c['lat'] - 1.0, f"{time_str}\n(预测)",
               fontsize=10, color='green', fontweight='bold',
               transform=ccrs.PlateCarree(), zorder=5,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='green', alpha=0.85))

    # 绘制每个时间点的误差连线（浅灰色虚线）
    prediction_centers_only = [c for c in true_centers if not c.get('is_input', True)]
    for i, (true_c, pred_c) in enumerate(zip(prediction_centers_only, predicted_centers)):
        ax.plot([true_c['lon'], pred_c['lon']], [true_c['lat'], pred_c['lat']],
               color='gray', linewidth=1.5, linestyle=':', alpha=0.5,
               transform=ccrs.PlateCarree(), zorder=3)

    # 标注起点（初始输入时刻）
    # 找到第一个输入时间点
    input_centers = [c for c in true_centers if c.get('is_input', True)]
    if len(input_centers) > 0:
        start_point = input_centers[-1]  # 使用最后一个输入点作为预测起点
        ax.scatter(start_point['lon'], start_point['lat'], marker='*', s=500,
                  c='red', edgecolors='black', linewidths=2,
                  transform=ccrs.PlateCarree(), zorder=6,
                  label=f"预测起点 ({start_point['time'].split()[-1]})")

    # 添加标题
    # 计算预测时长
    if len(predicted_centers) > 0:
        pred_hours = len(predicted_centers) * 6  # 假设每步6小时
        subtitle = f"预测时长: {pred_hours}小时 ({len(predicted_centers)}步)"
    else:
        subtitle = ""

    ax.set_title(f'{title}\n{subtitle}',
                fontsize=16, fontweight='bold', pad=20)

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='purple', linewidth=3.0, marker='o',
                  markersize=10, markerfacecolor='white', markeredgecolor='purple',
                  markeredgewidth=2.5, label='真实台风路径 (观测)'),
        plt.Line2D([0], [0], color='green', linewidth=3.0, linestyle='--', marker='s',
                  markersize=10, markerfacecolor='lightgreen', markeredgecolor='green',
                  markeredgewidth=2.5, label='AI预测路径 (GraphCast)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                  markersize=15, markeredgecolor='black', markeredgewidth=2,
                  label='预测起点'),
        plt.Line2D([0], [0], color='gray', linewidth=1.5, linestyle=':',
                  alpha=0.5, label='预测误差连线'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             framealpha=0.95, fancybox=True, shadow=True)

    # 计算并显示统计信息
    errors = [c['error_km'] for c in predicted_centers]
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)

    # 在图上添加误差统计文本框
    stats_text = f"预测误差统计:\n" \
                 f"平均误差: {mean_error:.1f} km\n" \
                 f"最小误差: {min_error:.1f} km\n" \
                 f"最大误差: {max_error:.1f} km\n" \
                 f"预测点数: {len(errors)}"

    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 台风路径对比图已保存: {save_path}")

    plt.show()

    # 打印详细统计信息
    print("\n预测误差统计:")
    print(f"  平均误差: {mean_error:.1f} km")
    print(f"  最小误差: {min_error:.1f} km")
    print(f"  最大误差: {max_error:.1f} km")
    print(f"  预测点数: {len(errors)}")
    print("=" * 70)


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n使用示例见文档头部的代码块。")
    print("本模块需要配合 GraphCast 模型和数据使用。")
