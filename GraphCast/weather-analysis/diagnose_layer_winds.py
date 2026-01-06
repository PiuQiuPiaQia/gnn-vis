#!/usr/bin/env python3
"""
诊断脚本：检查各气压层的风向，找出为什么平均后的引导气流方向不准确
"""

import sys
from pathlib import Path
import math

# 添加路径
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
PREPROCESS_DIR = SCRIPT_DIR.parent / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

import dataclasses
import numpy as np
import xarray

from graphcast import data_utils
from region_utils import extract_annulus_mean

# ==================== 配置 ====================

dir_path_dataset = "/root/data/dataset"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

CYCLONE_POSITIONS = [
    {"time": "2022-01-01 00Z", "lat": -21.2215, "lon": 156.7095, "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7810, "lon": 157.4565, "is_input": True, "input_time_idx": 1},
    {"time": "2022-01-01 12Z", "lat": -22.5571, "lon": 158.2946, "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9132, "lon": 158.8048, "is_input": False, "target_time_idx": 1},
]

STEERING_LEVELS = [850, 700, 500, 300, 200]

# 最佳参数
INNER_RADIUS = 2.0
OUTER_RADIUS = 5.0

# 台风实际移动方向（用于对比）
ACTUAL_ANGLES = [-36.8, -42.8, -69.4, -84.0]

# ==================== 加载数据 ====================

print("正在加载数据集...")
with open(f"{dir_path_dataset}/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

task_config_dict = {
    'input_variables': ['geopotential', 'specific_humidity', 'temperature',
                       'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                       '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
                       '10m_u_component_of_wind', 'total_precipitation_6hr'],
    'target_variables': ['geopotential', 'specific_humidity', 'temperature',
                        'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                        '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
                        '10m_u_component_of_wind', 'total_precipitation_6hr'],
    'forcing_variables': ['toa_incident_solar_radiation'],
    'pressure_levels': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    'input_duration': '12h',
}

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", "24h"),
    **task_config_dict
)

# ==================== 主流程 ====================

print("\n" + "=" * 100)
print("各气压层引导气流方向诊断")
print("=" * 100)

for idx, pos in enumerate(CYCLONE_POSITIONS):
    print(f"\n{'=' * 100}")
    print(f"图{idx+1}: {pos['time']} (台风实际移动方向: {ACTUAL_ANGLES[idx]:.1f}°)")
    print(f"{'=' * 100}")

    # 获取风场数据
    if pos.get('is_input', True):
        physics_data = train_inputs
        physics_time_idx = pos['input_time_idx']
    else:
        physics_data = train_targets
        physics_time_idx = pos['target_time_idx']

    u_wind_3d = physics_data['u_component_of_wind']
    v_wind_3d = physics_data['v_component_of_wind']

    # 处理 Batch 和 Time 维度
    if 'batch' in u_wind_3d.dims:
        u_wind_3d = u_wind_3d.isel(batch=0)
        v_wind_3d = v_wind_3d.isel(batch=0)
    if 'time' in u_wind_3d.dims:
        actual_time_idx = min(physics_time_idx, len(u_wind_3d.time) - 1)
        u_wind_3d = u_wind_3d.isel(time=actual_time_idx)
        v_wind_3d = v_wind_3d.isel(time=actual_time_idx)

    # 打印各层详情
    print(f"\n{'气压层':<10} {'u (m/s)':<12} {'v (m/s)':<12} {'风向 (°)':<12} {'风速 (m/s)':<12} {'与实际差 (°)':<15}")
    print("-" * 100)

    u_layers = []
    v_layers = []

    for level in STEERING_LEVELS:
        try:
            u_layer = u_wind_3d.sel(level=level, method='nearest')
            v_layer = v_wind_3d.sel(level=level, method='nearest')

            # 计算环形平均
            u_mean = extract_annulus_mean(u_layer, pos['lat'], pos['lon'],
                                          INNER_RADIUS, OUTER_RADIUS)
            v_mean = extract_annulus_mean(v_layer, pos['lat'], pos['lon'],
                                          INNER_RADIUS, OUTER_RADIUS)

            u_layers.append(u_mean)
            v_layers.append(v_mean)

            # 计算风向和风速
            wind_angle = math.atan2(v_mean, u_mean) * 180 / math.pi
            wind_speed = math.sqrt(u_mean**2 + v_mean**2)
            angle_diff = abs(wind_angle - ACTUAL_ANGLES[idx])

            print(f"{level:<10} {u_mean:<12.2f} {v_mean:<12.2f} {wind_angle:<12.1f} "
                  f"{wind_speed:<12.2f} {angle_diff:<15.1f}")

        except Exception as e:
            print(f"{level:<10} ERROR: {e}")

    # 计算平均
    if len(u_layers) > 0:
        u_avg = sum(u_layers) / len(u_layers)
        v_avg = sum(v_layers) / len(v_layers)
        avg_angle = math.atan2(v_avg, u_avg) * 180 / math.pi
        avg_speed = math.sqrt(u_avg**2 + v_avg**2)
        avg_diff = abs(avg_angle - ACTUAL_ANGLES[idx])

        print("-" * 100)
        print(f"{'平均':<10} {u_avg:<12.2f} {v_avg:<12.2f} {avg_angle:<12.1f} "
              f"{avg_speed:<12.2f} {avg_diff:<15.1f}")

        # 分析
        print(f"\n【分析】")
        angles = [math.atan2(v, u) * 180 / math.pi for u, v in zip(u_layers, v_layers)]
        angle_std = np.std(angles)
        print(f"  各层风向标准差: {angle_std:.1f}°")

        if angle_std > 30:
            print(f"  ⚠️  各层风向差异很大（标准差 > 30°），简单平均可能不合理")
            print(f"  建议: 考虑只使用低层（850-500hPa）或高层（500-200hPa）")

        # 找出最接近实际方向的层
        layer_diffs = [abs(ang - ACTUAL_ANGLES[idx]) for ang in angles]
        best_idx = layer_diffs.index(min(layer_diffs))
        print(f"  最接近实际方向的层: {STEERING_LEVELS[best_idx]} hPa (误差 {min(layer_diffs):.1f}°)")

print("\n" + "=" * 100)
print("诊断完成")
print("=" * 100)
