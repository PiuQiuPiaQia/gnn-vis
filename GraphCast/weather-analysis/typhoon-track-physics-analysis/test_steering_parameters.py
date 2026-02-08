#!/usr/bin/env python3
"""
测试不同环形区域半径参数对引导气流预测准确度的影响

目标：找到最佳的内外半径组合，使引导气流方向与台风实际移动方向最吻合
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

# 数据路径
dir_path_dataset = "/root/data/dataset"
dataset_file = "dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

# 台风位置数据（用于计算实际移动方向）
CYCLONE_POSITIONS = [
    {"time": "2022-01-01 00Z", "lat": -21.2215, "lon": 156.7095, "data_type": "输入(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7810, "lon": 157.4565, "data_type": "输入(0h)", "is_input": True, "input_time_idx": 1},
    {"time": "2022-01-01 12Z", "lat": -22.5571, "lon": 158.2946, "data_type": "预测(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9132, "lon": 158.8048, "data_type": "预测(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8306, "lon": 159.0052, "data_type": "预测(+18h)", "is_input": False, "target_time_idx": 2},
]

# 气压层配置
STEERING_LEVELS = [850, 700, 500, 300, 200]

# 参数扫描范围
INNER_RADIUS_RANGE = [1.0, 2.0, 3.0, 4.0, 5.0]  # 内半径（度）
OUTER_RADIUS_RANGE = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 外半径（度）

# ==================== 辅助函数 ====================

def compute_actual_movement_angle(pos1, pos2):
    """计算台风实际移动角度"""
    dlat = pos2['lat'] - pos1['lat']
    dlon = pos2['lon'] - pos1['lon']
    angle = math.atan2(dlat, dlon) * 180 / math.pi
    return angle


def compute_steering_wind(u_wind_3d, v_wind_3d, center_lat, center_lon,
                          inner_radius, outer_radius, time_idx):
    """计算给定参数下的引导气流"""

    # 处理 Batch 和 Time 维度
    if 'batch' in u_wind_3d.dims:
        u_wind_3d = u_wind_3d.isel(batch=0)
        v_wind_3d = v_wind_3d.isel(batch=0)
    if 'time' in u_wind_3d.dims:
        actual_time_idx = min(time_idx, len(u_wind_3d.time) - 1)
        u_wind_3d = u_wind_3d.isel(time=actual_time_idx)
        v_wind_3d = v_wind_3d.isel(time=actual_time_idx)

    # 对每个气压层计算环形平均
    u_layers = []
    v_layers = []

    for level in STEERING_LEVELS:
        try:
            u_layer = u_wind_3d.sel(level=level, method='nearest')
            v_layer = v_wind_3d.sel(level=level, method='nearest')

            u_mean = extract_annulus_mean(u_layer, center_lat, center_lon,
                                          inner_radius, outer_radius)
            v_mean = extract_annulus_mean(v_layer, center_lat, center_lon,
                                          inner_radius, outer_radius)

            u_layers.append(u_mean)
            v_layers.append(v_mean)
        except:
            pass

    # 简单平均
    if len(u_layers) > 0:
        u_center = sum(u_layers) / len(u_layers)
        v_center = sum(v_layers) / len(v_layers)
        wind_angle = math.atan2(v_center, u_center) * 180 / math.pi
        return wind_angle, u_center, v_center
    else:
        return None, None, None


# ==================== 主流程 ====================

print("正在加载数据集...")
with open(f"{dir_path_dataset}/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# 提取训练数据
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
    'input_duration': '12h',  # 使用过去12小时的数据作为输入
}

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", "24h"),
    **task_config_dict
)

print("\n" + "=" * 80)
print("参数扫描：测试不同的环形区域半径")
print("=" * 80)

# 存储结果
results = []

# 扫描参数空间
for inner_r in INNER_RADIUS_RANGE:
    for outer_r in OUTER_RADIUS_RANGE:
        if outer_r <= inner_r:
            continue  # 外半径必须大于内半径

        print(f"\n测试参数: 内半径={inner_r}°, 外半径={outer_r}°")

        total_error = 0.0
        n_valid = 0
        errors = []

        # 对前4个时刻（有下一时刻数据）进行测试
        for idx in range(len(CYCLONE_POSITIONS) - 1):
            pos1 = CYCLONE_POSITIONS[idx]
            pos2 = CYCLONE_POSITIONS[idx + 1]

            # 获取风场数据
            if pos1.get('is_input', True):
                physics_data = train_inputs
                physics_time_idx = pos1['input_time_idx']
            else:
                physics_data = train_targets
                physics_time_idx = pos1['target_time_idx']

            u_wind_3d = physics_data['u_component_of_wind']
            v_wind_3d = physics_data['v_component_of_wind']

            # 计算引导气流
            pred_angle, u_center, v_center = compute_steering_wind(
                u_wind_3d, v_wind_3d,
                pos1['lat'], pos1['lon'],
                inner_r, outer_r,
                physics_time_idx
            )

            if pred_angle is not None:
                # 计算实际移动方向
                actual_angle = compute_actual_movement_angle(pos1, pos2)

                # 计算误差
                angle_error = abs(pred_angle - actual_angle)
                errors.append(angle_error)
                total_error += angle_error
                n_valid += 1

        # 计算平均误差
        if n_valid > 0:
            avg_error = total_error / n_valid
            max_error = max(errors)
            min_error = min(errors)

            results.append({
                'inner_r': inner_r,
                'outer_r': outer_r,
                'avg_error': avg_error,
                'max_error': max_error,
                'min_error': min_error,
                'errors': errors
            })

            print(f"  平均误差: {avg_error:.1f}°  (最小: {min_error:.1f}°, 最大: {max_error:.1f}°)")

# ==================== 输出结果 ====================

print("\n" + "=" * 80)
print("参数扫描结果汇总（按平均误差排序）")
print("=" * 80)

# 按平均误差排序
results_sorted = sorted(results, key=lambda x: x['avg_error'])

print(f"\n{'排名':<6} {'内半径':<10} {'外半径':<10} {'平均误差':<12} {'最大误差':<12} {'最小误差':<12}")
print("-" * 80)

for rank, r in enumerate(results_sorted[:10], 1):  # 只显示前10名
    print(f"{rank:<6} {r['inner_r']:<10.1f} {r['outer_r']:<10.1f} {r['avg_error']:<12.1f} "
          f"{r['max_error']:<12.1f} {r['min_error']:<12.1f}")

print("\n" + "=" * 80)
print("推荐参数：")
best = results_sorted[0]
print(f"  内半径: {best['inner_r']}°")
print(f"  外半径: {best['outer_r']}°")
print(f"  平均误差: {best['avg_error']:.1f}°")
print("=" * 80)

print("\n详细误差（最佳参数）:")
for idx, err in enumerate(best['errors']):
    pos = CYCLONE_POSITIONS[idx]
    print(f"  图{idx+1} ({pos['time']}): {err:.1f}°")
