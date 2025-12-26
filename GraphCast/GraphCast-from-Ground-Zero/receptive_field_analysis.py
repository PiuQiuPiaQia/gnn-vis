#!/usr/bin/env python
# coding: utf-8

"""
GraphCast 有效感受野分析
=====================

目标：
1. 计算 GraphCast 的有效感受野（Effective Receptive Field, ERF）
2. 与物理过程的影响范围对比
3. 回答问题：模型实际利用了多大范围的输入信息？

方法：
- 输入扰动法（Input Perturbation）：遮挡不同区域观察输出变化
- 梯度方法（Gradient-based）：计算输出对输入的敏感性
- 注意力分析（Attention Analysis）：分析GNN的边权重（如果可访问）
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass
import pickle

# ============================================================================
# 配置与常量
# ============================================================================

@dataclass
class ReceptiveFieldConfig:
    """感受野分析配置"""

    # 目标预测点（关注哪个位置的输出）
    target_lat: float = 40.0  # 北纬40度（例如：北京、纽约）
    target_lon: float = 116.0  # 东经116度
    target_level: Optional[int] = None  # 如果是3D变量，指定层次（None表示2D变量）

    # 扰动配置
    perturbation_radius_degrees: List[float] = None  # 扰动半径列表（度）
    perturbation_magnitude: float = 1.0  # 扰动幅度（标准差的倍数）

    # 采样配置
    n_perturbations: int = 50  # 每个半径的扰动样本数

    # 变量配置
    input_variable: str = '2m_temperature'  # 扰动哪个输入变量
    output_variable: str = '2m_temperature'  # 观察哪个输出变量

    def __post_init__(self):
        if self.perturbation_radius_degrees is None:
            # 默认扫描半径：100km到2000km
            # 纬度1度约111km
            self.perturbation_radius_degrees = [
                1, 2, 5, 10, 15, 20, 30, 40, 50
            ]  # 对应约 111km 到 5550km


# ============================================================================
# 方法1：输入扰动法（Input Perturbation Method）
# ============================================================================

def create_spatial_mask(
    data: xr.DataArray,
    center_lat: float,
    center_lon: float,
    radius_degrees: float,
    mask_type: str = 'circle'
) -> xr.DataArray:
    """创建空间遮挡mask

    Args:
        data: 输入数据（用于获取坐标）
        center_lat: 中心纬度
        center_lon: 中心经度
        radius_degrees: 遮挡半径（度）
        mask_type: 遮挡形状 ('circle', 'square')

    Returns:
        mask: 1表示保留，0表示遮挡
    """
    lat = data.lat.values
    lon = data.lon.values

    # 创建网格
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    if mask_type == 'circle':
        # 球面距离近似（小角度近似）
        dlat = lat_grid - center_lat
        dlon = (lon_grid - center_lon) * np.cos(np.radians(center_lat))
        distance = np.sqrt(dlat**2 + dlon**2)

        mask = xr.DataArray(
            (distance > radius_degrees).astype(float),
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon']
        )

    elif mask_type == 'square':
        # 方形遮挡
        lat_mask = (np.abs(lat_grid - center_lat) > radius_degrees)
        lon_mask = (np.abs(lon_grid - center_lon) > radius_degrees)
        mask_array = (lat_mask | lon_mask).astype(float)

        mask = xr.DataArray(
            mask_array,
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon']
        )

    return mask


def apply_perturbation(
    inputs: xr.Dataset,
    variable: str,
    mask: xr.DataArray,
    magnitude: float = 1.0
) -> xr.Dataset:
    """对输入应用扰动

    Args:
        inputs: 原始输入数据集
        variable: 要扰动的变量名
        mask: 空间mask（0的位置将被扰动）
        magnitude: 扰动幅度（单位：标准差）

    Returns:
        perturbed_inputs: 扰动后的输入
    """
    inputs_perturbed = inputs.copy(deep=True)

    # 获取变量的标准差
    var_std = float(inputs[variable].std())

    # 生成随机扰动
    perturbation = np.random.randn(*inputs[variable].shape) * var_std * magnitude

    # 应用mask（只扰动mask=0的区域）
    if 'level' in inputs[variable].dims:
        # 3D变量：扩展mask到所有层
        mask_3d = mask.expand_dims({'level': inputs[variable].level}, axis=-1)
        perturbation_masked = perturbation * (1 - mask_3d.values)
    else:
        # 2D变量
        perturbation_masked = perturbation * (1 - mask.values)

    # 应用扰动
    inputs_perturbed[variable] = inputs[variable] + perturbation_masked

    return inputs_perturbed


def calculate_output_change(
    original_output: xr.DataArray,
    perturbed_output: xr.DataArray,
    target_lat: float,
    target_lon: float,
    target_level: Optional[int] = None
) -> float:
    """计算目标点输出的变化

    Args:
        original_output: 原始预测
        perturbed_output: 扰动后的预测
        target_lat: 目标纬度
        target_lon: 目标经度
        target_level: 目标压力层（可选）

    Returns:
        change: 输出变化量（RMSE）
    """
    # 提取目标点
    if target_level is not None and 'level' in original_output.dims:
        orig = original_output.sel(lat=target_lat, lon=target_lon, level=target_level, method='nearest')
        pert = perturbed_output.sel(lat=target_lat, lon=target_lon, level=target_level, method='nearest')
    else:
        orig = original_output.sel(lat=target_lat, lon=target_lon, method='nearest')
        pert = perturbed_output.sel(lat=target_lat, lon=target_lon, method='nearest')

    # 计算差异
    change = float(np.abs(orig - pert).mean())

    return change


def compute_receptive_field_perturbation(
    run_forward_fn: Callable,
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    eval_forcings: xr.Dataset,
    config: ReceptiveFieldConfig
) -> Dict[float, List[float]]:
    """使用输入扰动法计算感受野

    Args:
        run_forward_fn: GraphCast前向推理函数
        eval_inputs: 输入数据
        eval_targets: 目标数据（用作模板）
        eval_forcings: 强迫项
        config: 配置

    Returns:
        results: {radius: [changes]}，每个半径对应多次扰动的输出变化
    """
    print("="*80)
    print("方法1: 输入扰动法计算有效感受野")
    print("="*80)

    # 1. 计算原始预测
    print(f"\n[1/3] 计算原始预测...")
    original_predictions = run_forward_fn(
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings
    )

    original_output = original_predictions[config.output_variable].isel(time=0, batch=0)
    print(f"原始预测形状: {original_output.shape}")

    # 2. 对每个半径进行扰动实验
    results = {}

    for radius in config.perturbation_radius_degrees:
        print(f"\n[2/3] 测试半径: {radius:.1f}° (约 {radius*111:.0f} km)")

        changes = []

        # 创建mask（遮挡目标点周围区域）
        mask = create_spatial_mask(
            eval_inputs[config.input_variable].isel(time=-1, batch=0),
            config.target_lat,
            config.target_lon,
            radius,
            mask_type='circle'
        )

        # 多次扰动取平均
        for i in range(config.n_perturbations):
            if (i + 1) % 10 == 0:
                print(f"  扰动样本 {i+1}/{config.n_perturbations}...")

            # 应用扰动
            perturbed_inputs = apply_perturbation(
                eval_inputs,
                config.input_variable,
                mask,
                config.perturbation_magnitude
            )

            # 计算扰动后的预测
            perturbed_predictions = run_forward_fn(
                rng=jax.random.PRNGKey(i+1),
                inputs=perturbed_inputs,
                targets_template=eval_targets * np.nan,
                forcings=eval_forcings
            )

            perturbed_output = perturbed_predictions[config.output_variable].isel(time=0, batch=0)

            # 计算目标点的变化
            change = calculate_output_change(
                original_output,
                perturbed_output,
                config.target_lat,
                config.target_lon,
                config.target_level
            )

            changes.append(change)

        results[radius] = changes
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        print(f"  平均输出变化: {mean_change:.6f} ± {std_change:.6f}")

    print(f"\n[3/3] 扰动实验完成！")

    return results


# ============================================================================
# 方法2：梯度法（Gradient-based Method）
# ============================================================================

def compute_receptive_field_gradient(
    run_forward_fn: Callable,
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    eval_forcings: xr.Dataset,
    config: ReceptiveFieldConfig
) -> xr.DataArray:
    """使用梯度法计算感受野（计算输出对输入的敏感性）

    Args:
        run_forward_fn: GraphCast前向推理函数（需支持梯度）
        eval_inputs: 输入数据
        eval_targets: 目标数据
        eval_forcings: 强迫项
        config: 配置

    Returns:
        sensitivity_map: 输入敏感性地图（绝对值）
    """
    print("="*80)
    print("方法2: 梯度法计算有效感受野")
    print("="*80)

    # 注意：这需要 run_forward_fn 支持梯度计算
    # GraphCast 默认可能没有暴露梯度接口，这里提供框架

    print("[警告] 梯度法需要修改 GraphCast 代码以支持对输入的梯度计算")
    print("[提示] 使用 JAX 的 jax.grad 或 jax.jacrev 对输入变量求导")

    # 伪代码示例：
    # def loss_at_target(inputs):
    #     predictions = run_forward_fn(inputs=inputs, ...)
    #     return predictions[variable].sel(lat=target_lat, lon=target_lon)
    #
    # grad_fn = jax.grad(loss_at_target)
    # sensitivity = grad_fn(eval_inputs)

    # 这里返回一个占位结果
    sensitivity_map = xr.DataArray(
        np.zeros((len(eval_inputs.lat), len(eval_inputs.lon))),
        coords={'lat': eval_inputs.lat, 'lon': eval_inputs.lon},
        dims=['lat', 'lon']
    )

    print("梯度法尚未完全实现，需要修改模型代码")

    return sensitivity_map


# ============================================================================
# 物理过程影响范围计算
# ============================================================================

def calculate_physical_influence_range(
    eval_inputs: xr.Dataset,
    config: ReceptiveFieldConfig,
    time_horizon: float = 6 * 3600  # 6小时（秒）
) -> Dict[str, float]:
    """计算不同物理过程的影响范围

    Args:
        eval_inputs: 输入数据
        config: 配置
        time_horizon: 预报时长（秒）

    Returns:
        ranges: 各物理过程的影响半径（km）
    """
    print("="*80)
    print("计算物理过程的影响范围")
    print("="*80)

    # 提取目标点的气象要素
    current = eval_inputs.isel(time=-1, batch=0)

    # 1. 平流影响范围 = 风速 × 时间
    if config.input_variable in ['2m_temperature']:
        u = current['10m_u_component_of_wind'].sel(
            lat=config.target_lat, lon=config.target_lon, method='nearest'
        ).values
        v = current['10m_v_component_of_wind'].sel(
            lat=config.target_lat, lon=config.target_lon, method='nearest'
        ).values
    else:
        if 'level' in current['u_component_of_wind'].dims:
            level_idx = len(current.level) // 2
            u = current['u_component_of_wind'].isel(level=level_idx).sel(
                lat=config.target_lat, lon=config.target_lon, method='nearest'
            ).values
            v = current['v_component_of_wind'].isel(level=level_idx).sel(
                lat=config.target_lat, lon=config.target_lon, method='nearest'
            ).values
        else:
            u = v = 0

    wind_speed = float(np.sqrt(u**2 + v**2))  # m/s
    advection_distance = wind_speed * time_horizon / 1000  # km

    # 2. 辐射影响范围（局地过程，影响范围小）
    radiation_distance = 50  # km（经验值）

    # 3. 地表能量平衡影响范围（非常局地）
    seb_distance = 10  # km

    # 4. 波动传播影响范围
    # 重力波速度 ~ 10-50 m/s
    gravity_wave_speed = 30  # m/s
    wave_distance = gravity_wave_speed * time_horizon / 1000  # km

    # 5. 大尺度环流影响（罗斯贝波等）
    # 罗斯贝波速度 ~ 几 m/s，6小时可传播几百公里
    rossby_wave_speed = 5  # m/s
    rossby_distance = rossby_wave_speed * time_horizon / 1000  # km

    ranges = {
        '平流过程 (Advection)': advection_distance,
        '辐射过程 (Radiation)': radiation_distance,
        '地表能量平衡 (SEB)': seb_distance,
        '重力波 (Gravity Wave)': wave_distance,
        '罗斯贝波 (Rossby Wave)': rossby_distance
    }

    print(f"\n目标点: ({config.target_lat}°N, {config.target_lon}°E)")
    print(f"预报时长: {time_horizon/3600:.1f} 小时\n")

    for process, distance in ranges.items():
        print(f"{process:30s}: {distance:8.1f} km ({distance/111:.2f}°)")

    return ranges


# ============================================================================
# 可视化
# ============================================================================

def plot_receptive_field_curve(
    perturbation_results: Dict[float, List[float]],
    physical_ranges: Dict[str, float],
    config: ReceptiveFieldConfig,
    save_path: Optional[str] = None
):
    """绘制感受野曲线

    Args:
        perturbation_results: 扰动实验结果
        physical_ranges: 物理过程影响范围
        config: 配置
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # 准备数据
    radii = sorted(perturbation_results.keys())
    means = [np.mean(perturbation_results[r]) for r in radii]
    stds = [np.std(perturbation_results[r]) for r in radii]

    # 转换为km
    radii_km = [r * 111 for r in radii]

    # 1. 绘制感受野曲线
    ax.plot(radii_km, means, 'o-', linewidth=2, markersize=8,
            label='GraphCast ERF', color='#2E86AB')
    ax.fill_between(radii_km,
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.3, color='#2E86AB')

    # 2. 标记物理过程影响范围
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C']
    for i, (process, distance) in enumerate(physical_ranges.items()):
        ax.axvline(distance, linestyle='--', linewidth=2,
                   color=colors[i % len(colors)],
                   label=process, alpha=0.7)

    # 3. 标注有效感受野半径（输出变化达到饱和的80%处）
    max_change = max(means)
    threshold = 0.8 * max_change

    # 找到超过阈值的最小半径
    effective_radius_km = None
    for r_km, m in zip(radii_km, means):
        if m >= threshold:
            effective_radius_km = r_km
            break

    if effective_radius_km:
        ax.axvline(effective_radius_km, linestyle=':', linewidth=3,
                   color='red', label=f'有效感受野: {effective_radius_km:.0f} km')
        ax.axhline(threshold, linestyle=':', linewidth=1, color='gray', alpha=0.5)

    # 4. 设置坐标轴
    ax.set_xlabel('扰动半径 (km)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'目标点输出变化 ({config.output_variable})', fontsize=14, fontweight='bold')
    ax.set_title(
        f'GraphCast 有效感受野 vs 物理过程影响范围\n'
        f'目标点: ({config.target_lat}°N, {config.target_lon}°E)',
        fontsize=16, fontweight='bold'
    )

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 5. 添加说明文本
    textstr = (
        f'扰动变量: {config.input_variable}\n'
        f'输出变量: {config.output_variable}\n'
        f'扰动次数: {config.n_perturbations} per radius'
    )
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5), fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()

    # 打印关键结果
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    if effective_radius_km:
        print(f"GraphCast 有效感受野半径: ~{effective_radius_km:.0f} km ({effective_radius_km/111:.1f}°)")

    # 比较
    advection_range = physical_ranges.get('平流过程 (Advection)', 0)
    if effective_radius_km and advection_range > 0:
        ratio = effective_radius_km / advection_range
        print(f"\n感受野 vs 平流影响范围: {ratio:.2f}x")

        if ratio > 2:
            print("-> GraphCast 利用了远超平流过程的空间信息！")
            print("   可能捕捉了大尺度环流、波动传播等过程")
        elif ratio > 1:
            print("-> GraphCast 的感受野略大于平流过程")
        else:
            print("-> GraphCast 的感受野小于平流过程的理论影响范围")
            print("   可能是由于模型架构限制或训练策略导致")


def plot_sensitivity_map(
    sensitivity: xr.DataArray,
    config: ReceptiveFieldConfig,
    physical_ranges: Dict[str, float],
    save_path: Optional[str] = None
):
    """绘制输入敏感性地图

    Args:
        sensitivity: 敏感性地图
        config: 配置
        physical_ranges: 物理过程影响范围
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8),
                           subplot_kw={'projection': None})

    # 绘制敏感性
    im = ax.imshow(sensitivity, cmap='hot', origin='lower',
                   extent=[sensitivity.lon.min(), sensitivity.lon.max(),
                          sensitivity.lat.min(), sensitivity.lat.max()],
                   aspect='auto')

    # 标记目标点
    ax.plot(config.target_lon, config.target_lat, 'b*',
            markersize=20, label='目标点')

    # 绘制物理过程影响范围（圆圈）
    for process, distance_km in physical_ranges.items():
        radius_deg = distance_km / 111  # km转度
        circle = plt.Circle((config.target_lon, config.target_lat),
                           radius_deg, fill=False,
                           linestyle='--', linewidth=2,
                           label=process)
        ax.add_patch(circle)

    plt.colorbar(im, ax=ax, label='输入敏感性')
    ax.set_xlabel('经度 (°E)', fontsize=12)
    ax.set_ylabel('纬度 (°N)', fontsize=12)
    ax.set_title(f'GraphCast 输入敏感性地图\n目标点: ({config.target_lat}°N, {config.target_lon}°E)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# ============================================================================
# 主函数
# ============================================================================

def analyze_receptive_field(
    run_forward_jitted: Callable,
    eval_inputs: xr.Dataset,
    eval_targets: xr.Dataset,
    eval_forcings: xr.Dataset,
    config: Optional[ReceptiveFieldConfig] = None,
    method: str = 'perturbation',  # 'perturbation' or 'gradient'
    save_results: bool = True
) -> Dict:
    """完整的感受野分析流程

    Args:
        run_forward_jitted: GraphCast前向推理函数
        eval_inputs: 输入数据
        eval_targets: 目标数据
        eval_forcings: 强迫项
        config: 配置（None则使用默认）
        method: 分析方法
        save_results: 是否保存结果

    Returns:
        results: 分析结果字典
    """
    if config is None:
        config = ReceptiveFieldConfig()

    print("="*80)
    print("GraphCast 有效感受野分析")
    print("="*80)
    print(f"目标点: ({config.target_lat}°N, {config.target_lon}°E)")
    print(f"输入变量: {config.input_variable}")
    print(f"输出变量: {config.output_variable}")
    print(f"方法: {method}")
    print("="*80)

    results = {}

    # 1. 计算物理过程影响范围
    physical_ranges = calculate_physical_influence_range(eval_inputs, config)
    results['physical_ranges'] = physical_ranges

    # 2. 计算模型感受野
    if method == 'perturbation':
        perturbation_results = compute_receptive_field_perturbation(
            run_forward_jitted, eval_inputs, eval_targets, eval_forcings, config
        )
        results['perturbation_results'] = perturbation_results

        # 可视化
        plot_receptive_field_curve(
            perturbation_results, physical_ranges, config,
            save_path='receptive_field_analysis.png' if save_results else None
        )

    elif method == 'gradient':
        sensitivity = compute_receptive_field_gradient(
            run_forward_jitted, eval_inputs, eval_targets, eval_forcings, config
        )
        results['sensitivity_map'] = sensitivity

        # 可视化
        plot_sensitivity_map(
            sensitivity, config, physical_ranges,
            save_path='sensitivity_map.png' if save_results else None
        )

    # 3. 保存结果
    if save_results:
        with open('receptive_field_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\n结果已保存至: receptive_field_results.pkl")

    return results


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n使用示例:")
    print("```python")
    print("# 配置")
    print("config = ReceptiveFieldConfig(")
    print("    target_lat=40.0,  # 北京")
    print("    target_lon=116.0,")
    print("    input_variable='2m_temperature',")
    print("    output_variable='2m_temperature',")
    print("    n_perturbations=50")
    print(")")
    print("")
    print("# 运行分析")
    print("results = analyze_receptive_field(")
    print("    run_forward_jitted,")
    print("    eval_inputs,")
    print("    eval_targets,")
    print("    eval_forcings,")
    print("    config=config,")
    print("    method='perturbation'")
    print(")")
    print("```")
