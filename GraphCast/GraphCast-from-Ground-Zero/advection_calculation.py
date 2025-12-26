#!/usr/bin/env python
# coding: utf-8

"""
基于平流方程计算6小时后的气象值
公式: T(t+Δt)(x,y,z) = T(t)(x,y,z) - Δt[u∂T/∂x + v∂T/∂y + w∂T/∂z]
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional

# ------------------ 常量 ------------------
# 干空气定压比热 (J kg-1 K-1)
CP_AIR: float = 1004.0
# 近地层空气密度 (kg m-3)
RHO_AIR: float = 1.225
# 重力加速度 (m s-2)
G: float = 9.80665
# 一个标准柱质量 (kg m-2) —— 使用 1000 hPa
MASS_COLUMN: float = 1.0e5 / G  # ≈ 1.02e4
# 大气对入射短波的平均吸收系数（经验值 0.2~0.3）
SW_ABSORB_FRACTION: float = 0.25
# 水的汽化潜热 (J kg-1)
L_V: float = 2.5e6

# ------------------ 调试工具 ------------------
def _log_q_solar(q_solar: Optional[xr.DataArray]):
    """精简日志：仅打印形状、单位与均值。"""
    if q_solar is None:
        print("[警告] 缺少 'toa_incident_solar_radiation'，未添加辐射加热。")
        return

    units = q_solar.attrs.get("units", "未知")
    print(f"[辐射] shape:{q_solar.shape} unit:{units} mean:{float(q_solar.mean()):.2f}")

def calculate_advection_forecast(eval_inputs: xr.Dataset, dt: float = 6*3600) -> Dict[str, xr.DataArray]:
    """
    使用平流方程计算6小时后的气象场
    
    Args:
        eval_inputs: GraphCast输入数据集
        dt: 时间步长(秒)，默认6小时
    
    Returns:
        包含预报结果的字典
    """
    
    # 提取当前时刻的数据 (最后一个时间步)
    current_data = eval_inputs.isel(time=-1, batch=0)
    
    results = {}
    
    # 提取风场和太阳辐射数据
    u_wind = current_data.get('u_component_of_wind', None)
    v_wind = current_data.get('v_component_of_wind', None) 
    w_wind = current_data.get('vertical_velocity', None)
    q_solar = current_data.get('toa_incident_solar_radiation', None)
    q_latent = current_data.get('total_precipitation_6hr', None)
    
    if q_latent is not None:
        print(f"q_latent: {q_latent}")
        print(f"q_latent 最大值: {float(q_latent.max()):.4f}, 最小值: {float(q_latent.min()):.4f}")
    else:
        print("q_latent 为 None")
    
    # 日志输出
    _log_q_solar(q_solar)
    
    # 对温度场进行平流计算
    if 'temperature' in current_data and u_wind is not None and v_wind is not None:
        temp_forecast = advection_step(
            current_data['temperature'], 
            u_wind, v_wind, w_wind, dt,
            q_solar=q_solar,
            q_latent=q_latent
        )
        results['temperature'] = temp_forecast
    
    # 对2米温度进行平流计算 (使用10米风)
    if '2m_temperature' in current_data:
        u_10m = current_data.get('10m_u_component_of_wind', None)
        v_10m = current_data.get('10m_v_component_of_wind', None)
        
        if u_10m is not None and v_10m is not None:
            temp_2m_forecast = advection_step_2d(
                current_data['2m_temperature'],
                u_10m, v_10m, dt,
                q_solar=q_solar,
                q_latent=q_latent
            )
            results['2m_temperature'] = temp_2m_forecast
    
    # 对比湿进行平流计算
    if 'specific_humidity' in current_data and u_wind is not None and v_wind is not None:
        humidity_forecast = advection_step(
            current_data['specific_humidity'],
            u_wind, v_wind, w_wind, dt,
            q_solar=None,
            q_latent=None
        )
        results['specific_humidity'] = humidity_forecast
    
    return results

# -------------------------------------------------------------
# 3D 平流 + 太阳辐射加热
# -------------------------------------------------------------
def advection_step(field: xr.DataArray, 
                   u_wind: xr.DataArray, 
                   v_wind: xr.DataArray, 
                   w_wind: Optional[xr.DataArray],
                   dt: float,
                   q_solar: Optional[xr.DataArray] = None,
                   q_latent: Optional[xr.DataArray] = None) -> xr.DataArray:
    """
    使用平流方程计算一个时间步后的3D场
    
    Args:
        field: 当前时刻的标量场（温度、湿度等）
        u_wind: x方向(经度)风速分量
        v_wind: y方向(纬度)风速分量  
        w_wind: z方向(垂直)风速分量
        dt: 时间步长(秒)
        q_solar: 太阳辐射量
    
    Returns:
        下一时刻的标量场
    """
    
    # 地球半径
    R = 6.371e6  # 米
    
    # 初始化平流项
    advection_term = xr.zeros_like(field)
    
    # 计算经度方向平流 u * ∂T/∂x - 修正单位转换
    if 'lon' in field.dims:
        # 计算经度梯度 (K/度)
        dT_dx_deg = field.differentiate('lon')
        
        # 转换为 K/米 - 考虑纬度变化
        lat_rad = field.lat * np.pi / 180
        dlon_rad = np.pi / 180  # 1度对应的弧度
        dx_m = R * np.cos(lat_rad) * dlon_rad  # 每度经度对应的米数
        dT_dx_m = dT_dx_deg / dx_m
        
        # U平流项 (K/s)
        advection_term += u_wind * dT_dx_m
    
    # 计算纬度方向平流 v * ∂T/∂y - 修正单位转换
    if 'lat' in field.dims:
        # 计算纬度梯度 (K/度)
        dT_dy_deg = field.differentiate('lat')
        
        # 转换为 K/米
        dlat_rad = np.pi / 180  # 1度对应的弧度
        dy_m = R * dlat_rad  # 每度纬度对应的米数
        dT_dy_m = dT_dy_deg / dy_m
        
        # V平流项 (K/s)
        advection_term += v_wind * dT_dy_m
    
    # 计算垂直方向平流 w * ∂T/∂z - 简化处理
    if 'level' in field.dims and w_wind is not None:
        dT_dz = field.differentiate('level')
        # 垂直平流项添加缩放因子，因为压力坐标系比较复杂
        advection_term += w_wind * dT_dz * 0.01
    
    # 计算温度变化
    temperature_change = -dt * advection_term
    
    # 数值稳定性检查
    temperature_change = xr.where(np.isnan(temperature_change), 0, temperature_change)
    temperature_change = xr.where(np.isinf(temperature_change), 0, temperature_change)
    
    # 太阳辐射加热
    heating_solar = compute_radiative_heating(field, q_solar, dt)
    heating_latent = compute_latent_heating(field, q_latent, dt)
    heating_term = heating_solar + heating_latent

    # 应用平流方程: T(t+Δt) = T(t) + 温度变化 + 辐射加热
    new_field = field + temperature_change + heating_term
    
    return new_field

# -------------------------------------------------------------
# 2D 平流 + 太阳辐射加热
# -------------------------------------------------------------
def advection_step_2d(field: xr.DataArray, 
                      u_wind: xr.DataArray, 
                      v_wind: xr.DataArray, 
                      dt: float,
                      q_solar: Optional[xr.DataArray] = None,
                      q_latent: Optional[xr.DataArray] = None) -> xr.DataArray:
    """
    使用平流方程计算一个时间步后的2D场（如2米温度）
    
    Args:
        field: 当前时刻的2D标量场
        u_wind: x方向(经度)风速分量
        v_wind: y方向(纬度)风速分量  
        dt: 时间步长(秒)
        q_solar: 太阳辐射量
    
    Returns:
        下一时刻的2D标量场
    """
    
    # 地球半径
    R = 6.371e6  # 米
    
    # 初始化平流项
    advection_term = xr.zeros_like(field)
    
    # 计算经度方向平流 u * ∂T/∂x - 修正单位转换
    if 'lon' in field.dims:
        # 计算经度梯度 (K/度)
        dT_dx_deg = field.differentiate('lon')
        
        # 转换为 K/米 - 考虑纬度变化
        lat_rad = field.lat * np.pi / 180
        dlon_rad = np.pi / 180  # 1度对应的弧度
        dx_m = R * np.cos(lat_rad) * dlon_rad  # 每度经度对应的米数
        dT_dx_m = dT_dx_deg / dx_m
        
        # U平流项 (K/s)
        advection_term += u_wind * dT_dx_m
    
    # 计算纬度方向平流 v * ∂T/∂y - 修正单位转换
    if 'lat' in field.dims:
        # 计算纬度梯度 (K/度)
        dT_dy_deg = field.differentiate('lat')
        
        # 转换为 K/米
        dlat_rad = np.pi / 180  # 1度对应的弧度
        dy_m = R * dlat_rad  # 每度纬度对应的米数
        dT_dy_m = dT_dy_deg / dy_m
        
        # V平流项 (K/s)
        advection_term += v_wind * dT_dy_m
    
    # 计算温度变化
    temperature_change = -dt * advection_term
    
    # 数值稳定性检查
    temperature_change = xr.where(np.isnan(temperature_change), 0, temperature_change)
    temperature_change = xr.where(np.isinf(temperature_change), 0, temperature_change)
    
    # 太阳辐射加热（2D 场）
    heating_solar = compute_radiative_heating(field, q_solar, dt)
    heating_latent = compute_latent_heating(field, q_latent, dt)
    heating_term = heating_solar + heating_latent

    # 应用平流方程: T(t+Δt) = T(t) + 温度变化 + 辐射加热
    new_field = field + temperature_change + heating_term
    
    return new_field

def compute_radiative_heating(field: xr.DataArray,
                              q_solar: Optional[xr.DataArray],
                              dt: float,
                              absorb_frac: float = SW_ABSORB_FRACTION,
                              mass_column: float = MASS_COLUMN) -> xr.DataArray:
    """根据太阳辐射计算温度增量 ΔT。

    公式: ΔT = α · Q_energy / (c_p · m_column)
    * 若 q_solar 的最大值 <1e5，视为 W m⁻²，则 Q_energy = q_solar·dt。
    * 否则视为 J m⁻² 累积能量。
    * 若 field 含 'level' 维，则将 ΔT 均匀分配到各层。
    """

    if q_solar is None:
        return 0

    # 判断单位并换算能量
    q_energy = q_solar * dt if float(q_solar.max()) < 1e5 else q_solar

    # 吸收比例
    q_energy = absorb_frac * q_energy

    delta_t = q_energy / (CP_AIR * mass_column)

    if 'level' in field.dims:
        nlev = field.sizes['level']
        delta_t = (delta_t / nlev).expand_dims({'level': field.level}, axis=0)

    # 调试输出：打印能量与温升统计信息
    print(
        f"[辐射计算] q_energy[J m-2] min/max/mean: "
        f"{float(q_energy.min()):.2f} / {float(q_energy.max()):.2f} / {float(q_energy.mean()):.2f}" )
    print(
        f"[辐射计算] ΔT[K] min/max/mean: "
        f"{float(delta_t.min()):.4f} / {float(delta_t.max()):.4f} / {float(delta_t.mean()):.4f}" )

    return delta_t

# ------------------ 潜热加热计算 ------------------
def compute_latent_heating(field: xr.DataArray,
                           q_latent: Optional[xr.DataArray],
                           dt: float,
                           mass_column: float = MASS_COLUMN) -> xr.DataArray:
    """根据潜热释放计算温度增量 ΔT_latent。

    支持两类输入：
    1. 已为功率密度/通量 (W m-2 或 W m-3)。
    2. 6h 累积降水量 total_precipitation_6hr (kg m-2)。
    """

    if q_latent is None:
        return 0

    # 若数据像降水量 (最大值 < 10, 单位未写 W)
    if float(q_latent.max()) < 50:  # 粗略判断: mm 累积量
        # 转 kg m-2 6h-1 → kg m-2 s-1
        precip_rate = q_latent / dt
        q_power = L_V * precip_rate  # W m-2
    else:
        q_power = q_latent  # 已是 W m-2

    delta_t = q_power * dt / (CP_AIR * mass_column)

    if 'level' in field.dims:
        nlev = field.sizes['level']
        delta_t = (delta_t / nlev).expand_dims({'level': field.level}, axis=0)

    # 简短日志
    print(f"[潜热计算] ΔT[K] mean:{float(delta_t.mean()):.4f}")

    return delta_t

def calculate_correlation(advection_results: Dict[str, xr.DataArray], 
                         graphcast_predictions: xr.Dataset) -> Dict[str, float]:
    """
    计算平流方程结果与GraphCast预测结果的相关性
    
    Args:
        advection_results: 平流方程计算结果
        graphcast_predictions: GraphCast预测结果
    
    Returns:
        各变量的相关系数字典
    """
    
    correlations = {}
    
    for var_name, advection_data in advection_results.items():
        if var_name in graphcast_predictions:
            gc_data = graphcast_predictions[var_name].isel(time=0, batch=0)
            
            # 确保数据形状匹配
            if advection_data.shape == gc_data.shape:
                # 计算相关系数
                correlation = float(xr.corr(advection_data, gc_data))
                correlations[var_name] = correlation
            else:
                print(f"警告: {var_name} 形状不匹配 - Advection: {advection_data.shape}, GraphCast: {gc_data.shape}")
    
    return correlations

def print_correlation_results(correlations: Dict[str, float]):
    """打印相关性结果"""
    print("=" * 60)
    print("平流方程 vs GraphCast 相关性分析")
    print("=" * 60)
    
    for var_name, correlation in correlations.items():
        if correlation > 0.8:
            quality = "很好"
        elif correlation > 0.6:
            quality = "良好"
        elif correlation > 0.4:
            quality = "一般"
        else:
            quality = "较差"
        
        print(f"{var_name}: {correlation:.4f} ({quality})")
    
    # 计算平均相关性
    if correlations:
        avg_correlation = np.mean(list(correlations.values()))
        print(f"\n平均相关性: {avg_correlation:.4f}")

def calculate_enhanced_advection_forecast(eval_inputs: xr.Dataset, dt: float = 6*3600) -> Dict[str, xr.DataArray]:
    """计算 6 小时平流预报（已包含太阳辐射加热）。

    该封装函数等价于调用 :func:`calculate_advection_forecast`，仅在缺少
    ``toa_incident_solar_radiation`` 时打印一次警告，方便上层 Notebook 调用。
    """

    # 若输入缺失辐射量，给出提示
    if 'toa_incident_solar_radiation' not in eval_inputs.data_vars:
        print("[警告] 输入数据集中缺少 'toa_incident_solar_radiation'，将无法添加辐射加热。")

    # 直接调用主函数（内部已处理辐射加热）
    return calculate_advection_forecast(eval_inputs, dt)

if __name__ == "__main__":
    print("平流方程气象预报模块")
    print("主要功能:")
    print("1. calculate_advection_forecast() - 使用平流方程计算6小时预报")
    print("2. calculate_correlation() - 计算与GraphCast结果的相关性")
    print("3. print_correlation_results() - 打印相关性结果")
    print("4. calculate_enhanced_advection_forecast() - 扩展的平流预报") 