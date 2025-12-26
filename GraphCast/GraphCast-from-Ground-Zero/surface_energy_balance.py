#!/usr/bin/env python
# coding: utf-8

"""
基于地表能量平衡（Surface Energy Balance, SEB）
估算 6 小时后的气温（2 m 温度）

核心公式：

    ΔT = (R_n - H - LE - G) · dt / (ρ · c_p · z_heat)

其中
    R_n  – 净辐射（W m⁻²）
    H    – 感热通量（W m⁻²），此处采用经验系数估计
    LE   – 潜热通量（W m⁻²），此处采用经验系数估计
    G    – 地表热通量（W m⁻²），简化为 R_n 的固定比例
    ρ    – 空气密度（kg m⁻³）
    c_p  – 定压比热（J kg⁻¹ K⁻¹）
    z_heat – 受能量影响的有效厚度，这里取 2 m

在缺乏全部辐射量观测的情况下，
本实现仅依赖于示例数据集中常见的变量：
    * ``toa_incident_solar_radiation`` – 到达顶层大气的短波辐射
    * ``2m_temperature``
    * ``10m_u_component_of_wind`` 与 ``10m_v_component_of_wind``
    * ``specific_humidity``
    * ``land_sea_mask`` – 用于按下垫面区分反照率

所有系数均为经验取值，仅用于教学/对比实验。
如需科研级别的精度，请根据站点实测数据调参。
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Dict

# ------------------ 常量 ------------------
CP_AIR: float = 1004.0          # J kg⁻¹ K⁻¹
RHO_AIR: float = 1.225          # kg m⁻³
SIGMA: float = 5.670374419e-8   # W m⁻² K⁻⁴
Z_HEAT: float = 50.0            # m, 有效热容量深度（增加到50m，代表混合层）

# 反照率（可按下垫面调整）
ALBEDO_LAND: float = 0.25
ALBEDO_WATER: float = 0.06

# 地表放射率
EMISSIVITY_SURF: float = 0.95

# 大气放射率（经验值）
EMISSIVITY_ATM: float = 0.8

# 地热通量系数 G = fG * R_n
GROUND_FLUX_FRACTION: float = 0.1

# 涡动交换经验系数（Bulk transfer）- 降低系数
C_H: float = 0.0008  # 感热（降低）
C_E: float = 0.0008  # 潜热（降低）

# 潜热汽化 (J kg⁻¹)
L_V: float = 2.5e6

# 温度变化限制（防止极端值）
MAX_TEMP_CHANGE: float = 5.0   # K（降低限制）

# 新增：物理约束常量
MAX_SPECIFIC_HUMIDITY_DIFF: float = 0.010  # kg/kg，最大比湿差值
MAX_LATENT_HEAT_FLUX: float = 150.0        # W/m²，最大潜热通量
MIN_PRESSURE: float = 800.0                # hPa，最小压力（高海拔地区）

# ------------------ 辅助函数 ------------------

def _calc_wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """计算风速模长 (m s⁻¹)。"""
    return np.sqrt(u**2 + v**2)


def _calc_saturation_vapor_pressure(T: xr.DataArray) -> xr.DataArray:
    """克劳修斯–克拉佩龙近似 (hPa)。"""
    Tc = T - 273.15
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))


def _calc_specific_humidity_from_vapor_pressure(e: xr.DataArray, p: float = 1013.0) -> xr.DataArray:
    """由水汽压估算比湿 (kg kg⁻¹)，默认地面气压 1013 hPa。
    
    增加数值稳定性处理，避免分母过小的问题。
    """
    epsilon = 0.622
    
    # 确保压力不会太小
    p_safe = max(p, MIN_PRESSURE)
    
    # 限制水汽压不超过总压力的95%，避免分母过小
    e_safe = np.minimum(e, 0.95 * p_safe)
    
    # 计算比湿，增加数值稳定性
    denominator = p_safe - (1 - epsilon) * e_safe
    
    # 进一步确保分母不会太小
    denominator = np.maximum(denominator, p_safe * 0.1)
    
    q = epsilon * e_safe / denominator
    
    # 物理约束：比湿不能超过饱和值
    q_max = 0.040  # kg/kg，大约对应50°C时的饱和比湿
    q = np.minimum(q, q_max)
    
    return q


def _calc_relative_humidity_factor(t_air: xr.DataArray, q_actual: xr.DataArray) -> xr.DataArray:
    """计算相对湿度因子，用于修正潜热通量计算。
    
    参数
    ------
    t_air : xr.DataArray
        2米气温 (K)
    q_actual : xr.DataArray  
        实际比湿 (kg/kg)
        
    返回
    ------
    xr.DataArray
        相对湿度因子 (0-1)
    """
    # 计算饱和比湿
    e_s = _calc_saturation_vapor_pressure(t_air)
    q_s = _calc_specific_humidity_from_vapor_pressure(e_s)
    
    # 计算相对湿度
    rh = q_actual / (q_s + 1e-10)  # 避免除零
    
    # 限制在合理范围内
    rh = np.clip(rh, 0.0, 1.0)
    
    return rh


# ------------------ 主函数 ------------------

def calculate_surface_energy_balance_forecast(eval_inputs: xr.Dataset, dt: float = 6 * 3600) -> Dict[str, xr.DataArray]:
    """使用地表能量平衡公式估算 6 小时后的 2 m 温度。

    参数
    ------
    eval_inputs : xr.Dataset
        与 GraphCast 相同格式的输入数据集。
    dt : float, 可选
        时间步长（秒），默认 6 小时。

    返回
    ------
    Dict[str, xr.DataArray]
        目前仅包含 ``'2m_temperature'`` 的预报结果。
    """

    # 取最新时间步（batch 仅支持 1）
    current = eval_inputs.isel(time=-1, batch=0)

    # ===================== 变量检查 =====================
    if '2m_temperature' not in current:
        raise KeyError("输入数据缺少 '2m_temperature'，无法进行 SEB 预测。")
    if 'toa_incident_solar_radiation' not in current:
        print("[警告] 缺少 'toa_incident_solar_radiation'，返回原始温度。")
        return {'2m_temperature': current['2m_temperature']}

    t_air = current['2m_temperature']           # K
    q_solar = current['toa_incident_solar_radiation']  # W m⁻² 或 J m⁻²

    # 若 q_solar 单位为累计能量 (J m⁻²)，转换为平均功率 (W m⁻²)
    q_sw_in = q_solar / dt if float(q_solar.max()) > 1e5 else q_solar

    # 反照率
    if 'land_sea_mask' in current:
        albedo = (current['land_sea_mask'] * ALBEDO_LAND +
                   (1 - current['land_sea_mask']) * ALBEDO_WATER)
    else:
        albedo = ALBEDO_LAND

    sw_net = q_sw_in * (1 - albedo)

    # 长波辐射计算
    # 地表向外长波辐射
    lw_out = EMISSIVITY_SURF * SIGMA * t_air**4
    
    # 大气向下长波辐射（简化估计）
    # 假设大气温度比地表低约10K
    t_atm = t_air - 10.0
    lw_down = EMISSIVITY_ATM * SIGMA * t_atm**4
    
    # 净长波辐射（向外为正）
    lw_net = lw_out - lw_down

    # 净辐射
    r_net = sw_net - lw_net
    
    # 调试输出
    print(f"[SEB调试] 短波入射平均值: {float(q_sw_in.mean()):.2f} W/m²")
    print(f"[SEB调试] 短波净辐射平均值: {float(sw_net.mean()):.2f} W/m²")
    print(f"[SEB调试] 长波向外平均值: {float(lw_out.mean()):.2f} W/m²")
    print(f"[SEB调试] 长波向下平均值: {float(lw_down.mean()):.2f} W/m²")
    print(f"[SEB调试] 长波净辐射平均值: {float(lw_net.mean()):.2f} W/m²")
    print(f"[SEB调试] 总净辐射平均值: {float(r_net.mean()):.2f} W/m²")

    # ----------------- 感热与潜热 -----------------
    if {'10m_u_component_of_wind', '10m_v_component_of_wind'}.issubset(current.data_vars):
        wind = _calc_wind_speed(current['10m_u_component_of_wind'],
                                current['10m_v_component_of_wind'])
        # 限制风速在合理范围内
        wind = np.clip(wind, 0.1, 20.0)  # 0.1-20 m/s
    else:
        wind = xr.zeros_like(t_air) + 2.0  # 默认 2 m/s

    # 感热 H = ρ·c_p·C_H·W·(T_s - T_a)；此处假设 T_s ≈ T_a + 小幅调整
    # 基于净辐射的简单地表温度估计
    delta_T_surf = np.clip(r_net / 100.0, -2.0, 2.0)  # 简单估计地表与气温差
    H = RHO_AIR * CP_AIR * C_H * wind * delta_T_surf

    # 改进的潜热通量计算
    if 'specific_humidity' in current:
        q_a = current['specific_humidity'].isel(level=0) if 'level' in current['specific_humidity'].dims else current['specific_humidity']
        
        # 限制实际比湿在合理范围内
        q_a = np.clip(q_a, 0.0, 0.030)  # kg/kg
        
        # 计算饱和比湿（使用改进的函数）
        e_s = _calc_saturation_vapor_pressure(t_air)  # hPa
        q_s = _calc_specific_humidity_from_vapor_pressure(e_s)  # kg/kg
        
        # 计算相对湿度因子
        rh = _calc_relative_humidity_factor(t_air, q_a)
        
        # 比湿差值计算，考虑地表状态
        # 当相对湿度很高时，减少蒸发
        moisture_deficit = (q_s - q_a) * (1.0 - rh * 0.8)  # 相对湿度越高，蒸发越少
        
        # 限制比湿差值
        moisture_deficit = np.clip(moisture_deficit, 0.0, MAX_SPECIFIC_HUMIDITY_DIFF)
        
        # 潜热通量计算
        LE = RHO_AIR * L_V * C_E * wind * moisture_deficit
        
        # 应用最大潜热通量限制
        LE = np.clip(LE, 0.0, MAX_LATENT_HEAT_FLUX)
        
        # 调试信息
        print(f"[SEB调试] 实际比湿范围: [{float(q_a.min()):.6f}, {float(q_a.max()):.6f}] kg/kg")
        print(f"[SEB调试] 饱和比湿范围: [{float(q_s.min()):.6f}, {float(q_s.max()):.6f}] kg/kg") 
        print(f"[SEB调试] 比湿差值范围: [{float(moisture_deficit.min()):.6f}, {float(moisture_deficit.max()):.6f}] kg/kg")
        print(f"[SEB调试] 相对湿度范围: [{float(rh.min()):.3f}, {float(rh.max()):.3f}]")
        
    else:
        LE = xr.zeros_like(t_air)

    # 地热通量 G
    G = GROUND_FLUX_FRACTION * r_net

    # 能量平衡项
    net_energy = r_net - H - LE - G
    
    # 温度变化 ΔT (K) - 使用改进的有效热容
    # 考虑大气边界层的热容量
    effective_heat_capacity = RHO_AIR * CP_AIR * Z_HEAT
    delta_t = net_energy * dt / effective_heat_capacity

    # 调试输出 - 更详细的信息
    print(f"[SEB调试] 风速范围: [{float(wind.min()):.2f}, {float(wind.max()):.2f}] m/s")
    print(f"[SEB调试] 感热通量平均值: {float(H.mean()):.2f} W/m² (范围: [{float(H.min()):.2f}, {float(H.max()):.2f}])")
    print(f"[SEB调试] 潜热通量平均值: {float(LE.mean()):.2f} W/m² (范围: [{float(LE.min()):.2f}, {float(LE.max()):.2f}])")
    print(f"[SEB调试] 地热通量平均值: {float(G.mean()):.2f} W/m² (范围: [{float(G.min()):.2f}, {float(G.max()):.2f}])")
    print(f"[SEB调试] 净能量通量平均值: {float(net_energy.mean()):.2f} W/m² (范围: [{float(net_energy.min()):.2f}, {float(net_energy.max()):.2f}])")
    print(f"[SEB调试] 有效热容: {effective_heat_capacity:.2f} J/(m³·K)")
    print(f"[SEB调试] 温度变化平均值: {float(delta_t.mean()):.4f} K")
    print(f"[SEB调试] 温度变化范围: [{float(delta_t.min()):.4f}, {float(delta_t.max()):.4f}] K")
    
    # 检查是否有异常值
    extreme_heating = (delta_t > 2.0).sum()
    extreme_cooling = (delta_t < -2.0).sum()
    if extreme_heating > 0:
        print(f"[SEB警告] 发现 {int(extreme_heating)} 个格点温度变化 > 2K")
    if extreme_cooling > 0:
        print(f"[SEB警告] 发现 {int(extreme_cooling)} 个格点温度变化 < -2K")

    # 应用温度变化限制
    delta_t_original = delta_t.copy()
    delta_t = np.clip(delta_t, -MAX_TEMP_CHANGE, MAX_TEMP_CHANGE)
    
    clipped_points = (np.abs(delta_t_original) > MAX_TEMP_CHANGE).sum()
    if clipped_points > 0:
        print(f"[SEB信息] 限制了 {int(clipped_points)} 个格点的极端温度变化")
    
    print(f"[SEB调试] 限制后温度变化平均值: {float(delta_t.mean()):.4f} K")
    print(f"[SEB调试] 限制后温度变化范围: [{float(delta_t.min()):.4f}, {float(delta_t.max()):.4f}] K")

    # 新温度
    t_future = t_air + delta_t
    
    print(f"[SEB调试] 原始温度范围: [{float(t_air.min()):.2f}, {float(t_air.max()):.2f}] K")
    print(f"[SEB调试] 预测温度范围: [{float(t_future.min()):.2f}, {float(t_future.max()):.2f}] K")
    print(f"[SEB总结] 平均温度变化: {float(delta_t.mean()):.3f} K，标准差: {float(delta_t.std()):.3f} K")

    # 组装 DataArray，保持坐标一致
    t_future = t_future.assign_attrs(t_air.attrs)  # 继承原属性

    return {'2m_temperature': t_future}


# ------------------ 评估辅助 ------------------

def calculate_correlation(seb_results: Dict[str, xr.DataArray],
                          graphcast_predictions: xr.Dataset) -> Dict[str, float]:
    """同 advection_calculation 中的相关性评估。"""
    import numpy as np
    correlations: Dict[str, float] = {}
    for var, seb in seb_results.items():
        if var in graphcast_predictions:
            gc = graphcast_predictions[var].isel(time=0, batch=0)
            if seb.shape == gc.shape:
                correlations[var] = float(xr.corr(seb, gc))
    return correlations


def print_correlation_results(correlations: Dict[str, float]):
    """简单打印结果。"""
    for k, v in correlations.items():
        print(f"{k}: {v:.4f}")


# ------------------ CLI ------------------
if __name__ == "__main__":
    print("Surface Energy Balance forecasting module")
    print("主要函数:")
    print("  calculate_surface_energy_balance_forecast()  – SEB 6h 预报")
    print("  calculate_correlation()                     – 相关性评估")
    print("  print_correlation_results()                 – 打印相关性") 