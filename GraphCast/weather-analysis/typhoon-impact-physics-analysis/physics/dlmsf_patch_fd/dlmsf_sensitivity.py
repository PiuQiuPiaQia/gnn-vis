from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DLMSFSensitivityResult:
    S_map: np.ndarray            # 2D (n_lat, n_lon)
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    center_lat: float
    center_lon: float
    target_time_idx: int
    d_hat: Tuple[float, float]   # (d_u, d_v)
    J_phys_baseline: float
    U_dlmsf: float
    V_dlmsf: float
    n_patches: int
    elapsed_sec: float


def compute_d_hat(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> Tuple[float, float]:
    """计算台风移动方向的单位向量 (d_u, d_v)。

    d_u 对应经度方向位移（东向为正），d_v 对应纬度方向位移（北向为正）。
    若台风静止（lat0==lat1 且 lon0==lon1），返回 (0.0, 0.0)。

    注意：d_hat 是在角度空间（degree）中归一化的单位向量，不是球面距离空间。
    这种近似适用于台风路径的定性方向判断。
    """
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    # 经度差折叠到 [-180, 180]
    dlon = ((dlon + 180.0) % 360.0) - 180.0
    mag = math.hypot(dlon, dlat)
    if mag < 1e-10:
        return 0.0, 0.0
    return float(dlon / mag), float(dlat / mag)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 公式计算两点之间的球面距离（km）。
    
    注意：与 physics/swe/steering.py 中的 _haversine_distance_km 逻辑等价，
    此处使用 math 库以避免跨模块依赖。
    """
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat_r = math.radians(lat2 - lat1)
    dlon_r = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dlat_r / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon_r / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_dlmsf_925_300(
    u_levels: np.ndarray,
    v_levels: np.ndarray,
    levels_hpa: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    core_radius_deg: float,
    annulus_inner_km: float,
    annulus_outer_km: float,
    min_env_points: int = 10,
) -> Tuple[float, float]:
    """计算 925–300 hPa 深层环境引导气流 (U_dlmsf, V_dlmsf)。

    层结选取：300 ≤ p ≤ 925 hPa（含端点）。注意：此范围比 physics/swe/steering.py
    中的 850–300 hPa 更深，包含近地面边界层影响。

    权重：梯形压厚权重，归一化。
    环境点：距中心 annulus_inner_km–annulus_outer_km 且排除核心区域的格点。

    Args:
        u_levels: 3D zonal wind array (n_level, n_lat, n_lon) in m/s
        v_levels: 3D meridional wind array (n_level, n_lat, n_lon) in m/s
        levels_hpa: 1D pressure levels in hPa（可升序或降序）
        lat_vals: 1D latitude array in degrees
        lon_vals: 1D longitude array in degrees
        center_lat: Cyclone center latitude in degrees
        center_lon: Cyclone center longitude in degrees
        core_radius_deg: Core exclusion radius in degrees（0.0 = 不排除）
        annulus_inner_km: Inner radius of environmental annulus in km
        annulus_outer_km: Outer radius of environmental annulus in km
        min_env_points: Minimum valid env points; if below, falls back to full domain

    Returns:
        (U_dlmsf, V_dlmsf) pressure-thickness weighted environmental mean winds (m/s)
    """
    if u_levels.ndim != 3 or v_levels.ndim != 3:
        raise ValueError(
            f"u_levels and v_levels must be 3D arrays, "
            f"got shapes {u_levels.shape} and {v_levels.shape}"
        )
    if u_levels.shape != v_levels.shape:
        raise ValueError(
            f"u_levels and v_levels shape mismatch: {u_levels.shape} vs {v_levels.shape}"
        )
    if u_levels.shape[0] != len(levels_hpa):
        raise ValueError(
            f"u_levels first dimension ({u_levels.shape[0]}) must match "
            f"levels_hpa length ({len(levels_hpa)})"
        )
    if annulus_outer_km <= annulus_inner_km:
        raise ValueError(
            f"annulus_outer_km ({annulus_outer_km}) must be greater than "
            f"annulus_inner_km ({annulus_inner_km})"
        )

    # 选取 300–925 hPa 层（含端点）
    level_mask = (levels_hpa >= 300.0) & (levels_hpa <= 925.0)
    sel_idx = np.where(level_mask)[0]
    if len(sel_idx) == 0:
        raise ValueError("No levels found in 300–925 hPa range")

    u_sel = u_levels[sel_idx]
    v_sel = v_levels[sel_idx]
    levels_sel = levels_hpa[sel_idx]
    n_sel = len(levels_sel)

    # 梯形压厚权重
    weights = np.zeros(n_sel, dtype=np.float64)
    if n_sel == 1:
        weights[0] = 1.0
    else:
        for i in range(n_sel):
            if i == 0:
                weights[i] = 0.5 * abs(float(levels_sel[1]) - float(levels_sel[0]))
            elif i == n_sel - 1:
                weights[i] = 0.5 * abs(float(levels_sel[i]) - float(levels_sel[i - 1]))
            else:
                weights[i] = 0.5 * (
                    abs(float(levels_sel[i + 1]) - float(levels_sel[i]))
                    + abs(float(levels_sel[i]) - float(levels_sel[i - 1]))
                )
    weights /= weights.sum()

    nlat, nlon = len(lat_vals), len(lon_vals)

    # 距离矩阵（Haversine）
    dist_km = np.zeros((nlat, nlon), dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            dist_km[i, j] = _haversine_km(
                center_lat, center_lon, float(lat_vals[i]), float(lon_vals[j])
            )

    # 有效点掩膜（所有选定层均有限值）
    finite_mask = np.ones((nlat, nlon), dtype=bool)
    for k in range(n_sel):
        finite_mask &= np.isfinite(u_sel[k]) & np.isfinite(v_sel[k])

    # 核心排除 & 环状掩膜
    core_km = core_radius_deg * 111.0
    core_mask = dist_km < max(core_km, annulus_inner_km)
    env_mask = finite_mask & (~core_mask) & (dist_km <= annulus_outer_km)

    if int(np.sum(env_mask)) < min_env_points:
        env_mask = finite_mask  # 降级：使用全域有效点

    # 压厚加权空间均值
    U_sum, V_sum = 0.0, 0.0
    count = 0
    for k in range(n_sel):
        u_env = u_sel[k][env_mask]
        v_env = v_sel[k][env_mask]
        if len(u_env) > 0:
            U_sum += float(weights[k]) * float(np.mean(u_env))
            V_sum += float(weights[k]) * float(np.mean(v_env))
            count += 1
    
    if count == 0:
        return float("nan"), float("nan")
    
    return U_sum, V_sum
