from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xarray


@dataclass
class DLMSFSensitivityResult:
    S_map: np.ndarray            # 2D (n_lat, n_lon), signed sensitivity
    S_abs_map: np.ndarray        # 2D (n_lat, n_lon), absolute sensitivity
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
    levels_bottom_hpa: float = 925.0,
    levels_top_hpa: float = 300.0,
) -> Tuple[float, float]:
    """计算深层环境引导气流 (U_dlmsf, V_dlmsf)。

    层结选取：levels_top_hpa ≤ p ≤ levels_bottom_hpa（含端点）。
    默认范围为 300–925 hPa，比 physics/swe/steering.py 中的 850–300 hPa
    更深，包含近地面边界层影响。

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
        levels_bottom_hpa: DLMSF 层结底层气压（hPa），默认 925
        levels_top_hpa: DLMSF 层结顶层气压（hPa），默认 300

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
    if levels_bottom_hpa <= levels_top_hpa:
        raise ValueError(
            f"levels_bottom_hpa ({levels_bottom_hpa}) must be greater than "
            f"levels_top_hpa ({levels_top_hpa})"
        )

    # 选取 levels_top_hpa–levels_bottom_hPa 层（含端点）
    # Note: bottom > top (e.g., 925 > 300)
    level_mask = (levels_hpa >= levels_top_hpa) & (levels_hpa <= levels_bottom_hpa)
    sel_idx = np.where(level_mask)[0]
    if len(sel_idx) == 0:
        raise ValueError(
            f"No levels found in {levels_top_hpa}–{levels_bottom_hpa} hPa range"
        )

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


_U_VAR = "u_component_of_wind"
_V_VAR = "v_component_of_wind"


def _extract_uv_levels(
    eval_inputs: xarray.Dataset,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    time_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 eval_inputs 提取 u/v 多层场。

    返回 (u, v, levels_hpa)，shape 均为 (n_lev, n_lat, n_lon)。

    Args:
        eval_inputs: xarray Dataset，含 u/v 多层场
        lat_vals: 目标子域纬度坐标
        lon_vals: 目标子域经度坐标
        time_idx: 时间步索引（默认 1，对应 t=0h（评估时刻的初始条件））

    Raises:
        ValueError: 如果缺少 'level' 维度
    """
    u_da = eval_inputs[_U_VAR]
    v_da = eval_inputs[_V_VAR]
    if "batch" in u_da.dims:
        u_da = u_da.isel(batch=0)
        v_da = v_da.isel(batch=0)
    if "time" in u_da.dims:
        # DLMSF requires time[1] (initial condition at t=0h)
        if u_da.sizes["time"] < 2:
            raise ValueError(
                f"{_U_VAR} 'time' dimension has {u_da.sizes['time']} slice(s); "
                f"DLMSF requires at least 2 time slices (uses time[1])."
            )
        u_da = u_da.isel(time=time_idx)
        v_da = v_da.isel(time=time_idx)
    if "level" not in u_da.dims:
        raise ValueError(
            f"{_U_VAR} has no 'level' dimension; DLMSF requires multi-level data."
        )
    u_da = (
        u_da.sel(lat=lat_vals, method="nearest")
        .sel(lon=lon_vals, method="nearest")
        .transpose("level", "lat", "lon")
    )
    v_da = (
        v_da.sel(lat=lat_vals, method="nearest")
        .sel(lon=lon_vals, method="nearest")
        .transpose("level", "lat", "lon")
    )
    levels = np.asarray(u_da.coords["level"].values, dtype=np.float32)
    return (
        np.asarray(u_da.values, dtype=np.float32),
        np.asarray(v_da.values, dtype=np.float32),
        levels,
    )


def _build_patches(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    patch_size_deg: float,
) -> list:
    """构建覆盖整个 ROI 的矩形 patch 列表。

    每个 patch 为一个布尔掩膜 (n_lat, n_lon)，覆盖 patch_size_deg × patch_size_deg 的区域。
    Patch 沿经纬度方向滑动，起点为各方向最小值，步长等于 patch_size_deg。

    Args:
        lat_vals: 1D latitude array
        lon_vals: 1D longitude array
        patch_size_deg: patch 尺寸（度），必须为正数

    Returns:
        List of boolean masks, each shape (n_lat, n_lon)

    Raises:
        ValueError: If patch_size_deg <= 0
    """
    if patch_size_deg <= 0:
        raise ValueError(
            f"patch_size_deg must be positive, got {patch_size_deg}"
        )

    lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())

    patches = []
    p_lat = lat_min
    while p_lat <= lat_max + patch_size_deg * 0.5:
        p_lon = lon_min
        while p_lon <= lon_max + patch_size_deg * 0.5:
            lat_mask = (lat_vals >= p_lat) & (lat_vals < p_lat + patch_size_deg)
            lon_mask = (lon_vals >= p_lon) & (lon_vals < p_lon + patch_size_deg)
            mask = np.outer(lat_mask, lon_mask)
            if mask.any():
                patches.append(mask)
            p_lon += patch_size_deg
        p_lat += patch_size_deg
    return patches


def compute_dlmsf_patch_fd(
    eval_inputs: xarray.Dataset,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    d_hat: Tuple[float, float],
    target_time_idx: int,
    patch_size_deg: float = 2.0,
    eps: float = 1.0,
    core_radius_deg: float = 3.0,
    annulus_inner_km: float = 300.0,
    annulus_outer_km: float = 900.0,
    levels_bottom_hpa: float = 925.0,
    levels_top_hpa: float = 300.0,
) -> DLMSFSensitivityResult:
    """有限差分 patch 扰动计算 DLMSF 敏感度图。

    对 ROI 划分为 patch_size_deg × patch_size_deg 的矩形 patch，对每个 patch
    施加方向性扰动，通过中心差分估计该 patch 对 J_phys 的贡献：
        plus:  u += eps*d_u, v += eps*d_v
        minus: u -= eps*d_u, v -= eps*d_v
        S_P = (J_plus - J_minus) / (2*eps)

    J_phys = DLMSF · d_hat，其中 DLMSF 为 [levels_top_hpa, levels_bottom_hpa]
    范围内的压厚加权环境引导气流向量。

    Args:
        eval_inputs: xarray Dataset，含 u/v 多层场
        lat_vals: 子域纬度坐标（通常与 SWE 子域一致）
        lon_vals: 子域经度坐标
        center_lat: 台风中心纬度
        center_lon: 台风中心经度
        d_hat: 台风移动方向单位向量 (d_u, d_v)，由 compute_d_hat 计算
        target_time_idx: 预报时次索引（仅用于标注结果，不影响数据提取；
            DLMSF 始终使用 eval_inputs 中 time[1] 的初始条件风场）
        patch_size_deg: patch 尺寸（度），默认 2.0°
        eps: 有限差分扰动量（m/s），默认 1.0
        core_radius_deg: 核心排除半径（度）
        annulus_inner_km: 环境环内径（km）
        annulus_outer_km: 环境环外径（km）
        levels_bottom_hpa: DLMSF 层结底层气压（hPa），默认 925
        levels_top_hpa: DLMSF 层结顶层气压（hPa），默认 300

    Returns:
        DLMSFSensitivityResult 含 S_map（有符号敏度）和 S_abs_map（绝对敏度）
    """
    t0 = time.perf_counter()
    d_u, d_v = d_hat

    # Validate patch_size_deg at entrypoint (fail-fast before any work)
    if patch_size_deg <= 0:
        raise ValueError(
            f"patch_size_deg must be positive, got {patch_size_deg}"
        )

    # Validate eps (must be positive for finite differences)
    if eps <= 0:
        raise ValueError(
            f"eps must be positive for finite differences, got {eps}"
        )

    # Normalize d_hat to unit vector (if non-zero)
    # Store the normalized d_hat for use in result
    d_mag = math.hypot(d_u, d_v)
    if d_mag > 1e-10:
        d_u = d_u / d_mag
        d_v = d_v / d_mag
    # Store normalized (or zero) vector for result
    d_hat_normalized = (d_u, d_v)

    # d_hat 为零向量时，J 恒为 0，S_map 全零
    # CHECK ZERO d_hat BEFORE NaN-baseline check (short-circuit priority)
    if abs(d_u) < 1e-10 and abs(d_v) < 1e-10:
        # 提取 u/v 多层场仅用于获取 lat_vals/lon_vals 形状验证
        # 零 d_hat 无需实际风场数据即可返回结果
        elapsed = time.perf_counter() - t0
        patches = _build_patches(lat_vals, lon_vals, patch_size_deg)
        S_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)
        return DLMSFSensitivityResult(
            S_map=S_map, S_abs_map=np.abs(S_map),
            lat_vals=lat_vals, lon_vals=lon_vals,
            center_lat=center_lat, center_lon=center_lon,
            target_time_idx=target_time_idx,
            d_hat=d_hat_normalized, J_phys_baseline=0.0,
            U_dlmsf=0.0, V_dlmsf=0.0,  # Zero d_hat means no meaningful DLMSF direction
            n_patches=len(patches), elapsed_sec=elapsed,
        )

    # 提取 u/v 多层场
    u_base, v_base, levels = _extract_uv_levels(eval_inputs, lat_vals, lon_vals, time_idx=1)

    # 基线 J_phys
    U0, V0 = compute_dlmsf_925_300(
        u_base, v_base, levels, lat_vals, lon_vals,
        center_lat, center_lon,
        core_radius_deg=core_radius_deg,
        annulus_inner_km=annulus_inner_km,
        annulus_outer_km=annulus_outer_km,
        levels_bottom_hpa=levels_bottom_hpa,
        levels_top_hpa=levels_top_hpa,
    )
    J0 = float(U0) * d_u + float(V0) * d_v
    
    if math.isnan(J0):
        print("[DLMSF-FD] warn: baseline J0 is NaN (likely all-NaN wind field). "
              "Returning zero S_map.")
        elapsed = time.perf_counter() - t0
        patches = _build_patches(lat_vals, lon_vals, patch_size_deg)
        S_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)
        return DLMSFSensitivityResult(
            S_map=S_map, S_abs_map=np.abs(S_map),
            lat_vals=lat_vals, lon_vals=lon_vals,
            center_lat=center_lat, center_lon=center_lon,
            target_time_idx=target_time_idx,
            d_hat=d_hat_normalized, J_phys_baseline=float("nan"),
            U_dlmsf=float(U0), V_dlmsf=float(V0),
            n_patches=len(patches), elapsed_sec=elapsed,
        )

    S_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)

    # 构建 patch 列表并逐 patch 有限差分
    patches = _build_patches(lat_vals, lon_vals, patch_size_deg)
    for mask in patches:
        # Plus perturbation: u += eps*d_u, v += eps*d_v
        u_plus = u_base.copy()
        v_plus = v_base.copy()
        u_plus[:, mask] += eps * d_u
        v_plus[:, mask] += eps * d_v
        U_p, V_p = compute_dlmsf_925_300(
            u_plus, v_plus, levels, lat_vals, lon_vals,
            center_lat, center_lon,
            core_radius_deg=core_radius_deg,
            annulus_inner_km=annulus_inner_km,
            annulus_outer_km=annulus_outer_km,
            levels_bottom_hpa=levels_bottom_hpa,
            levels_top_hpa=levels_top_hpa,
        )
        J_plus = float(U_p) * d_u + float(V_p) * d_v

        # Minus perturbation: u -= eps*d_u, v -= eps*d_v
        u_minus = u_base.copy()
        v_minus = v_base.copy()
        u_minus[:, mask] -= eps * d_u
        v_minus[:, mask] -= eps * d_v
        U_m, V_m = compute_dlmsf_925_300(
            u_minus, v_minus, levels, lat_vals, lon_vals,
            center_lat, center_lon,
            core_radius_deg=core_radius_deg,
            annulus_inner_km=annulus_inner_km,
            annulus_outer_km=annulus_outer_km,
            levels_bottom_hpa=levels_bottom_hpa,
            levels_top_hpa=levels_top_hpa,
        )
        J_minus = float(U_m) * d_u + float(V_m) * d_v

        # Central difference: S_P = (J_plus - J_minus) / (2*eps)
        S_P = (J_plus - J_minus) / (2.0 * eps)
        S_map[mask] = S_P

    elapsed = time.perf_counter() - t0
    n_patches = len(patches)
    print(
        f"[DLMSF-FD] +{(target_time_idx + 1) * 6}h  {n_patches} patches  "
        f"{elapsed:.1f}s  U_dlmsf={U0:+.2f} V_dlmsf={V0:+.2f}  J0={J0:+.4f}"
    )

    return DLMSFSensitivityResult(
        S_map=S_map, S_abs_map=np.abs(S_map),
        lat_vals=lat_vals, lon_vals=lon_vals,
        center_lat=center_lat, center_lon=center_lon,
        target_time_idx=target_time_idx,
        d_hat=d_hat_normalized, J_phys_baseline=float(J0),
        U_dlmsf=float(U0), V_dlmsf=float(V0),
        n_patches=n_patches, elapsed_sec=elapsed,
    )
