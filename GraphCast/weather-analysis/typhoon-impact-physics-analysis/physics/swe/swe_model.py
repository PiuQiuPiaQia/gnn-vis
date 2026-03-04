# -*- coding: utf-8 -*-
"""JAX 线性化浅水方程 (SWE) 前向模型——完全可微，供 jax.grad 穿透。

方程（f-plane 线性化 SWE）：
    ∂h/∂t = -H (∂u/∂x + ∂v/∂y)
    ∂u/∂t = -g ∂h/∂x + f₀ v
    ∂v/∂t = -g ∂h/∂y - f₀ u

状态量 (h, u, v) 均为扰动量（已减去参考态均值）。
边界条件：周期性（jnp.roll 实现）。
时间积分：4 阶 Runge-Kutta，由 jax.lax.scan 展开（O(1) 内存）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ─── 物理参数容器 ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SWEPhysicsConfig:
    """SWE 积分所需的全部物理/格点参数。"""

    H: float    # 参考深度 (m)，等效水深 ≈ z_500 / g
    f0: float   # Coriolis 参数 (s⁻¹)，f-plane，南半球为负
    g: float    # 重力加速度 (m s⁻²)
    dx: float   # 纬向格点距离 (m)
    dy: float   # 经向格点距离 (m)
    dt: float   # 时间步长 (s)
    n_lat: int  # 子域纬度格点数
    n_lon: int  # 子域经度格点数
    U_bar: float = 0.0  # 背景纬向引导风 (m/s)；=0 退化为原始行为
    V_bar: float = 0.0  # 背景经向引导风 (m/s)；=0 退化为原始行为
    r_m: float = 0.0    # 动量 Rayleigh 阻尼系数 (s^-1)
    r_h: float = 0.0    # 高度 Rayleigh 阻尼系数 (s^-1)
    nu: float = 0.0     # 二阶扩散系数 (m^2 s^-1)
    sponge_width: int = 0
    sponge_rate: float = 0.0  # 边界海绵层最大阻尼 (s^-1)


def make_physics_config(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    h0_mean: float,
    g: float = 9.81,
    dt: float = 300.0,
    U_bar: float = 0.0,
    V_bar: float = 0.0,
    H_eq: Optional[float] = None,
    rayleigh_momentum_h: float = 0.0,
    rayleigh_height_h: float = 0.0,
    diffusion_coeff: float = 0.0,
    sponge_width: int = 0,
    sponge_efold_h: float = 0.0,
) -> SWEPhysicsConfig:
    """根据子域坐标和初始高度场均值构造 SWEPhysicsConfig。

    Args:
        lat_vals: 子域纬度坐标 (度), shape (n_lat,)
        lon_vals: 子域经度坐标 (度), shape (n_lon,)
        h0_mean:  初始 h 场均值 (m)，用于估计 H = z_500 / g
        g:        重力加速度
        dt:       时间步长 (s)
        U_bar:    背景纬向引导风 (m/s)，默认 0
        V_bar:    背景经向引导风 (m/s)，默认 0

    Returns:
        SWEPhysicsConfig
    """
    OMEGA = 7.2921e-5  # 地球自转角速度 (rad/s)

    center_lat_deg = float(np.mean(lat_vals))
    center_lat_rad = np.deg2rad(center_lat_deg)
    f0 = float(2.0 * OMEGA * np.sin(center_lat_rad))

    # 格距（适配任意分辨率）
    dlat_deg = float(abs(lat_vals[1] - lat_vals[0])) if len(lat_vals) > 1 else 1.0
    dlon_deg = float(abs(lon_vals[1] - lon_vals[0])) if len(lon_vals) > 1 else 1.0
    dy = dlat_deg * 110540.0        # 1° 纬度 ≈ 110540 m
    dx = dlon_deg * 111320.0 * abs(float(np.cos(center_lat_rad)))

    # 等效水深显式可控；未指定时退回旧行为。
    H = max(float(h0_mean if H_eq is None else H_eq), 1.0)
    r_m = 0.0 if rayleigh_momentum_h <= 0.0 else 1.0 / (float(rayleigh_momentum_h) * 3600.0)
    r_h = 0.0 if rayleigh_height_h <= 0.0 else 1.0 / (float(rayleigh_height_h) * 3600.0)
    nu = max(float(diffusion_coeff), 0.0)
    sp_w = max(int(sponge_width), 0)
    sponge_rate = 0.0 if sponge_efold_h <= 0.0 else 1.0 / (float(sponge_efold_h) * 3600.0)

    return SWEPhysicsConfig(
        H=H,
        f0=f0,
        g=g,
        dx=dx,
        dy=dy,
        dt=dt,
        n_lat=int(len(lat_vals)),
        n_lon=int(len(lon_vals)),
        U_bar=float(U_bar),
        V_bar=float(V_bar),
        r_m=r_m,
        r_h=r_h,
        nu=nu,
        sponge_width=sp_w,
        sponge_rate=sponge_rate,
    )


# ─── 数值算子 ────────────────────────────────────────────────────────────────


def _centered_diff_x(field: jax.Array, dx: float) -> jax.Array:
    """纬向中心差分 ∂/∂x，周期边界。"""
    return (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2.0 * dx)


def _centered_diff_y(field: jax.Array, dy: float) -> jax.Array:
    """经向中心差分 ∂/∂y，周期边界。"""
    return (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / (2.0 * dy)


def _laplacian(field: jax.Array, dx: float, dy: float) -> jax.Array:
    """二维拉普拉斯算子，周期边界。"""
    d2x = (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)) / (dx * dx)
    d2y = (jnp.roll(field, -1, axis=0) - 2.0 * field + jnp.roll(field, 1, axis=0)) / (dy * dy)
    return d2x + d2y


def _upwind_diff_x(field: jax.Array, dx: float, velocity_x: float) -> jax.Array:
    """一阶迎风差分 ∂/∂x（由速度符号决定方向）。"""
    if velocity_x >= 0.0:
        return (field - jnp.roll(field, 1, axis=1)) / dx
    return (jnp.roll(field, -1, axis=1) - field) / dx


def _upwind_diff_y(field: jax.Array, dy: float, velocity_y: float) -> jax.Array:
    """一阶迎风差分 ∂/∂y（由速度符号决定方向）。"""
    if velocity_y >= 0.0:
        return (field - jnp.roll(field, 1, axis=0)) / dy
    return (jnp.roll(field, -1, axis=0) - field) / dy


def _advect(field: jax.Array, cfg: SWEPhysicsConfig) -> jax.Array:
    """背景平流项 -U d/dx - V d/dy（迎风格式）。"""
    return -cfg.U_bar * _upwind_diff_x(field, cfg.dx, cfg.U_bar) - cfg.V_bar * _upwind_diff_y(field, cfg.dy, cfg.V_bar)


def _sponge_mask(cfg: SWEPhysicsConfig) -> jax.Array:
    """构造边界海绵层掩膜（0~1，边界最大）。"""
    if cfg.sponge_width <= 0 or cfg.sponge_rate <= 0.0:
        return jnp.zeros((cfg.n_lat, cfg.n_lon), dtype=jnp.float32)

    yi = jnp.arange(cfg.n_lat)
    xi = jnp.arange(cfg.n_lon)
    y_edge = jnp.minimum(yi, cfg.n_lat - 1 - yi)
    x_edge = jnp.minimum(xi, cfg.n_lon - 1 - xi)
    edge_dist = jnp.minimum(y_edge[:, None], x_edge[None, :]).astype(jnp.float32)

    w = float(cfg.sponge_width)
    strength = jnp.clip((w - edge_dist) / w, 0.0, 1.0)
    return strength * strength


def _swe_tendency(
    h: jax.Array,
    u: jax.Array,
    v: jax.Array,
    cfg: SWEPhysicsConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """计算线性化 SWE 的时间倾向（右端项）。"""
    du_dx = _centered_diff_x(u, cfg.dx)
    dv_dy = _centered_diff_y(v, cfg.dy)
    dh_dx = _centered_diff_x(h, cfg.dx)
    dh_dy = _centered_diff_y(h, cfg.dy)
    lap_h = _laplacian(h, cfg.dx, cfg.dy)
    lap_u = _laplacian(u, cfg.dx, cfg.dy)
    lap_v = _laplacian(v, cfg.dx, cfg.dy)
    sponge = _sponge_mask(cfg) * float(cfg.sponge_rate)

    dh_dt = (
        _advect(h, cfg)
        - cfg.H * (du_dx + dv_dy)
        - cfg.r_h * h
        + cfg.nu * lap_h
        - sponge * h
    )
    du_dt = (
        _advect(u, cfg)
        - cfg.g * dh_dx
        + cfg.f0 * v
        - cfg.r_m * u
        + cfg.nu * lap_u
        - sponge * u
    )
    dv_dt = (
        _advect(v, cfg)
        - cfg.g * dh_dy
        - cfg.f0 * u
        - cfg.r_m * v
        + cfg.nu * lap_v
        - sponge * v
    )
    return dh_dt, du_dt, dv_dt


def _swe_rk4_step(
    h: jax.Array,
    u: jax.Array,
    v: jax.Array,
    cfg: SWEPhysicsConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """4 阶 Runge-Kutta 单步推进。"""
    dt = cfg.dt

    k1h, k1u, k1v = _swe_tendency(h, u, v, cfg)

    h2 = h + 0.5 * dt * k1h
    u2 = u + 0.5 * dt * k1u
    v2 = v + 0.5 * dt * k1v
    k2h, k2u, k2v = _swe_tendency(h2, u2, v2, cfg)

    h3 = h + 0.5 * dt * k2h
    u3 = u + 0.5 * dt * k2u
    v3 = v + 0.5 * dt * k2v
    k3h, k3u, k3v = _swe_tendency(h3, u3, v3, cfg)

    h4 = h + dt * k3h
    u4 = u + dt * k3u
    v4 = v + dt * k3v
    k4h, k4u, k4v = _swe_tendency(h4, u4, v4, cfg)

    h_new = h + (dt / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
    u_new = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return h_new, u_new, v_new


# ─── 完整积分 ────────────────────────────────────────────────────────────────


def swe_forward(
    h0: jax.Array,
    u0: jax.Array,
    v0: jax.Array,
    cfg: SWEPhysicsConfig,
    n_steps: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """将 (h0, u0, v0) 积分 n_steps 步，返回终态 (h_t, u_t, v_t)。

    使用 jax.lax.scan 展开循环：
    - 编译后 O(1) 内存（不存储中间态）
    - 梯度可通过反向传播自动穿透所有时间步
    """
    def step_fn(carry: Tuple, _):
        h, u, v = carry
        h, u, v = _swe_rk4_step(h, u, v, cfg)
        return (h, u, v), None

    (h_t, u_t, v_t), _ = jax.lax.scan(step_fn, (h0, u0, v0), None, length=n_steps)
    return h_t, u_t, v_t


# ─── 目标函数 J ───────────────────────────────────────────────────────────────


def make_gaussian_weights(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    sigma_deg: float = 3.0,
) -> jax.Array:
    """构造以台风中心为峰值的归一化高斯权重矩阵，shape (n_lat, n_lon)。

    J = Σ w(i,j) · h_t(i,j)  越小 = 中心气压越低 = 台风越强。
    """
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    dlat = lat2d - center_lat
    # 处理经度周期性
    dlon = ((lon2d - center_lon + 180.0) % 360.0) - 180.0
    dist2 = dlat ** 2 + dlon ** 2
    w = np.exp(-dist2 / (2.0 * sigma_deg ** 2))
    w = w / w.sum()
    return jnp.array(w, dtype=jnp.float32)


def make_target_J_fn(
    weights: jax.Array,
    cfg: SWEPhysicsConfig,
    n_steps: int,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """返回 J(h0, u0, v0) → scalar 的闭包，供 jax.grad 使用。

    J = Σ_{i,j} w(i,j) · h_t(i,j)
    """
    def J(h0: jax.Array, u0: jax.Array, v0: jax.Array) -> jax.Array:
        h_t, _, _ = swe_forward(h0, u0, v0, cfg, n_steps)
        return jnp.sum(weights * h_t)

    return J


# ─── 地转风映射 ────────────────────────────────────────────────────────────────


def geostrophic_wind_from_height(
    height: jax.Array,
    cfg: SWEPhysicsConfig,
    f0_floor: float = 1e-5,
) -> Tuple[jax.Array, jax.Array]:
    """从高度扰动场计算地转风扰动。

    地转平衡关系（线性化）：
        delta_u = -(g / f0) * ∂(height) / ∂y
        delta_v =  (g / f0) * ∂(height) / ∂x

    Args:
        height: 高度扰动场 (m)，shape (n_lat, n_lon)。
        cfg: SWE 物理配置，包含 g, f0, dx, dy 等参数。
        f0_floor: f0 最小绝对值阈值，防止除零。

    Returns:
        (delta_u, delta_v): 地转风扰动分量 (m/s)，shape 均为 (n_lat, n_lon)。

    Raises:
        ValueError: 如果 |cfg.f0| < f0_floor。
    """
    if abs(cfg.f0) < f0_floor:
        raise ValueError(f"f0 too small: |f0|={abs(cfg.f0):.2e} < {f0_floor:.2e}")

    # 地转风系数
    coeff = cfg.g / cfg.f0

    # 地转平衡：u_g = -(g/f0) * ∂h/∂y, v_g = (g/f0) * ∂h/∂x
    delta_u = -coeff * _centered_diff_y(height, cfg.dy)
    delta_v = coeff * _centered_diff_x(height, cfg.dx)

    return delta_u, delta_v
