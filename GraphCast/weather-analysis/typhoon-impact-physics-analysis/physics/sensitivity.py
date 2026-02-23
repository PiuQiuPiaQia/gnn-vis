from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from physics.swe_model import (
    SWEPhysicsConfig,
    make_gaussian_weights,
    make_physics_config,
    make_target_J_fn,
    swe_forward,
)

_ERA5_Z_VAR = "geopotential"
_ERA5_U_VAR = "u_component_of_wind"
_ERA5_V_VAR = "v_component_of_wind"
_TARGET_LEVEL_HPA = 500


@dataclass
class SWESensitivityResult:
    target_time_idx: int
    n_steps: int
    method: str

    S_h: np.ndarray
    S_u: np.ndarray
    S_v: np.ndarray
    S_uv: np.ndarray
    S_total: np.ndarray

    lat_vals: np.ndarray
    lon_vals: np.ndarray
    center_lat: float
    center_lon: float
    physics_cfg: SWEPhysicsConfig
    elapsed_sec: float


def extract_swe_initial_conditions(
    eval_inputs: xarray.Dataset,
    center_lat: float,
    center_lon: float,
    domain_half_deg: float = 20.0,
    time_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_all = eval_inputs.coords["lat"].values
    lon_all = eval_inputs.coords["lon"].values

    lat_mask = (lat_all >= center_lat - domain_half_deg) & \
               (lat_all <= center_lat + domain_half_deg)
    dlon = ((lon_all - center_lon + 180.0) % 360.0) - 180.0
    lon_mask = np.abs(dlon) <= domain_half_deg

    lat_vals = lat_all[lat_mask]
    lon_vals = lon_all[lon_mask]

    def _extract(var_name: str) -> np.ndarray:
        da = eval_inputs[var_name]
        if "level" in da.dims:
            levels = da.coords["level"].values
            lvl_idx = int(np.argmin(np.abs(levels - _TARGET_LEVEL_HPA)))
            da = da.isel(level=lvl_idx)
        if "batch" in da.dims:
            da = da.isel(batch=0)
        if "time" in da.dims:
            da = da.isel(time=time_idx)
        da = da.sel(lat=lat_vals, method="nearest").sel(lon=lon_vals, method="nearest")
        return np.asarray(da.values, dtype=np.float32)

    z_500 = _extract(_ERA5_Z_VAR)
    u_500 = _extract(_ERA5_U_VAR)
    v_500 = _extract(_ERA5_V_VAR)

    g = 9.81
    h0 = z_500 / g
    return h0, u_500, v_500, lat_vals, lon_vals


def _n_steps_for(target_time_idx: int, dt: float = 300.0) -> int:
    lead_hours = (target_time_idx + 1) * 6
    return int(round(lead_hours * 3600.0 / dt))


def compute_dlm_background_wind(
    eval_inputs: xarray.Dataset,
    center_lat: float,
    center_lon: float,
    inner_radius_km: float = 300.0,
    outer_radius_km: float = 800.0,
    p_bot_hpa: float = 850.0,
    p_top_hpa: float = 300.0,
    time_idx: int = 1,
) -> tuple:
    """深层平均引导风 (DLM Steering) — 环形平均 + 质量加权垂直积分。

    Args:
        eval_inputs:      ERA5 xarray Dataset，含 u/v_component_of_wind。
        center_lat/lon:   台风中心坐标（度）。
        inner_radius_km:  环形内半径 (km)。
        outer_radius_km:  环形外半径 (km)。
        p_bot_hpa:        垂直积分下界压力层 (hPa)。
        p_top_hpa:        垂直积分上界压力层 (hPa)。
        time_idx:         时间步索引（0 = 分析时刻，1 = +6h）。

    Returns:
        (U_bar, V_bar) in m/s — Python floats.
    """
    DEG_TO_M = 111320.0

    lat_all = eval_inputs.coords["lat"].values.astype(float)
    lon_all = eval_inputs.coords["lon"].values.astype(float)

    # 计算每个格点到台风中心的距离 (km)
    dlat_m = (lat_all - center_lat) * DEG_TO_M
    dlon_deg = ((lon_all - center_lon + 180.0) % 360.0) - 180.0
    cos_lat = np.cos(np.deg2rad(center_lat))
    dlon_m = dlon_deg * DEG_TO_M * cos_lat

    lat2d, lon2d = np.meshgrid(dlat_m, dlon_m, indexing="ij")
    dist_km = np.sqrt(lat2d ** 2 + lon2d ** 2) / 1000.0
    annulus_mask = (dist_km >= inner_radius_km) & (dist_km <= outer_radius_km)

    if not np.any(annulus_mask):
        return 0.0, 0.0

    def _extract_var(var_name: str) -> xarray.DataArray:
        da = eval_inputs[var_name]
        if "batch" in da.dims:
            da = da.isel(batch=0)
        if "time" in da.dims:
            da = da.isel(time=time_idx)
        return da  # shape (level, lat, lon)

    u_da = _extract_var(_ERA5_U_VAR)
    v_da = _extract_var(_ERA5_V_VAR)
    levels = u_da.coords["level"].values.astype(float)

    # 选择 p_top ≤ level ≤ p_bot 的层
    lev_mask = (levels >= p_top_hpa) & (levels <= p_bot_hpa)
    if not np.any(lev_mask):
        return 0.0, 0.0
    selected_levels = levels[lev_mask]

    u_arr = np.asarray(u_da.values)[lev_mask]  # (n_lev, n_lat, n_lon)
    v_arr = np.asarray(v_da.values)[lev_mask]

    # 环形空间平均 → (n_lev,)
    u_prof = np.array([u_arr[k][annulus_mask].mean() for k in range(u_arr.shape[0])])
    v_prof = np.array([v_arr[k][annulus_mask].mean() for k in range(v_arr.shape[0])])

    # 质量加权垂直积分（梯形法）
    dp = np.abs(np.diff(selected_levels))
    u_bar = float(np.sum((u_prof[:-1] + u_prof[1:]) / 2.0 * dp) / dp.sum())
    v_bar = float(np.sum((v_prof[:-1] + v_prof[1:]) / 2.0 * dp) / dp.sum())
    return u_bar, v_bar


def compute_sensitivity_jax(
    h0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    target_time_idx: int,
    sigma_deg: float = 3.0,
    dt: float = 300.0,
    constraint_mode: str = "none",
) -> SWESensitivityResult:
    n_steps = _n_steps_for(target_time_idx, dt)
    lead_h = (target_time_idx + 1) * 6

    h0_jax = jnp.array(h0)
    u0_jax = jnp.array(u0)
    v0_jax = jnp.array(v0)

    if constraint_mode == "none":
        cfg = make_physics_config(lat_vals, lon_vals, h0_mean=float(np.mean(h0)), dt=dt)
        weights = make_gaussian_weights(lat_vals, lon_vals, center_lat, center_lon, sigma_deg)
        J_fn = make_target_J_fn(weights, cfg, n_steps)
        grad_fn = jax.jit(jax.grad(J_fn, argnums=(0, 1, 2)))

        t0 = time.perf_counter()
        print(f"[SWE-JAX] +{lead_h}h ({n_steps} steps) — compiling & computing...")
        dJ_dh, dJ_du, dJ_dv = grad_fn(h0_jax, u0_jax, v0_jax)
        elapsed = time.perf_counter() - t0
        print(f"[SWE-JAX] done in {elapsed:.1f}s")

        return _pack_result(
            dJ_dh, dJ_du, dJ_dv,
            target_time_idx, n_steps, "jax",
            lat_vals, lon_vals, center_lat, center_lon, cfg, elapsed,
        )

    elif constraint_mode == "geostrophic_hard":
        from physics.swe_model import geostrophic_wind_from_height

        cfg = make_physics_config(lat_vals, lon_vals, h0_mean=float(np.mean(h0)), dt=dt)
        weights = make_gaussian_weights(lat_vals, lon_vals, center_lat, center_lon, sigma_deg)

        def J_geo(delta_h):
            dh_u, dh_v = geostrophic_wind_from_height(delta_h, cfg)
            h_in = h0_jax + delta_h
            u_in = u0_jax + dh_u
            v_in = v0_jax + dh_v
            h_t, _, _ = swe_forward(h_in, u_in, v_in, cfg, n_steps)
            return jnp.sum(weights * h_t)

        geo_grad_fn = jax.jit(jax.grad(J_geo))
        t0 = time.perf_counter()
        print(f"[SWE-JAX-GEO-HARD] +{lead_h}h ({n_steps} steps) — compiling & computing...")
        dJ_ddh = geo_grad_fn(jnp.zeros_like(h0_jax))
        elapsed = time.perf_counter() - t0
        print(f"[SWE-JAX-GEO-HARD] done in {elapsed:.1f}s")

        raw_u, raw_v = geostrophic_wind_from_height(dJ_ddh, cfg)

        return _pack_result(
            dJ_ddh, raw_u, raw_v,
            target_time_idx, n_steps, "jax_geo_hard",
            lat_vals, lon_vals, center_lat, center_lon, cfg, elapsed,
        )

    else:
        raise ValueError(f"Unknown constraint_mode: {constraint_mode}")


def compute_sensitivity_fd(
    h0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    target_time_idx: int,
    eps_fraction: float = 1e-4,
    max_points: int = 200,
    seed: int = 0,
    dt: float = 300.0,
) -> Dict[str, np.ndarray]:
    n_steps = _n_steps_for(target_time_idx, dt)

    cfg = make_physics_config(lat_vals, lon_vals, h0_mean=float(np.mean(h0)), dt=dt)
    weights = make_gaussian_weights(lat_vals, lon_vals, center_lat, center_lon, sigma_deg=3.0)

    eps_h = eps_fraction * (float(np.std(h0)) + 1e-6)
    eps_u = eps_fraction * (float(np.std(u0)) + 1e-6)
    eps_v = eps_fraction * (float(np.std(v0)) + 1e-6)

    n_lat, n_lon = h0.shape
    rng = np.random.RandomState(seed)
    n_sample = min(max_points, n_lat * n_lon)
    flat_idx = rng.choice(n_lat * n_lon, size=n_sample, replace=False)
    lat_idxs = flat_idx // n_lon
    lon_idxs = flat_idx % n_lon

    def _J(h_, u_, v_) -> float:
        h_t, _, _ = swe_forward(jnp.array(h_), jnp.array(u_), jnp.array(v_), cfg, n_steps)
        return float(jnp.sum(weights * h_t))

    fd_S_h = np.zeros(n_sample)
    fd_S_u = np.zeros(n_sample)
    fd_S_v = np.zeros(n_sample)

    print(f"[SWE-FD] validating {n_sample} points (6×N = {6*n_sample} fwd runs)...")
    for k, (li, lj) in enumerate(zip(lat_idxs, lon_idxs)):
        h_p = h0.copy(); h_p[li, lj] += eps_h
        h_m = h0.copy(); h_m[li, lj] -= eps_h
        fd_S_h[k] = abs((_J(h_p, u0, v0) - _J(h_m, u0, v0)) / (2 * eps_h))

        u_p = u0.copy(); u_p[li, lj] += eps_u
        u_m = u0.copy(); u_m[li, lj] -= eps_u
        fd_S_u[k] = abs((_J(h0, u_p, v0) - _J(h0, u_m, v0)) / (2 * eps_u))

        v_p = v0.copy(); v_p[li, lj] += eps_v
        v_m = v0.copy(); v_m[li, lj] -= eps_v
        fd_S_v[k] = abs((_J(h0, u0, v_p) - _J(h0, u0, v_m)) / (2 * eps_v))

        if (k + 1) % 50 == 0 or k + 1 == n_sample:
            print(f"  FD: {k+1}/{n_sample}")

    return {
        "lat_idx": lat_idxs,
        "lon_idx": lon_idxs,
        "fd_S_h": fd_S_h,
        "fd_S_u": fd_S_u,
        "fd_S_v": fd_S_v,
    }


def compute_sensitivity_spsa(
    h0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    target_time_idx: int,
    n_directions: int = 64,
    eps_fraction: float = 5e-4,
    seed: int = 42,
    dt: float = 300.0,
) -> SWESensitivityResult:
    n_steps = _n_steps_for(target_time_idx, dt)
    lead_h = (target_time_idx + 1) * 6

    cfg = make_physics_config(lat_vals, lon_vals, h0_mean=float(np.mean(h0)), dt=dt)
    weights = make_gaussian_weights(lat_vals, lon_vals, center_lat, center_lon, sigma_deg=3.0)

    eps_h = eps_fraction * (float(np.std(h0)) + 1e-6)
    eps_u = eps_fraction * (float(np.std(u0)) + 1e-6)
    eps_v = eps_fraction * (float(np.std(v0)) + 1e-6)

    rng = np.random.RandomState(seed)
    grad_h = np.zeros_like(h0, dtype=np.float64)
    grad_u = np.zeros_like(u0, dtype=np.float64)
    grad_v = np.zeros_like(v0, dtype=np.float64)

    def _J(h_, u_, v_) -> float:
        h_t, _, _ = swe_forward(jnp.array(h_), jnp.array(u_), jnp.array(v_), cfg, n_steps)
        return float(jnp.sum(weights * h_t))

    t0 = time.perf_counter()
    print(f"[SWE-SPSA] +{lead_h}h, {n_directions} directions ({2*n_directions} fwd runs)...")
    for k in range(n_directions):
        dh = rng.choice([-1.0, 1.0], size=h0.shape).astype(np.float64)
        du = rng.choice([-1.0, 1.0], size=u0.shape).astype(np.float64)
        dv = rng.choice([-1.0, 1.0], size=v0.shape).astype(np.float64)

        J_plus  = _J((h0 + eps_h * dh).astype(np.float32),
                     (u0 + eps_u * du).astype(np.float32),
                     (v0 + eps_v * dv).astype(np.float32))
        J_minus = _J((h0 - eps_h * dh).astype(np.float32),
                     (u0 - eps_u * du).astype(np.float32),
                     (v0 - eps_v * dv).astype(np.float32))
        g_hat = (J_plus - J_minus) / 2.0

        grad_h += g_hat * dh / eps_h
        grad_u += g_hat * du / eps_u
        grad_v += g_hat * dv / eps_v

        if (k + 1) % 16 == 0 or k + 1 == n_directions:
            print(f"  SPSA: {k+1}/{n_directions}")

    grad_h /= n_directions
    grad_u /= n_directions
    grad_v /= n_directions
    elapsed = time.perf_counter() - t0

    return _pack_result(
        grad_h, grad_u, grad_v,
        target_time_idx, n_steps, "spsa",
        lat_vals, lon_vals, center_lat, center_lon, cfg, elapsed,
    )


def _pack_result(
    raw_h, raw_u, raw_v,
    target_time_idx, n_steps, method,
    lat_vals, lon_vals, center_lat, center_lon, cfg, elapsed,
) -> SWESensitivityResult:
    S_h = np.abs(np.asarray(raw_h, dtype=np.float64))
    S_u = np.abs(np.asarray(raw_u, dtype=np.float64))
    S_v = np.abs(np.asarray(raw_v, dtype=np.float64))
    return SWESensitivityResult(
        target_time_idx=target_time_idx,
        n_steps=n_steps,
        method=method,
        S_h=S_h,
        S_u=S_u,
        S_v=S_v,
        S_uv=np.sqrt(S_u ** 2 + S_v ** 2),
        S_total=S_h + np.sqrt(S_u ** 2 + S_v ** 2),
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        center_lat=center_lat,
        center_lon=center_lon,
        physics_cfg=cfg,
        elapsed_sec=elapsed,
    )


def sensitivity_field_to_dataarray(
    field: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    name: str,
) -> xarray.DataArray:
    return xarray.DataArray(
        field,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name=name,
    )
