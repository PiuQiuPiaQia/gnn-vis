from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

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

    S_h: np.ndarray       # |∂J/∂h₀|  (无符号)
    S_u: np.ndarray
    S_v: np.ndarray
    S_uv: np.ndarray
    S_total: np.ndarray
    dJ_dh_signed: np.ndarray  # raw ∂J/∂h₀，保留符号（用于上游拉伸诊断）

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


def _n_steps_for(
    target_time_idx: int,
    dt: float = 300.0,
    lead_hours_override: Optional[float] = None,
) -> int:
    lead_hours = float((target_time_idx + 1) * 6) if lead_hours_override is None else float(lead_hours_override)
    return int(round(lead_hours * 3600.0 / dt))


def compute_environmental_steering_flow(
    u0: np.ndarray,
    v0: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    core_radius_deg: float,
    min_env_points: int = 9,
) -> Tuple[float, float, int, int, float]:
    """Compute steering flow from environmental wind outside cyclone core."""
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    dlat = lat2d - center_lat
    dlon = ((lon2d - center_lon + 180.0) % 360.0) - 180.0
    radius = np.sqrt(dlat ** 2 + dlon ** 2)

    if core_radius_deg <= 0.0:
        env_mask = np.ones_like(radius, dtype=bool)
    else:
        env_mask = radius > core_radius_deg

    finite_mask = np.isfinite(u0) & np.isfinite(v0)
    total_points = int(u0.size)
    valid_env_mask = env_mask & finite_mask
    env_points = int(np.count_nonzero(valid_env_mask))

    if env_points < min_env_points:
        valid_env_mask = finite_mask
        env_points = int(np.count_nonzero(valid_env_mask))

    if env_points == 0:
        return 0.0, 0.0, 0, total_points, 0.0

    U_bar = float(np.mean(u0[valid_env_mask]))
    V_bar = float(np.mean(v0[valid_env_mask]))
    masked_ratio = 1.0 - float(env_points / max(total_points, 1))
    return U_bar, V_bar, env_points, total_points, masked_ratio


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
    core_radius_deg: float = 3.0,
    forced_U_bar: Optional[float] = None,
    forced_V_bar: Optional[float] = None,
    lead_hours_override: Optional[float] = None,
    constraint_mode: str = "none",
    H_eq: Optional[float] = None,
    rayleigh_momentum_h: float = 0.0,
    rayleigh_height_h: float = 0.0,
    diffusion_coeff: float = 0.0,
    sponge_width: int = 0,
    sponge_efold_h: float = 0.0,
) -> SWESensitivityResult:
    n_steps = _n_steps_for(target_time_idx, dt, lead_hours_override=lead_hours_override)
    lead_h = float((target_time_idx + 1) * 6) if lead_hours_override is None else float(lead_hours_override)

    h0_jax = jnp.array(h0)
    u0_jax = jnp.array(u0)
    v0_jax = jnp.array(v0)

    U_bar, V_bar, n_env, n_total, masked_ratio = compute_environmental_steering_flow(
        u0=u0,
        v0=v0,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        center_lat=center_lat,
        center_lon=center_lon,
        core_radius_deg=core_radius_deg,
    )
    if forced_U_bar is not None:
        U_bar = float(forced_U_bar)
    if forced_V_bar is not None:
        V_bar = float(forced_V_bar)

    print(f"  Steering flow: U_bar={U_bar:+.2f} m/s  V_bar={V_bar:+.2f} m/s")
    print(f"  Core mask radius={core_radius_deg:.2f}°  env_points={n_env}/{n_total}  masked_ratio={masked_ratio:.3f}")
    if H_eq is not None:
        print(
            f"  SWE controls: H_eq={float(H_eq):.1f} m  rm={rayleigh_momentum_h:.1f}h  "
            f"rh={rayleigh_height_h:.1f}h  nu={diffusion_coeff:.1e}  sponge={int(sponge_width)}@{sponge_efold_h:.1f}h"
        )
    if (forced_U_bar is not None) or (forced_V_bar is not None):
        print("  Steering override is active for advection-shift diagnostics.")

    if constraint_mode == "none":
        cfg = make_physics_config(
            lat_vals,
            lon_vals,
            h0_mean=float(np.mean(h0)),
            dt=dt,
            U_bar=U_bar,
            V_bar=V_bar,
            H_eq=H_eq,
            rayleigh_momentum_h=rayleigh_momentum_h,
            rayleigh_height_h=rayleigh_height_h,
            diffusion_coeff=diffusion_coeff,
            sponge_width=sponge_width,
            sponge_efold_h=sponge_efold_h,
        )
        weights = make_gaussian_weights(lat_vals, lon_vals, center_lat, center_lon, sigma_deg)
        J_fn = make_target_J_fn(weights, cfg, n_steps)
        grad_fn = jax.jit(jax.grad(J_fn, argnums=(0, 1, 2)))

        t0 = time.perf_counter()
        print(f"[SWE-JAX] +{lead_h:g}h ({n_steps} steps) — compiling & computing...")
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

        cfg = make_physics_config(
            lat_vals,
            lon_vals,
            h0_mean=float(np.mean(h0)),
            dt=dt,
            U_bar=U_bar,
            V_bar=V_bar,
            H_eq=H_eq,
            rayleigh_momentum_h=rayleigh_momentum_h,
            rayleigh_height_h=rayleigh_height_h,
            diffusion_coeff=diffusion_coeff,
            sponge_width=sponge_width,
            sponge_efold_h=sponge_efold_h,
        )
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
        print(f"[SWE-JAX-GEO-HARD] +{lead_h:g}h ({n_steps} steps) — compiling & computing...")
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


def _pack_result(
    raw_h, raw_u, raw_v,
    target_time_idx, n_steps, method,
    lat_vals, lon_vals, center_lat, center_lon, cfg, elapsed,
) -> SWESensitivityResult:
    dJ_dh_signed = np.asarray(raw_h, dtype=np.float64)   # 保留符号
    S_h = np.abs(dJ_dh_signed)
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
        dJ_dh_signed=dJ_dh_signed,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        center_lat=center_lat,
        center_lon=center_lon,
        physics_cfg=cfg,
        elapsed_sec=elapsed,
    )
