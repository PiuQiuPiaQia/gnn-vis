from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import xarray

from shared.patch_geometry import CenteredWindow, build_sliding_patches, patch_scores_to_grid


@dataclass
class DLMSFSensitivityResult:
    S_abs_map: np.ndarray
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    center_lat: float
    center_lon: float
    target_time_idx: int
    d_hat: Tuple[float, float]
    axis_name: str
    J_phys_baseline: float
    U_dlmsf: float
    V_dlmsf: float
    n_patches: int
    patch_parallel_scores: np.ndarray
    elapsed_sec: float


def compute_d_hat(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> Tuple[float, float]:
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    dlon = ((dlon + 180.0) % 360.0) - 180.0
    mag = math.hypot(dlon, dlat)
    if mag < 1e-10:
        return 0.0, 0.0
    return float(dlon / mag), float(dlat / mag)


def _projection_axis(d_hat: Tuple[float, float]) -> Tuple[float, float]:
    """Return the along-track projection axis from the unit direction vector."""
    return float(d_hat[0]), float(d_hat[1])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    if u_levels.ndim != 3 or v_levels.ndim != 3:
        raise ValueError(
            f"u_levels and v_levels must be 3D arrays, got shapes {u_levels.shape} and {v_levels.shape}"
        )
    if u_levels.shape != v_levels.shape:
        raise ValueError(
            f"u_levels and v_levels shape mismatch: {u_levels.shape} vs {v_levels.shape}"
        )
    if u_levels.shape[0] != len(levels_hpa):
        raise ValueError(
            f"u_levels first dimension ({u_levels.shape[0]}) must match levels_hpa length ({len(levels_hpa)})"
        )
    if annulus_outer_km <= annulus_inner_km:
        raise ValueError(
            f"annulus_outer_km ({annulus_outer_km}) must be greater than annulus_inner_km ({annulus_inner_km})"
        )
    if levels_bottom_hpa <= levels_top_hpa:
        raise ValueError(
            f"levels_bottom_hpa ({levels_bottom_hpa}) must be greater than levels_top_hpa ({levels_top_hpa})"
        )

    level_mask = (levels_hpa >= levels_top_hpa) & (levels_hpa <= levels_bottom_hpa)
    sel_idx = np.where(level_mask)[0]
    if len(sel_idx) == 0:
        raise ValueError(f"No levels found in {levels_top_hpa}–{levels_bottom_hpa} hPa range")

    u_sel = u_levels[sel_idx]
    v_sel = v_levels[sel_idx]
    levels_sel = levels_hpa[sel_idx]
    n_sel = len(levels_sel)

    weights = np.zeros(n_sel, dtype=np.float64)
    if n_sel == 1:
        weights[0] = 1.0
    else:
        for idx in range(n_sel):
            if idx == 0:
                weights[idx] = 0.5 * abs(float(levels_sel[1]) - float(levels_sel[0]))
            elif idx == n_sel - 1:
                weights[idx] = 0.5 * abs(float(levels_sel[idx]) - float(levels_sel[idx - 1]))
            else:
                weights[idx] = 0.5 * (
                    abs(float(levels_sel[idx + 1]) - float(levels_sel[idx]))
                    + abs(float(levels_sel[idx]) - float(levels_sel[idx - 1]))
                )
    weights /= weights.sum()

    nlat, nlon = len(lat_vals), len(lon_vals)
    dist_km = np.zeros((nlat, nlon), dtype=np.float32)
    for row in range(nlat):
        for col in range(nlon):
            dist_km[row, col] = _haversine_km(
                center_lat,
                center_lon,
                float(lat_vals[row]),
                float(lon_vals[col]),
            )

    finite_mask = np.ones((nlat, nlon), dtype=bool)
    for lev in range(n_sel):
        finite_mask &= np.isfinite(u_sel[lev]) & np.isfinite(v_sel[lev])

    core_km = core_radius_deg * 111.0
    core_mask = dist_km < max(core_km, annulus_inner_km)
    env_mask = finite_mask & (~core_mask) & (dist_km <= annulus_outer_km)
    n_env = int(np.sum(env_mask))
    if n_env < min_env_points:
        raise ValueError(
            f"Steering annulus has only {n_env} valid grid points "
            f"(< min_env_points={min_env_points}). "
            f"Cannot compute DLMSF. Check annulus_inner_km / annulus_outer_km configuration."
        )

    U_sum = 0.0
    V_sum = 0.0
    count = 0
    for lev in range(n_sel):
        u_env = u_sel[lev][env_mask]
        v_env = v_sel[lev][env_mask]
        if len(u_env) > 0:
            U_sum += float(weights[lev]) * float(np.mean(u_env))
            V_sum += float(weights[lev]) * float(np.mean(v_env))
            count += 1

    if count == 0:
        return float("nan"), float("nan")
    return float(U_sum), float(V_sum)


_U_VAR = "u_component_of_wind"
_V_VAR = "v_component_of_wind"


def _extract_uv_levels(
    inputs: xarray.Dataset,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    *,
    time_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_da = inputs[_U_VAR]
    v_da = inputs[_V_VAR]
    if "batch" in u_da.dims:
        u_da = u_da.isel(batch=0)
        v_da = v_da.isel(batch=0)
    if "time" in u_da.dims:
        n = u_da.sizes["time"]
        if n <= time_idx:
            raise ValueError(
                f"{_U_VAR} 'time' dimension has {n} slice(s); "
                f"cannot select time_idx={time_idx} (requires at least {time_idx + 1} time slices, got {n})."
            )
        u_da = u_da.isel(time=time_idx)
        v_da = v_da.isel(time=time_idx)
    if "level" not in u_da.dims:
        raise ValueError(f"{_U_VAR} has no 'level' dimension; DLMSF requires multi-level data.")

    u_da = u_da.sel(lat=lat_vals, method="nearest").sel(lon=lon_vals, method="nearest").transpose("level", "lat", "lon")
    v_da = v_da.sel(lat=lat_vals, method="nearest").sel(lon=lon_vals, method="nearest").transpose("level", "lat", "lon")
    levels = np.asarray(u_da.coords["level"].values, dtype=np.float32)
    return (
        np.asarray(u_da.values, dtype=np.float32),
        np.asarray(v_da.values, dtype=np.float32),
        levels,
    )


def compute_dlmsf_patch_fd(
    *,
    eval_inputs: xarray.Dataset,
    baseline_inputs: xarray.Dataset,
    window: CenteredWindow,
    center_lat: float,
    center_lon: float,
    d_hat: Tuple[float, float],
    target_time_idx: int,
    patch_size: int = 5,
    stride: int = 2,
    direction_mode: str = "along",
    core_radius_deg: float = 3.0,
    annulus_inner_km: float = 300.0,
    annulus_outer_km: float = 900.0,
    levels_bottom_hpa: float = 925.0,
    levels_top_hpa: float = 300.0,
) -> DLMSFSensitivityResult:
    t0 = time.perf_counter()
    axis_u, axis_v = _projection_axis(d_hat)
    patches = build_sliding_patches(window, patch_size=patch_size, stride=stride)

    if not patches or (abs(axis_u) < 1e-12 and abs(axis_v) < 1e-12):
        zeros = np.zeros(window.shape, dtype=np.float64)
        patch_zeros = np.zeros(len(patches), dtype=np.float64)
        return DLMSFSensitivityResult(
            S_abs_map=np.abs(zeros),
            lat_vals=window.lat_vals,
            lon_vals=window.lon_vals,
            center_lat=center_lat,
            center_lon=center_lon,
            target_time_idx=target_time_idx,
            d_hat=(float(axis_u), float(axis_v)),
            axis_name=str(direction_mode).lower().strip(),
            J_phys_baseline=0.0,
            U_dlmsf=0.0,
            V_dlmsf=0.0,
            n_patches=len(patches),
            patch_parallel_scores=patch_zeros,
            elapsed_sec=time.perf_counter() - t0,
        )

    u_full, v_full, levels = _extract_uv_levels(
        eval_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=target_time_idx,
    )
    u_bg, v_bg, bg_levels = _extract_uv_levels(
        baseline_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=target_time_idx,
    )
    if not np.array_equal(levels, bg_levels):
        raise ValueError("baseline_inputs and eval_inputs have inconsistent pressure levels for DLMSF")

    U_full, V_full = compute_dlmsf_925_300(
        u_full,
        v_full,
        levels,
        window.lat_vals,
        window.lon_vals,
        center_lat,
        center_lon,
        core_radius_deg=core_radius_deg,
        annulus_inner_km=annulus_inner_km,
        annulus_outer_km=annulus_outer_km,
        levels_bottom_hpa=levels_bottom_hpa,
        levels_top_hpa=levels_top_hpa,
    )
    J_full = float(U_full) * float(axis_u) + float(V_full) * float(axis_v)

    patch_parallel = np.zeros(len(patches), dtype=np.float64)

    for idx, patch in enumerate(patches):
        u_masked = np.array(u_full, copy=True)
        v_masked = np.array(v_full, copy=True)
        patch_mask = np.asarray(patch.mask, dtype=bool)
        u_masked[:, patch_mask] = u_bg[:, patch_mask]
        v_masked[:, patch_mask] = v_bg[:, patch_mask]

        U_minus, V_minus = compute_dlmsf_925_300(
            u_masked,
            v_masked,
            levels,
            window.lat_vals,
            window.lon_vals,
            center_lat,
            center_lon,
            core_radius_deg=core_radius_deg,
            annulus_inner_km=annulus_inner_km,
            annulus_outer_km=annulus_outer_km,
            levels_bottom_hpa=levels_bottom_hpa,
            levels_top_hpa=levels_top_hpa,
        )

        delta_u = float(U_full) - float(U_minus)
        delta_v = float(V_full) - float(V_minus)
        patch_parallel[idx] = delta_u * float(axis_u) + delta_v * float(axis_v)
    S_abs_map = patch_scores_to_grid(
        np.abs(patch_parallel),
        patches,
        window.shape,
        core_mask=window.core_mask,
    )

    elapsed = time.perf_counter() - t0
    print(
        f"[DLMSF] +{(target_time_idx + 1) * 6}h  axis={direction_mode}  "
        f"patches={len(patches)}  {elapsed:.1f}s  "
        f"V_full=({U_full:+.2f}, {V_full:+.2f})  J={J_full:+.4f}"
    )
    return DLMSFSensitivityResult(
        S_abs_map=S_abs_map,
        lat_vals=window.lat_vals,
        lon_vals=window.lon_vals,
        center_lat=center_lat,
        center_lon=center_lon,
        target_time_idx=target_time_idx,
        d_hat=(float(axis_u), float(axis_v)),
        axis_name=str(direction_mode).lower().strip(),
        J_phys_baseline=float(J_full),
        U_dlmsf=float(U_full),
        V_dlmsf=float(V_full),
        n_patches=len(patches),
        patch_parallel_scores=patch_parallel,
        elapsed_sec=elapsed,
    )
