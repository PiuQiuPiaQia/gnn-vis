# -*- coding: utf-8 -*-
"""Environmental steering flow computation for deep-layer analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SteeringFlowResult:
    """Result container for environmental steering flow computation.

    Attributes:
        U_bar: Zonal (eastward) component of steering flow (m/s)
        V_bar: Meridional (northward) component of steering flow (m/s)
        n_env: Number of environmental points used in average
        n_total: Total number of finite points in domain
        masked_ratio: Fraction of points masked out (core + annulus boundary)
    """
    U_bar: float
    V_bar: float
    n_env: int
    n_total: int
    masked_ratio: float


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_deep_layer_environmental_steering(
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
) -> SteeringFlowResult:
    """Compute deep-layer (850-300 hPa) environmental steering flow.

    Selects pressure levels between 850 and 300 hPa (inclusive), computes
    layer-averaged winds with pressure-thickness weights, and masks out
    the vortex core region to extract environmental steering flow.

    Args:
        u_levels: 3D array of zonal winds (level, lat, lon) in m/s
        v_levels: 3D array of meridional winds (level, lat, lon) in m/s
        levels_hpa: 1D array of pressure levels in hPa
        lat_vals: 1D array of latitudes in degrees
        lon_vals: 1D array of longitudes in degrees
        center_lat: Latitude of vortex center in degrees
        center_lon: Longitude of vortex center in degrees
        core_radius_deg: Core radius in degrees to exclude
        annulus_inner_km: Inner radius of environmental annulus in km
        annulus_outer_km: Outer radius of environmental annulus in km
        min_env_points: Minimum valid environmental points for reliable estimate

    Returns:
        SteeringFlowResult with U_bar, V_bar, n_env, n_total, masked_ratio
    """
    # Input validation
    if u_levels.ndim != 3 or v_levels.ndim != 3:
        raise ValueError("u_levels and v_levels must be 3D arrays")
    if u_levels.shape != v_levels.shape:
        raise ValueError("u_levels and v_levels shape mismatch")
    if levels_hpa.ndim != 1:
        raise ValueError("levels_hpa must be 1D array")
    if u_levels.shape[0] != len(levels_hpa):
        raise ValueError("u_levels first dimension must match levels_hpa length")
    level_diffs = np.diff(levels_hpa.astype(np.float64))
    monotonic_inc = np.all(level_diffs >= 0.0)
    monotonic_dec = np.all(level_diffs <= 0.0)
    if not (monotonic_inc or monotonic_dec):
        raise ValueError("levels_hpa must be monotonic")

    n_levels, nlat, nlon = u_levels.shape
    if len(lat_vals) != nlat:
        raise ValueError("lat_vals length must match u_levels lat dimension")
    if len(lon_vals) != nlon:
        raise ValueError("lon_vals length must match u_levels lon dimension")
    if annulus_inner_km < 0.0:
        raise ValueError("annulus_inner_km must be non-negative")
    if annulus_outer_km <= annulus_inner_km:
        raise ValueError("annulus_outer_km must be greater than annulus_inner_km")

    # Step 1: Select 850-300 hPa levels (inclusive)
    level_mask = (levels_hpa >= 300.0) & (levels_hpa <= 850.0)
    selected_indices = np.where(level_mask)[0]

    if len(selected_indices) == 0:
        raise ValueError("No levels found in 850-300 hPa range")

    # Extract selected levels
    u_sel = u_levels[selected_indices, :, :]
    v_sel = v_levels[selected_indices, :, :]
    levels_sel = levels_hpa[selected_indices]

    # Step 2: Compute pressure-thickness weights
    # Simple approach: use half-distance to adjacent levels
    n_sel = len(levels_sel)
    weights = np.zeros(n_sel, dtype=np.float32)

    if n_sel == 1:
        weights[0] = 1.0
    else:
        for i in range(n_sel):
            if i == 0:
                # First level: trapezoidal half-width to next level
                weights[i] = 0.5 * abs(levels_sel[1] - levels_sel[0])
            elif i == n_sel - 1:
                # Last level: trapezoidal half-width from previous level
                weights[i] = 0.5 * abs(levels_sel[i] - levels_sel[i - 1])
            else:
                # Middle: average of distances to neighbors
                weights[i] = 0.5 * (abs(levels_sel[i + 1] - levels_sel[i]) +
                                    abs(levels_sel[i] - levels_sel[i - 1]))

    # Normalize weights
    weight_sum = float(weights.sum())
    if (not np.isfinite(weight_sum)) or weight_sum <= 0.0:
        raise ValueError("Invalid pressure-thickness weights")
    weights = weights / weight_sum

    # Step 3: Create 2D lat/lon grids
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    # Step 4: Compute distance from center in km for each grid point
    dist_km = np.zeros((nlat, nlon), dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            dist_km[i, j] = _haversine_distance_km(
                center_lat, center_lon, lat2d[i, j], lon2d[i, j]
            )

    # Step 5: Compute finite mask (non-NaN in both u and v at any selected level)
    # A point is finite if ALL selected levels have finite values
    finite_mask = np.ones((nlat, nlon), dtype=bool)
    for k in range(n_sel):
        finite_mask &= np.isfinite(u_sel[k, :, :]) & np.isfinite(v_sel[k, :, :])

    n_total = int(np.sum(finite_mask))

    if n_total == 0:
        return SteeringFlowResult(
            U_bar=float("nan"),
            V_bar=float("nan"),
            n_env=0,
            n_total=0,
            masked_ratio=1.0,
        )

    # Step 6: Create core mask and annulus mask
    # Core exclusion: within core_radius_deg OR within annulus_inner_km
    # Note: core_radius_deg is in degrees, need to convert roughly to km
    # 1 degree ~ 111 km at equator
    core_radius_km_approx = core_radius_deg * 111.0
    core_mask = dist_km < max(core_radius_km_approx, annulus_inner_km)

    # Environmental annulus mask
    env_mask = finite_mask & (~core_mask) & (dist_km <= annulus_outer_km)

    n_env = int(np.sum(env_mask))

    # Step 7: Check if we have enough environmental points
    if n_env < min_env_points:
        # Fallback: use full finite-domain mean (no core masking)
        env_mask = finite_mask
        n_env = n_total

    # Step 8: Compute layer-averaged winds over environmental points
    U_sum = 0.0
    V_sum = 0.0
    count = 0

    for k in range(n_sel):
        w_k = float(weights[k])
        u_k = u_sel[k, :, :]
        v_k = v_sel[k, :, :]

        # Apply environmental mask
        u_env = u_k[env_mask]
        v_env = v_k[env_mask]

        if len(u_env) > 0:
            U_sum += w_k * float(np.mean(u_env))
            V_sum += w_k * float(np.mean(v_env))
            count += 1

    if count == 0:
        U_bar = float("nan")
        V_bar = float("nan")
    else:
        U_bar = U_sum
        V_bar = V_sum

    # Compute masked ratio
    masked_ratio = 1.0 - (n_env / n_total) if n_total > 0 else 1.0

    # In fallback mode, masked_ratio should be 0
    if n_env == n_total:
        masked_ratio = 0.0

    return SteeringFlowResult(
        U_bar=U_bar,
        V_bar=V_bar,
        n_env=n_env,
        n_total=n_total,
        masked_ratio=masked_ratio,
    )
