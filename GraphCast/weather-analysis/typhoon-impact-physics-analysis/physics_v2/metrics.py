# -*- coding: utf-8 -*-
"""Physics alignment metrics."""
from __future__ import annotations

import numpy as np


def compute_anisotropy_ratio_km(weights: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray) -> float:
    """
    Compute anisotropy ratio in km-space.
    
    Converts lat/lon to km offsets around weighted centroid, computes
    weighted covariance, and returns ratio of major to minor std.
    
    Supports both 1D and 2D weight inputs:
    - 1D: weights, lat_vals, lon_vals all same length 1D arrays
    - 2D: weights is 2D (nlat x nlon), lat_vals is 1D (nlat,), lon_vals is 1D (nlon,)
    
    Args:
        weights: Weight array (1D or 2D grid)
        lat_vals: Latitude values in degrees (1D)
        lon_vals: Longitude values in degrees (1D)
    
    Returns:
        Anisotropy ratio >= 1.0
    
    Raises:
        ValueError: If shapes are incompatible, weights contain non-finite values,
                   or weights are negative
    """
    weights = np.asarray(weights, dtype=np.float64)
    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    lon_vals = np.asarray(lon_vals, dtype=np.float64)
    
    # Validate inputs
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite (no NaN or inf)")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.all(np.isfinite(lat_vals)) or not np.all(np.isfinite(lon_vals)):
        raise ValueError("lat_vals and lon_vals must be finite")
    
    # Handle 2D grid case vs 1D case
    if weights.ndim == 2:
        nlat, nlon = weights.shape
        if lat_vals.ndim != 1 or lon_vals.ndim != 1:
            raise ValueError(
                f"Shape mismatch: weights is 2D {(nlat, nlon)}, "
                f"lat_vals must be 1D (got {lat_vals.shape}), lon_vals must be 1D (got {lon_vals.shape})"
            )
        if lat_vals.shape[0] != nlat:
            raise ValueError(
                f"Shape mismatch: weights has {nlat} lat rows, lat_vals has {lat_vals.shape[0]} elements"
            )
        if lon_vals.shape[0] != nlon:
            raise ValueError(
                f"Shape mismatch: weights has {nlon} lon cols, lon_vals has {lon_vals.shape[0]} elements"
            )
        
        # Flatten for processing, creating meshgrid for lat/lon
        LON, LAT = np.meshgrid(lon_vals, lat_vals)
        weights_flat = weights.ravel()
        lat_flat = LAT.ravel()
        lon_flat = LON.ravel()
    elif weights.ndim == 1:
        # 1D case: all arrays should have same length
        if lat_vals.shape != weights.shape or lon_vals.shape != weights.shape:
            raise ValueError(
                f"Shape mismatch: weights {weights.shape}, lat_vals {lat_vals.shape}, lon_vals {lon_vals.shape}"
            )
        weights_flat = weights
        lat_flat = lat_vals
        lon_flat = lon_vals
    else:
        raise ValueError(f"weights must be 1D or 2D, got {weights.ndim}D")
    
    # Normalize weights
    weight_sum = np.sum(weights_flat)
    if weight_sum <= 0:
        # Degenerate case: return 1.0
        return 1.0
    w = weights_flat / weight_sum
    
    # Approximate km per degree
    KM_PER_DEG_LAT = 111.0
    
    # Weighted centroid
    centroid_lat = np.sum(w * lat_flat)

    # Circular mean for longitude keeps dateline-crossing samples coherent.
    lon_rad = np.radians(lon_flat)
    mean_sin = np.sum(w * np.sin(lon_rad))
    mean_cos = np.sum(w * np.cos(lon_rad))
    centroid_lon = np.degrees(np.arctan2(mean_sin, mean_cos))
    
    # Convert to km offsets from centroid
    # Use pointwise cos(latitude) for lon conversion
    cos_lat = np.cos(np.radians(lat_flat))
    cos_lat = np.maximum(cos_lat, 1e-10)  # Floor to avoid division by zero at poles
    
    y_km = (lat_flat - centroid_lat) * KM_PER_DEG_LAT
    dlon = ((lon_flat - centroid_lon + 180.0) % 360.0) - 180.0
    x_km = dlon * KM_PER_DEG_LAT * cos_lat
    
    # Weighted covariance matrix (2x2)
    cov_xx = np.sum(w * x_km * x_km)
    cov_yy = np.sum(w * y_km * y_km)
    cov_xy = np.sum(w * x_km * y_km)
    
    # Eigenvalues of covariance matrix give variances along principal axes
    # For 2x2 symmetric matrix: lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    discriminant = max(0.0, trace * trace - 4.0 * det)
    
    lambda_major = (trace + np.sqrt(discriminant)) / 2.0
    lambda_minor = (trace - np.sqrt(discriminant)) / 2.0
    
    # Standard deviations along principal axes
    major_std = np.sqrt(max(0.0, lambda_major))
    minor_std = np.sqrt(max(1e-20, lambda_minor))  # Floor at small value
    
    # Floor minor_std at 1e-10 as per spec
    minor_std = max(minor_std, 1e-10)
    
    # Anisotropy ratio
    ratio = major_std / minor_std

    return max(1.0, ratio)


def compute_upstream_fraction(
    weights: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    U_bar: float,
    V_bar: float,
    lead_h: float,
    core_radius_deg: float = 0.0,
) -> float:
    """Compute upstream half-plane mass fraction.
    
    Measures the fraction of total weight that lies in the upstream half-plane
    relative to the cyclone center, based on expected advection displacement.
    
    Args:
        weights: 2D weight array (nlat x nlon)
        lat_vals: 1D latitude values in degrees
        lon_vals: 1D longitude values in degrees
        center_lat: Cyclone center latitude
        center_lon: Cyclone center longitude
        U_bar: Zonal steering wind (m/s)
        V_bar: Meridional steering wind (m/s)
        lead_h: Lead time in hours
        core_radius_deg: Optional radius to exclude from denominator (degrees)
    
    Returns:
        Upstream fraction in [0, 1], or NaN if no valid weights or zero displacement.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("weights must be 2D")
    if np.any(~np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")

    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    lon_vals = np.asarray(lon_vals, dtype=np.float64)
    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("lat_vals and lon_vals must be 1D")
    if w.shape != (lat_vals.size, lon_vals.size):
        raise ValueError("weights shape must match (lat, lon)")

    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    
    # Compute expected upstream displacement direction
    seconds = float(lead_h) * 3600.0
    cos_lat = max(abs(float(np.cos(np.radians(center_lat)))), 1e-6)
    dlat_upstream = -float(V_bar) * seconds / 111000.0
    dlon_upstream = -float(U_bar) * seconds / (111000.0 * cos_lat)

    expected_norm = float(np.hypot(dlat_upstream, dlon_upstream))
    if expected_norm <= 1e-8:
        return float("nan")

    # Build projection: p = dlat*dlat_upstream + dlon*dlon_upstream
    # Offsets from cyclone center
    dlat = lat2d - float(center_lat)
    dlon = ((lon2d - float(center_lon) + 180.0) % 360.0) - 180.0
    p = dlat * dlat_upstream + dlon * dlon_upstream

    # Compute radius from center for optional core exclusion
    radius_deg = np.sqrt(dlat**2 + dlon**2)
    
    # Valid mask: exclude core if requested
    if core_radius_deg > 0.0:
        valid_mask = radius_deg > core_radius_deg
    else:
        valid_mask = np.ones_like(w, dtype=bool)
    
    # Upstream weights: p > 0 and valid
    upstream_mask = (p > 0) & valid_mask
    
    # Compute fractions
    upstream_sum = float(np.sum(w[upstream_mask]))
    valid_sum = float(np.sum(w[valid_mask]))
    
    if valid_sum <= 0.0:
        return float("nan")
    
    frac = upstream_sum / valid_sum
    return float(np.clip(frac, 0.0, 1.0))
