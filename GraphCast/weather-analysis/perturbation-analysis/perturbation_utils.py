# -*- coding: utf-8 -*-
"""Utilities for occlusion-based perturbation scanning."""

import numpy as np
import xarray
from typing import List, Tuple


def _annulus_mask(lat_vals: np.ndarray, lon_vals: np.ndarray, center_lat: float, center_lon: float, inner_deg: float, outer_deg: float) -> xarray.DataArray:
    lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lon_diff = ((lon2d - center_lon + 180.0) % 360.0) - 180.0
    lat_rad = np.deg2rad(lat2d)
    center_lat_rad = np.deg2rad(center_lat)
    lon_diff_rad = np.deg2rad(lon_diff)
    # Great-circle central angle in degrees to avoid latitude-dependent distortion.
    cos_d = np.sin(lat_rad) * np.sin(center_lat_rad) + np.cos(lat_rad) * np.cos(center_lat_rad) * np.cos(lon_diff_rad)
    cos_d = np.clip(cos_d, -1.0, 1.0)
    dist_deg = np.rad2deg(np.arccos(cos_d))
    mask = (dist_deg >= float(inner_deg)) & (dist_deg <= float(outer_deg))
    return xarray.DataArray(mask, dims=("lat", "lon"), coords={"lat": lat_vals, "lon": lon_vals})


def compute_baseline(
    inputs: xarray.Dataset,
    variables: List[str],
    baseline_mode: str,
    center_lat: float = None,
    center_lon: float = None,
    inner_deg: float = 5.0,
    outer_deg: float = 12.0,
    min_points: int = 120,
) -> xarray.Dataset:
    data_vars = {}
    annulus_mask = None
    annulus_points = 0

    if baseline_mode in {"local_annulus_mean", "local_annulus_median"}:
        if center_lat is None or center_lon is None:
            raise ValueError("center_lat/center_lon are required for local annulus baseline")
        if outer_deg <= inner_deg:
            raise ValueError(f"outer_deg ({outer_deg}) must be greater than inner_deg ({inner_deg})")
        lat_vals = inputs.coords["lat"].values
        lon_vals = inputs.coords["lon"].values
        annulus_mask = _annulus_mask(lat_vals, lon_vals, center_lat, center_lon, inner_deg, outer_deg)
        annulus_points = int(annulus_mask.sum().item())

    for var in variables:
        da = inputs[var]
        if baseline_mode == "spatial_mean":
            data_vars[var] = da.mean(dim=("lat", "lon"), keepdims=True)
        elif baseline_mode == "spatial_median":
            data_vars[var] = da.median(dim=("lat", "lon"), keepdims=True)
        elif baseline_mode == "local_annulus_mean":
            if annulus_points < int(min_points):
                data_vars[var] = da.mean(dim=("lat", "lon"), keepdims=True)
            else:
                data_vars[var] = da.where(annulus_mask).mean(dim=("lat", "lon"), keepdims=True, skipna=True)
        elif baseline_mode == "local_annulus_median":
            if annulus_points < int(min_points):
                data_vars[var] = da.median(dim=("lat", "lon"), keepdims=True)
            else:
                data_vars[var] = da.where(annulus_mask).median(dim=("lat", "lon"), keepdims=True, skipna=True)
        else:
            raise ValueError(f"unsupported BASELINE_MODE: {baseline_mode}")
    return xarray.Dataset(data_vars)


def select_region_indices(lat_vals: np.ndarray, lon_vals: np.ndarray, center_lat: float, center_lon: float, radius_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lat_mask = (lat_vals >= center_lat - radius_deg) & (lat_vals <= center_lat + radius_deg)
    lon_dist = ((lon_vals - center_lon + 180) % 360) - 180
    lon_mask = np.abs(lon_dist) <= radius_deg
    return np.where(lat_mask)[0], np.where(lon_mask)[0]


def build_indexer(da: xarray.DataArray, lat_slice, lon_slice, time_sel, level_sel):
    indexer = []
    for dim in da.dims:
        if dim == "time":
            indexer.append(time_sel)
        elif dim == "level":
            indexer.append(level_sel)
        elif dim == "lat":
            indexer.append(lat_slice)
        elif dim == "lon":
            indexer.append(lon_slice)
        else:
            indexer.append(slice(None))
    return tuple(indexer)


def build_baseline_indexer(da: xarray.DataArray, time_sel, level_sel):
    indexer = []
    for dim in da.dims:
        if dim == "time":
            indexer.append(time_sel)
        elif dim == "level":
            indexer.append(level_sel)
        elif dim == "lat":
            indexer.append(slice(0, 1))
        elif dim == "lon":
            indexer.append(slice(0, 1))
        else:
            indexer.append(slice(None))
    return tuple(indexer)


def resolve_level_sel(da: xarray.DataArray, perturb_levels):
    if "level" not in da.dims:
        return slice(None)
    if perturb_levels is None:
        return slice(None)
    levels = da.coords["level"].values
    level_idx = [int(np.where(levels == lvl)[0][0]) for lvl in perturb_levels if lvl in levels]
    if not level_idx:
        return slice(None)
    return level_idx
