# -*- coding: utf-8 -*-
"""Utilities for occlusion-based perturbation scanning."""

import numpy as np
import xarray
from typing import List, Tuple


def compute_baseline(inputs: xarray.Dataset, variables: List[str], baseline_mode: str) -> xarray.Dataset:
    data_vars = {}
    for var in variables:
        da = inputs[var]
        if baseline_mode == "spatial_mean":
            data_vars[var] = da.mean(dim=("lat", "lon"), keepdims=True)
        elif baseline_mode == "spatial_median":
            data_vars[var] = da.median(dim=("lat", "lon"), keepdims=True)
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
