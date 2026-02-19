# -*- coding: utf-8 -*-
"""Patch 级候选打分的通用数值工具。"""

from __future__ import annotations

import numpy as np


def _window_sum_2d(data: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if radius <= 0:
        return values.copy()

    n_lat, n_lon = values.shape
    padded = np.pad(values, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    integral = padded.cumsum(axis=0).cumsum(axis=1)

    out = np.empty_like(values, dtype=np.float64)
    for lat_idx in range(n_lat):
        lat_start = max(lat_idx - radius, 0)
        lat_end = min(lat_idx + radius + 1, n_lat)
        for lon_idx in range(n_lon):
            lon_start = max(lon_idx - radius, 0)
            lon_end = min(lon_idx + radius + 1, n_lon)
            out[lat_idx, lon_idx] = (
                integral[lat_end, lon_end]
                - integral[lat_start, lon_end]
                - integral[lat_end, lon_start]
                + integral[lat_start, lon_start]
            )
    return out


def window_reduce_2d(data: np.ndarray, radius: int, agg: str) -> np.ndarray:
    """对 2D 网格按 Chebyshev 邻域做窗口聚合（sum/mean）。"""
    reducer = str(agg).lower().strip()
    if reducer not in {"mean", "sum"}:
        raise ValueError(f"unsupported PATCH_SCORE_AGG: {agg}")

    values = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(values)
    summed = _window_sum_2d(np.where(finite, values, 0.0), radius)
    if reducer == "sum":
        return summed

    counts = _window_sum_2d(finite.astype(np.float64), radius)
    out = np.full_like(summed, -np.inf, dtype=np.float64)
    np.divide(summed, counts, out=out, where=(counts > 0))
    return out
