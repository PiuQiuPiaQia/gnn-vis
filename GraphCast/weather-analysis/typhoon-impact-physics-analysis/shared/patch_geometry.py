from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class CenteredWindow:
    lat_indices: np.ndarray
    lon_indices: np.ndarray
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    center_row: int
    center_col: int
    core_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.lat_vals.size), int(self.lon_vals.size)


@dataclass(frozen=True)
class PatchDefinition:
    patch_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    mask: np.ndarray
    n_cells: int


def _resolve_centered_index_bounds(
    coord_vals: np.ndarray,
    center_val: float,
    window_size: int,
) -> tuple[np.ndarray, int]:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    vals = np.asarray(coord_vals, dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        raise ValueError("coord_vals must be non-empty")

    center_idx = int(np.argmin(np.abs(vals - float(center_val))))
    actual_size = min(window_size, n)
    half = actual_size // 2
    start = max(0, center_idx - half)
    end = start + actual_size
    if end > n:
        end = n
        start = max(0, end - actual_size)

    indices = np.arange(start, end, dtype=np.int64)
    local_center_idx = int(center_idx - start)
    return indices, local_center_idx


def build_centered_window(
    full_lat_vals: np.ndarray,
    full_lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    window_size: int,
    core_size: int,
) -> CenteredWindow:
    if core_size <= 0:
        raise ValueError(f"core_size must be positive, got {core_size}")
    if core_size % 2 == 0:
        raise ValueError(f"core_size must be odd, got {core_size}")
    if core_size > window_size:
        raise ValueError(
            f"core_size ({core_size}) must not exceed window_size ({window_size})"
        )

    lat_indices, center_row = _resolve_centered_index_bounds(
        full_lat_vals,
        center_lat,
        window_size,
    )
    lon_indices, center_col = _resolve_centered_index_bounds(
        full_lon_vals,
        center_lon,
        window_size,
    )
    lat_vals = np.asarray(full_lat_vals, dtype=np.float64)[lat_indices]
    lon_vals = np.asarray(full_lon_vals, dtype=np.float64)[lon_indices]

    core_mask = np.zeros((lat_vals.size, lon_vals.size), dtype=bool)
    core_half = core_size // 2
    row_start = max(0, center_row - core_half)
    row_end = min(lat_vals.size, center_row + core_half + 1)
    col_start = max(0, center_col - core_half)
    col_end = min(lon_vals.size, center_col + core_half + 1)
    core_mask[row_start:row_end, col_start:col_end] = True

    return CenteredWindow(
        lat_indices=lat_indices,
        lon_indices=lon_indices,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        center_row=center_row,
        center_col=center_col,
        core_mask=core_mask,
    )


def build_sliding_patches(
    window: CenteredWindow,
    patch_size: int,
    stride: int,
) -> List[PatchDefinition]:
    n_lat, n_lon = window.shape
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if patch_size > n_lat or patch_size > n_lon:
        raise ValueError(
            f"patch_size ({patch_size}) must not exceed window shape {window.shape}"
        )

    patches: List[PatchDefinition] = []
    patch_id = 0
    for row_start in range(0, n_lat - patch_size + 1, stride):
        row_end = row_start + patch_size
        for col_start in range(0, n_lon - patch_size + 1, stride):
            col_end = col_start + patch_size
            mask = np.zeros((n_lat, n_lon), dtype=bool)
            mask[row_start:row_end, col_start:col_end] = True
            mask &= ~window.core_mask
            n_cells = int(mask.sum())
            if n_cells <= 0:
                continue
            patches.append(
                PatchDefinition(
                    patch_id=patch_id,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    mask=mask,
                    n_cells=n_cells,
                )
            )
            patch_id += 1
    return patches


def patch_scores_to_grid(
    scores: Sequence[float],
    patches: Sequence[PatchDefinition],
    shape: tuple[int, int],
    core_mask: np.ndarray | None = None,
) -> np.ndarray:
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.shape[0] != len(patches):
        raise ValueError(
            f"scores length ({score_arr.shape[0]}) must match patches ({len(patches)})"
        )

    accum = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    for score, patch in zip(score_arr, patches):
        mask = np.asarray(patch.mask, dtype=bool)
        accum[mask] += float(score)
        counts[mask] += 1.0

    out = np.full(shape, np.nan, dtype=np.float64)
    np.divide(accum, counts, out=out, where=(counts > 0))
    if core_mask is not None:
        out = np.where(core_mask, np.nan, out)
    return out


def patch_topk_mask(
    scores: Sequence[float],
    patches: Sequence[PatchDefinition],
    shape: tuple[int, int],
    k: int,
) -> np.ndarray:
    if k <= 0 or not patches:
        return np.zeros(shape, dtype=bool)

    score_arr = np.asarray(scores, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(score_arr))
    if finite_idx.size == 0:
        return np.zeros(shape, dtype=bool)

    actual_k = min(int(k), int(finite_idx.size))
    top_local = np.argpartition(score_arr[finite_idx], -actual_k)[-actual_k:]
    top_idx = finite_idx[top_local]

    mask = np.zeros(shape, dtype=bool)
    for idx in top_idx.tolist():
        mask |= np.asarray(patches[idx].mask, dtype=bool)
    return mask
