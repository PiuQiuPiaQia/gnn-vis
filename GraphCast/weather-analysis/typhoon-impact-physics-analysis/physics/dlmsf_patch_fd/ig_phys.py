"""Analytical Integrated Gradients for the linear DLMSF-along scalar (E1).

The DLMSF-along scalar is:

    J_along = Σ_l w_l * (1/N_env) * Σ_{cell ∈ env} (u_cell * d̂_u + v_cell * d̂_v)

where
  - w_l are pressure-layer weights (trapezoidal rule, sum to 1),
  - env is the annular environment mask (same as used in compute_dlmsf_925_300),
  - d̂ = (d̂_u, d̂_v) is the unit direction vector (along or cross track),
  - the baseline is u=v=0 everywhere.

Because J_along is linear in u and v, the Integrated Gradients attribution
equals input × gradient:

    IG_u(cell, l)  = u_cell * w_l * env_mask_cell / N_env * d̂_u
    IG_v(cell, l)  = v_cell * w_l * env_mask_cell / N_env * d̂_v

This module collapses (sums) the level dimension to produce a 2-D lat×lon map.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _level_weights(levels_hpa: np.ndarray) -> np.ndarray:
    """Trapezoidal pressure-level weights that sum to 1.

    Parameters
    ----------
    levels_hpa:
        1-D array of pressure levels in hPa (any order).

    Returns
    -------
    np.ndarray
        1-D weight array of the same length, summing to 1.
    """
    levels = np.asarray(levels_hpa, dtype=np.float64)
    n = len(levels)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            weights[i] = 0.5 * abs(float(levels[1]) - float(levels[0]))
        elif i == n - 1:
            weights[i] = 0.5 * abs(float(levels[i]) - float(levels[i - 1]))
        else:
            weights[i] = 0.5 * (
                abs(float(levels[i + 1]) - float(levels[i]))
                + abs(float(levels[i]) - float(levels[i - 1]))
            )
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def compute_ig_phys_dlmsf_along(
    *,
    u: np.ndarray,
    v: np.ndarray,
    levels_hpa: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    d_hat: Tuple[float, float],
    env_mask: np.ndarray,
) -> Dict[str, object]:
    """Compute the analytical IG attribution for the linear DLMSF-along scalar.

    Parameters
    ----------
    u, v:
        Wind component arrays of shape ``(n_levels, n_lat, n_lon)``.
    levels_hpa:
        1-D array of pressure levels in hPa, length ``n_levels``.
    lat_vals, lon_vals:
        1-D coordinate arrays of length ``n_lat`` and ``n_lon``.
    center_lat, center_lon:
        Typhoon center coordinates (unused in the pure linear formula but
        retained for API consistency with future extensions).
    d_hat:
        ``(d_u, d_v)`` — the unit along-track direction vector, where
        ``d_u`` is the zonal (lon) component and ``d_v`` is the meridional
        (lat) component.
    env_mask:
        Boolean array of shape ``(n_lat, n_lon)`` — ``True`` for cells
        included in the environmental annulus.

    Returns
    -------
    dict with keys:
        ``"ig_u_latlon"``  — 2-D ndarray (n_lat, n_lon), IG for u summed over levels.
        ``"ig_v_latlon"``  — 2-D ndarray (n_lat, n_lon), IG for v summed over levels.
        ``"j_along"``      — float, the scalar value of J_along (useful for E1 verification).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    env_mask = np.asarray(env_mask, dtype=bool)
    levels_hpa = np.asarray(levels_hpa, dtype=np.float64)

    if u.ndim != 3 or v.ndim != 3:
        raise ValueError(
            f"u and v must be 3-D (n_levels, n_lat, n_lon), got {u.shape} and {v.shape}"
        )
    if u.shape != v.shape:
        raise ValueError(f"u and v shape mismatch: {u.shape} vs {v.shape}")
    if u.shape[0] != len(levels_hpa):
        raise ValueError(
            f"u first dimension ({u.shape[0]}) must match levels_hpa length ({len(levels_hpa)})"
        )
    if env_mask.shape != u.shape[1:]:
        raise ValueError(
            f"env_mask shape {env_mask.shape} must match spatial grid {u.shape[1:]}"
        )

    d_u, d_v = float(d_hat[0]), float(d_hat[1])
    weights = _level_weights(levels_hpa)  # shape: (n_levels,)

    n_env = float(max(int(env_mask.sum()), 1))

    # Normalised env mask: 1/N_env inside, 0 outside
    env_float = env_mask.astype(np.float64) / n_env  # (n_lat, n_lon)

    # IG_u = Σ_l w_l * u_l * env_float * d̂_u
    # IG_v = Σ_l w_l * v_l * env_float * d̂_v
    ig_u = np.zeros(u.shape[1:], dtype=np.float64)
    ig_v = np.zeros(v.shape[1:], dtype=np.float64)

    for lev_idx, w in enumerate(weights):
        ig_u += w * u[lev_idx] * env_float * d_u
        ig_v += w * v[lev_idx] * env_float * d_v

    # J_along = Σ_l w_l * (1/N_env) * Σ_{cell ∈ env} (u_cell * d̂_u + v_cell * d̂_v)
    # = Σ_cell (ig_u[cell] + ig_v[cell])
    j_along = float(np.sum(ig_u) + np.sum(ig_v))

    return {
        "ig_u_latlon": ig_u,
        "ig_v_latlon": ig_v,
        "j_along": j_along,
    }
