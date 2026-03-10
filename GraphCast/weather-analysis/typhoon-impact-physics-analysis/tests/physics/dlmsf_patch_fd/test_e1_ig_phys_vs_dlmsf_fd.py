"""Tests for E1: ig_phys_vs_dlmsf_fd (analytical IG on physical scalar vs FD).

Tests the helper _compute_dlmsf_env_mask which constructs the annular
environment mask used by both DLMSF and the E1 analytical IG comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import _compute_dlmsf_env_mask


class TestComputeDlmsfEnvMask:
    """_compute_dlmsf_env_mask returns boolean (n_lat, n_lon) annulus mask."""

    def _grid_around(self, center_lat: float, center_lon: float, span_deg: float = 5.0):
        n = 11
        lat = np.linspace(center_lat - span_deg, center_lat + span_deg, n)
        lon = np.linspace(center_lon - span_deg, center_lon + span_deg, n)
        return lat, lon

    def test_output_shape_matches_grid(self):
        lat, lon = self._grid_around(20.0, 130.0)
        mask = _compute_dlmsf_env_mask(
            lat_vals=lat,
            lon_vals=lon,
            center_lat=20.0,
            center_lon=130.0,
            core_radius_deg=2.0,
            annulus_inner_km=200.0,
            annulus_outer_km=1000.0,
        )
        assert mask.shape == (len(lat), len(lon))
        assert mask.dtype == bool

    def test_center_cell_excluded_by_core(self):
        """Center cell should be within core and therefore excluded."""
        lat = np.array([20.0])
        lon = np.array([130.0])
        mask = _compute_dlmsf_env_mask(
            lat_vals=lat,
            lon_vals=lon,
            center_lat=20.0,
            center_lon=130.0,
            core_radius_deg=3.0,
            annulus_inner_km=300.0,
            annulus_outer_km=900.0,
        )
        # The center cell is at distance 0, which is < core_km → excluded
        assert mask[0, 0] == False

    def test_far_cell_excluded_by_outer_radius(self):
        """Cell beyond outer_km should be excluded."""
        # Put a cell far away
        lat = np.array([20.0, 40.0])   # 40°N is ~2200 km from 20°N
        lon = np.array([130.0, 130.0])
        mask = _compute_dlmsf_env_mask(
            lat_vals=lat,
            lon_vals=lon,
            center_lat=20.0,
            center_lon=130.0,
            core_radius_deg=2.0,
            annulus_inner_km=200.0,
            annulus_outer_km=900.0,
        )
        # The far cell at ~40°N (dist ~2200 km) should be False
        assert mask[1, 0] == False

    def test_annulus_cell_included(self):
        """A cell at ~500 km (within 200-900 km band, outside core) should be True."""
        # 4.5° ≈ 500 km (at 1°=111 km)
        lat = np.array([20.0, 24.5])
        lon = np.array([130.0, 130.0])
        mask = _compute_dlmsf_env_mask(
            lat_vals=lat,
            lon_vals=lon,
            center_lat=20.0,
            center_lon=130.0,
            core_radius_deg=2.0,
            annulus_inner_km=200.0,
            annulus_outer_km=900.0,
        )
        # dist ~500 km > inner_km=200 and < outer_km=900 and > core_km → True
        assert mask[1, 0] == True

    def test_empty_grid_returns_all_false(self):
        """An empty 1-cell grid at center returns False."""
        lat = np.array([20.0])
        lon = np.array([130.0])
        mask = _compute_dlmsf_env_mask(
            lat_vals=lat,
            lon_vals=lon,
            center_lat=20.0,
            center_lon=130.0,
            core_radius_deg=2.0,
            annulus_inner_km=100.0,
            annulus_outer_km=1000.0,
        )
        assert not mask.any()  # center is at 0 km, inside core_km=222 km


def test_j_along_fd_anomaly_differs_from_j_full():
    """j_along_fd_anomaly = J_full - J_base must be computable and differ from J_full."""
    from physics.dlmsf_patch_fd.dlmsf_sensitivity import compute_dlmsf_925_300

    W = 7
    levels = np.array([925.0, 600.0, 300.0])
    lat = np.linspace(18.0, 26.0, W)
    lon = np.linspace(118.0, 126.0, W)
    center_lat, center_lon = 22.0, 122.0

    rng = np.random.default_rng(7)
    u_eval = rng.standard_normal((3, W, W)).astype(np.float32) * 5 + 3.0
    v_eval = rng.standard_normal((3, W, W)).astype(np.float32) * 3
    u_base = rng.standard_normal((3, W, W)).astype(np.float32) * 2 + 1.0
    v_base = rng.standard_normal((3, W, W)).astype(np.float32) * 1.5

    U_full, V_full = compute_dlmsf_925_300(
        u_eval, v_eval, levels, lat, lon, center_lat, center_lon,
        core_radius_deg=2.0, annulus_inner_km=200.0, annulus_outer_km=800.0,
    )
    U_base, V_base = compute_dlmsf_925_300(
        u_base, v_base, levels, lat, lon, center_lat, center_lon,
        core_radius_deg=2.0, annulus_inner_km=200.0, annulus_outer_km=800.0,
    )
    j_full = float(U_full) * 1.0 + float(V_full) * 0.0
    j_base = float(U_base) * 1.0 + float(V_base) * 0.0
    j_anomaly = j_full - j_base

    assert abs(j_base) > 0.01, "Baseline J should be non-zero"
    assert abs(j_anomaly - j_full) > 0.01, "Anomaly should differ from full J"
    assert np.isfinite(j_anomaly)
    assert np.isfinite(j_full)
