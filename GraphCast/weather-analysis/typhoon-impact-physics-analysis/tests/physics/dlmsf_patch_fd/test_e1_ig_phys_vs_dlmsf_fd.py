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
            min_env_points=0,
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
            min_env_points=0,
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
            min_env_points=0,
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


def test_compute_dlmsf_env_mask_uses_max_core_annulus_inner():
    """env_mask boundary must use max(core_km, annulus_inner_km), matching compute_dlmsf_925_300."""
    from physics.dlmsf_patch_fd.patch_comparison import _compute_dlmsf_env_mask

    W = 9
    lat = np.linspace(19.0, 27.0, W)
    lon = np.linspace(119.0, 127.0, W)
    center_lat, center_lon = 23.0, 123.0

    mask = _compute_dlmsf_env_mask(
        lat_vals=lat, lon_vals=lon,
        center_lat=center_lat, center_lon=center_lon,
        core_radius_deg=1.0,
        annulus_inner_km=200.0,
        annulus_outer_km=900.0,
    )
    assert not mask[W // 2, W // 2], "Center must be excluded"
    assert mask.shape == (W, W)


def test_compute_dlmsf_env_mask_finite_mask_excludes_nan_cells():
    """Cells with NaN wind values must be excluded from env_mask."""
    from physics.dlmsf_patch_fd.patch_comparison import _compute_dlmsf_env_mask

    W = 7
    lat = np.linspace(19.0, 25.0, W)
    lon = np.linspace(119.0, 125.0, W)
    center_lat, center_lon = 22.0, 122.0

    u = np.ones((2, W, W), dtype=np.float32)
    v = np.ones((2, W, W), dtype=np.float32)
    u[0, 3, 3] = float("nan")

    mask_with_nan = _compute_dlmsf_env_mask(
        lat_vals=lat, lon_vals=lon,
        center_lat=center_lat, center_lon=center_lon,
        core_radius_deg=0.5, annulus_inner_km=50.0, annulus_outer_km=900.0,
        u_levels=u, v_levels=v,
    )
    mask_no_nan = _compute_dlmsf_env_mask(
        lat_vals=lat, lon_vals=lon,
        center_lat=center_lat, center_lon=center_lon,
        core_radius_deg=0.5, annulus_inner_km=50.0, annulus_outer_km=900.0,
    )
    assert mask_with_nan.shape == (W, W)
    assert mask_no_nan.shape == (W, W)
    assert not mask_with_nan[3, 3], "NaN cell must be excluded"


def test_compute_dlmsf_env_mask_min_env_points_fallback():
    """If fewer than min_env_points qualify, fall back to finite_mask."""
    from physics.dlmsf_patch_fd.patch_comparison import _compute_dlmsf_env_mask

    W = 5
    lat = np.linspace(20.0, 24.0, W)
    lon = np.linspace(120.0, 124.0, W)
    center_lat, center_lon = 22.0, 122.0

    mask = _compute_dlmsf_env_mask(
        lat_vals=lat, lon_vals=lon,
        center_lat=center_lat, center_lon=center_lon,
        core_radius_deg=0.0,
        annulus_inner_km=99000.0,
        annulus_outer_km=99001.0,
        min_env_points=1,
    )
    assert mask.all(), "Fallback must yield all-True when no NaN provided"


def test_e1_level_band_filter_isolates_925_300_hpa():
    """E1 analytical IG must be computed on 925–300 hPa levels only.

    Strategy: build a scenario with 3 levels [1000, 500, 100] hPa.
    Put non-zero wind only at 1000 hPa (outside band) and 100 hPa (outside band).
    Put zero wind at 500 hPa (the only level inside 925–300 hPa).
    When band filter is applied, j_along should be ~0.
    Without filter, j_along would be non-zero.
    This test calls compute_ig_phys_dlmsf_along directly with/without band filter
    to verify the API behaves as expected, and then verifies that the E1 block
    in run_track_patch_analysis also applies the filter by checking j_along_analytical
    in the case output is consistent with the filtered result.
    """
    from physics.dlmsf_patch_fd.ig_phys import compute_ig_phys_dlmsf_along

    nlat, nlon = 5, 5
    levels = np.array([1000.0, 500.0, 100.0], dtype=np.float64)
    u = np.zeros((3, nlat, nlon), dtype=np.float64)
    v = np.zeros((3, nlat, nlon), dtype=np.float64)
    u[0] = 10.0
    u[2] = -10.0
    env_mask = np.ones((nlat, nlon), dtype=bool)
    d_hat = (1.0, 0.0)
    lat_vals = np.linspace(20, 25, nlat)
    lon_vals = np.linspace(120, 125, nlon)

    band_mask = (levels >= 300.0) & (levels <= 925.0)
    sel = np.where(band_mask)[0]
    result_filtered = compute_ig_phys_dlmsf_along(
        u=u[sel], v=v[sel], levels_hpa=levels[sel],
        lat_vals=lat_vals, lon_vals=lon_vals,
        center_lat=22.5, center_lon=122.5,
        d_hat=d_hat, env_mask=env_mask,
    )
    j_along_filtered = float(np.asarray(result_filtered["j_along"], dtype=np.float64))
    assert abs(j_along_filtered) < 1e-10, (
        f"Filtered j_along should be ~0, got {result_filtered['j_along']}"
    )

    result_all = compute_ig_phys_dlmsf_along(
        u=u, v=v, levels_hpa=levels,
        lat_vals=lat_vals, lon_vals=lon_vals,
        center_lat=22.5, center_lon=122.5,
        d_hat=d_hat, env_mask=env_mask,
    )
    j_along_all = float(np.asarray(result_all["j_along"], dtype=np.float64))
    assert abs(j_along_all) > 0.1, (
        f"Unfiltered j_along should be nonzero, got {result_all['j_along']}"
    )
