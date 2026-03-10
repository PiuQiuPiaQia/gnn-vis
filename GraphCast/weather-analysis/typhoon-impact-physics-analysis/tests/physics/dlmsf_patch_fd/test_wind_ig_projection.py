"""Tests for _project_wind_ig_along_track (E2: wind-along signed IG projection).

The function projects the raw (per-variable) IG arrays for u/v wind onto the
along-track direction d_hat, optionally filtering to a pressure-level band,
and returns a signed lat×lon cell map for the given window.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from physics.dlmsf_patch_fd.patch_comparison import _case_summary, _compute_dlmsf_env_mask, _project_wind_ig_along_track
from shared.patch_geometry import CenteredWindow


def _make_window(shape=(2, 3), *, lat_vals=None, lon_vals=None) -> CenteredWindow:
    if isinstance(shape, int):
        n_lat = n_lon = shape
    else:
        n_lat, n_lon = shape
    if lat_vals is None:
        lat_vals = np.linspace(10.0, 10.0 + n_lat - 1, n_lat)
    if lon_vals is None:
        lon_vals = np.linspace(120.0, 120.0 + n_lon - 1, n_lon)
    return CenteredWindow(
        lat_indices=np.arange(n_lat, dtype=np.int64),
        lon_indices=np.arange(n_lon, dtype=np.int64),
        lat_vals=np.asarray(lat_vals, dtype=np.float64),
        lon_vals=np.asarray(lon_vals, dtype=np.float64),
        center_row=n_lat // 2,
        center_col=n_lon // 2,
        core_mask=np.zeros((n_lat, n_lon), dtype=bool),
    )


def _make_wind_da(data: np.ndarray, levels: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Create a (level, lat, lon) DataArray for a wind component."""
    return xr.DataArray(
        data,
        dims=("level", "lat", "lon"),
        coords={"level": levels, "lat": lat, "lon": lon},
    )


class TestProjectWindIgAlongTrack:
    """Tests for _project_wind_ig_along_track."""

    def _base_inputs(self):
        levels = np.array([1000.0, 925.0, 850.0, 500.0, 300.0, 200.0])
        lat = np.array([10.0, 11.0])
        lon = np.array([120.0, 121.0, 122.0])
        n_lev = len(levels)
        u_data = np.ones((n_lev, 2, 3), dtype=np.float64)
        v_data = np.zeros((n_lev, 2, 3), dtype=np.float64)
        u_da = _make_wind_da(u_data, levels, lat, lon)
        v_da = _make_wind_da(v_data, levels, lat, lon)
        window = _make_window((2, 3))
        return u_da, v_da, window

    def test_output_shape_matches_window(self):
        u_da, v_da, window = self._base_inputs()
        result = _project_wind_ig_along_track(
            ig_u_full=np.asarray(u_da.values),
            ig_v_full=np.asarray(v_da.values),
            u_da=u_da,
            v_da=v_da,
            window=window,
            d_hat=(1.0, 0.0),
        )
        assert result.shape == window.shape

    def test_due_east_projects_only_u(self):
        """d̂=(1,0): signed_cell_map = sum over levels of ig_u * 1 + ig_v * 0."""
        u_da, v_da, window = self._base_inputs()
        # u=1 at all levels, v=0; d̂=(1,0) → project = u
        # 925-300 hPa filter keeps levels: 925, 850, 500, 300 → 4 levels
        result = _project_wind_ig_along_track(
            ig_u_full=np.asarray(u_da.values),
            ig_v_full=np.asarray(v_da.values),
            u_da=u_da,
            v_da=v_da,
            window=window,
            d_hat=(1.0, 0.0),
            levels_bottom_hpa=925.0,
            levels_top_hpa=300.0,
        )
        # Pressure weights are normalized; constant per-level IG remains 1.0 per cell
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, np.full((2, 3), 1.0))

    def test_due_north_projects_only_v(self):
        """d̂=(0,1): signed_cell_map = sum over levels of ig_v."""
        u_da, v_da, window = self._base_inputs()
        # set v=2 everywhere
        v_da2 = v_da + 2.0
        result = _project_wind_ig_along_track(
            ig_u_full=np.asarray(u_da.values),
            ig_v_full=np.asarray(v_da2.values),
            u_da=u_da,
            v_da=v_da2,
            window=window,
            d_hat=(0.0, 1.0),
            levels_bottom_hpa=925.0,
            levels_top_hpa=300.0,
        )
        # Pressure weights are normalized; constant per-level IG remains 2.0 per cell
        np.testing.assert_allclose(result, np.full((2, 3), 2.0))

    def test_level_filter_excludes_out_of_range(self):
        """Only levels in [300, 925] hPa are summed."""
        levels = np.array([1000.0, 925.0, 300.0, 200.0])
        lat = np.array([10.0, 11.0])
        lon = np.array([120.0, 121.0, 122.0])
        # level values as the IG: 1000→10, 925→1, 300→1, 200→100
        u_data = np.zeros((4, 2, 3), dtype=np.float64)
        for i, lev_val in enumerate([10.0, 1.0, 1.0, 100.0]):
            u_data[i] = lev_val
        u_da = _make_wind_da(u_data, levels, lat, lon)
        v_da = _make_wind_da(np.zeros_like(u_data), levels, lat, lon)
        window = _make_window((2, 3))

        result = _project_wind_ig_along_track(
            ig_u_full=u_data,
            ig_v_full=np.zeros_like(u_data),
            u_da=u_da,
            v_da=v_da,
            window=window,
            d_hat=(1.0, 0.0),
            levels_bottom_hpa=925.0,
            levels_top_hpa=300.0,
        )
        # Only levels 925 and 300 are kept; normalized trapezoid weights preserve value=1
        np.testing.assert_allclose(result, np.full((2, 3), 1.0))

    def test_negative_values_give_negative_projection(self):
        """Negative IG produces negative signed cell map."""
        u_da, v_da, window = self._base_inputs()
        # Negate u IG
        result = _project_wind_ig_along_track(
            ig_u_full=-np.asarray(u_da.values),
            ig_v_full=np.asarray(v_da.values),
            u_da=u_da,
            v_da=v_da,
            window=window,
            d_hat=(1.0, 0.0),
            levels_bottom_hpa=925.0,
            levels_top_hpa=300.0,
        )
        np.testing.assert_allclose(result, np.full((2, 3), -1.0))

    def test_no_level_dim_sums_only_spatial(self):
        """Variables without level dim are summed spatially only."""
        lat = np.array([10.0, 11.0])
        lon = np.array([120.0, 121.0, 122.0])
        u_data = np.ones((2, 3), dtype=np.float64)
        v_data = np.zeros((2, 3), dtype=np.float64)
        u_da = xr.DataArray(u_data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        v_da = xr.DataArray(v_data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        window = _make_window((2, 3))
        result = _project_wind_ig_along_track(
            ig_u_full=u_data,
            ig_v_full=v_data,
            u_da=u_da,
            v_da=v_da,
            window=window,
            d_hat=(1.0, 0.0),
        )
        # No level filtering needed; u=1, d̂_u=1 → 1.0 per cell
        np.testing.assert_allclose(result, np.full((2, 3), 1.0))


def test_wind_ig_projection_uses_time_idx_not_sum():
    """Projection must select time=1 (DLMSF convention), not sum over all time slices."""
    import xarray

    W = 5
    nlev = 3
    ig_u = np.zeros((2, nlev, W, W), dtype=np.float64)
    ig_u[0] = 10.0
    ig_u[1] = -10.0
    ig_v = np.zeros_like(ig_u)

    lat = np.linspace(20, 24, W)
    lon = np.linspace(120, 124, W)
    coords = {"time": [0, 1], "level": [925.0, 600.0, 300.0], "lat": lat, "lon": lon}
    u_da = xarray.DataArray(
        np.zeros((2, nlev, W, W), dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords=coords,
    )
    v_da = u_da.copy()
    window = _make_window(W, lat_vals=lat, lon_vals=lon)
    result = _project_wind_ig_along_track(
        ig_u_full=ig_u,
        ig_v_full=ig_v,
        u_da=u_da,
        v_da=v_da,
        window=window,
        d_hat=(1.0, 0.0),
        time_idx=1,
    )
    assert result.mean() < -1.0, "Should use time=1 (negative), not sum over time"


def test_wind_ig_projection_uses_pressure_weights():
    """Projection must apply trapezoid pressure weights matching DLMSF."""
    import xarray

    W = 5
    ig_u = np.zeros((1, 3, W, W), dtype=np.float64)
    ig_u[0, 0] = 1.0
    ig_v = np.zeros_like(ig_u)

    lat = np.linspace(20, 24, W)
    lon = np.linspace(120, 124, W)
    coords = {"time": [1], "level": [925.0, 600.0, 300.0], "lat": lat, "lon": lon}
    u_da = xarray.DataArray(
        np.zeros((1, 3, W, W), dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords=coords,
    )
    v_da = u_da.copy()
    window = _make_window(W, lat_vals=lat, lon_vals=lon)
    result = _project_wind_ig_along_track(
        ig_u_full=ig_u,
        ig_v_full=ig_v,
        u_da=u_da,
        v_da=v_da,
        window=window,
        d_hat=(1.0, 0.0),
        time_idx=0,
    )
    expected = 162.5 / 625.0
    assert abs(result.mean() - expected) < 1e-4, (
        f"Expected pressure-weighted mean ≈ {expected:.4f}, got {result.mean():.4f}"
    )


def test_wind_along_signed_case_has_multi_threshold_metrics():
    """compute_sign_agreement and compute_topk_iou_signed work for k30 and k40."""
    from physics.dlmsf_patch_fd.patch_comparison import compute_topk_iou_signed, compute_sign_agreement

    rng = np.random.default_rng(42)
    n = 50
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    for frac in (0.30, 0.40):
        k = max(1, int(np.ceil(frac * n)))
        iou_pos = compute_topk_iou_signed(a, b, k=k, sign="pos")
        iou_neg = compute_topk_iou_signed(a, b, k=k, sign="neg")
        sign_agr = compute_sign_agreement(a, b, k=k)
        assert 0.0 <= iou_pos <= 1.0
        assert 0.0 <= iou_neg <= 1.0
        assert 0.0 <= sign_agr <= 1.0


def test_signed_spearman_direction():
    """Signed Spearman should be positive for positively correlated signed scores."""
    import scipy.stats

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    b = np.array([0.5, -1.0, 2.0, -3.0, 4.0])
    a_rank = scipy.stats.rankdata(a)
    b_rank = scipy.stats.rankdata(b)
    rho = np.corrcoef(a_rank, b_rank)[0, 1]
    assert rho > 0.9


def test_annulus_mask_zeroes_core_in_wind_ig():
    """Multiplying wind_signed_cell_map by env_mask must zero core cells."""
    import xarray

    W = 9
    lat = np.linspace(19.0, 27.0, W)
    lon = np.linspace(119.0, 127.0, W)
    ig_u = np.ones((1, 1, W, W), dtype=np.float64)
    ig_v = np.zeros_like(ig_u)
    u_da = xarray.DataArray(
        np.zeros((1, 1, W, W), dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords={"time": [1], "level": [500.0], "lat": lat, "lon": lon},
    )
    window = _make_window(W, lat_vals=lat, lon_vals=lon)
    cell_map = _project_wind_ig_along_track(
        ig_u_full=ig_u, ig_v_full=ig_v,
        u_da=u_da, v_da=u_da.copy(),
        window=window, d_hat=(1.0, 0.0), time_idx=0,
    )
    env_mask = _compute_dlmsf_env_mask(
        lat_vals=lat, lon_vals=lon,
        center_lat=23.0, center_lon=123.0,
        core_radius_deg=2.0, annulus_inner_km=200.0, annulus_outer_km=900.0,
    )
    masked = cell_map * env_mask.astype(np.float64)
    assert masked[W // 2, W // 2] == 0.0, "Core cell must be zero"
    assert masked.sum() > 0.0, "Some annulus cells must be non-zero"


def test_wind_along_signed_case_has_visualization_key():
    """_case_summary must preserve visualization key from wind_along_signed case."""
    case = {
        "direction": "along", "patch_size": 3, "window_size": 7,
        "core_size": 1, "stride": 2,
        "sign_agreement_at_20": 0.5, "iou_pos_at_20": 0.3, "iou_neg_at_20": 0.2,
        "iou_pos_at_30": 0.35, "iou_neg_at_30": 0.25,
        "iou_pos_at_40": 0.4, "iou_neg_at_40": 0.28,
        "sign_agreement_at_30": 0.55, "sign_agreement_at_40": 0.6,
        "signed_spearman": 0.45, "levels_bottom_hpa": 925.0, "levels_top_hpa": 300.0,
        "visualization": {
            "meta": {"direction": "along", "patch_size": 3, "target_time_idx": 1,
                     "topq_fraction": 0.2, "source": "wind_along_signed"},
            "sign_map": {"lat_vals": [20.0], "lon_vals": [120.0],
                         "sign_class_map": [[1]], "overlap_mask": [[False]],
                         "sign_agreement_at_20": 0.5},
            "scatter": {"x_patch_abs_scores": [0.1], "y_patch_abs_scores": [0.15],
                        "spearman_rho": 0.45},
        },
    }
    summary = _case_summary(case)
    assert "visualization" in summary
    assert summary["visualization"]["meta"]["source"] == "wind_along_signed"
    assert "sign_map" in summary["visualization"]
    assert "scatter" in summary["visualization"]
