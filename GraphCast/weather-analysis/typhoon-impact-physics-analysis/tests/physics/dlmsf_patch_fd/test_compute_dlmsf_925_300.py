from __future__ import annotations
import numpy as np
import pytest
from physics.dlmsf_patch_fd.dlmsf_sensitivity import compute_dlmsf_925_300


def _make_inputs(levels_hpa, u_val=5.0, v_val=3.0, nlat=10, nlon=10):
    """构造均匀风场（所有格点、所有层同值）。"""
    n = len(levels_hpa)
    u = np.full((n, nlat, nlon), u_val, dtype=np.float32)
    v = np.full((n, nlat, nlon), v_val, dtype=np.float32)
    lat = np.linspace(-5, 5, nlat)
    lon = np.linspace(115, 125, nlon)
    return u, v, lat, lon


class TestComputeDlmsf925300:

    def test_uniform_wind_returns_that_wind(self):
        """全域均匀风场（无核心掩膜）应返回该风速值。"""
        levels = np.array([925, 850, 700, 500, 400, 300], dtype=np.float32)
        u, v, lat, lon = _make_inputs(levels, u_val=5.0, v_val=3.0)
        U, V = compute_dlmsf_925_300(
            u, v, levels, lat, lon,
            center_lat=0.0, center_lon=120.0,
            core_radius_deg=0.0,
            annulus_inner_km=0.0,
            annulus_outer_km=9999.0,
        )
        assert abs(U - 5.0) < 0.1
        assert abs(V - 3.0) < 0.1

    def test_levels_outside_925_300_excluded(self):
        """200 hPa 和 1000 hPa 的层不应纳入计算。"""
        levels = np.array([1000, 925, 300, 200], dtype=np.float32)
        u = np.zeros((4, 5, 5), dtype=np.float32)
        v = np.zeros((4, 5, 5), dtype=np.float32)
        u[0] = 999.0  # 1000 hPa（应被排除）
        u[1] = 5.0    # 925 hPa
        u[2] = 5.0    # 300 hPa
        u[3] = 999.0  # 200 hPa（应被排除）
        lat = np.linspace(-2, 2, 5)
        lon = np.linspace(118, 122, 5)
        U, V = compute_dlmsf_925_300(
            u, v, levels, lat, lon,
            center_lat=0.0, center_lon=120.0,
            core_radius_deg=0.0,
            annulus_inner_km=0.0,
            annulus_outer_km=9999.0,
        )
        assert abs(U - 5.0) < 0.5

    def test_no_valid_levels_raises(self):
        """若没有 300–925 hPa 范围内的层，应抛出 ValueError。"""
        levels = np.array([200, 150], dtype=np.float32)
        u = np.ones((2, 5, 5), dtype=np.float32)
        v = np.ones((2, 5, 5), dtype=np.float32)
        lat = np.linspace(-2, 2, 5)
        lon = np.linspace(118, 122, 5)
        with pytest.raises(ValueError, match="No levels found"):
            compute_dlmsf_925_300(
                u, v, levels, lat, lon,
                center_lat=0.0, center_lon=120.0,
                core_radius_deg=0.0,
                annulus_inner_km=0.0,
                annulus_outer_km=9999.0,
            )

    def test_shape_mismatch_raises(self):
        """u/v 与 levels_hpa 长度不匹配应抛出 ValueError。"""
        levels = np.array([925, 500, 300], dtype=np.float32)
        u = np.ones((2, 5, 5), dtype=np.float32)
        v = np.ones((2, 5, 5), dtype=np.float32)
        lat = np.linspace(-2, 2, 5)
        lon = np.linspace(118, 122, 5)
        with pytest.raises(ValueError):
            compute_dlmsf_925_300(
                u, v, levels, lat, lon,
                center_lat=0.0, center_lon=120.0,
                core_radius_deg=0.0,
                annulus_inner_km=0.0,
                annulus_outer_km=9999.0,
            )

    def test_fallback_when_env_points_insufficient(self):
        """当 annulus 内有效点不足时，应抛出 ValueError 而非静默退化。"""
        levels = np.array([925, 500, 300], dtype=np.float32)
        u, v, lat, lon = _make_inputs(levels, u_val=4.0, v_val=2.0)
        # annulus_outer_km=1.0 极小，几乎没有环境点 → 触发 ValueError
        with pytest.raises(ValueError, match="Steering annulus"):
            compute_dlmsf_925_300(
                u, v, levels, lat, lon,
                center_lat=0.0, center_lon=120.0,
                core_radius_deg=0.0,
                annulus_inner_km=0.0,
                annulus_outer_km=1.0,    # 极小环状域
                min_env_points=10,       # 需要 10 个点，但 1km 环内几乎没有
            )

    def test_custom_level_bounds_shallow_only(self):
        """Custom level bounds should change which levels contribute.

        Using levels_bottom_hpa=850, levels_top_hpa=500 should exclude
        925 hPa and 300 hPa layers that would normally be included.
        """
        levels = np.array([925, 850, 500, 300], dtype=np.float32)
        u = np.zeros((4, 5, 5), dtype=np.float32)
        v = np.zeros((4, 5, 5), dtype=np.float32)
        # Give each level a unique value to detect which are used
        u[0] = 99.0   # 925 hPa - should be EXCLUDED (below bottom=850)
        u[1] = 5.0    # 850 hPa - should be INCLUDED
        u[2] = 5.0    # 500 hPa - should be INCLUDED
        u[3] = 88.0   # 300 hPa - should be EXCLUDED (above top=500)
        lat = np.linspace(-2, 2, 5)
        lon = np.linspace(118, 122, 5)
        U, V = compute_dlmsf_925_300(
            u, v, levels, lat, lon,
            center_lat=0.0, center_lon=120.0,
            core_radius_deg=0.0,
            annulus_inner_km=0.0,
            annulus_outer_km=9999.0,
            levels_bottom_hpa=850.0,
            levels_top_hpa=500.0,
        )
        # Should be ~5.0, not ~99.0 or ~88.0
        assert abs(U - 5.0) < 0.5, \
            f"Expected ~5.0 from 850-500 hPa levels, got {U}"

    def test_custom_level_bounds_top_only(self):
        """Test with only upper-level data (400-200 hPa)."""
        levels = np.array([500, 400, 300, 200], dtype=np.float32)
        u = np.zeros((4, 5, 5), dtype=np.float32)
        v = np.zeros((4, 5, 5), dtype=np.float32)
        u[0] = 11.0   # 500 hPa - EXCLUDED (below bottom=400)
        u[1] = 5.0    # 400 hPa - INCLUDED
        u[2] = 5.0    # 300 hPa - INCLUDED
        u[3] = 5.0    # 200 hPa - INCLUDED
        lat = np.linspace(-2, 2, 5)
        lon = np.linspace(118, 122, 5)
        U, V = compute_dlmsf_925_300(
            u, v, levels, lat, lon,
            center_lat=0.0, center_lon=120.0,
            core_radius_deg=0.0,
            annulus_inner_km=0.0,
            annulus_outer_km=9999.0,
            levels_bottom_hpa=400.0,
            levels_top_hpa=200.0,
        )
        assert abs(U - 5.0) < 0.5, \
            f"Expected ~5.0 from 400-200 hPa levels, got {U}"
