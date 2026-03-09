from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch

from physics.dlmsf_patch_fd.dlmsf_sensitivity import (
    compute_dlmsf_patch_fd,
    DLMSFSensitivityResult,
)


def _make_eval_inputs(levels=(925, 850, 500, 300), nlat=21, nlon=21,
                      u_val=5.0, v_val=3.0):
    """最小 eval_inputs：只含 u/v 多层场，lat/lon 围绕 (0, 120) 的 ±10° 子域。"""
    lat = np.linspace(-10, 10, nlat)
    lon = np.linspace(110, 130, nlon)
    lev = np.array(levels, dtype=np.float32)
    n_lev = len(lev)

    u_data = np.full((1, 2, n_lev, nlat, nlon), u_val, dtype=np.float32)
    v_data = np.full((1, 2, n_lev, nlat, nlon), v_val, dtype=np.float32)

    ds = xr.Dataset(
        {
            "u_component_of_wind": xr.DataArray(
                u_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                v_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
        }
    )
    return ds


class TestComputeDlmsfPatchFd:

    def test_output_type_and_shape(self):
        """返回 DLMSFSensitivityResult，S_map 形状与输入子域一致。"""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
        )
        assert isinstance(result, DLMSFSensitivityResult)
        assert result.S_map.shape == (len(lat), len(lon))

    def test_zero_d_hat_gives_zero_smap(self):
        """d_hat=(0,0) 时 J_phys 始终为 0，S_map 和 S_abs_map 应全零。"""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(0.0, 0.0),
            target_time_idx=0,
        )
        assert np.all(result.S_map == 0.0)
        # NEW: Also verify S_abs_map exists and is all zeros
        assert hasattr(result, 'S_abs_map'), "Result should have S_abs_map attribute"
        assert np.all(result.S_abs_map == 0.0), "S_abs_map should be all zeros for zero d_hat"

    def test_zero_d_hat_short_circuits_before_nan_check(self):
        """Zero d_hat should short-circuit to J_phys_baseline=0.0 even with all-NaN wind field.

        The zero d_hat check must happen BEFORE the NaN-baseline check, ensuring:
        - J_phys_baseline = 0.0 (not NaN)
        - S_map is all zeros
        - U_dlmsf and V_dlmsf can be NaN (since wind field is invalid)

        This prevents the bug where NaN-baseline branch would win first.
        """
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        lev = np.array([925, 850, 500, 300], dtype=np.float32)
        n_lev = len(lev)
        nlat, nlon = 21, 21

        # Create dataset with ALL-NaN wind field
        u_data = np.full((1, 2, n_lev, nlat, nlon), np.nan, dtype=np.float32)
        v_data = np.full((1, 2, n_lev, nlat, nlon), np.nan, dtype=np.float32)

        ds = xr.Dataset({
            "u_component_of_wind": xr.DataArray(
                u_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                v_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
        })

        # With zero d_hat, result should have J_phys_baseline=0.0 (not NaN)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(0.0, 0.0),
            target_time_idx=0,
        )

        # CRITICAL: J_phys_baseline should be 0.0, NOT NaN
        assert result.J_phys_baseline == 0.0, \
            f"Zero d_hat should give J_phys_baseline=0.0, got {result.J_phys_baseline}"
        assert np.all(result.S_map == 0.0), "S_map should be all zeros for zero d_hat"
        assert np.all(result.S_abs_map == 0.0), "S_abs_map should be all zeros for zero d_hat"

    def test_patch_fill_consistency(self):
        """同一个 patch 内所有格点的 S_map 值应相同（patch 回填）。

        使用显式已知的同 patch 格点组，避免依赖 _build_patches 私有辅助函数。
        网格: lat/lon 各 21 点，间距 1°
        patch_size_deg=2.0: 每个 patch 覆盖 2x2 格点
        例如：格点 (0,0), (0,1), (1,0), (1,1) 属于同一个 patch
        """
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
        )
        # S_map 不全为 0
        assert not np.all(result.S_map == 0.0)

        # 显式验证同 patch 格点值相同（不依赖 _build_patches）
        # Patch 1: cells (0,0), (0,1), (1,0), (1,1) - northwest corner
        patch1_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        patch1_vals = [result.S_map[i, j] for i, j in patch1_cells]
        assert all(v == patch1_vals[0] for v in patch1_vals), \
            f"Patch 1 cells should have same value, got {patch1_vals}"

        # Patch 2: cells (10,10), (10,11), (11,10), (11,11) - near center
        patch2_cells = [(10, 10), (10, 11), (11, 10), (11, 11)]
        patch2_vals = [result.S_map[i, j] for i, j in patch2_cells]
        assert all(v == patch2_vals[0] for v in patch2_vals), \
            f"Patch 2 cells should have same value, got {patch2_vals}"

        # Patch 3: cells (19,19), (19,20), (20,19), (20,20) - southeast corner
        patch3_cells = [(19, 19), (19, 20), (20, 19), (20, 20)]
        patch3_vals = [result.S_map[i, j] for i, j in patch3_cells]
        assert all(v == patch3_vals[0] for v in patch3_vals), \
            f"Patch 3 cells should have same value, got {patch3_vals}"

    def test_n_patches_positive(self):
        """至少有一个有效 patch。"""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
        )
        assert result.n_patches > 0

    def test_smap_nonnegative(self):
        """S_abs_map 为绝对值，不应有负数；S_map 可正可负（符号敏感）。"""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(0.7, 0.7),
            target_time_idx=0,
        )
        # NEW BEHAVIOR: S_abs_map is the absolute value map (always >= 0)
        # The old S_map test is updated to check S_abs_map instead
        assert hasattr(result, 'S_abs_map'), "Result should have S_abs_map attribute"
        assert np.all(result.S_abs_map >= 0.0), "S_abs_map should be non-negative"

    def test_missing_level_dimension_raises(self):
        """eval_inputs 中 u/v 没有 level 维度时应抛出 ValueError。"""
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        # 构造无 level 维度的 Dataset（只有 lat/lon）
        ds = xr.Dataset({
            "u_component_of_wind": xr.DataArray(
                np.ones((21, 21), dtype=np.float32),
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                np.ones((21, 21), dtype=np.float32),
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            ),
        })
        with pytest.raises(ValueError, match="level"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
            )


class TestSignedDlmsfBehavior:
    """Tests for the NEW signed DLMSF behavior.

    These tests verify the central difference implementation that preserves
    the sign of the sensitivity, rather than using the old unsigned absolute
    value approach.
    """

    def test_sabs_map_equals_abs_of_smap(self):
        """S_abs_map equals np.abs(S_map) for nonzero d_hat.

        The absolute value map should always equal the element-wise absolute
        value of the signed sensitivity map.
        """
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(0.7, 0.7),  # Nonzero direction
            target_time_idx=0,
        )
        assert hasattr(result, 'S_abs_map'), "Result should have S_abs_map attribute"
        # S_abs_map should equal |S_map| element-wise
        expected_abs = np.abs(result.S_map)
        np.testing.assert_allclose(
            result.S_abs_map, expected_abs,
            rtol=1e-6,
            err_msg="S_abs_map should equal np.abs(S_map)"
        )

    def test_signed_smap_tracks_d_hat_direction(self):
        """Directional toy case proves sign tracks d_hat and uses central difference.

        This test runs the same toy setup for THREE different d_hat directions
        (eastward, northward, and oblique) to prove:
        1. Sensitivity sign depends on projection onto d_hat, not a fixed axis
        2. The implementation uses CENTRAL difference, not forward difference
        3. Mixed-component projection works correctly (oblique d_hat)

        The mock uses a quadratic response: J = J0 + k*projection*perturbation + m*perturbation^2

        For this model:
        - Central diff: S_central = (J(+eps) - J(-eps)) / (2*eps) = k * projection
        - Forward diff: S_forward = (J(+eps) - J0) / eps = k * projection + m * eps

        With m != 0, these differ. The test numerically verifies central difference is used.
        """
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)  # 21 points, spacing 1 degree
        lon = np.linspace(110, 130, 21)
        center_lat, center_lon = 0.0, 120.0

        # Grid/patch geometry (explicit, no _build_patches dependency):
        # - lat[i] = -10 + i, lon[j] = 110 + j
        # - patch_size_deg=2.0: patches start at even indices
        # - Cell (i,j) belongs to patch starting at (i - i%2, j - j%2)
        # - Patch centroid lat = (lat[i_start] + lat[i_start+1]) / 2
        # - Patch centroid lon = (lon[j_start] + lon[j_start+1]) / 2

        # Parameters for the quadratic response model
        J0 = 5.0
        k = 0.1   # Linear coefficient (sensitivity magnitude)
        m = 0.5   # Quadratic coefficient (distinguishes central from forward)
        eps = 1.0
        baseline_u = 5.0
        baseline_v = 3.0

        def make_mock_compute_dlmsf(d_u, d_v):
            """Create a mock that depends on projection onto the given d_hat."""
            d_hat_norm = np.sqrt(d_u**2 + d_v**2)

            def mock_compute_dlmsf(u_levels, v_levels, levels, lat_vals, lon_vals,
                                   center_lat, center_lon, **kwargs):
                u_diff = u_levels - baseline_u
                v_diff = v_levels - baseline_v
                perturbed_mask = (np.abs(u_diff) > 0.1) | (np.abs(v_diff) > 0.1)

                if not np.any(perturbed_mask):
                    if d_u != 0:
                        return (J0 / d_u, 0.0)
                    else:
                        return (0.0, J0 / d_v)

                perturbed_indices = np.where(perturbed_mask)
                lat_indices = perturbed_indices[1]
                lon_indices = perturbed_indices[2]
                patch_centroid_lat = float(np.mean(lat_vals[lat_indices]))
                patch_centroid_lon = float(np.mean(lon_vals[lon_indices]))

                delta_lon = patch_centroid_lon - center_lon
                delta_lat = patch_centroid_lat - center_lat
                projection = (delta_lon * d_u + delta_lat * d_v) / d_hat_norm

                # Signed perturbation amplitude projected onto d_hat.
                # This tracks the new directional kernel where u/v are perturbed
                # by (eps*d_u, eps*d_v), not by a fixed diagonal (+eps, +eps).
                u_mean = float(np.mean(u_diff[perturbed_mask]))
                v_mean = float(np.mean(v_diff[perturbed_mask]))
                parallel_perturbation = (u_mean * d_u + v_mean * d_v) / d_hat_norm
                J = J0 + k * projection * parallel_perturbation + m * parallel_perturbation**2

                if d_u != 0:
                    return (J / d_u, 0.0)
                else:
                    return (0.0, J / d_v)

            return mock_compute_dlmsf

        # ============================================================
        # Test 1: Eastward direction d_hat = (1, 0)
        # ============================================================
        # Cell (10, 4) -> patch starts at (10, 4) -> centroid lat=0.5, lon=114.5
        # Cell (10, 16) -> patch starts at (10, 16) -> centroid lat=0.5, lon=126.5

        d_hat_east = (1.0, 0.0)
        with patch(
            'physics.dlmsf_patch_fd.dlmsf_sensitivity.compute_dlmsf_925_300',
            side_effect=make_mock_compute_dlmsf(*d_hat_east)
        ):
            result_east = compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=center_lat, center_lon=center_lon,
                d_hat=d_hat_east,
                target_time_idx=0,
                patch_size_deg=2.0,
                eps=eps,
            )

        # Verify S_abs_map exists
        assert hasattr(result_east, 'S_abs_map'), "Result should have S_abs_map attribute"

        west_cell = (10, 4)   # centroid lat=0.5, lon=114.5
        east_cell = (10, 16)  # centroid lat=0.5, lon=126.5

        west_s = result_east.S_map[west_cell]
        east_s = result_east.S_map[east_cell]

        # Correct projection values (centroid - center)
        west_projection = 114.5 - center_lon  # -5.5
        east_projection = 126.5 - center_lon  # +6.5

        expected_west_central = k * west_projection  # -0.55
        expected_east_central = k * east_projection  # +0.65
        expected_west_forward = k * west_projection + m * eps  # -0.05
        expected_east_forward = k * east_projection + m * eps  # +1.15

        # SIGN assertions: West negative, East positive (longitude-driven)
        assert west_s < 0, \
            f"Eastward d_hat: west cell should have NEGATIVE S, got {west_s}"
        assert east_s > 0, \
            f"Eastward d_hat: east cell should have POSITIVE S, got {east_s}"

        # NUMERICAL central difference assertion
        np.testing.assert_allclose(
            west_s, expected_west_central, rtol=0.1,
            err_msg=f"West cell: expected central diff S={expected_west_central:.4f}, "
                    f"got {west_s:.4f}. Forward diff would give {expected_west_forward:.4f}"
        )
        np.testing.assert_allclose(
            east_s, expected_east_central, rtol=0.1,
            err_msg=f"East cell: expected central diff S={expected_east_central:.4f}, "
                    f"got {east_s:.4f}. Forward diff would give {expected_east_forward:.4f}"
        )

        # VERIFY values are DIFFERENT from forward difference
        assert abs(west_s - expected_west_central) < abs(west_s - expected_west_forward), \
            f"West S={west_s} closer to forward diff ({expected_west_forward}) than central ({expected_west_central})"
        assert abs(east_s - expected_east_central) < abs(east_s - expected_east_forward), \
            f"East S={east_s} closer to forward diff ({expected_east_forward}) than central ({expected_east_central})"

        # ============================================================
        # Test 2: Northward direction d_hat = (0, 1)
        # ============================================================
        # Cell (4, 10) -> patch starts at (4, 10) -> centroid lat=-5.5, lon=120.5
        # Cell (16, 10) -> patch starts at (16, 10) -> centroid lat=6.5, lon=120.5

        d_hat_north = (0.0, 1.0)
        with patch(
            'physics.dlmsf_patch_fd.dlmsf_sensitivity.compute_dlmsf_925_300',
            side_effect=make_mock_compute_dlmsf(*d_hat_north)
        ):
            result_north = compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=center_lat, center_lon=center_lon,
                d_hat=d_hat_north,
                target_time_idx=0,
                patch_size_deg=2.0,
                eps=eps,
            )

        south_cell = (4, 10)   # centroid lat=-5.5, lon=120.5
        north_cell = (16, 10)  # centroid lat=6.5, lon=120.5

        south_s = result_north.S_map[south_cell]
        north_s = result_north.S_map[north_cell]

        # Correct projection values (only lat matters for d_hat=(0,1))
        south_projection = -5.5 - center_lat  # -5.5
        north_projection = 6.5 - center_lat   # +6.5

        expected_south_central = k * south_projection  # -0.55
        expected_north_central = k * north_projection  # +0.65
        expected_south_forward = k * south_projection + m * eps  # -0.05
        expected_north_forward = k * north_projection + m * eps  # +1.15

        # SIGN assertions: South negative, North positive (latitude-driven)
        assert south_s < 0, \
            f"Northward d_hat: south cell should have NEGATIVE S, got {south_s}"
        assert north_s > 0, \
            f"Northward d_hat: north cell should have POSITIVE S, got {north_s}"

        # NUMERICAL central difference assertion
        np.testing.assert_allclose(
            south_s, expected_south_central, rtol=0.1,
            err_msg=f"South cell: expected central diff S={expected_south_central:.4f}, "
                    f"got {south_s:.4f}. Forward diff would give {expected_south_forward:.4f}"
        )
        np.testing.assert_allclose(
            north_s, expected_north_central, rtol=0.1,
            err_msg=f"North cell: expected central diff S={expected_north_central:.4f}, "
                    f"got {north_s:.4f}. Forward diff would give {expected_north_forward:.4f}"
        )

        # VERIFY values are DIFFERENT from forward difference
        assert abs(south_s - expected_south_central) < abs(south_s - expected_south_forward), \
            f"South S={south_s} closer to forward diff ({expected_south_forward}) than central ({expected_south_central})"
        assert abs(north_s - expected_north_central) < abs(north_s - expected_north_forward), \
            f"North S={north_s} closer to forward diff ({expected_north_forward}) than central ({expected_north_central})"

        # ============================================================
        # Test 3: Oblique direction d_hat = (1, 1) normalized
        # This catches mixed-component projection bugs where the implementation
        # might only use lon or lat, but not both.
        # ============================================================
        # d_hat = (1, 1) normalized = (1/sqrt(2), 1/sqrt(2))
        # Projection = (delta_lon * d_u + delta_lat * d_v) / ||d_hat||
        #
        # Cell (4, 4) -> patch starts at (4, 4) -> centroid lat=-5.5, lon=114.5
        # Cell (16, 16) -> patch starts at (16, 16) -> centroid lat=6.5, lon=126.5
        # Both have positive contribution from lat AND lon -> should be distinguishable
        # from axis-aligned direction tests

        d_u, d_v = 1.0, 1.0
        d_hat_norm = np.sqrt(d_u**2 + d_v**2)
        d_hat_oblique = (d_u / d_hat_norm, d_v / d_hat_norm)

        with patch(
            'physics.dlmsf_patch_fd.dlmsf_sensitivity.compute_dlmsf_925_300',
            side_effect=make_mock_compute_dlmsf(*d_hat_oblique)
        ):
            result_oblique = compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=center_lat, center_lon=center_lon,
                d_hat=d_hat_oblique,
                target_time_idx=0,
                patch_size_deg=2.0,
                eps=eps,
            )

        # Cell at southwest (both lat and lon negative projection)
        sw_cell = (4, 4)  # centroid lat=-5.5, lon=114.5
        # Cell at northeast (both lat and lon positive projection)
        ne_cell = (16, 16)  # centroid lat=6.5, lon=126.5

        sw_s = result_oblique.S_map[sw_cell]
        ne_s = result_oblique.S_map[ne_cell]

        # Correct projection: (delta_lon * d_u + delta_lat * d_v) / ||d_hat||
        # SW: delta_lon = 114.5 - 120 = -5.5, delta_lat = -5.5 - 0 = -5.5
        # projection = (-5.5 * 1 + -5.5 * 1) / sqrt(2) = -11 / sqrt(2) ≈ -7.778
        sw_projection = (-5.5 * d_u + -5.5 * d_v) / d_hat_norm
        # NE: delta_lon = 126.5 - 120 = +6.5, delta_lat = 6.5 - 0 = +6.5
        # projection = (6.5 * 1 + 6.5 * 1) / sqrt(2) = 13 / sqrt(2) ≈ +9.192
        ne_projection = (6.5 * d_u + 6.5 * d_v) / d_hat_norm

        expected_sw_central = k * sw_projection
        expected_ne_central = k * ne_projection
        expected_sw_forward = k * sw_projection + m * eps
        expected_ne_forward = k * ne_projection + m * eps

        # SIGN assertions: SW negative, NE positive (mixed projection)
        assert sw_s < 0, \
            f"Oblique d_hat: SW cell should have NEGATIVE S, got {sw_s}"
        assert ne_s > 0, \
            f"Oblique d_hat: NE cell should have POSITIVE S, got {ne_s}"

        # NUMERICAL central difference assertion
        np.testing.assert_allclose(
            sw_s, expected_sw_central, rtol=0.1,
            err_msg=f"SW cell: expected central diff S={expected_sw_central:.4f}, "
                    f"got {sw_s:.4f}. Forward diff would give {expected_sw_forward:.4f}"
        )
        np.testing.assert_allclose(
            ne_s, expected_ne_central, rtol=0.1,
            err_msg=f"NE cell: expected central diff S={expected_ne_central:.4f}, "
                    f"got {ne_s:.4f}. Forward diff would give {expected_ne_forward:.4f}"
        )

        # VERIFY values are DIFFERENT from forward difference
        assert abs(sw_s - expected_sw_central) < abs(sw_s - expected_sw_forward), \
            f"SW S={sw_s} closer to forward diff ({expected_sw_forward}) than central ({expected_sw_central})"
        assert abs(ne_s - expected_ne_central) < abs(ne_s - expected_ne_forward), \
            f"NE S={ne_s} closer to forward diff ({expected_ne_forward}) than central ({expected_ne_central})"

        # CRITICAL: Verify oblique values differ from axis-aligned (mixed-component projection)
        # If implementation ignored one component, oblique results would match eastward/northward
        # For a pure lon-only bug: oblique SW would match eastward west (different centroids)
        # For a pure lat-only bug: oblique SW would match northward south (different centroids)
        # The centroids are different, so values should differ if both components are used

        # ============================================================
        # Verify S_abs_map = |S_map| for all directions
        # ============================================================
        np.testing.assert_allclose(
            result_east.S_abs_map, np.abs(result_east.S_map), rtol=1e-6,
            err_msg="S_abs_map should equal |S_map| for eastward d_hat"
        )
        np.testing.assert_allclose(
            result_north.S_abs_map, np.abs(result_north.S_map), rtol=1e-6,
            err_msg="S_abs_map should equal |S_map| for northward d_hat"
        )
        np.testing.assert_allclose(
            result_oblique.S_abs_map, np.abs(result_oblique.S_map), rtol=1e-6,
            err_msg="S_abs_map should equal |S_map| for oblique d_hat"
        )


class TestDHatNormalization:
    """Tests for d_hat normalization behavior.

    The implementation should either:
    a) Normalize non-unit d_hat internally, OR
    b) Produce the same signed sensitivity as if d_hat were normalized

    This test verifies that using d_hat=(2, 0) gives the same result as
    d_hat=(1, 0) (normalized), proving the implementation handles non-unit vectors.
    """

    def test_non_unit_d_hat_eastward(self):
        """d_hat=(2, 0) should give same S_map as d_hat=(1, 0).

        The direction is the same (eastward), only the magnitude differs.
        If the implementation normalizes internally, results should match.
        """
        ds = _make_eval_inputs(u_val=5.0, v_val=3.0)
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        # Unit vector: eastward
        result_unit = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        # Non-unit vector: still eastward, but magnitude 2
        result_nonunit = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(2.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        # S_map values should be equal (or very close)
        np.testing.assert_allclose(
            result_unit.S_map, result_nonunit.S_map, rtol=1e-6,
            err_msg="d_hat=(2,0) should give same S_map as d_hat=(1,0) after normalization"
        )

    def test_non_unit_d_hat_oblique(self):
        """d_hat=(2, 2) should give same S_map as d_hat=(1/sqrt(2), 1/sqrt(2))."""
        ds = _make_eval_inputs(u_val=5.0, v_val=3.0)
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        # Unit vector: northeast diagonal
        d_unit = (1.0 / np.sqrt(2), 1.0 / np.sqrt(2))
        result_unit = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=d_unit,
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        # Non-unit vector: same direction, magnitude sqrt(8) ≈ 2.83
        result_nonunit = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(2.0, 2.0),
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        # S_map values should be equal
        np.testing.assert_allclose(
            result_unit.S_map, result_nonunit.S_map, rtol=1e-6,
            err_msg="d_hat=(2,2) should give same S_map as normalized unit vector"
        )

    def test_non_unit_d_hat_preserves_sign(self):
        """S_map should be EXACTLY the same for opposite d_hat directions.

        When d_hat direction reverses:
        - Perturbation direction flips (u += eps*d_u vs u += eps*(-d_u))
        - Projection flips (J = U*d_u vs J = U*(-d_u))

        These two flips cancel:
        - J'_plus = U_minus * (-d_u) + V_minus * (-d_v) = -J_minus
        - J'_minus = U_plus * (-d_u) + V_plus * (-d_v) = -J_plus
        - S' = (J'_plus - J'_minus) / (2*eps) = (-J_minus - (-J_plus)) / (2*eps) = S

        This test verifies that the normalization handles negative d_hat correctly
        (e.g., d_hat=(-3, 0) normalizes to (-1, 0), not (1, 0)).

        CRITICAL: The assertion checks actual signed S_map values, not just magnitudes.
        """
        ds = _make_eval_inputs(u_val=5.0, v_val=3.0)
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        result_pos = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        result_neg = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(-3.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
            eps=1.0,
        )

        # S_map values should be EXACTLY the SAME (signed comparison)
        # because both perturbation and projection flip together
        np.testing.assert_allclose(
            result_pos.S_map, result_neg.S_map, rtol=1e-6,
            err_msg="S_map should be identical for opposite d_hat directions (both flips cancel)"
        )

        # Also verify d_hat normalization preserves sign
        assert result_pos.d_hat == (1.0, 0.0), f"Expected d_hat=(1.0, 0.0), got {result_pos.d_hat}"
        assert result_neg.d_hat == (-1.0, 0.0), f"Expected d_hat=(-1.0, 0.0), got {result_neg.d_hat}"


class TestPatchSizeEntrypointFailFast:
    """Tests for fail-fast validation order of patch_size_deg.

    patch_size_deg validation should occur at function entry, BEFORE any
    dataset extraction or other expensive work. This ensures users get
    immediate feedback on invalid parameters.
    """

    def test_patch_size_validated_before_time_dim_check(self):
        """patch_size_deg=0 should raise before time dimension validation.

        Uses a dataset with insufficient time slices (would fail later),
        but asserts that patch_size_deg=0 error is raised FIRST at entrypoint.
        """
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        lev = np.array([925, 850, 500, 300], dtype=np.float32)
        n_lev = len(lev)
        nlat, nlon = 21, 21

        # Dataset with time dimension of size 1 (insufficient for DLMSF)
        # This would trigger ValueError from _extract_uv_levels if we got that far
        u_data = np.full((1, 1, n_lev, nlat, nlon), 5.0, dtype=np.float32)
        v_data = np.full((1, 1, n_lev, nlat, nlon), 3.0, dtype=np.float32)

        ds = xr.Dataset({
            "u_component_of_wind": xr.DataArray(
                u_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                v_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
        })

        # patch_size_deg=0 is invalid - should raise at function entry
        # BEFORE the time dimension check in _extract_uv_levels
        with pytest.raises(ValueError, match="patch_size_deg"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                patch_size_deg=0.0,  # Invalid - should fail fast
            )


class TestInvalidPatchSize:
    """Tests for invalid patch_size_deg parameter validation.

    The implementation should fail fast with a clear ValueError when patch_size_deg
    is invalid (<= 0), rather than hanging in an infinite loop.
    """

    def test_zero_patch_size_raises_valueerror(self):
        """patch_size_deg=0 should raise ValueError, not hang in infinite loop."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        with pytest.raises(ValueError, match="patch_size_deg"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                patch_size_deg=0.0,
            )

    def test_negative_patch_size_raises_valueerror(self):
        """patch_size_deg=-1 should raise ValueError, not hang in infinite loop."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        with pytest.raises(ValueError, match="patch_size_deg"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                patch_size_deg=-1.0,
            )


class TestEpsZeroValidation:
    """Tests for eps=0 validation.

    eps=0 would cause division by zero in the central difference formula:
        S_P = (J_plus - J_minus) / (2*eps)

    The implementation should fail fast with a clear ValueError.
    """

    def test_eps_zero_raises_valueerror(self):
        """eps=0 should raise ValueError (division by zero)."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        with pytest.raises(ValueError, match="eps"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                eps=0.0,
            )

    def test_eps_negative_raises_valueerror(self):
        """eps<0 should raise ValueError (invalid perturbation)."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        with pytest.raises(ValueError, match="eps"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                eps=-1.0,
            )


class TestTimeDimensionValidation:
    """Tests for time dimension validation in _extract_uv_levels.

    DLMSF requires time[1] (the initial condition at t=0h), so the time
    dimension must have at least 2 slices. If time exists but has fewer
    than 2 slices, the function should raise a clear ValueError.
    """

    def test_time_dim_with_one_slice_raises_valueerror(self):
        """time dimension with only 1 slice should raise ValueError.

        DLMSF uses time[1], which requires at least 2 time slices.
        """
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        lev = np.array([925, 850, 500, 300], dtype=np.float32)
        n_lev = len(lev)
        nlat, nlon = 21, 21

        # Dataset with time dimension of size 1 (insufficient)
        u_data = np.full((1, 1, n_lev, nlat, nlon), 5.0, dtype=np.float32)
        v_data = np.full((1, 1, n_lev, nlat, nlon), 3.0, dtype=np.float32)

        ds = xr.Dataset({
            "u_component_of_wind": xr.DataArray(
                u_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                v_data, dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
        })

        with pytest.raises(ValueError, match="time.*requires.*2"):
            compute_dlmsf_patch_fd(
                eval_inputs=ds,
                lat_vals=lat, lon_vals=lon,
                center_lat=0.0, center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
            )

    def test_time_dim_with_two_slices_works(self):
        """time dimension with 2 slices should work (uses time[1])."""
        ds = _make_eval_inputs()  # Default has time dimension with 2 slices
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)
        # Should not raise
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
        )
        assert isinstance(result, DLMSFSensitivityResult)


class TestDHatInResult:
    """Tests for d_hat attribute in DLMSFSensitivityResult.

    The returned d_hat should reflect the normalized vector actually used in
    the computation, not the raw input.
    """

    def test_non_unit_d_hat_is_normalized_in_result(self):
        """result.d_hat should be normalized when input is non-unit."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        # Input d_hat is non-unit (magnitude 2)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(2.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
        )

        # Result d_hat should be normalized to (1.0, 0.0)
        assert abs(result.d_hat[0] - 1.0) < 1e-6, \
            f"Expected d_hat[0]=1.0, got {result.d_hat[0]}"
        assert abs(result.d_hat[1]) < 1e-6, \
            f"Expected d_hat[1]=0.0, got {result.d_hat[1]}"

    def test_oblique_d_hat_is_normalized_in_result(self):
        """result.d_hat should be normalized for oblique direction."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        # Input d_hat is non-unit (magnitude sqrt(8) ≈ 2.83)
        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(2.0, 2.0),
            target_time_idx=0,
            patch_size_deg=2.0,
        )

        # Result d_hat should be normalized to (1/sqrt(2), 1/sqrt(2))
        expected = 1.0 / np.sqrt(2)
        assert abs(result.d_hat[0] - expected) < 1e-6, \
            f"Expected d_hat[0]={expected}, got {result.d_hat[0]}"
        assert abs(result.d_hat[1] - expected) < 1e-6, \
            f"Expected d_hat[1]={expected}, got {result.d_hat[1]}"

    def test_zero_d_hat_preserved_in_result(self):
        """result.d_hat should remain (0, 0) when input is (0, 0)."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(0.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
        )

        # Result d_hat should still be (0.0, 0.0)
        assert result.d_hat == (0.0, 0.0), \
            f"Expected d_hat=(0.0, 0.0), got {result.d_hat}"

    def test_unit_d_hat_unchanged_in_result(self):
        """result.d_hat should remain unchanged when input is already unit."""
        ds = _make_eval_inputs()
        lat = np.linspace(-10, 10, 21)
        lon = np.linspace(110, 130, 21)

        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            lat_vals=lat, lon_vals=lon,
            center_lat=0.0, center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
            patch_size_deg=2.0,
        )

        # Result d_hat should be (1.0, 0.0)
        assert result.d_hat == (1.0, 0.0), \
            f"Expected d_hat=(1.0, 0.0), got {result.d_hat}"
