from __future__ import annotations

from unittest.mock import patch

import numpy as np
import xarray as xr

from physics.dlmsf_patch_fd.dlmsf_sensitivity import (
    DLMSFSensitivityResult,
    compute_dlmsf_patch_fd,
)
from shared.patch_geometry import build_centered_window, build_sliding_patches


def _make_eval_inputs(
    *,
    levels=(925, 850, 500, 300),
    nlat: int = 21,
    nlon: int = 21,
    u_val: float = 5.0,
    v_val: float = 3.0,
) -> xr.Dataset:
    lat = np.linspace(-10, 10, nlat)
    lon = np.linspace(110, 130, nlon)
    lev = np.asarray(levels, dtype=np.float32)
    n_lev = len(lev)
    u_data = np.full((1, 2, n_lev, nlat, nlon), u_val, dtype=np.float32)
    v_data = np.full((1, 2, n_lev, nlat, nlon), v_val, dtype=np.float32)
    return xr.Dataset(
        {
            "u_component_of_wind": xr.DataArray(
                u_data,
                dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
            "v_component_of_wind": xr.DataArray(
                v_data,
                dims=("batch", "time", "level", "lat", "lon"),
                coords={"level": lev, "lat": lat, "lon": lon},
            ),
        }
    )


def _make_window(ds: xr.Dataset):
    lat = np.asarray(ds.coords["lat"].values, dtype=np.float64)
    lon = np.asarray(ds.coords["lon"].values, dtype=np.float64)
    return build_centered_window(
        lat,
        lon,
        center_lat=0.0,
        center_lon=120.0,
        window_size=21,
        core_size=3,
    )


class TestDlmsfPatchAblation:
    def test_returns_new_patch_level_fields(self):
        ds = _make_eval_inputs()
        baseline = _make_eval_inputs(u_val=0.0, v_val=0.0)
        window = _make_window(ds)

        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            baseline_inputs=baseline,
            window=window,
            center_lat=0.0,
            center_lon=120.0,
            d_hat=(1.0, 0.0),
            target_time_idx=0,
            patch_size=5,
            stride=2,
        )

        assert isinstance(result, DLMSFSensitivityResult)
        assert result.S_abs_map.shape == window.shape
        assert result.patch_parallel_scores.shape == (result.n_patches,)
        assert result.axis_name == "along"
        assert result.n_patches == 81

    def test_zero_direction_short_circuits_to_zero_maps(self):
        ds = _make_eval_inputs()
        baseline = _make_eval_inputs(u_val=0.0, v_val=0.0)
        window = _make_window(ds)

        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            baseline_inputs=baseline,
            window=window,
            center_lat=0.0,
            center_lon=120.0,
            d_hat=(0.0, 0.0),
            target_time_idx=0,
            patch_size=5,
            stride=2,
        )

        assert np.all(result.S_abs_map == 0.0)
        assert np.all(result.patch_parallel_scores == 0.0)

    def test_background_replacement_scales_with_patch_cell_count(self):
        ds = _make_eval_inputs(u_val=5.0, v_val=3.0)
        baseline = _make_eval_inputs(u_val=0.0, v_val=0.0)
        window = _make_window(ds)
        patches = build_sliding_patches(window, patch_size=5, stride=2)
        total_cells = float(np.prod(window.shape))

        def fake_compute(u_levels, v_levels, levels_hpa, lat_vals, lon_vals, center_lat, center_lon, **kwargs):
            return float(np.mean(u_levels)), float(np.mean(v_levels))

        with patch(
            "physics.dlmsf_patch_fd.dlmsf_sensitivity.compute_dlmsf_925_300",
            side_effect=fake_compute,
        ):
            result = compute_dlmsf_patch_fd(
                eval_inputs=ds,
                baseline_inputs=baseline,
                window=window,
                center_lat=0.0,
                center_lon=120.0,
                d_hat=(1.0, 0.0),
                target_time_idx=0,
                patch_size=5,
                stride=2,
            )

        expected = np.array(
            [(5.0 * patch.n_cells) / total_cells for patch in patches],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result.patch_parallel_scores, expected, rtol=1e-6)

    def test_overlap_abs_map_is_nonnegative(self):
        ds = _make_eval_inputs()
        baseline = _make_eval_inputs(u_val=0.0, v_val=0.0)
        window = _make_window(ds)

        result = compute_dlmsf_patch_fd(
            eval_inputs=ds,
            baseline_inputs=baseline,
            window=window,
            center_lat=0.0,
            center_lon=120.0,
            d_hat=(0.7, 0.7),
            target_time_idx=0,
            patch_size=5,
            stride=2,
        )

        finite_abs = result.S_abs_map[np.isfinite(result.S_abs_map)]
        assert np.all(finite_abs >= 0.0)


class TestDlmsfTimeIdxPropagation:
    """time_idx 必须被传播到 _extract_uv_levels，不能写死成 1。

    Note: This only tests propagation via compute_dlmsf_patch_fd().
    The env_mask and E1 call sites in patch_comparison.py are not directly
    tested due to requiring full AnalysisContext infrastructure.
    """

    def test_different_target_time_idx_gives_different_j_phys(self):
        """用 t=0 和 t=1 放不同风速，确认 J_phys_baseline 不同。"""
        lev = np.asarray([925, 500, 300], dtype=np.float32)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(115, 125, 11)
        # t=0: u=2, t=1: u=8
        u_data = np.zeros((1, 2, 3, 11, 11), dtype=np.float32)
        v_data = np.zeros((1, 2, 3, 11, 11), dtype=np.float32)
        u_data[:, 0, :, :, :] = 2.0
        u_data[:, 1, :, :, :] = 8.0
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
        baseline = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray(
                    np.zeros_like(u_data), dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
                "v_component_of_wind": xr.DataArray(
                    np.zeros_like(v_data), dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
            }
        )
        window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                       window_size=11, core_size=3)

        result0 = compute_dlmsf_patch_fd(
            eval_inputs=ds, baseline_inputs=baseline, window=window,
            center_lat=0.0, center_lon=120.0, d_hat=(1.0, 0.0),
            target_time_idx=0, patch_size=3, stride=2,
        )
        result1 = compute_dlmsf_patch_fd(
            eval_inputs=ds, baseline_inputs=baseline, window=window,
            center_lat=0.0, center_lon=120.0, d_hat=(1.0, 0.0),
            target_time_idx=1, patch_size=3, stride=2,
        )

        assert np.isfinite(result0.J_phys_baseline), (
            f"result0.J_phys_baseline should be finite, got {result0.J_phys_baseline}"
        )
        assert np.isfinite(result1.J_phys_baseline), (
            f"result1.J_phys_baseline should be finite, got {result1.J_phys_baseline}"
        )
        assert result0.J_phys_baseline != result1.J_phys_baseline, (
            "J_phys_baseline 应随 target_time_idx 变化；若相同说明 time_idx 仍被写死"
        )


class TestEnvMaskInsufficientRaisesError:
    """annulus 点数不足时必须 raise ValueError，不能静默退化。"""

    def test_raises_when_annulus_outer_km_too_small(self):
        """把 annulus_outer_km 设为极小值，使得 env_mask 没有足够点。"""
        import pytest
        from physics.dlmsf_patch_fd.dlmsf_sensitivity import compute_dlmsf_925_300

        lev = np.asarray([925, 500, 300], dtype=np.float32)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(115, 125, 11)

        # 3D numpy arrays: shape (n_levels, nlat, nlon)
        u_levels = np.ones((3, 11, 11), dtype=np.float32)
        v_levels = np.zeros((3, 11, 11), dtype=np.float32)

        with pytest.raises(ValueError, match="Steering annulus"):
            compute_dlmsf_925_300(
                u_levels=u_levels,
                v_levels=v_levels,
                levels_hpa=lev,
                lat_vals=lat,
                lon_vals=lon,
                center_lat=0.0,
                center_lon=120.0,
                core_radius_deg=0.0,
                annulus_inner_km=0.0,
                annulus_outer_km=1.0,   # 极小值：几乎没有 annulus 点
                min_env_points=10,
            )
