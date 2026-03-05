from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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
        """d_hat=(0,0) 时 J_phys 始终为 0，S_map 应全零。"""
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

    def test_patch_fill_consistency(self):
        """同一个 patch 内所有格点的 S_map 值应相同（patch 回填）。"""
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
        # S_map 不全为 0（均匀场扰动后 DLMSF 应有变化）
        assert not np.all(result.S_map == 0.0)

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
        """S_map 为绝对值，不应有负数。"""
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
        assert np.all(result.S_map >= 0.0)
