"""Tests for compute_ig_phys_dlmsf_along (E1 analytical IG).

The function computes the analytical Integrated Gradients attribution for the
linear DLMSF scalar J_along = Σ_l w_l * Σ_cell env_mask_cell * (u_cell * d̂_u + v_cell * d̂_v),
where the baseline is zero. Because J is linear in the inputs, the IG equals
the input × gradient everywhere, i.e. IG_u(cell, l) = u_cell * w_l * env_mask_cell * d̂_u.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from physics.dlmsf_patch_fd.ig_phys import compute_ig_phys_dlmsf_along


class TestComputeIgPhysDlmsfAlong:
    """Unit tests for compute_ig_phys_dlmsf_along."""

    def _minimal_inputs(self):
        """Return a minimal valid set of inputs: 1 level, 2×2 lat/lon grid."""
        levels_hpa = np.array([500.0])
        u = np.ones((1, 2, 2), dtype=np.float64)  # (levels, lat, lon)
        v = np.zeros((1, 2, 2), dtype=np.float64)
        lat_vals = np.array([10.0, 11.0])
        lon_vals = np.array([120.0, 121.0])
        center_lat = 10.5
        center_lon = 120.5
        # along direction = due east (d̂ = (1, 0) in lon,lat)
        d_hat = (1.0, 0.0)  # (d_u, d_v) where d_u is lon-component, d_v is lat-component
        env_mask = np.ones((2, 2), dtype=bool)
        return {
            "u": u,
            "v": v,
            "levels_hpa": levels_hpa,
            "lat_vals": lat_vals,
            "lon_vals": lon_vals,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "d_hat": d_hat,
            "env_mask": env_mask,
        }

    def test_returns_dict_with_required_keys(self):
        inp = self._minimal_inputs()
        result = compute_ig_phys_dlmsf_along(**inp)
        assert "ig_u_latlon" in result
        assert "ig_v_latlon" in result
        assert "j_along" in result

    def test_ig_shape_matches_latlon_grid(self):
        inp = self._minimal_inputs()
        result = compute_ig_phys_dlmsf_along(**inp)
        assert result["ig_u_latlon"].shape == (2, 2)
        assert result["ig_v_latlon"].shape == (2, 2)

    def test_due_east_direction_ig_u_equals_u_ig_v_zero(self):
        """When d̂ = (1, 0), IG_u = u / N_env (single level, weight=1) and IG_v = 0."""
        inp = self._minimal_inputs()
        # 2×2 full env → N_env=4; u=1, d̂_u=1, w=1 → IG_u = 1/4 per cell
        result = compute_ig_phys_dlmsf_along(**inp)
        np.testing.assert_allclose(result["ig_u_latlon"], np.full((2, 2), 0.25))
        np.testing.assert_allclose(result["ig_v_latlon"], np.zeros((2, 2)))

    def test_due_north_direction_ig_v_equals_v_ig_u_zero(self):
        """When d̂ = (0, 1), IG_v = v / N_env and IG_u = 0."""
        inp = self._minimal_inputs()
        inp["u"] = np.zeros((1, 2, 2), dtype=np.float64)
        inp["v"] = np.ones((1, 2, 2), dtype=np.float64)
        inp["d_hat"] = (0.0, 1.0)
        result = compute_ig_phys_dlmsf_along(**inp)
        np.testing.assert_allclose(result["ig_u_latlon"], np.zeros((2, 2)))
        np.testing.assert_allclose(result["ig_v_latlon"], np.full((2, 2), 0.25))

    def test_env_mask_zeros_out_masked_cells(self):
        inp = self._minimal_inputs()
        # Only 2 cells active out of 4 → N_env=2
        inp["env_mask"] = np.array([[True, False], [False, True]])
        result = compute_ig_phys_dlmsf_along(**inp)
        # Masked cells (False) should have zero IG
        assert result["ig_u_latlon"][0, 1] == pytest.approx(0.0)
        assert result["ig_u_latlon"][1, 0] == pytest.approx(0.0)
        # Active cells: u=1, w=1, d̂_u=1, N_env=2 → IG_u = 1/2
        assert result["ig_u_latlon"][0, 0] == pytest.approx(0.5)
        assert result["ig_u_latlon"][1, 1] == pytest.approx(0.5)

    def test_two_equal_levels_weight_sums_to_one(self):
        """Two equal-spaced levels: each level gets weight ~0.5 via trapezoidal rule."""
        levels_hpa = np.array([925.0, 300.0])
        u = np.ones((2, 1, 1), dtype=np.float64)
        v = np.zeros((2, 1, 1), dtype=np.float64)
        env_mask = np.ones((1, 1), dtype=bool)
        d_hat = (1.0, 0.0)
        result = compute_ig_phys_dlmsf_along(
            u=u,
            v=v,
            levels_hpa=levels_hpa,
            lat_vals=np.array([10.0]),
            lon_vals=np.array([120.0]),
            center_lat=10.0,
            center_lon=120.0,
            d_hat=d_hat,
            env_mask=env_mask,
        )
        # IG_u = Σ_l w_l * u_l * d̂_u = 1.0 (weights sum to 1, u=1, d̂_u=1)
        assert result["ig_u_latlon"][0, 0] == pytest.approx(1.0)

    def test_j_along_is_scalar_float(self):
        inp = self._minimal_inputs()
        result = compute_ig_phys_dlmsf_along(**inp)
        assert isinstance(result["j_along"], float)

    def test_j_along_equals_u_dot_d_hat_when_single_level_full_env(self):
        """With single level weight=1, J_along = mean env-cell u*d̂_u + v*d̂_v."""
        inp = self._minimal_inputs()
        # u=1, v=0, d̂=(1,0), all cells in env → J = mean(1*1+0*0) = 1.0
        result = compute_ig_phys_dlmsf_along(**inp)
        assert result["j_along"] == pytest.approx(1.0)

    def test_diagonal_d_hat_splits_contribution(self):
        """d̂ = (1/√2, 1/√2) with u=v=1 → each contributes equally per cell."""
        d_u = 1.0 / math.sqrt(2)
        d_v = 1.0 / math.sqrt(2)
        inp = self._minimal_inputs()
        inp["u"] = np.ones((1, 2, 2), dtype=np.float64)
        inp["v"] = np.ones((1, 2, 2), dtype=np.float64)
        inp["d_hat"] = (d_u, d_v)
        result = compute_ig_phys_dlmsf_along(**inp)
        # N_env=4, w=1 → IG_u = d_u / 4 per cell
        np.testing.assert_allclose(result["ig_u_latlon"], np.full((2, 2), d_u / 4.0))
        np.testing.assert_allclose(result["ig_v_latlon"], np.full((2, 2), d_v / 4.0))
