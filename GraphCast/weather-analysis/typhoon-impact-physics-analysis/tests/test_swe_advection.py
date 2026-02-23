# -*- coding: utf-8 -*-
"""Unit tests for SWE background advection terms and DLM wind extraction."""

import unittest
import numpy as np
import jax.numpy as jnp
import xarray

from physics.swe_model import SWEPhysicsConfig, _swe_tendency
from physics.sensitivity import compute_dlm_background_wind


class TestSWETendencyAdvection(unittest.TestCase):
    """Tests that _swe_tendency includes background advection when U_bar != 0."""

    def _cfg(self, U_bar=0.0, V_bar=0.0):
        return SWEPhysicsConfig(
            H=5500.0, f0=1e-4, g=9.81,
            dx=50000.0, dy=50000.0, dt=300.0,
            n_lat=6, n_lon=6,
            U_bar=U_bar, V_bar=V_bar,
        )

    def test_zero_background_wind_unchanged(self):
        """With U_bar=V_bar=0, tendency should match original formula."""
        cfg0 = self._cfg(U_bar=0.0, V_bar=0.0)
        n = 6
        h = jnp.ones((n, n)) * 10.0
        u = jnp.zeros((n, n))
        v = jnp.zeros((n, n))
        dh, du, dv = _swe_tendency(h, u, v, cfg0)
        # All zero: uniform field, no gradients
        np.testing.assert_allclose(dh, 0.0, atol=1e-10)
        np.testing.assert_allclose(du, 0.0, atol=1e-10)
        np.testing.assert_allclose(dv, 0.0, atol=1e-10)

    def test_nonzero_ubar_changes_tendency(self):
        """With U_bar != 0 and a non-uniform h, dh/dt must differ from U_bar=0 case."""
        cfg_no_adv = self._cfg(U_bar=0.0)
        cfg_adv    = self._cfg(U_bar=10.0)
        n = 6
        # Linear ramp in x so dh/dx != 0
        h = jnp.tile(jnp.arange(n, dtype=float), (n, 1))
        u = jnp.zeros((n, n))
        v = jnp.zeros((n, n))
        dh_no, _, _ = _swe_tendency(h, u, v, cfg_no_adv)
        dh_ad, _, _ = _swe_tendency(h, u, v, cfg_adv)
        self.assertFalse(
            jnp.allclose(dh_no, dh_ad),
            msg="Non-zero U_bar must change dh/dt for a non-uniform h field",
        )


class TestComputeDLMBackgroundWind(unittest.TestCase):
    """Tests for compute_dlm_background_wind."""

    def _make_eval_inputs(self, u_val=5.0, v_val=-3.0, n_lat=41, n_lon=41):
        """Build a minimal xarray Dataset with uniform u/v at two pressure levels."""
        lat = np.linspace(-42, -2, n_lat)
        lon = np.linspace(138, 178, n_lon)
        levels = np.array([850, 500, 300], dtype=float)
        data = np.ones((1, 2, len(levels), n_lat, n_lon), dtype=np.float32)
        u_data = data * u_val
        v_data = data * v_val
        coords = {"batch": [0], "time": [0, 1], "level": levels, "lat": lat, "lon": lon}
        return xarray.Dataset({
            "u_component_of_wind": xarray.DataArray(u_data, dims=["batch","time","level","lat","lon"], coords=coords),
            "v_component_of_wind": xarray.DataArray(v_data, dims=["batch","time","level","lat","lon"], coords=coords),
        })

    def test_uniform_field_returns_input_wind(self):
        """Uniform u/v field → DLM result equals that uniform value."""
        ds = self._make_eval_inputs(u_val=5.0, v_val=-3.0)
        U_bar, V_bar = compute_dlm_background_wind(
            ds, center_lat=-22.0, center_lon=158.0,
            inner_radius_km=300.0, outer_radius_km=800.0,
            p_bot_hpa=850.0, p_top_hpa=300.0,
        )
        self.assertAlmostEqual(U_bar, 5.0, places=1)
        self.assertAlmostEqual(V_bar, -3.0, places=1)

    def test_returns_floats(self):
        """Return types must be Python floats."""
        ds = self._make_eval_inputs()
        U_bar, V_bar = compute_dlm_background_wind(
            ds, center_lat=-22.0, center_lon=158.0,
        )
        self.assertIsInstance(U_bar, float)
        self.assertIsInstance(V_bar, float)


if __name__ == "__main__":
    unittest.main()
