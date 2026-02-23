# -*- coding: utf-8 -*-
"""Unit tests for geostrophic wind mapping from height field.

Tests for the `geostrophic_wind_from_height` function to be implemented
in physics/swe_model.py.

Geostrophic balance relation:
    delta_u = -(g / f0) * ∂(delta_h) / ∂y
    delta_v =  (g / f0) * ∂(delta_h) / ∂x
"""

import unittest

import jax.numpy as jnp
import numpy as np

from physics.swe_model import SWEPhysicsConfig, geostrophic_wind_from_height


class TestGeostrophicMapping(unittest.TestCase):
    """Test suite for geostrophic_wind_from_height function."""

    def _make_minimal_config(self, f0: float = 1e-4) -> SWEPhysicsConfig:
        """Create a minimal SWEPhysicsConfig for testing.

        Args:
            f0: Coriolis parameter (s⁻¹). Default 1e-4 is typical mid-latitude value.

        Returns:
            SWEPhysicsConfig with minimal grid (4x4) and typical physics params.
        """
        return SWEPhysicsConfig(
            H=5500.0,       # Reference depth (m)
            f0=f0,          # Coriolis parameter (s⁻¹)
            g=9.81,         # Gravity (m s⁻²)
            dx=50000.0,     # 50 km grid spacing
            dy=50000.0,     # 50 km grid spacing
            dt=300.0,       # Time step (s)
            n_lat=4,        # Small grid for tests
            n_lon=4,
        )

    def test_constant_field_gives_zero_wind(self):
        """A spatially-constant h field should produce delta_u=0, delta_v=0.

        Rationale: centered difference of a constant field is zero everywhere.
        """
        cfg = self._make_minimal_config(f0=1e-4)

        # Constant height perturbation field
        delta_h = jnp.ones((cfg.n_lat, cfg.n_lon)) * 10.0  # 10 m everywhere

        delta_u, delta_v = geostrophic_wind_from_height(delta_h, cfg)

        # Both wind components should be zero (within numerical precision)
        np.testing.assert_allclose(delta_u, 0.0, atol=1e-10,
                                   err_msg="Constant field should give zero delta_u")
        np.testing.assert_allclose(delta_v, 0.0, atol=1e-10,
                                   err_msg="Constant field should give zero delta_v")

    def test_sinusoidal_field_gives_finite_wind(self):
        """A sinusoidal delta_h should give finite, nonzero delta_v with correct sign.

        Geostrophic relation: delta_v = (g / f0) * d(delta_h) / dx

        For delta_h = sin(2π * x / Lx), we have:
            d(delta_h) / dx = (2π / Lx) * cos(2π * x / Lx)

        At x=0 (first column), cos(0)=1, so gradient is positive.
        With g>0, f0>0: delta_v should be positive at x=0.
        """
        cfg = self._make_minimal_config(f0=1e-4)

        # Create sinusoidal height perturbation: sin(2π * x / Lx)
        Lx = cfg.dx * cfg.n_lon
        x = jnp.arange(cfg.n_lon) * cfg.dx
        x_2d = jnp.broadcast_to(x, (cfg.n_lat, cfg.n_lon))

        # delta_h varies sinusoidally in x, constant in y
        delta_h = jnp.sin(2.0 * np.pi * x_2d / Lx)

        delta_u, delta_v = geostrophic_wind_from_height(delta_h, cfg)

        # delta_v should be finite and nonzero
        self.assertFalse(jnp.allclose(delta_v, 0.0),
                         msg="Sinusoidal field should give nonzero delta_v")

        # Check sign at x near 0: gradient is positive, so delta_v should be positive
        # (g > 0, f0 > 0, and d(delta_h)/dx > 0 near x=0)
        expected_dh_dx_at_0 = (2.0 * np.pi / Lx) * jnp.cos(0.0)  # positive
        expected_delta_v_at_0 = (cfg.g / cfg.f0) * expected_dh_dx_at_0

        # Check that the computed delta_v at column 0 has the correct sign
        # (exact match is tricky due to centered diff discretization)
        self.assertGreater(float(delta_v[0, 0]), 0.0,
                           msg="delta_v should be positive where d(delta_h)/dx > 0")

        # Verify delta_u is approximately zero (field is constant in y)
        np.testing.assert_allclose(delta_u, 0.0, atol=1e-8,
                                   err_msg="delta_u should be ~0 for x-only variation")

    def test_small_f0_raises_valueerror(self):
        """Calling with |cfg.f0| < f0_floor should raise ValueError.

        This prevents division by (near-)zero in the geostrophic relation.
        """
        cfg = self._make_minimal_config(f0=1e-8)  # Very small f0

        delta_h = jnp.ones((cfg.n_lat, cfg.n_lon))

        # Default f0_floor should be 1e-5
        with self.assertRaises(ValueError) as context:
            geostrophic_wind_from_height(delta_h, cfg, f0_floor=1e-5)

        self.assertIn("f0", str(context.exception).lower(),
                      msg="Error message should mention f0")


if __name__ == "__main__":
    unittest.main()
