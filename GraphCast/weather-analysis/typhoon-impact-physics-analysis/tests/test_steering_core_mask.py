from __future__ import annotations

import inspect
import unittest

import numpy as np

import config as cfg
from physics import sensitivity


class SteeringCoreMaskTests(unittest.TestCase):
    def test_config_exposes_default_core_radius(self) -> None:
        self.assertTrue(hasattr(cfg, "SWE_CORE_RADIUS_DEG"))
        self.assertAlmostEqual(float(cfg.SWE_CORE_RADIUS_DEG), 3.0, places=6)

    def test_compute_sensitivity_exposes_core_radius_parameter(self) -> None:
        sig = inspect.signature(sensitivity.compute_sensitivity_jax)
        self.assertIn("core_radius_deg", sig.parameters)

    def test_environmental_steering_excludes_core_points(self) -> None:
        lat = np.linspace(-4.0, 4.0, 9, dtype=np.float32)
        lon = np.linspace(-4.0, 4.0, 9, dtype=np.float32)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

        u = np.full_like(lat2d, 6.0, dtype=np.float32)
        v = np.full_like(lat2d, -3.0, dtype=np.float32)

        r = np.sqrt(lat2d**2 + lon2d**2)
        core = r <= 2.0
        u[core] += -10.0
        v[core] += 5.0

        full_u = float(np.mean(u))
        full_v = float(np.mean(v))

        env_u, env_v, n_env, n_total, masked_ratio = sensitivity.compute_environmental_steering_flow(
            u0=u,
            v0=v,
            lat_vals=lat,
            lon_vals=lon,
            center_lat=0.0,
            center_lon=0.0,
            core_radius_deg=2.0,
        )

        self.assertEqual(n_total, u.size)
        self.assertGreater(n_env, 0)
        self.assertLess(n_env, n_total)
        self.assertGreater(masked_ratio, 0.0)
        self.assertAlmostEqual(env_u, 6.0, places=6)
        self.assertAlmostEqual(env_v, -3.0, places=6)
        self.assertGreater(abs(full_u - 6.0), 0.5)
        self.assertGreater(abs(full_v + 3.0), 0.25)


if __name__ == "__main__":
    unittest.main()
