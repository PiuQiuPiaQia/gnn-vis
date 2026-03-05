from __future__ import annotations
import math
import pytest
from physics.dlmsf_patch_fd.dlmsf_sensitivity import compute_d_hat


class TestComputeDHat:

    def test_northward_movement(self):
        """台风向正北移动 → d_u≈0, d_v>0, 单位向量。"""
        d_u, d_v = compute_d_hat(0.0, 0.0, 5.0, 0.0)
        assert abs(d_u) < 1e-6
        assert d_v > 0.0
        assert abs(math.hypot(d_u, d_v) - 1.0) < 1e-6

    def test_eastward_movement(self):
        """台风向正东移动 → d_u>0, d_v≈0, 单位向量。"""
        d_u, d_v = compute_d_hat(0.0, 0.0, 0.0, 5.0)
        assert d_u > 0.0
        assert abs(d_v) < 1e-6
        assert abs(math.hypot(d_u, d_v) - 1.0) < 1e-6

    def test_diagonal_movement_is_normalized(self):
        """斜向移动时结果为单位向量。"""
        d_u, d_v = compute_d_hat(0.0, 0.0, 3.0, 4.0)
        assert abs(math.hypot(d_u, d_v) - 1.0) < 1e-6

    def test_stationary_cyclone_returns_zero(self):
        """台风静止（相同起终点）→ (0.0, 0.0)，不报错。"""
        d_u, d_v = compute_d_hat(10.0, 120.0, 10.0, 120.0)
        assert d_u == 0.0
        assert d_v == 0.0

    def test_southward_movement(self):
        """台风向正南移动 → d_v < 0。"""
        d_u, d_v = compute_d_hat(5.0, 0.0, 0.0, 0.0)
        assert d_v < 0.0
