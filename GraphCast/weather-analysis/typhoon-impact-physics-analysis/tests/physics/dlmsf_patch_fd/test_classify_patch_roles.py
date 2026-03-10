"""Tests for classify_patch_roles (top-K role classification).

classify_patch_roles assigns each patch a role code based on whether it appears
in the top-k IG set, top-k DLMSF set, or both:
  0 = neither
  1 = model_only  (top-k IG only)
  2 = physics_only (top-k DLMSF only)
  3 = consensus (top-k of both)
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import classify_patch_roles


class TestClassifyPatchRoles:

    def test_consensus_patches_get_role_3(self):
        ig = np.array([5.0, 4.0, 1.0, 2.0])
        dlmsf = np.array([5.0, 4.0, 1.0, 2.0])  # same top-2: indices {0,1}
        roles = classify_patch_roles(ig, dlmsf, k=2)
        assert roles[0] == 3
        assert roles[1] == 3

    def test_model_only_gets_role_1(self):
        ig = np.array([5.0, 0.0, 0.0, 0.0])    # top-1: {0}
        dlmsf = np.array([0.0, 5.0, 0.0, 0.0])  # top-1: {1}
        roles = classify_patch_roles(ig, dlmsf, k=1)
        assert roles[0] == 1  # only in IG top-1
        assert roles[1] == 2  # only in DLMSF top-1

    def test_physics_only_gets_role_2(self):
        ig = np.array([1.0, 10.0, 1.0])
        dlmsf = np.array([10.0, 1.0, 1.0])
        roles = classify_patch_roles(ig, dlmsf, k=1)
        assert roles[0] == 2  # physics_only
        assert roles[1] == 1  # model_only

    def test_neither_gets_role_0(self):
        ig = np.array([5.0, 4.0, 1.0, 0.5])
        dlmsf = np.array([5.0, 4.0, 1.0, 0.5])
        roles = classify_patch_roles(ig, dlmsf, k=2)
        # Indices 2 and 3 are not in top-2 of either
        assert roles[2] == 0
        assert roles[3] == 0

    def test_output_length_matches_input(self):
        ig = np.array([3.0, 2.0, 1.0, 4.0, 5.0])
        dlmsf = np.array([1.0, 4.0, 2.0, 3.0, 5.0])
        roles = classify_patch_roles(ig, dlmsf, k=2)
        assert len(roles) == 5

    def test_all_roles_in_valid_set(self):
        ig = np.random.default_rng(0).random(10)
        dlmsf = np.random.default_rng(1).random(10)
        roles = classify_patch_roles(ig, dlmsf, k=3)
        assert set(roles.tolist()).issubset({0, 1, 2, 3})

    def test_nan_excluded_from_top_k(self):
        ig = np.array([float("nan"), 5.0, 3.0, 2.0])
        dlmsf = np.array([float("nan"), 5.0, 3.0, 2.0])
        roles = classify_patch_roles(ig, dlmsf, k=2)
        # NaN index excluded; top-2 are indices {1, 2}
        assert roles[0] == 0  # NaN → neither
        assert roles[1] == 3  # consensus
        assert roles[2] == 3  # consensus

    def test_k_larger_than_n_uses_all_patches(self):
        ig = np.array([1.0, 2.0])
        dlmsf = np.array([2.0, 1.0])
        # k=5 > n=2: all patches should be either 1, 2, or 3
        roles = classify_patch_roles(ig, dlmsf, k=5)
        assert all(r > 0 for r in roles)
