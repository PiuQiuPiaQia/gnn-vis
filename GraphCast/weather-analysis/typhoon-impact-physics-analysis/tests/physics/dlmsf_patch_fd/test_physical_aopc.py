"""Tests for _compute_physical_aopc (E4: union-mask AOPC on physical scalar J_along).

_compute_physical_aopc uses the analytical IG cell map to compute cumulative
delta curves. At each step the delta = sum(ig_phys_cell_map[union_mask]).
This avoids double-counting overlapping patches.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import _compute_physical_aopc


class _FakePatch:
    def __init__(self, mask: np.ndarray):
        self.mask = mask


def _single_cell_patches(n: int, shape=(4, 4)):
    """n non-overlapping single-cell patches at positions (0,0),(0,1),..."""
    patches = []
    for i in range(n):
        m = np.zeros(shape, dtype=bool)
        row, col = divmod(i, shape[1])
        if row < shape[0]:
            m[row, col] = True
        patches.append(_FakePatch(m))
    return patches


class TestComputePhysicalAopc:

    def test_returns_required_keys(self):
        shape = (2, 2)
        cell_map = np.array([[1.0, -0.5], [0.3, -0.8]])
        patches = _single_cell_patches(4, shape)
        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=np.abs([1.0, -0.5, 0.3, -0.8]),
            dlmsf_signed_patch_scores=np.array([1.0, -0.5, 0.3, -0.8]),
            patches=patches,
        )
        for key in ("high_dlmsf_cumulative", "ig_cumulative", "random_mean_cumulative",
                    "aopc_dlmsf", "aopc_ig", "aopc_random_mean", "n_patches"):
            assert key in result, f"Missing key: {key}"

    def test_n_patches(self):
        shape = (1, 3)
        cell_map = np.zeros(shape)
        patches = _single_cell_patches(3, shape)
        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=np.ones(3),
            dlmsf_signed_patch_scores=np.ones(3),
            patches=patches,
        )
        assert result["n_patches"] == 3

    def test_non_overlapping_cumulative_equals_running_sum_of_cells(self):
        """For non-overlapping patches, delta at step k = sum of k highest-priority cell values."""
        shape = (2, 2)
        # cell contributions: patch 0=[0,0]=3, patch 1=[0,1]=-1, patch 2=[1,0]=2, patch 3=[1,1]=-4
        cell_map = np.array([[3.0, -1.0], [2.0, -4.0]])
        patches = _single_cell_patches(4, shape)
        dlmsf_scores = np.array([3.0, -1.0, 2.0, -4.0])

        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=np.abs(dlmsf_scores),
            dlmsf_signed_patch_scores=dlmsf_scores,
            patches=patches,
        )
        # DLMSF order by |score| desc: |[3,-1,2,-4]| → idx [3,0,2,1]
        # step 1: union=[1,1]  → sum = -4.0
        # step 2: union=[0,0]+[1,1] → sum = -4+3 = -1.0
        # step 3: union=[1,0]+[0,0]+[1,1] → sum = -4+3+2 = 1.0
        # step 4: all → sum = -4+3+2+(-1) = 0.0
        expected = [-4.0, -1.0, 1.0, 0.0]
        np.testing.assert_allclose(result["high_dlmsf_cumulative"], expected, atol=1e-9)

    def test_overlapping_patches_no_double_counting(self):
        """Overlapping patches: union mask ensures cells counted once."""
        shape = (1, 2)
        cell_map = np.array([[5.0, 3.0]])
        # patch 0: covers [0,0] only; patch 1: covers [0,0] AND [0,1]
        m0 = np.array([[True, False]])
        m1 = np.array([[True, True]])
        patches = [_FakePatch(m0), _FakePatch(m1)]
        dlmsf_scores = np.array([1.0, 2.0])

        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=np.abs(dlmsf_scores),
            dlmsf_signed_patch_scores=dlmsf_scores,
            patches=patches,
        )
        # DLMSF order: |[1,2]| → idx [1,0]
        # step 1: union=m1 = [[True,True]] → sum = 5+3 = 8.0
        # step 2: union=m1|m0 = [[True,True]] (same, since m0 ⊆ m1) → sum = 8.0
        np.testing.assert_allclose(result["high_dlmsf_cumulative"], [8.0, 8.0], atol=1e-9)

    def test_aopc_dlmsf_equals_mean_of_cumulative(self):
        shape = (1, 2)
        cell_map = np.array([[4.0, -2.0]])
        patches = _single_cell_patches(2, shape)
        dlmsf_scores = np.array([4.0, -2.0])
        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=np.abs(dlmsf_scores),
            dlmsf_signed_patch_scores=dlmsf_scores,
            patches=patches,
        )
        # DLMSF order: |[4,-2]| → idx [0,1]
        # cumul = [4.0, 4+(-2)=2.0] → aopc_dlmsf = mean(4,2) = 3.0
        assert result["aopc_dlmsf"] == pytest.approx(3.0)

    def test_ig_ordering_uses_ig_abs_scores(self):
        """IG ordering follows ig_abs_patch_scores, not |dlmsf_signed|."""
        shape = (1, 2)
        cell_map = np.array([[1.0, 5.0]])
        patches = _single_cell_patches(2, shape)
        dlmsf_scores = np.array([1.0, 5.0])
        ig_scores = np.array([10.0, 1.0])   # IG top-1 = idx 0 (cell value 1.0)
        result = _compute_physical_aopc(
            ig_phys_cell_map=cell_map,
            ig_abs_patch_scores=ig_scores,
            dlmsf_signed_patch_scores=dlmsf_scores,
            patches=patches,
        )
        # IG order: idx [0,1] → step 1 delta = cell_map[0,0] = 1.0
        assert result["ig_cumulative"][0] == pytest.approx(1.0)

    def test_deterministic_with_same_seed(self):
        shape = (2, 2)
        cell_map = np.array([[1.0, -3.0], [2.0, -0.5]])
        patches = _single_cell_patches(4, shape)
        scores = np.array([1.0, -3.0, 2.0, -0.5])
        kwargs = dict(ig_phys_cell_map=cell_map, ig_abs_patch_scores=np.abs(scores),
                      dlmsf_signed_patch_scores=scores, patches=patches, seed=7)
        r1 = _compute_physical_aopc(**kwargs)
        r2 = _compute_physical_aopc(**kwargs)
        np.testing.assert_allclose(r1["random_mean_cumulative"], r2["random_mean_cumulative"])
