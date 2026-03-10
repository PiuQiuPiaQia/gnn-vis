"""Tests for _compute_physical_aopc (E4: analytical AOPC on physical scalar).

_compute_physical_aopc uses pre-computed DLMSF per-patch ΔJ values to produce
AOPC curves analytically (no forward passes needed).  The cumulative delta at
step k equals the cumulative sum of per-patch ΔJ values in the chosen ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import _compute_physical_aopc


class TestComputePhysicalAopc:

    def test_returns_required_keys(self):
        scores = np.array([3.0, -1.0, 2.0, -4.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        for key in ("high_dlmsf_cumulative", "ig_cumulative", "random_mean_cumulative",
                    "aopc_dlmsf", "aopc_ig", "aopc_random_mean", "n_patches"):
            assert key in result, f"Missing key: {key}"

    def test_n_patches_equals_input_length(self):
        scores = np.array([1.0, 2.0, 3.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        assert result["n_patches"] == 3

    def test_high_dlmsf_cumulative_length_equals_n_patches(self):
        scores = np.array([3.0, -1.0, 2.0, -4.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        assert len(result["high_dlmsf_cumulative"]) == 4
        assert len(result["ig_cumulative"]) == 4

    def test_high_dlmsf_cumulative_first_step_is_largest_abs_score(self):
        """First step in DLMSF-topK order picks the patch with highest |ΔJ|."""
        scores = np.array([3.0, -1.0, 2.0, -4.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        # Abs values: [3,1,2,4] → top-1 is index 3 (value=-4)
        # Cumulative delta at step 1 = dlmsf_signed[3] = -4.0
        assert result["high_dlmsf_cumulative"][0] == pytest.approx(-4.0)

    def test_high_dlmsf_cumulative_is_monotone_in_abs_value(self):
        """Each step adds the next-largest |ΔJ| value."""
        scores = np.array([3.0, -1.0, 2.0, -4.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        cumul = result["high_dlmsf_cumulative"]
        abs_diffs = [abs(cumul[0])]
        for i in range(1, len(cumul)):
            abs_diffs.append(abs(cumul[i] - cumul[i - 1]))
        # Each step's increment must be ≤ the previous one (largest-first)
        for i in range(1, len(abs_diffs)):
            assert abs_diffs[i] <= abs_diffs[i - 1] + 1e-9

    def test_aopc_dlmsf_equals_mean_of_cumulative(self):
        scores = np.array([4.0, -2.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        # DLMSF order: abs=[4,2] → idx [0,1]
        # cumul[0]=4.0, cumul[1]=4+(-2)=2.0 → aopc = mean(4, 2) = 3.0
        assert result["aopc_dlmsf"] == pytest.approx(3.0)

    def test_ig_ordering_uses_ig_abs_scores(self):
        """IG ordering follows ig_abs_scores, not |dlmsf_signed|."""
        dlmsf_scores = np.array([1.0, 5.0])
        ig_scores = np.array([10.0, 1.0])  # ig-top1 is index 0
        result = _compute_physical_aopc(
            dlmsf_signed_scores=dlmsf_scores,
            ig_abs_scores=ig_scores,
        )
        # IG order: idx [0, 1] → cumul[0] = dlmsf[0] = 1.0
        assert result["ig_cumulative"][0] == pytest.approx(1.0)

    def test_random_mean_cumulative_has_correct_length(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
            random_repeats=5,
            seed=0,
        )
        assert len(result["random_mean_cumulative"]) == 4

    def test_deterministic_with_same_seed(self):
        scores = np.array([1.0, -3.0, 2.0, -0.5])
        r1 = _compute_physical_aopc(dlmsf_signed_scores=scores, ig_abs_scores=np.abs(scores), seed=7)
        r2 = _compute_physical_aopc(dlmsf_signed_scores=scores, ig_abs_scores=np.abs(scores), seed=7)
        np.testing.assert_allclose(r1["random_mean_cumulative"], r2["random_mean_cumulative"])

    def test_all_same_score_gives_constant_cumulative(self):
        """When all patches have the same value, cumulative is linear."""
        scores = np.array([2.0, 2.0, 2.0])
        result = _compute_physical_aopc(
            dlmsf_signed_scores=scores,
            ig_abs_scores=np.abs(scores),
        )
        expected = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(result["high_dlmsf_cumulative"], expected)
