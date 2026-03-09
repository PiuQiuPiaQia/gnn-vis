# tests/physics/test_alignment_signed_vs_magnitude.py
# -*- coding: utf-8 -*-
"""Tests for signed-vs-magnitude preprocessing in alignment utilities.

TDD: These tests verify that:
- signed correlation/scatter path preserves negative values instead of forcing abs
- Top-K overlap / IoU path uses magnitude-style preprocessing
- plotting API still works with refactored helpers
"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace
from typing import Any, cast

import matplotlib
matplotlib.use("Agg")

from physics.swe.alignment import (
    AlignmentReport,
    GroupMetrics,
    compute_alignment_report,
    compute_spearman,
    compute_topk_iou,
    plot_alignment_scatter,
    plot_topk_iou_curves,
    plot_topk_overlap_maps,
)


def _make_signed_map(shape=(10, 12), seed=42) -> np.ndarray:
    """Create a map with both positive and negative values."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float64)
    # Ensure we have a mix of positive and negative values
    arr = arr * 2.0  # Spread values
    return arr


def _make_positive_map(shape=(10, 12), seed=42) -> np.ndarray:
    """Create a map with only positive values."""
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float64)


# =============================================================================
# Test: compute_spearman preserves signed values
# =============================================================================

class TestComputeSpearmanSignedPreprocessing:
    """Spearman correlation should use signed-preserving preprocessing."""

    def test_spearman_preserves_negative_values(self):
        """Signed values should be preserved, not abs'd, for correlation."""
        # Create maps with strong negative correlation
        swe_map = _make_signed_map(seed=0)
        # Create gnn_map that's negatively correlated (flip sign)
        gnn_map = -swe_map.copy() + np.random.default_rng(1).standard_normal(swe_map.shape) * 0.1
        
        # If values are preserved (not abs'd), we should get strong negative correlation
        rho, pval = compute_spearman(swe_map, gnn_map, patch_radius=1)
        
        # With signed preservation, rho should be strongly negative
        # With abs-forcing, rho would be near +1
        assert rho < 0, f"Expected negative rho for negatively correlated maps, got {rho}"
        assert rho < -0.5, f"Expected strongly negative rho (< -0.5), got {rho}"

    def test_spearman_with_patch_radius_zero(self):
        """Even with patch_radius=0, signed values should be preserved."""
        swe_map = _make_signed_map(seed=2)
        gnn_map = -swe_map.copy()  # Perfect negative correlation
        
        rho, pval = compute_spearman(swe_map, gnn_map, patch_radius=0)
        
        # Should be near -1 (perfect negative correlation)
        assert rho < -0.95, f"Expected rho near -1, got {rho}"

    def test_spearman_with_all_positive_values_still_works(self):
        """All-positive maps should work correctly."""
        swe_map = _make_positive_map(seed=0)
        gnn_map = swe_map * 2.0  # Positive correlation
        
        rho, pval = compute_spearman(swe_map, gnn_map, patch_radius=1)
        
        assert rho > 0.9, f"Expected strong positive correlation, got {rho}"


# =============================================================================
# Test: compute_topk_iou uses magnitude preprocessing
# =============================================================================

class TestComputeTopkIouMagnitudePreprocessing:
    """Top-K IoU should use magnitude (abs) preprocessing for hotspot detection."""

    def test_topk_iou_treats_negative_as_magnitude(self):
        """Large negative values should be treated as 'hot' for IoU."""
        # Create map where largest magnitude is negative
        swe_map = np.zeros((5, 5))
        swe_map[0, 0] = -10.0  # Large negative = large magnitude
        swe_map[1, 1] = 5.0
        
        gnn_map = np.zeros((5, 5))
        gnn_map[0, 0] = 9.0   # Matches location of large magnitude
        gnn_map[1, 1] = 4.0
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(2,), patch_radius=0)

        # Both top-2 should be exactly {(0,0), (1,1)} by magnitude.
        assert iou[2] == 1.0, f"Expected exact IoU=1.0 for matching magnitudes, got {iou[2]}"

    def test_topk_iou_ignores_sign_for_ranking(self):
        """Top-K ranking should be based on |value|, not raw value."""
        # Map with large negative that should be top by magnitude
        swe_map = np.array([[5.0, -10.0], [3.0, 2.0]])
        gnn_map = np.array([[4.0, 9.0], [2.0, 3.0]])  # Similar pattern
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(2,), patch_radius=0)

        # Top-2 by magnitude are exactly {(0,1), (0,0)} in both maps.
        assert iou[2] == 1.0, f"Expected exact IoU=1.0 for magnitude ranking, got {iou[2]}"

    def test_topk_iou_with_patch_aggregation(self):
        """IoU should work with patch aggregation (magnitude mode)."""
        swe_map = _make_signed_map(shape=(20, 24), seed=10)
        gnn_map = np.abs(swe_map) + np.random.default_rng(11).random((20, 24)) * 0.5
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(10, 20), patch_radius=2)
        
        # Should compute without error
        assert 10 in iou
        assert 20 in iou
        assert all(0.0 <= v <= 1.0 for v in iou.values())


# =============================================================================
# Test: _topk_overlap_code uses magnitude
# =============================================================================

class TestTopkOverlapMagnitudePreprocessing:
    """Top-K overlap maps should use magnitude preprocessing."""

    def test_overlap_detects_large_negative_as_hotspot(self, tmp_path):
        """Large negative values should appear in overlap map as hotspots."""
        lat = np.linspace(20.0, 30.0, 10)
        lon = np.linspace(120.0, 132.0, 12)
        
        # Create maps where top hotspot has large negative value
        swe_map = np.zeros((10, 12))
        swe_map[2, 3] = -100.0  # Large negative = hotspot by magnitude
        
        gnn_map = np.zeros((10, 12))
        gnn_map[2, 3] = 95.0    # Same location, positive
        
        gnn_ig_maps = {"z_500": gnn_map}
        pairs = [("h", swe_map, "z_500")]
        
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="test",
            patch_radius=0, topk_overlap_k=1,
        )
        
        # Should create a file without error
        files = list(tmp_path.glob("test_overlap_h_*.png"))
        assert len(files) == 1


# =============================================================================
# Test: scatter plot uses signed preprocessing
# =============================================================================

class TestScatterPlotSignedPreprocessing:
    """Scatter plots should show signed values, not abs'd."""

    def test_scatter_preserves_signed_values(self, tmp_path):
        """Scatter plot should show both positive and negative ranges."""
        swe_map = _make_signed_map(seed=20)
        gnn_map = _make_signed_map(seed=21)
        
        gnn_ig_maps = {"z_500": gnn_map, "uv_500": gnn_map}
        pairs = [
            ("h", swe_map, "z_500", "SWE", "GNN"),
            ("uv", swe_map, "uv_500", "SWE", "GNN"),
        ]
        report = AlignmentReport(0, 6, 1, "mean", 3.0)
        report.groups.append(GroupMetrics("h", 0.3, 0.1, {50: 0.2}, 100))
        report.groups.append(GroupMetrics("uv", 0.4, 0.05, {50: 0.3}, 100))
        
        plot_alignment_scatter(
            pairs, gnn_ig_maps, report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="test",
            patch_radius=0,
        )
        
        assert (tmp_path / "test_scatter_t0.png").exists()


# =============================================================================
# Test: IoU curves use magnitude preprocessing
# =============================================================================

class TestIouCurvesMagnitudePreprocessing:
    """IoU curve computation should use magnitude preprocessing."""

    def test_iou_curves_with_signed_maps(self, tmp_path):
        """IoU curves should handle signed maps correctly (magnitude mode)."""
        swe_map = _make_signed_map(seed=30)
        gnn_map = _make_signed_map(seed=31)
        
        gnn_ig_maps = {"z_500": gnn_map}
        pairs = [("h", swe_map, "z_500")]
        
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="test",
            k_values=(5, 10, 20),
            patch_radius=1,
        )
        
        assert (tmp_path / "test_iou_t0.png").exists()


# =============================================================================
# Test: _topk_overlap_code consistency with compute_topk_iou
# =============================================================================

class TestTopkOverlapConsistency:
    """Top-K overlap must use jointly-finite population like compute_topk_iou."""

    def test_overlap_ignores_asymmetric_nan_cells(self):
        """Cells that are NaN in only one map must be excluded from overlap."""
        from physics.swe.alignment import _topk_overlap_code
        
        # Create maps with asymmetric NaNs
        swe_map = np.array([
            [10.0, 9.0, np.nan],
            [8.0, 7.0, 6.0],
            [5.0, np.nan, 4.0],
        ])
        
        gnn_map = np.array([
            [np.nan, 9.0, 8.0],
            [7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0],
        ])
        
        # Use magnitude preprocessing (as plotting function does)
        from physics.swe.alignment import _patch_magnitude
        swe_mag = _patch_magnitude(swe_map, radius=0, agg="mean")
        gnn_mag = _patch_magnitude(gnn_map, radius=0, agg="mean")
        
        # Get overlap code
        code, actual_k = _topk_overlap_code(swe_mag, gnn_mag, k=5)
        
        # Jointly-finite cells: (0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,2)
        # Asymmetric NaNs: (0,2), (2,1) - these should be excluded
        # Total jointly-finite = 7 cells
        
        # Compute what IoU sees for comparison
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(5,), patch_radius=0)
        
        # Count how many cells are marked as top-k in overlap code
        # (codes 1, 2, 3 indicate top-k in at least one map)
        nonzero_codes = np.sum(code > 0)
        
        # If asymmetric NaNs are properly excluded:
        # - compute_topk_iou uses 7 jointly-finite cells
        # - _topk_overlap_code should also consider only jointly-finite cells
        # - With k=5, each map selects top 5 from its finite cells
        # - But overlap should only consider jointly-finite population
        
        # The key invariant: cells with asymmetric NaNs should have code 0
        # (not top-k in either map's valid population)
        # (0,2) is NaN in swe -> code should be 0
        # (2,1) is NaN in gnn -> code should be 0
        assert code[0, 2] == 0, "Cell (0,2) with SWE=NaN should not be marked as top-k"
        assert code[2, 1] == 0, "Cell (2,1) with GNN=NaN should not be marked as top-k"

    def test_overlap_matches_iou_population(self):
        """Overlap code must use same population as compute_topk_iou."""
        from physics.swe.alignment import _topk_overlap_code, _patch_magnitude
        
        # Maps where asymmetric NaNs affect which cells are top-k
        swe_map = np.array([
            [100.0, 90.0, np.nan],
            [80.0, 70.0, 60.0],
            [50.0, np.nan, 40.0],
        ])
        
        gnn_map = np.array([
            [np.nan, 95.0, 85.0],
            [75.0, 65.0, 55.0],
            [45.0, 35.0, 25.0],
        ])
        
        # Compute IoU
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(3,), patch_radius=0)
        
        # Get overlap with same preprocessing
        swe_mag = _patch_magnitude(swe_map, radius=0, agg="mean")
        gnn_mag = _patch_magnitude(gnn_map, radius=0, agg="mean")
        code, actual_k = _topk_overlap_code(swe_mag, gnn_mag, k=3)
        
        # Jointly-finite: (0,0), (0,1), (1,0), (1,1), (1,2), (2,0), (2,2) = 7 cells
        # With k=3, each map picks top 3 from jointly-finite
        
        # Count overlap (code=3) cells
        overlap_count = np.sum(code == 3)
        
        # Compute expected overlap from IoU formula
        # IoU = intersection / union
        # With actual_k=3 and IoU value, we can verify consistency
        assert actual_k <= 7, f"actual_k should be <= jointly-finite count (7), got {actual_k}"
        
        # The overlap count should be derivable from IoU
        # If IoU[3] = i / u where i = intersection, u = union
        # For self-consistency, overlap cells (code=3) should match intersection
        # This is a sanity check that both use same population
        
        # Count cells marked as top-k in each map (codes 1, 2, 3)
        swe_topk_count = np.sum((code == 1) | (code == 3))
        gnn_topk_count = np.sum((code == 2) | (code == 3))
        
        # Both should equal actual_k if population is consistent
        assert swe_topk_count == actual_k, f"SWE top-k count {swe_topk_count} != actual_k {actual_k}"
        assert gnn_topk_count == actual_k, f"GNN top-k count {gnn_topk_count} != actual_k {actual_k}"

    def test_overlap_code_matches_iou_ranking_on_jointly_finite_cells(self):
        """Overlap map must use the same jointly-finite top-k ranking as IoU."""
        from physics.swe.alignment import _topk_overlap_code, _patch_magnitude

        swe_map = np.array([[3.0, 3.0, 3.0, 1.0]])
        gnn_map = np.array([[3.0, 3.0, 1.0, 3.0]])

        swe_mag = _patch_magnitude(swe_map, radius=0, agg="mean")
        gnn_mag = _patch_magnitude(gnn_map, radius=0, agg="mean")
        code, actual_k = _topk_overlap_code(swe_mag, gnn_mag, k=2)

        # All cells are jointly finite. To keep PNG semantics aligned with IoU semantics,
        # overlap selection should follow the same ranking rule used in compute_topk_iou.
        # For this tied case, that yields SWE top-k {1, 2} and GNN top-k {1, 3}.
        expected = np.array([[0, 3, 1, 2]], dtype=np.int8)
        assert actual_k == 2
        assert np.array_equal(code, expected), f"Expected exact overlap code {expected}, got {code}"


# =============================================================================
# Test: magnitude ranking with exact expectations
# =============================================================================

class TestMagnitudeRankingExact:
    """Magnitude ranking should have exact, testable behavior."""

    def test_topk_iou_exact_overlap(self):
        """Test exact IoU for known top-k sets."""
        # Create maps where we know exactly which cells are top-k
        swe_map = np.array([
            [100.0, 80.0, 60.0],
            [40.0, 20.0, 10.0],
            [5.0, 2.0, 1.0],
        ])
        
        # Same ranking order -> perfect overlap
        gnn_map = np.array([
            [99.0, 79.0, 59.0],
            [39.0, 19.0, 9.0],
            [4.0, 2.0, 1.0],
        ])
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(3,), patch_radius=0)
        
        # Top 3 in both maps are (0,0), (0,1), (0,2) -> perfect match
        assert iou[3] == 1.0, f"Expected IoU=1.0 for identical ranking, got {iou[3]}"
        
    def test_topk_iou_partial_overlap_exact(self):
        """Test IoU with known partial overlap."""
        swe_map = np.array([
            [100.0, 90.0, 80.0],  # top 3: (0,0), (0,1), (0,2)
            [70.0, 60.0, 50.0],
            [40.0, 30.0, 20.0],
        ])
        
        gnn_map = np.array([
            [100.0, 70.0, 80.0],  # top 3: (0,0), (0,2), (1,0)
            [90.0, 60.0, 50.0],
            [40.0, 30.0, 20.0],
        ])
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(3,), patch_radius=0)
        
        # SWE top-3 indices: [0,1,2] (raveled: 0, 1, 2)
        # GNN top-3 indices: [0, 2, 3] (raveled: 0, 2, 3)
        # Intersection: {0, 2} -> 2 elements
        # Union: {0, 1, 2, 3} -> 4 elements
        # IoU = 2/4 = 0.5
        expected_iou = 0.5
        assert abs(iou[3] - expected_iou) < 0.01, f"Expected IoU≈{expected_iou}, got {iou[3]}"

    def test_topk_iou_negative_magnitude_ranking(self):
        """Large negatives should be ranked high by magnitude."""
        # Negative values ranked by |value|
        swe_map = np.array([
            [-100.0, 50.0, 30.0],  # |-100| is largest
            [20.0, 10.0, 5.0],
            [2.0, 1.0, 0.5],
        ])
        
        gnn_map = np.array([
            [95.0, 45.0, 25.0],  # Same pattern, positive values
            [15.0, 8.0, 4.0],
            [1.5, 1.0, 0.5],
        ])
        
        iou = compute_topk_iou(swe_map, gnn_map, k_values=(2,), patch_radius=0)
        
        # Top 2 by magnitude:
        # SWE: |-100|=100, 50 -> cells (0,0), (0,1)
        # GNN: 95, 45 -> cells (0,0), (0,1)
        # Perfect match!
        assert iou[2] == 1.0, f"Expected IoU=1.0 for matching magnitude ranking, got {iou[2]}"


# =============================================================================
# Test: scatter preprocessing direct assertions
# =============================================================================

class TestScatterPreprocessingDirect:
    """Directly assert scatter preprocessing semantics."""

    def test_scatter_uses_patch_signed_not_magnitude(self):
        """Scatter must use signed patching, not magnitude."""
        from physics.swe.alignment import _patch_signed, _patch_magnitude
        
        # Create a map with strong negative values
        arr = np.array([
            [100.0, -100.0],
            [50.0, -50.0],
        ])
        
        signed_result = _patch_signed(arr, radius=0, agg="mean")
        magnitude_result = _patch_magnitude(arr, radius=0, agg="mean")
        
        # Signed should preserve negative values
        assert signed_result[0, 1] == -100.0, "Signed preprocessing should preserve -100"
        assert signed_result[1, 1] == -50.0, "Signed preprocessing should preserve -50"
        
        # Magnitude should use abs
        assert magnitude_result[0, 1] == 100.0, "Magnitude preprocessing should give |−100|=100"
        assert magnitude_result[1, 1] == 50.0, "Magnitude preprocessing should give |−50|=50"

    def test_scatter_data_matches_spearman_input(self):
        """Scatter plot data must match what compute_spearman uses."""
        from physics.swe.alignment import _patch_signed, _safe_finite_pair
        
        swe_map = np.array([
            [10.0, -5.0, np.nan, 7.0],
            [3.0, 2.0, 1.0, 6.0],
        ])
        
        gnn_map = np.array([
            [8.0, -4.0, 2.0, 5.0],
            [np.nan, 1.5, 0.5, 4.0],
        ])
        
        # What scatter uses (via _patch_signed and _safe_finite_pair)
        s_patch = _patch_signed(swe_map, radius=0, agg="mean")
        g_patch = _patch_signed(gnn_map, radius=0, agg="mean")
        s_scatter, g_scatter = _safe_finite_pair(s_patch, g_patch)
        
        # What spearman uses
        rho, pval = compute_spearman(swe_map, gnn_map, patch_radius=0)
        
        # They should have same number of points
        # Jointly finite: (0,0), (0,1), (0,3), (1,1), (1,2), (1,3) = 6 points
        assert len(s_scatter) == 6, f"Expected 6 jointly-finite points, got {len(s_scatter)}"
        
        # Spearman should return valid result with same data
        assert not np.isnan(rho), f"Spearman should return valid value with {len(s_scatter)} points"

    def test_scatter_preserves_sign_in_values(self, tmp_path):
        """Scatter must show both positive and negative values."""
        from physics.swe.alignment import _patch_signed, _safe_finite_pair
        
        # Maps with clear positive/negative correlation pattern
        swe_map = np.array([
            [10.0, -10.0, 5.0, -5.0],
            [8.0, -8.0, 4.0, -4.0],
        ])
        
        gnn_map = -swe_map * 0.9  # Negative correlation, preserved sign pattern
        
        s_patch = _patch_signed(swe_map, radius=0, agg="mean")
        g_patch = _patch_signed(gnn_map, radius=0, agg="mean")
        s_vals, g_vals = _safe_finite_pair(s_patch, g_patch)
        
        # Verify both positive and negative values exist in scatter data
        assert np.any(s_vals > 0), "Scatter should have positive SWE values"
        assert np.any(s_vals < 0), "Scatter should have negative SWE values"
        assert np.any(g_vals > 0), "Scatter should have positive GNN values"
        assert np.any(g_vals < 0), "Scatter should have negative GNN values"

    def test_plot_alignment_scatter_passes_signed_joint_finite_vectors(self, tmp_path, monkeypatch):
        """The plotting path should pass signed jointly-finite values into scatter."""
        captured: list[tuple[np.ndarray, np.ndarray]] = []

        def _capture(self, x, y, *args, **kwargs):
            captured.append((np.asarray(x), np.asarray(y)))
            return None

        monkeypatch.setattr("matplotlib.axes._axes.Axes.scatter", _capture)

        swe_map = np.array([[10.0, -5.0, np.nan], [2.0, -1.0, 3.0]])
        gnn_map = np.array([[-8.0, 4.0, 7.0], [np.nan, 1.0, -2.0]])
        gnn_ig_maps = {"z_500": gnn_map}
        pairs = [("h", swe_map, "z_500", "SWE", "GNN")]
        report = AlignmentReport(0, 6, 0, "mean", 3.0)
        report.groups.append(GroupMetrics("h", -1.0, 0.0, {50: 0.0}, 4))

        plot_alignment_scatter(
            pairs, gnn_ig_maps, report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="test",
            patch_radius=0,
        )

        assert len(captured) == 1
        x_vals, y_vals = captured[0]
        np.testing.assert_array_equal(x_vals, np.array([10.0, -5.0, -1.0, 3.0]))
        np.testing.assert_array_equal(y_vals, np.array([-8.0, 4.0, 1.0, -2.0]))

    def test_plot_alignment_scatter_can_abs_gnn_values_for_display(self, tmp_path, monkeypatch):
        """Display-only mode can abs the GNN axis without changing scatter pairing."""
        captured: list[tuple[np.ndarray, np.ndarray]] = []

        def _capture(self, x, y, *args, **kwargs):
            captured.append((np.asarray(x), np.asarray(y)))
            return None

        monkeypatch.setattr("matplotlib.axes._axes.Axes.scatter", _capture)

        swe_map = np.array([[10.0, -5.0, np.nan], [2.0, -1.0, 3.0]])
        gnn_map = np.array([[-8.0, 4.0, 7.0], [np.nan, 1.0, -2.0]])
        gnn_ig_maps = {"z_500": gnn_map}
        pairs = [("h", swe_map, "z_500", "SWE", "|GNN|")]
        report = AlignmentReport(0, 6, 0, "mean", 3.0)
        report.groups.append(GroupMetrics("h", -1.0, 0.0, {50: 0.0}, 4))

        plot_alignment_scatter(
            pairs, gnn_ig_maps, report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="test",
            patch_radius=0,
            abs_gnn_for_display=True,
        )

        assert len(captured) == 1
        x_vals, y_vals = captured[0]
        np.testing.assert_array_equal(x_vals, np.array([10.0, -5.0, -1.0, 3.0]))
        np.testing.assert_array_equal(y_vals, np.array([8.0, 4.0, 1.0, 2.0]))


# =============================================================================
# Test: existing API behavior preserved
# =============================================================================

class TestExistingBehaviorPreserved:
    """Ensure refactoring doesn't break existing usage patterns."""

    def test_spearman_returns_nan_for_insufficient_data(self):
        """Less than 5 valid points should return NaN."""
        tiny_map = np.full((2, 2), np.nan)
        tiny_map[0, 0] = 1.0
        tiny_map[0, 1] = 2.0
        
        rho, pval = compute_spearman(tiny_map, tiny_map, patch_radius=0)
        
        assert np.isnan(rho)
        assert np.isnan(pval)

    def test_topk_iou_returns_zero_for_empty_maps(self):
        """All-NaN maps should return IoU of 0."""
        empty_map = np.full((5, 5), np.nan)
        
        iou = compute_topk_iou(empty_map, empty_map, k_values=(10,), patch_radius=0)
        
        assert iou[10] == 0.0

    def test_topk_iou_clips_k_to_valid_count(self):
        """K should be clipped to actual valid count."""
        small_map = _make_signed_map(shape=(3, 3), seed=40)
        
        # Request K=100 but only 9 valid points
        iou = compute_topk_iou(small_map, small_map, k_values=(100,), patch_radius=0)
        
        # Should not crash, should compute for k=9
        assert 100 in iou
        assert iou[100] >= 0.0  # Should be valid IoU (self-comparison = 1.0)


class TestComputeAlignmentReportContracts:
    def test_magnitude_only_grouped_maps_require_explicit_signed_main_maps(self):
        swe_result = SimpleNamespace(
            target_time_idx=0,
            S_h=np.array([[1.0, -1.0], [0.5, -0.5]]),
            S_uv=np.array([[2.0, 1.0], [0.5, 0.25]]),
        )
        grouped_maps = {
            "z_500": np.array([[1.0, 1.0], [0.5, 0.5]]),
            "uv_500": np.array([[2.0, 2.0], [1.0, 1.0]]),
        }

        with pytest.raises(ValueError, match="gnn_main_maps"):
            compute_alignment_report(
                swe_result=cast(Any, swe_result),
                gnn_ig_maps=grouped_maps,
                patch_radius=0,
            )

    def test_empty_signed_map_dict_also_rejects_magnitude_only_grouped_maps(self):
        swe_result = SimpleNamespace(
            target_time_idx=0,
            S_h=np.array([[1.0, -1.0], [0.5, -0.5]]),
            S_uv=np.array([[2.0, 1.0], [0.5, 0.25]]),
        )
        grouped_maps = {
            "z_500": np.array([[1.0, 1.0], [0.5, 0.5]]),
        }

        with pytest.raises(ValueError, match="signed z_500"):
            compute_alignment_report(
                swe_result=cast(Any, swe_result),
                gnn_ig_maps=grouped_maps,
                gnn_main_maps={},
                patch_radius=0,
            )

    def test_signed_main_maps_enable_signed_report_groups(self):
        swe_result = SimpleNamespace(
            target_time_idx=0,
            S_h=np.array([[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]]),
            S_uv=np.array([[2.0, 1.0], [0.5, 0.25], [0.1, 0.05]]),
        )
        grouped_maps = {
            "z_500": np.abs(np.array([[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]])),
            "uv_500": np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 0.5]]),
        }
        signed_maps = {
            "z_500": np.array([[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]]),
        }

        report = compute_alignment_report(
            swe_result=cast(Any, swe_result),
            gnn_ig_maps=grouped_maps,
            gnn_main_maps=signed_maps,
            patch_radius=0,
        )

        assert [group.group_name for group in report.groups] == ["h"]
