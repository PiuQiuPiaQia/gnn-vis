# -*- coding: utf-8 -*-
"""Tests for IG sanity check module."""
from __future__ import annotations

import json
import numpy as np
import pytest
import xarray as xr
from pathlib import Path
from types import SimpleNamespace

from physics.swe.ig_sanity import (
    sample_random_grid_indices,
    build_point_score_da,
    compute_sanity_summary,
    write_ig_sanity_report,
    _sanitize_for_json,
)


class TestSampleRandomGridIndices:
    """Tests for sample_random_grid_indices helper."""

    def test_returns_unique_indices(self):
        """Should return unique indices."""
        result = sample_random_grid_indices(
            n_lat=10,
            n_lon=10,
            k=5,
            excluded=set(),
            seed=42,
        )
        assert len(result) == 5
        assert len(set(result)) == 5  # All unique

    def test_excludes_given_set(self):
        """Should exclude indices in the excluded set."""
        excluded = {(0, 0), (1, 1), (2, 2)}
        result = sample_random_grid_indices(
            n_lat=5,
            n_lon=5,
            k=10,
            excluded=excluded,
            seed=42,
        )
        for idx in result:
            assert idx not in excluded

    def test_deterministic_with_seed(self):
        """Should be deterministic with same seed."""
        result1 = sample_random_grid_indices(
            n_lat=10,
            n_lon=10,
            k=5,
            excluded=set(),
            seed=123,
        )
        result2 = sample_random_grid_indices(
            n_lat=10,
            n_lon=10,
            k=5,
            excluded=set(),
            seed=123,
        )
        assert result1 == result2

    def test_degrades_with_insufficient_points(self):
        """Should return fewer points if insufficient available."""
        excluded = {(i, j) for i in range(10) for j in range(10) if i < 9 or j < 9}
        # Only (9, 9) is available
        result = sample_random_grid_indices(
            n_lat=10,
            n_lon=10,
            k=5,
            excluded=excluded,
            seed=42,
        )
        assert len(result) <= 1  # Only one point available


class TestComputeSanitySummary:
    """Tests for compute_sanity_summary helper - stricter non-finite behavior."""

    def test_pass_rule_topk_greater_and_lift_ratio_met(self):
        """Should pass when topk_mean > rand_mean AND lift_ratio >= min_lift_ratio."""
        result = compute_sanity_summary(
            topk_deltas=[2.0, 3.0, 2.5],
            rand_deltas=[1.0, 1.0, 1.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is True
        assert result["topk_mean"] == pytest.approx(2.5)
        assert result["rand_mean"] == pytest.approx(1.0)
        assert result["lift_ratio"] == pytest.approx(2.5)

    def test_fail_rule_topk_smaller(self):
        """Should fail when topk_mean < rand_mean (even if lift_ratio met)."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, 1.1, 1.0],
            rand_deltas=[2.0, 2.0, 2.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert "reason" in result

    def test_fail_rule_lift_ratio_not_met(self):
        """Should fail when lift_ratio < min_lift_ratio (even if topk > rand)."""
        result = compute_sanity_summary(
            topk_deltas=[1.1, 1.1, 1.1],
            rand_deltas=[1.0, 1.0, 1.0],
            min_lift_ratio=1.5,
        )
        assert result["passed"] is False
        assert result["lift_ratio"] == pytest.approx(1.1)
        assert "reason" in result

    def test_handles_empty_topk_list(self):
        """Should fail with reason='empty_inputs' when topk list empty."""
        result = compute_sanity_summary(
            topk_deltas=[],
            rand_deltas=[1.0, 2.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "empty_inputs"

    def test_handles_empty_rand_list(self):
        """Should fail with reason='empty_inputs' when rand list empty."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, 2.0],
            rand_deltas=[],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "empty_inputs"

    def test_handles_non_finite_in_topk(self):
        """Should fail with reason='non_finite_input' when topk has non-finite."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, float("nan"), 2.0],
            rand_deltas=[1.0, 1.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "non_finite_input"

    def test_handles_non_finite_in_rand(self):
        """Should fail with reason='non_finite_input' when rand has non-finite."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, 2.0],
            rand_deltas=[float("inf"), 1.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "non_finite_input"

    def test_handles_zero_rand_mean(self):
        """Should fail with reason='rand_mean_too_small' when rand_mean <= 1e-12."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, 2.0],
            rand_deltas=[0.0, 0.0],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "rand_mean_too_small"

    def test_handles_very_small_rand_mean(self):
        """Should fail when rand_mean is very small (<= 1e-12)."""
        result = compute_sanity_summary(
            topk_deltas=[1.0, 2.0],
            rand_deltas=[1e-13, 1e-13],
            min_lift_ratio=1.1,
        )
        assert result["passed"] is False
        assert result.get("reason") == "rand_mean_too_small"


class TestBuildPointScoreDa:
    """Tests for build_point_score_da helper."""

    def test_builds_correct_shape(self):
        """Should produce DataArray with correct shape and dims."""
        lat_vals = np.linspace(-10, 10, 5)
        lon_vals = np.linspace(-10, 10, 6)
        ig_maps = {
            "var1": np.random.randn(5, 6),
            "var2": np.random.randn(5, 6),
        }
        result = build_point_score_da(ig_maps, lat_vals, lon_vals)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (5, 6)
        assert result.dims == ("lat", "lon")
        assert result.name == "ig_candidate_score"

    def test_sums_abs_values(self):
        """Should sum absolute values of input maps."""
        lat_vals = np.array([0.0, 1.0])
        lon_vals = np.array([0.0, 1.0])
        ig_maps = {
            "var1": np.array([[1.0, -2.0], [3.0, -4.0]]),
            "var2": np.array([[0.5, 0.5], [0.5, 0.5]]),
        }
        result = build_point_score_da(ig_maps, lat_vals, lon_vals)
        expected = np.array([[1.5, 2.5], [3.5, 4.5]])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_raises_on_shape_mismatch(self):
        """Should raise ValueError listing var names with mismatched shapes."""
        lat_vals = np.array([0.0, 1.0, 2.0])
        lon_vals = np.array([0.0, 1.0])
        ig_maps = {
            "var1": np.array([[1.0, -2.0], [3.0, -4.0]]),  # shape (2, 2), expected (3, 2)
            "var2": np.array([[0.5, 0.5, 0.5]]),  # shape (1, 3), expected (3, 2)
        }
        with pytest.raises(ValueError) as exc_info:
            build_point_score_da(ig_maps, lat_vals, lon_vals)
        
        error_msg = str(exc_info.value)
        assert "mismatched shapes" in error_msg
        assert "var1" in error_msg
        assert "var2" in error_msg


class TestSanitizeForJson:
    """Tests for _sanitize_for_json helper."""

    def test_sanitize_nan_to_none(self):
        """Should convert NaN to None."""
        assert _sanitize_for_json(float("nan")) is None

    def test_sanitize_inf_to_none(self):
        """Should convert inf/-inf to None."""
        assert _sanitize_for_json(float("inf")) is None
        assert _sanitize_for_json(float("-inf")) is None

    def test_preserve_finite_floats(self):
        """Should preserve finite floats."""
        assert _sanitize_for_json(3.14) == 3.14
        assert _sanitize_for_json(-100.0) == -100.0

    def test_sanitize_nested_dict(self):
        """Should recursively sanitize nested dicts."""
        data = {
            "a": float("nan"),
            "b": 1.0,
            "c": {"d": float("inf"), "e": 2.0}
        }
        result = _sanitize_for_json(data)
        assert result["a"] is None
        assert result["b"] == 1.0
        assert result["c"]["d"] is None
        assert result["c"]["e"] == 2.0

    def test_sanitize_list(self):
        """Should recursively sanitize lists."""
        data = [float("nan"), 1.0, float("inf")]
        result = _sanitize_for_json(data)
        assert result == [None, 1.0, None]


class TestWriteIgSanityReport:
    """Tests for write_ig_sanity_report helper."""

    def test_creates_json_file(self, tmp_path: Path):
        """Should create JSON file with correct content."""
        payload = {
            "status": "ok",
            "passed": True,
            "topk_mean": 2.5,
            "rand_mean": 1.0,
        }
        output_path = tmp_path / "subdir" / "report.json"
        write_ig_sanity_report(payload, output_path)
        
        assert output_path.exists()
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert content["status"] == "ok"
        assert content["passed"] is True
        assert content["topk_mean"] == 2.5

    def test_creates_parent_directories(self, tmp_path: Path):
        """Should create parent directories if needed."""
        payload = {"test": "value"}
        output_path = tmp_path / "a" / "b" / "c" / "report.json"
        write_ig_sanity_report(payload, output_path)
        assert output_path.exists()

    def test_sanitizes_nan_inf_to_null(self, tmp_path: Path):
        """Should sanitize NaN/Inf to null (None) in JSON output."""
        payload = {
            "status": "ok",
            "lift_ratio": float("inf"),
            "topk_mean": float("nan"),
            "rand_mean": -float("inf"),
        }
        output_path = tmp_path / "report.json"
        write_ig_sanity_report(payload, output_path)
        
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert content["lift_ratio"] is None
        assert content["topk_mean"] is None
        assert content["rand_mean"] is None

    def test_uses_allow_nan_false(self, tmp_path: Path):
        """Should use allow_nan=False (would raise if non-finite not sanitized)."""
        payload = {
            "status": "ok",
            "value": float("nan"),
        }
        output_path = tmp_path / "report.json"
        # Should not raise ValueError
        write_ig_sanity_report(payload, output_path)
        assert output_path.exists()

    def test_handles_unicode_in_reason(self, tmp_path: Path):
        """Should handle unicode characters in reason field."""
        payload = {
            "status": "failed",
            "reason": "错误: 无效的输入数据",
        }
        output_path = tmp_path / "report.json"
        
        write_ig_sanity_report(payload, output_path)
        
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "错误" in content["reason"]


class TestAnalysisConfigFromModule:
    """Tests for AnalysisConfig.from_module reading new IG sanity fields."""

    def test_reads_ig_sanity_fields(self):
        """Should read new IG sanity config fields."""
        cfg_module = SimpleNamespace(
            DATASET_CONFIGS={"low_res": {}},
            DATASET_TYPE="low_res",
            TARGET_TIME_IDX=0,
            TARGET_VARIABLE="msl",
            TARGET_LEVEL=None,
            PATCH_RADIUS=2,
            PATCH_SCORE_AGG="mean",
            PERTURB_TIME="all",
            PERTURB_VARIABLES=None,
            PERTURB_LEVELS=None,
            BASELINE_MODE="local_annulus_median",
            LOCAL_BASELINE_INNER_DEG=5.0,
            LOCAL_BASELINE_OUTER_DEG=12.0,
            LOCAL_BASELINE_MIN_POINTS=120,
            HEATMAP_DPI=200,
            IG_STEPS=50,
            INCLUDE_TARGET_INPUTS=False,
            GRADIENT_VMAX_QUANTILE=0.9,
            GRADIENT_CMAP="RdBu_r",
            GRADIENT_CENTER_WINDOW_DEG=10.0,
            GRADIENT_CENTER_SCALE_QUANTILE=0.99,
            GRADIENT_ALPHA_QUANTILE=0.9,
            GRADIENT_TIME_AGG="single",
            DIR_PATH_PARAMS="/params",
            DIR_PATH_DATASET="/dataset",
            DIR_PATH_STATS="/stats",
            IG_SANITY_ENABLE=True,
            IG_SANITY_TOPK=15,
            IG_SANITY_RANDOM_K=20,
            IG_SANITY_SEED=123,
            IG_SANITY_MIN_LIFT_RATIO=1.5,
        )
        
        from shared.analysis_pipeline import AnalysisConfig
        config = AnalysisConfig.from_module(cfg_module)
        
        assert config.ig_sanity_enable is True
        assert config.ig_sanity_topk == 15
        assert config.ig_sanity_random_k == 20
        assert config.ig_sanity_seed == 123
        assert config.ig_sanity_min_lift_ratio == 1.5

    def test_uses_defaults_when_missing(self):
        """Should use defaults when IG sanity fields are missing."""
        cfg_module = SimpleNamespace(
            DATASET_CONFIGS={"low_res": {}},
            DATASET_TYPE="low_res",
            TARGET_TIME_IDX=0,
            TARGET_VARIABLE="msl",
            TARGET_LEVEL=None,
            PATCH_RADIUS=2,
            PATCH_SCORE_AGG="mean",
            PERTURB_TIME="all",
            PERTURB_VARIABLES=None,
            PERTURB_LEVELS=None,
            BASELINE_MODE="local_annulus_median",
            LOCAL_BASELINE_INNER_DEG=5.0,
            LOCAL_BASELINE_OUTER_DEG=12.0,
            LOCAL_BASELINE_MIN_POINTS=120,
            HEATMAP_DPI=200,
            IG_STEPS=50,
            INCLUDE_TARGET_INPUTS=False,
            GRADIENT_VMAX_QUANTILE=0.9,
            GRADIENT_CMAP="RdBu_r",
            GRADIENT_CENTER_WINDOW_DEG=10.0,
            GRADIENT_CENTER_SCALE_QUANTILE=0.99,
            GRADIENT_ALPHA_QUANTILE=0.9,
            GRADIENT_TIME_AGG="single",
            DIR_PATH_PARAMS="/params",
            DIR_PATH_DATASET="/dataset",
            DIR_PATH_STATS="/stats",
        )
        
        from shared.analysis_pipeline import AnalysisConfig
        config = AnalysisConfig.from_module(cfg_module)
        
        assert config.ig_sanity_enable is True  # default
        assert config.ig_sanity_topk == 10  # default
        assert config.ig_sanity_random_k == 10  # default
        assert config.ig_sanity_seed == 42  # default
        assert config.ig_sanity_min_lift_ratio == 1.1  # default
