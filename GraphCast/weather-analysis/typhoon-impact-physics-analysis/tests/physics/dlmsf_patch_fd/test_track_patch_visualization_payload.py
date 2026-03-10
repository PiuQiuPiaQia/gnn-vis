from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import (
    _build_case_visualization_payload,
    _case_summary,
    PatchAlignmentMetrics,
)
from shared.patch_geometry import CenteredWindow, PatchDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(shape, cell: tuple) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[cell] = True
    return m


def _make_window(shape=(1, 3)) -> CenteredWindow:
    n_lat, n_lon = shape
    return CenteredWindow(
        lat_indices=np.arange(n_lat, dtype=np.int64),
        lon_indices=np.arange(n_lon, dtype=np.int64),
        lat_vals=np.linspace(10.0, 10.0 + n_lat - 1, n_lat),
        lon_vals=np.linspace(120.0, 120.0 + n_lon - 1, n_lon),
        center_row=0,
        center_col=1,
        core_mask=np.zeros(shape, dtype=bool),
    )


def _make_patches(shape=(1, 3)) -> List[PatchDefinition]:
    """Three non-overlapping single-cell patches in a 1x3 grid."""
    return [
        PatchDefinition(patch_id=0, row_start=0, row_end=1, col_start=0, col_end=1,
                        mask=_make_mask(shape, (0, 0)), n_cells=1),
        PatchDefinition(patch_id=1, row_start=0, row_end=1, col_start=1, col_end=2,
                        mask=_make_mask(shape, (0, 1)), n_cells=1),
        PatchDefinition(patch_id=2, row_start=0, row_end=1, col_start=2, col_end=3,
                        mask=_make_mask(shape, (0, 2)), n_cells=1),
    ]


SHAPE = (1, 3)


def _dummy_payload_args(**overrides):
    """
    Default setup: 3 patches, shape (1, 3), topq_fraction=0.5.
    k = ceil(0.5 * 3) = 2
    IG top-2 (by abs):   patch 0 (5.0), patch 1 (4.0)
    DLMSF top-2 (by abs): patch 0 (6.0), patch 1 (3.0)
    Overlap: {0, 1}

    Patch 0: ig_signed=+5, dlmsf_signed=+2  -> same-sign positive  (class 1)
    Patch 1: ig_signed=-4, dlmsf_signed=-1  -> same-sign negative   (class 2)
    sign_agreement = 2/2 = 1.0
    """
    defaults = dict(
        window=_make_window(SHAPE),
        patches=_make_patches(SHAPE),
        direction="along",
        patch_size=3,
        target_time_idx=0,
        topq_fraction=0.5,
        ig_abs_map=np.array([[1.0, 2.0, 3.0]]),
        ig_abs_scores=np.array([5.0, 4.0, 1.0]),
        ig_signed_scores=np.array([5.0, -4.0, 1.0]),
        dlmsf_abs_map=np.array([[2.0, 3.0, 1.0]]),
        dlmsf_abs_scores=np.array([6.0, 3.0, 1.0]),
        dlmsf_signed_scores=np.array([2.0, -1.0, -1.0]),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Tests: payload top-level structure
# ---------------------------------------------------------------------------

class TestVisualizationPayloadStructure:

    def test_payload_has_required_top_level_keys(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        for key in ("meta", "overlap", "scatter", "sign_map", "deletion"):
            assert key in payload, f"Missing key: {key!r}"

    def test_deletion_is_none_by_default(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["deletion"] is None


# ---------------------------------------------------------------------------
# Tests: meta section
# ---------------------------------------------------------------------------

class TestVisualizationMeta:

    def test_meta_contains_direction(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["meta"]["direction"] == "along"

    def test_meta_contains_patch_size(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["meta"]["patch_size"] == 3

    def test_meta_contains_target_time_idx(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["meta"]["target_time_idx"] == 0

    def test_meta_contains_topq_fraction(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["meta"]["topq_fraction"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: overlap section
# ---------------------------------------------------------------------------

class TestVisualizationOverlap:

    def test_overlap_contains_spearman_rho(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "spearman_rho" in payload["overlap"]
        assert np.isfinite(payload["overlap"]["spearman_rho"])

    def test_overlap_contains_iou_at_20(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "iou_at_20" in payload["overlap"]
        assert np.isfinite(payload["overlap"]["iou_at_20"])

    def test_overlap_contains_serializable_ig_abs_map(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert isinstance(payload["overlap"]["ig_abs_map"], list)

    def test_overlap_contains_serializable_dlmsf_abs_map(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert isinstance(payload["overlap"]["dlmsf_abs_map"], list)

    def test_overlap_contains_overlap_mask(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "overlap_mask" in payload["overlap"]

    def test_overlap_contains_lat_lon_vals(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "lat_vals" in payload["overlap"]
        assert "lon_vals" in payload["overlap"]


# ---------------------------------------------------------------------------
# Tests: scatter section
# ---------------------------------------------------------------------------

class TestVisualizationScatter:

    def test_scatter_x_is_dlmsf_abs(self):
        args = _dummy_payload_args()
        payload = _build_case_visualization_payload(**args)
        x = np.array(payload["scatter"]["x_patch_abs_scores"])
        np.testing.assert_array_equal(x, args["dlmsf_abs_scores"])

    def test_scatter_y_is_ig_abs(self):
        args = _dummy_payload_args()
        payload = _build_case_visualization_payload(**args)
        y = np.array(payload["scatter"]["y_patch_abs_scores"])
        np.testing.assert_array_equal(y, args["ig_abs_scores"])

    def test_scatter_contains_spearman_rho(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "spearman_rho" in payload["scatter"]
        assert np.isfinite(payload["scatter"]["spearman_rho"])


# ---------------------------------------------------------------------------
# Tests: sign_map section — sign_agreement_at_20
# ---------------------------------------------------------------------------

class TestSignAgreement:

    def test_sign_agreement_perfect_when_all_overlap_same_sign(self):
        # Default: both overlap patches are same-sign -> agreement = 1.0
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert payload["sign_map"]["sign_agreement_at_20"] == pytest.approx(1.0)

    def test_sign_agreement_zero_when_all_overlap_opposite_sign(self):
        args = _dummy_payload_args(
            ig_signed_scores=np.array([5.0, -4.0, 1.0]),
            dlmsf_signed_scores=np.array([-2.0, 1.0, -1.0]),
        )
        payload = _build_case_visualization_payload(**args)
        assert payload["sign_map"]["sign_agreement_at_20"] == pytest.approx(0.0)

    def test_sign_agreement_nan_when_no_overlap(self):
        # Force k=1: with 3 patches, fraction=0.33 -> k = max(1, ceil(0.33*3)) = max(1,1) = 1
        # IG top-1: patch 0 (score=5), DLMSF top-1: patch 1 (score=6) -> overlap empty
        args = _dummy_payload_args(
            topq_fraction=0.33,
            ig_abs_scores=np.array([5.0, 1.0, 1.0]),
            dlmsf_abs_scores=np.array([1.0, 6.0, 1.0]),
        )
        payload = _build_case_visualization_payload(**args)
        assert math.isnan(payload["sign_map"]["sign_agreement_at_20"])

    def test_sign_agreement_ignores_nonfinite_as_opposite(self):
        # patch 0 in overlap: ig=nan -> opposite; patch 1: ig=-4, dlmsf=-1 -> same
        # agreement = 1/2 = 0.5
        args = _dummy_payload_args(
            ig_signed_scores=np.array([float("nan"), -4.0, 1.0]),
            dlmsf_signed_scores=np.array([2.0, -1.0, -1.0]),
        )
        payload = _build_case_visualization_payload(**args)
        assert payload["sign_map"]["sign_agreement_at_20"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: sign_map section — sign_class_map
# ---------------------------------------------------------------------------

class TestSignClassMap:

    def test_sign_class_map_non_overlap_cells_are_zero(self):
        # Default: overlap = {0, 1}, non-overlap = patch 2 -> cell (0,2)
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        cmap = np.array(payload["sign_map"]["sign_class_map"])
        assert cmap[0, 2] == 0, "Non-overlap cell should be 0"

    def test_sign_class_map_same_sign_positive_is_one(self):
        # Default: patch 0 -> ig=+5, dlmsf=+2 -> class 1, cell (0,0)
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        cmap = np.array(payload["sign_map"]["sign_class_map"])
        assert cmap[0, 0] == 1, "Same-sign positive cell should be 1"

    def test_sign_class_map_same_sign_negative_is_two(self):
        # Default: patch 1 -> ig=-4, dlmsf=-1 -> class 2, cell (0,1)
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        cmap = np.array(payload["sign_map"]["sign_class_map"])
        assert cmap[0, 1] == 2, "Same-sign negative cell should be 2"

    def test_sign_class_map_opposite_sign_is_three(self):
        # patch 0: ig=+5, dlmsf=-2 -> opposite (class 3)
        args = _dummy_payload_args(
            ig_signed_scores=np.array([5.0, -4.0, 1.0]),
            dlmsf_signed_scores=np.array([-2.0, -1.0, -1.0]),
        )
        payload = _build_case_visualization_payload(**args)
        cmap = np.array(payload["sign_map"]["sign_class_map"])
        assert cmap[0, 0] == 3, "Opposite-sign cell should be 3"

    def test_sign_class_map_has_correct_shape(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        cmap = np.array(payload["sign_map"]["sign_class_map"])
        assert cmap.shape == SHAPE

    def test_sign_class_map_contains_overlap_mask(self):
        payload = _build_case_visualization_payload(**_dummy_payload_args())
        assert "overlap_mask" in payload["sign_map"]
        mask = np.array(payload["sign_map"]["overlap_mask"])
        assert mask.shape == SHAPE
        # overlap patches 0 and 1 cover cells (0,0) and (0,1)
        assert mask[0, 0] is np.bool_(True)
        assert mask[0, 1] is np.bool_(True)
        assert mask[0, 2] is np.bool_(False)


# ---------------------------------------------------------------------------
# Task 2 tests: _case_summary must expose visualization payload
# ---------------------------------------------------------------------------

def _dummy_metrics() -> PatchAlignmentMetrics:
    return PatchAlignmentMetrics(
        direction="along",
        patch_size=3,
        n_patches=4,
        pearson_r=0.6,
        pearson_pval=0.1,
        spearman_rho=0.7,
        spearman_pval=0.05,
        iou_topq=0.5,
        topq_fraction=0.2,
        topq_k=1,
    )


def _dummy_visualization_payload() -> dict:
    return {
        "meta": {"direction": "along", "patch_size": 3, "target_time_idx": 0, "topq_fraction": 0.2},
        "overlap": {"spearman_rho": 0.7, "iou_at_20": 0.5, "ig_abs_map": [[1.0]], "dlmsf_abs_map": [[2.0]],
                    "overlap_mask": [[True]], "lat_vals": [10.0], "lon_vals": [120.0]},
        "scatter": {"x_patch_abs_scores": [1.0], "y_patch_abs_scores": [2.0], "spearman_rho": 0.7},
        "sign_map": {"sign_class_map": [[1]], "sign_agreement_at_20": 1.0,
                     "overlap_mask": [[True]], "lat_vals": [10.0], "lon_vals": [120.0]},
        "deletion": None,
    }


def _dummy_deletion_summary() -> dict:
    """Minimal DeletionCurveSummary-compatible dict that asdict() can reproduce."""
    from physics.dlmsf_patch_fd.patch_comparison import DeletionCurveSummary
    return DeletionCurveSummary(
        step_fraction=[0.5, 1.0],
        masked_fraction=[0.3, 0.6],
        high_ig_delta=[0.5, 1.0],
        low_ig_delta=[0.05, 0.1],
        random_mean_delta=[0.2, 0.4],
        high_ig_auc=0.6,
        high_ig_aopc=0.57,
        low_ig_auc=0.07,
        low_ig_aopc=0.09,
        random_mean_auc=0.2,
        random_mean_aopc=0.19,
        random_repeats=3,
        seed=42,
    )


def _make_dummy_case(with_deletion: bool = False) -> dict:
    return {
        "direction": "along",
        "patch_size": 3,
        "window_size": 21,
        "core_size": 3,
        "stride": 2,
        "ig": {
            "full_scalar": 1.0,
            "baseline_scalar": 0.5,
            "lhs": 0.49,
            "rhs": 0.50,
            "rel_err": 0.02,
        },
        "metrics": _dummy_metrics(),
        "track_diagnostics": {},
        "plot": {},
        "visualization": _dummy_visualization_payload(),
        "deletion": _dummy_deletion_summary() if with_deletion else None,
    }


class TestCaseSummaryExposesVisualization:
    """_case_summary() must expose visualization and deletion display fields."""

    def test_case_summary_has_visualization_key(self):
        summary = _case_summary(_make_dummy_case())
        assert "visualization" in summary

    def test_case_summary_visualization_meta_direction(self):
        summary = _case_summary(_make_dummy_case())
        assert summary["visualization"]["meta"]["direction"] == "along"

    def test_case_summary_visualization_overlap_has_spearman(self):
        summary = _case_summary(_make_dummy_case())
        assert "spearman_rho" in summary["visualization"]["overlap"]

    def test_case_summary_visualization_sign_map_has_agreement(self):
        summary = _case_summary(_make_dummy_case())
        assert "sign_agreement_at_20" in summary["visualization"]["sign_map"]

    def test_case_summary_without_deletion_has_no_deletion_key(self):
        summary = _case_summary(_make_dummy_case(with_deletion=False))
        # deletion should be absent or None when not computed
        assert summary.get("deletion") is None

    def test_case_summary_with_deletion_exposes_aopc_fields(self):
        summary = _case_summary(_make_dummy_case(with_deletion=True))
        assert "deletion" in summary
        del_summary = summary["deletion"]
        for field in ("high_ig_aopc", "low_ig_aopc", "random_mean_aopc"):
            assert field in del_summary, f"Missing deletion field: {field!r}"

    def test_case_summary_visualization_deletion_wired_from_deletion(self):
        """visualization['deletion'] must be populated when case['deletion'] exists."""
        summary = _case_summary(_make_dummy_case(with_deletion=True))
        vis_del = summary["visualization"]["deletion"]
        assert vis_del is not None
        assert "aopc_high" in vis_del
        assert "aopc_random" in vis_del
        assert "aopc_low" in vis_del
        assert vis_del["aopc_high"] == pytest.approx(0.57)
        assert vis_del["aopc_random"] == pytest.approx(0.19)
        assert vis_del["aopc_low"] == pytest.approx(0.09)
