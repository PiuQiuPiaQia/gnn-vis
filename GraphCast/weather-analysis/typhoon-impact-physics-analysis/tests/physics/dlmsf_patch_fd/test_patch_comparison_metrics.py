from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import (
    _compute_single_deletion_curve,
    _compute_alignment_metrics,
    _patch_scores_from_maps,
    compute_topk_iou_signed,
    compute_sign_agreement,
)
from shared.patch_geometry import CenteredWindow


def _window_no_core(shape: tuple[int, int]) -> CenteredWindow:
    n_lat, n_lon = shape
    return CenteredWindow(
        lat_indices=np.arange(n_lat, dtype=np.int64),
        lon_indices=np.arange(n_lon, dtype=np.int64),
        lat_vals=np.linspace(10.0, 10.0 + n_lat - 1, n_lat),
        lon_vals=np.linspace(120.0, 120.0 + n_lon - 1, n_lon),
        center_row=n_lat // 2,
        center_col=n_lon // 2,
        core_mask=np.zeros(shape, dtype=bool),
    )


def test_patch_scores_from_maps_uses_sum_aggregation():
    window = _window_no_core((2, 3))
    signed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    abs_map = np.abs(signed)

    patch_scores = _patch_scores_from_maps(
        window=window,
        patch_size=2,
        stride=1,
        signed_cell_map=signed,
        abs_cell_map=abs_map,
    )

    # patch_size=2 along lon, stride=1 → 2 patches covering cols [0,1] and [1,2]
    # sum over all rows for each patch (2 rows × 2 cols)
    # patch 0: 1+2+4+5=12, patch 1: 2+3+5+6=16
    np.testing.assert_allclose(patch_scores["signed_scores"], np.array([12.0, 16.0]))
    np.testing.assert_allclose(patch_scores["abs_scores"], np.array([12.0, 16.0]))


def test_compute_alignment_metrics_returns_valid_fields():
    metrics = _compute_alignment_metrics(
        direction_mode="along",
        patch_size=5,
        ig_abs_scores=np.array([4.0, 3.0, 1.0]),
        dlmsf_parallel_scores=np.array([4.0, 2.0, 3.0]),
        topk_fraction=0.34,
    )

    # topk_fraction=0.34 * 3 patches → ceil(1.02)=2
    assert metrics.topq_k == 2
    # top-2 of ig=[4,3,1] → indices {0,1}; top-2 of |dlmsf|=[4,2,3] → indices {0,2}; IoU=1/3
    assert metrics.iou_topq == pytest.approx(1.0 / 3.0)
    assert metrics.direction == "along"
    assert metrics.patch_size == 5
    assert metrics.n_patches == 3


def test_compute_single_deletion_curve_produces_correct_deltas(monkeypatch):
    window = _window_no_core((1, 3))
    context = SimpleNamespace(eval_inputs=object())
    runtime_cfg = SimpleNamespace()
    patches = []
    for col in range(3):
        mask = np.zeros(window.shape, dtype=bool)
        mask[0, col] = True
        patches.append(SimpleNamespace(mask=mask))

    monkeypatch.setattr(
        "physics.dlmsf_patch_fd.patch_comparison._mask_inputs_by_window_mask",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "physics.dlmsf_patch_fd.patch_comparison._run_forward_track_scalar",
        lambda *args, **kwargs: (0.5, None),
    )

    curve = _compute_single_deletion_curve(
        context=context,
        runtime_cfg=runtime_cfg,
        baseline_inputs=object(),
        vars_to_use=[],
        window=window,
        patches=patches,
        order=np.arange(len(patches)),
        center_field_name="mean_sea_level_pressure",
        direction_mode="along",
        softmin_temperature=1.0,
        base_scalar=1.0,
    )

    assert len(curve["deltas"]) == 3
    # base_scalar=1.0, new_scalar=0.5 → delta=0.5 at every step
    assert all(abs(d - 0.5) < 1e-9 for d in curve["deltas"])


# ---------------------------------------------------------------------------
# Task 1: IoU_pos/neg and sign_agreement
# ---------------------------------------------------------------------------

class TestIouSignedPos:
    def test_iou_pos_perfect_overlap(self):
        a = np.array([3.0, 2.0, 1.0, -1.0])
        b = np.array([3.0, 2.0, 1.0, -1.0])
        assert compute_topk_iou_signed(a, b, k=2, sign="pos") == pytest.approx(1.0)

    def test_iou_pos_no_overlap(self):
        a = np.array([5.0, 1.0, 2.0, 3.0])
        b = np.array([1.0, 5.0, 2.0, 3.0])
        assert compute_topk_iou_signed(a, b, k=1, sign="pos") == pytest.approx(0.0)

    def test_iou_pos_returns_zero_when_no_positive_entries(self):
        a = np.array([-1.0, -2.0])
        b = np.array([-1.0, -2.0])
        assert compute_topk_iou_signed(a, b, k=1, sign="pos") == pytest.approx(0.0)

    def test_iou_pos_invalid_sign_raises(self):
        with pytest.raises(ValueError, match="sign must be"):
            compute_topk_iou_signed(np.array([1.0]), np.array([1.0]), k=1, sign="bad")


class TestIouSignedNeg:
    def test_iou_neg_perfect_overlap(self):
        a = np.array([-3.0, -2.0, 1.0, 2.0])
        b = np.array([-3.0, -2.0, 1.0, 2.0])
        assert compute_topk_iou_signed(a, b, k=2, sign="neg") == pytest.approx(1.0)

    def test_iou_neg_returns_zero_when_no_negative_entries(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        assert compute_topk_iou_signed(a, b, k=1, sign="neg") == pytest.approx(0.0)

    def test_iou_neg_partial_overlap(self):
        # Top-2 neg from a: {0,1} (-3,-2), from b: {0,2} (-3,-1) → inter=1, union=3 → 1/3
        a = np.array([-3.0, -2.0, -1.0, 1.0])
        b = np.array([-3.0, -0.5, -1.0, 1.0])
        result = compute_topk_iou_signed(a, b, k=2, sign="neg")
        assert 0.0 < result < 1.0


class TestSignAgreement:
    def test_sign_agreement_all_same_sign(self):
        a = np.array([3.0, 2.0, 1.0])
        b = np.array([4.0, 3.0, 2.0])
        assert compute_sign_agreement(a, b, k=2) == pytest.approx(1.0)

    def test_sign_agreement_all_opposite_sign(self):
        a = np.array([3.0, 2.0, 1.0])
        b = np.array([-4.0, -3.0, -2.0])
        # Top-2 by |val|: a={0,1}, b={0,1} → overlap={0,1}, all opposite → 0.0
        assert compute_sign_agreement(a, b, k=2) == pytest.approx(0.0)

    def test_sign_agreement_returns_nan_when_no_overlap(self):
        a = np.array([5.0, 1.0])
        b = np.array([1.0, 5.0])
        assert math.isnan(compute_sign_agreement(a, b, k=1))

    def test_sign_agreement_with_nan_excluded(self):
        a = np.array([3.0, float("nan"), 1.0])
        b = np.array([3.0, float("nan"), 1.0])
        result = compute_sign_agreement(a, b, k=2)
        assert np.isfinite(result)

    def test_sign_agreement_empty_input_returns_nan(self):
        a = np.array([float("nan")])
        b = np.array([float("nan")])
        assert math.isnan(compute_sign_agreement(a, b, k=1))
