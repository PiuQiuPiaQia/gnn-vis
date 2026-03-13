from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from physics.dlmsf_patch_fd.patch_comparison import (
    _compute_single_deletion_curve,
    _compute_alignment_metrics,
    _patch_scores_from_maps,
)
from shared.patch_geometry import CenteredWindow, build_centered_window


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
        patch_radius=0,
        patch_score_agg="mean",
        ig_abs_map=np.array([[4.0, 3.0, 1.0]]),
        dlmsf_abs_map=np.array([[4.0, 2.0, 3.0]]),
        topk_k=2,
    )

    assert metrics.topk_k == 2
    # top-2 of |ig|=[4,3,1] → indices {0,1}; top-2 of |dlmsf|=[4,2,3] → indices {0,2}; IoU=1/3
    assert metrics.iou_topk == pytest.approx(1.0 / 3.0)
    assert metrics.direction == "along"
    assert metrics.patch_size == 5
    assert metrics.n_valid == 3


def test_compute_alignment_metrics_caps_topk_at_valid_patch_count():
    metrics = _compute_alignment_metrics(
        direction_mode="along",
        patch_size=5,
        patch_radius=0,
        patch_score_agg="mean",
        ig_abs_map=np.array([[4.0, 3.0, 1.0]]),
        dlmsf_abs_map=np.array([[4.0, 2.0, 3.0]]),
        topk_k=50,
    )

    assert metrics.topk_k == 3
    assert metrics.iou_topk == pytest.approx(1.0)


def test_compute_alignment_metrics_preserves_sign_for_spearman():
    metrics = _compute_alignment_metrics(
        direction_mode="along",
        patch_size=3,
        patch_radius=0,
        patch_score_agg="mean",
        ig_abs_map=np.array([[1.0, 2.0, 3.0]]),
        dlmsf_abs_map=np.array([[1.0, 2.0, 3.0]]),
        topk_k=2,
    )

    assert metrics.spearman_rho == pytest.approx(1.0)


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


def test_patch_scores_from_maps_without_annulus_mask_uses_sum():
    """Without annulus_mask, behavior is unchanged: sum over full patch."""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    ones = np.ones(window.shape)
    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=ones, abs_cell_map=ones,
    )
    assert result["abs_scores"].shape[0] == len(result["patches"])
    # Without annulus_mask: each patch score = sum of 1.0 per cell = n_cells
    for i, patch in enumerate(result["patches"]):
        assert result["abs_scores"][i] == pytest.approx(float(patch.n_cells))


def test_patch_scores_with_annulus_mask_restricts_scoring_to_annulus():
    """With annulus_mask, only annulus cells count."""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    abs_map = np.ones(window.shape)
    # annulus = only top-left corner cell
    annulus_mask = np.zeros(window.shape, dtype=bool)
    annulus_mask[0, 0] = True

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=abs_map, abs_cell_map=abs_map,
        annulus_mask=annulus_mask,
    )
    for i, patch in enumerate(result["patches"]):
        covers_corner = bool(np.asarray(patch.mask, dtype=bool)[0, 0])
        if covers_corner:
            assert result["abs_scores"][i] > 0.0
        else:
            assert result["abs_scores"][i] == 0.0, f"patch {i} has no annulus overlap, expected 0"


def test_patch_scores_with_full_annulus_uses_restricted_sum():
    """With full annulus mask and all-1 cell map, score = number of annulus cells in patch."""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    ones = np.ones(window.shape)
    full_annulus = np.ones(window.shape, dtype=bool)

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=ones, abs_cell_map=ones,
        annulus_mask=full_annulus,
    )
    # With all-1 cell map and full annulus: score = sum of 1.0 per annulus cell = n_cells
    for i, patch in enumerate(result["patches"]):
        expected = float(np.sum(np.asarray(patch.mask, dtype=bool) & full_annulus))
        assert result["abs_scores"][i] == pytest.approx(expected), (
            f"patch {i}: expected sum={expected}, got {result['abs_scores'][i]}"
        )


def test_no_overlap_patch_scores_zero_with_annulus_mask():
    """Patches with no annulus intersection score 0.0."""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    abs_map = np.ones(window.shape)
    # annulus = only bottom-right corner
    annulus_mask = np.zeros(window.shape, dtype=bool)
    annulus_mask[-1, -1] = True

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=abs_map, abs_cell_map=abs_map,
        annulus_mask=annulus_mask,
    )
    for i, patch in enumerate(result["patches"]):
        covers = bool(np.asarray(patch.mask, dtype=bool)[-1, -1])
        if not covers:
            assert result["abs_scores"][i] == 0.0, f"patch {i} has no annulus cell, expected 0"
