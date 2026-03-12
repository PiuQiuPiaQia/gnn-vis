from __future__ import annotations

import json
import matplotlib
import numpy as np

matplotlib.use("Agg")

from physics.dlmsf_patch_fd.plot_track_patch_report import write_track_patch_figures


# ---------------------------------------------------------------------------
# Dummy report factory
# ---------------------------------------------------------------------------

def _dummy_visualization_payload():
    """Minimal visualization payload matching the figure schema."""
    lat_vals = [10.0, 11.0, 12.0]
    lon_vals = [120.0, 121.0, 122.0]
    shape33 = [[0.0] * 3 for _ in range(3)]
    mask33 = [[False, True, False], [True, False, False], [False, False, True]]
    return {
        "meta": {
            "direction": "along",
            "patch_size": 3,
            "target_time_idx": 0,
            "topk_k": 50,
        },
        "overlap": {
            "lat_vals": lat_vals,
            "lon_vals": lon_vals,
            "ig_abs_map": [[0.2, 0.4, 0.1], [0.5, float("nan"), 0.3], [0.1, 0.2, 0.6]],
            "dlmsf_abs_map": [[0.1, 0.6, 0.2], [0.4, float("nan"), 0.2], [0.2, 0.3, 0.5]],
            "overlap_mask": mask33,
            "spearman_rho": 0.7,
            "iou_at_50": 0.5,
        },
        "scatter": {
            "x_patch_abs_scores": [0.6, 0.4, 0.2, 0.3, 0.5],
            "y_patch_abs_scores": [0.5, 0.3, 0.2, 0.4, 0.6],
            "spearman_rho": 0.7,
        },
        "deletion": {
            "masked_fraction": [0.1, 0.3, 0.6, 1.0],
            "high_ig_delta": [0.2, 0.4, 0.7, 1.0],
            "low_ig_delta": [0.05, 0.08, 0.1, 0.15],
            "random_mean_delta": [0.1, 0.15, 0.22, 0.3],
            "aopc_high": 0.575,
            "aopc_random": 0.1925,
            "aopc_low": 0.095,
        },
    }


def _dummy_report():
    return {
        "source_pipeline": "swe",
        "main_case": "along_p3",
        "window_size": 21,
        "core_size": 3,
        "stride": 2,
        "track_center_field": "mean_sea_level_pressure",
        "softmin_temperature": 1.0,
        "topk_k": 50,
        "cases": {
            "along_p3": {
                "metrics": {
                    "spearman_rho": 0.7,
                    "iou_topk": 0.5,
                    "topk_k": 50,
                },
                "plot": {},
                "visualization": _dummy_visualization_payload(),
                "deletion": {
                    "masked_fraction": [0.1, 0.3, 0.6, 1.0],
                    "high_ig_delta": [0.2, 0.4, 0.7, 1.0],
                    "low_ig_delta": [0.05, 0.08, 0.1, 0.15],
                    "random_mean_delta": [0.1, 0.15, 0.22, 0.3],
                    "high_ig_auc": 0.56,
                    "high_ig_aopc": 0.575,
                    "low_ig_auc": 0.09,
                    "low_ig_aopc": 0.095,
                    "random_mean_auc": 0.18,
                    "random_mean_aopc": 0.1925,
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Tests: fixed output filenames
# ---------------------------------------------------------------------------

EXPECTED_NAMES = [
    "dlmsf_along_overlap_k50_t0.png",
    "dlmsf_along_scatter_t0.png",
    "deletion_validation_along_p3.png",
]


def test_write_track_patch_figures_from_dict(tmp_path):
    outputs = write_track_patch_figures(
        _dummy_report(),
        output_dir=tmp_path,
        prefix="dlmsf_track_patch",  # prefix is now ignored for filename
        dpi=100,
    )

    assert [path.name for path in outputs] == EXPECTED_NAMES
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_track_patch_figures_from_json_path(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_dummy_report()), encoding="utf-8")

    outputs = write_track_patch_figures(report_path, prefix="any_old_prefix", dpi=100)

    assert [path.name for path in outputs] == EXPECTED_NAMES


def test_prefix_does_not_change_fixed_output_names(tmp_path):
    """prefix argument must be ignored for filenames."""
    outputs = write_track_patch_figures(
        _dummy_report(), output_dir=tmp_path, prefix="custom", dpi=100
    )
    assert outputs[0].name == "dlmsf_along_overlap_k50_t0.png"


def test_write_track_patch_figures_returns_exactly_three_files(tmp_path):
    outputs = write_track_patch_figures(_dummy_report(), output_dir=tmp_path, dpi=100)
    assert len(outputs) == 3


# ---------------------------------------------------------------------------
# Tests: annotations stay Spearman-only
# ---------------------------------------------------------------------------

def test_overlap_figure_annotation_uses_spearman_and_iou(tmp_path):
    """Overlap figure annotation should show Spearman and IoU text only."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_overlap_annotation
    text = _format_overlap_annotation(spearman_rho=0.72, iou_at_50=0.45)
    assert "Spearman" in text
    assert "0.72" in text or "ρ" in text or "rho" in text.lower()
    assert "IoU@50" in text


def test_scatter_figure_annotation_uses_spearman_only(tmp_path):
    """Scatter figure annotation should contain the Spearman label."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_scatter_annotation
    text = _format_scatter_annotation(spearman_rho=0.65)
    assert "Spearman" in text


def test_deletion_figure_annotation_does_not_contain_auc(tmp_path):
    """Deletion figure annotation must not include AUC."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_deletion_annotation
    text = _format_deletion_annotation(aopc_high=0.57, aopc_random=0.19, aopc_low=0.09)
    assert "AUC" not in text
    assert "auc" not in text.lower()
    assert "0.57" in text  # AOPC values are present


def test_overlap_figure_uses_single_subplot(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import physics.dlmsf_patch_fd.plot_track_patch_report as plot_track_patch_report

    original_subplots = plt.subplots
    subplots_calls = []

    def mock_subplots(*args, **kwargs):
        subplots_calls.append((args, kwargs))
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(plot_track_patch_report.plt, "subplots", mock_subplots)

    plot_track_patch_report.plot_track_patch_overlap_k50(
        _dummy_report(),
        tmp_path / "overlap.png",
        dpi=100,
    )

    assert subplots_calls, "subplots mock was not triggered"
    args, kwargs = subplots_calls[-1]
    nrows = kwargs.get("nrows", args[0] if len(args) > 0 else 1)
    ncols = kwargs.get("ncols", args[1] if len(args) > 1 else 1)
    assert (nrows, ncols) == (1, 1)


def test_scatter_xlabel_is_dlmsf(monkeypatch):
    """x-axis (data = dlmsf_abs) label should contain 'DLMSF', y-axis should contain 'IG'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import physics.dlmsf_patch_fd.plot_track_patch_report as plot_track_patch_report

    captured_axes = []
    original_subplots = plt.subplots

    def mock_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plot_track_patch_report.plt, "subplots", mock_subplots)

    report = {
        "main_case": "along_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {"direction": "along"},
                    "scatter": {
                        "x_patch_abs_scores": [1.0, 2.0, 3.0],
                        "y_patch_abs_scores": [1.5, 2.5, 0.5],
                        "spearman_rho": 0.5,
                    },
                }
            }
        },
    }
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "scatter.png")
        plot_track_patch_report.plot_track_patch_scatter(report, out)

    assert captured_axes, "subplots mock was not triggered"
    ax = captured_axes[-1]
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    assert "DLMSF" in xlabel, f"x-axis label should contain 'DLMSF' (data is DLMSF), got: {xlabel!r}"
    assert "IG" in ylabel, f"y-axis label should contain 'IG' (data is IG), got: {ylabel!r}"
