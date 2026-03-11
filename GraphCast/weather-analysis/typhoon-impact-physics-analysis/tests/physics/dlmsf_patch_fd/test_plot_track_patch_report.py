from __future__ import annotations

import json
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")

from physics.dlmsf_patch_fd.plot_track_patch_report import write_track_patch_figures


# ---------------------------------------------------------------------------
# Dummy report factory
# ---------------------------------------------------------------------------

def _dummy_visualization_payload():
    """Minimal visualization payload matching the four-figure schema."""
    lat_vals = [10.0, 11.0, 12.0]
    lon_vals = [120.0, 121.0, 122.0]
    shape33 = [[0.0] * 3 for _ in range(3)]
    mask33 = [[False, True, False], [True, False, False], [False, False, True]]
    return {
        "meta": {
            "direction": "along",
            "patch_size": 3,
            "target_time_idx": 0,
            "topq_fraction": 0.2,
        },
        "overlap": {
            "lat_vals": lat_vals,
            "lon_vals": lon_vals,
            "ig_abs_map": [[0.2, 0.4, 0.1], [0.5, float("nan"), 0.3], [0.1, 0.2, 0.6]],
            "dlmsf_abs_map": [[0.1, 0.6, 0.2], [0.4, float("nan"), 0.2], [0.2, 0.3, 0.5]],
            "overlap_mask": mask33,
            "spearman_rho": 0.7,
            "iou_at_20": 0.5,
        },
        "scatter": {
            "x_patch_abs_scores": [0.6, 0.4, 0.2, 0.3, 0.5],
            "y_patch_abs_scores": [0.5, 0.3, 0.2, 0.4, 0.6],
            "spearman_rho": 0.7,
        },
        "sign_map": {
            "lat_vals": lat_vals,
            "lon_vals": lon_vals,
            "sign_class_map": [[0, 1, 2], [3, 0, 1], [2, 3, 0]],
            "sign_agreement_at_20": 0.75,
            "overlap_mask": mask33,
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
        "topq_fraction": 0.2,
        "cases": {
            "along_p3": {
                "metrics": {
                    "pearson_r": 0.6,
                    "spearman_rho": 0.7,
                    "iou_topq": 0.5,
                    "topq_fraction": 0.2,
                    "topq_k": 4,
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
# Tests: four fixed output filenames
# ---------------------------------------------------------------------------

EXPECTED_NAMES = [
    "dlmsf_along_overlap_q20_t0.png",
    "dlmsf_along_scatter_t0.png",
    "deletion_validation_along_p3.png",
    "dlmsf_along_sign_map_t0.png",
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
    assert outputs[0].name == "dlmsf_along_overlap_q20_t0.png"


def test_write_track_patch_figures_returns_exactly_four_files(tmp_path):
    outputs = write_track_patch_figures(_dummy_report(), output_dir=tmp_path, dpi=100)
    assert len(outputs) == 4


# ---------------------------------------------------------------------------
# Tests: Pearson must not appear in rendered text annotations
# ---------------------------------------------------------------------------

def test_overlap_figure_annotation_does_not_contain_pearson(tmp_path):
    """Overlap figure must only show Spearman and IoU, never Pearson."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_overlap_annotation
    text = _format_overlap_annotation(spearman_rho=0.72, iou_at_20=0.45)
    assert "Pearson" not in text
    assert "pearson" not in text.lower()
    assert "0.72" in text or "ρ" in text or "rho" in text.lower()


def test_scatter_figure_annotation_does_not_contain_pearson(tmp_path):
    """Scatter figure must only show Spearman, never Pearson."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_scatter_annotation
    text = _format_scatter_annotation(spearman_rho=0.65)
    assert "Pearson" not in text
    assert "pearson" not in text.lower()


def test_deletion_figure_annotation_does_not_contain_auc(tmp_path):
    """Deletion figure annotation must not include AUC."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_deletion_annotation
    text = _format_deletion_annotation(aopc_high=0.57, aopc_random=0.19, aopc_low=0.09)
    assert "AUC" not in text
    assert "auc" not in text.lower()
    assert "0.57" in text  # AOPC values are present


def test_sign_map_annotation_does_not_contain_pearson(tmp_path):
    """Sign map annotation must only show Sign agreement, never Pearson."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import _format_sign_map_annotation
    text = _format_sign_map_annotation(sign_agreement_at_20=0.80)
    assert "Pearson" not in text
    assert "pearson" not in text.lower()
    assert "0.80" in text or "80" in text


def test_plot_sign_map_uses_wind_case_when_present(tmp_path):
    """plot_track_patch_sign_map reads wind_case, not main_case."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import plot_track_patch_sign_map

    nlat, nlon = 5, 5
    sign_map = np.zeros((nlat, nlon), dtype=int)
    sign_map[2, 2] = 1

    report = {
        "main_case": "along_p3",
        "wind_case": "wind_along_signed_p3",
        "cases": {
            "along_p3": {
                "visualization": None,
            },
            "wind_along_signed_p3": {
                "visualization": {
                    "meta": {
                        "direction": "along", "patch_size": 3,
                        "target_time_idx": 1, "topq_fraction": 0.2,
                        "source": "wind_along_signed",
                    },
                    "sign_map": {
                        "lat_vals": np.linspace(20, 25, nlat).tolist(),
                        "lon_vals": np.linspace(120, 125, nlon).tolist(),
                        "sign_class_map": sign_map.tolist(),
                        "overlap_mask": (sign_map > 0).tolist(),
                        "sign_agreement_at_20": 0.75,
                    },
                    "scatter": {
                        "x_patch_abs_scores": [1.0],
                        "y_patch_abs_scores": [1.0],
                        "spearman_rho": 0.5,
                    },
                },
            },
        },
    }

    out = tmp_path / "sign_map.png"
    result = plot_track_patch_sign_map(report, out)
    assert result is not None, "Should produce a figure from wind_case"
    assert out.exists(), "PNG file should be created"


def test_plot_sign_map_falls_back_to_main_case_if_wind_case_missing(tmp_path):
    """If wind_case is absent, falls back to main_case."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import plot_track_patch_sign_map

    nlat, nlon = 4, 4
    sign_map = np.ones((nlat, nlon), dtype=int)

    report = {
        "main_case": "along_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {
                        "direction": "along", "patch_size": 3,
                        "target_time_idx": 1, "topq_fraction": 0.2,
                        "source": "ig",
                    },
                    "sign_map": {
                        "lat_vals": np.linspace(20, 24, nlat).tolist(),
                        "lon_vals": np.linspace(120, 124, nlon).tolist(),
                        "sign_class_map": sign_map.tolist(),
                        "overlap_mask": (sign_map > 0).tolist(),
                        "sign_agreement_at_20": 0.6,
                    },
                    "scatter": {
                        "x_patch_abs_scores": [1.0],
                        "y_patch_abs_scores": [1.0],
                        "spearman_rho": 0.4,
                    },
                },
            },
        },
    }

    out = tmp_path / "sign_map_fallback.png"
    result = plot_track_patch_sign_map(report, out)
    assert result is not None, "Should produce a figure from main_case fallback"
    assert out.exists()


def test_plot_sign_map_falls_back_when_wind_case_not_in_cases(tmp_path):
    """wind_case key present in report but case not in cases → fall back to main_case."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import plot_track_patch_sign_map

    nlat, nlon = 4, 4
    sign_map = np.ones((nlat, nlon), dtype=int)

    report = {
        "main_case": "along_p3",
        "wind_case": "wind_along_signed_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {
                        "direction": "along", "patch_size": 3,
                        "target_time_idx": 1, "topq_fraction": 0.2, "source": "ig",
                    },
                    "sign_map": {
                        "lat_vals": np.linspace(20, 24, nlat).tolist(),
                        "lon_vals": np.linspace(120, 124, nlon).tolist(),
                        "sign_class_map": sign_map.tolist(),
                        "overlap_mask": (sign_map > 0).tolist(),
                        "sign_agreement_at_20": 0.6,
                    },
                    "scatter": {
                        "x_patch_abs_scores": [1.0], "y_patch_abs_scores": [1.0], "spearman_rho": 0.4,
                    },
                },
            },
        },
    }

    out = tmp_path / "sign_map_fallback2.png"
    result = plot_track_patch_sign_map(report, out)
    assert result is not None, "Should fall back to main_case and produce figure"
    assert out.exists()


def test_plot_sign_map_falls_back_when_wind_case_has_no_sign_map(tmp_path):
    """wind_case case exists but has no sign_map → fall back to main_case."""
    from physics.dlmsf_patch_fd.plot_track_patch_report import plot_track_patch_sign_map

    nlat, nlon = 4, 4
    sign_map = np.ones((nlat, nlon), dtype=int)

    report = {
        "main_case": "along_p3",
        "wind_case": "wind_along_signed_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {
                        "direction": "along", "patch_size": 3,
                        "target_time_idx": 1, "topq_fraction": 0.2, "source": "ig",
                    },
                    "sign_map": {
                        "lat_vals": np.linspace(20, 24, nlat).tolist(),
                        "lon_vals": np.linspace(120, 124, nlon).tolist(),
                        "sign_class_map": sign_map.tolist(),
                        "overlap_mask": (sign_map > 0).tolist(),
                        "sign_agreement_at_20": 0.6,
                    },
                    "scatter": {
                        "x_patch_abs_scores": [1.0], "y_patch_abs_scores": [1.0], "spearman_rho": 0.4,
                    },
                },
            },
            "wind_along_signed_p3": {
                "visualization": {
                    "meta": {
                        "direction": "along", "patch_size": 3,
                        "target_time_idx": 1, "topq_fraction": 0.2, "source": "wind",
                    },
                    "scatter": {
                        "x_patch_abs_scores": [1.0], "y_patch_abs_scores": [1.0], "spearman_rho": 0.5,
                    },
                },
            },
        },
    }

    out = tmp_path / "sign_map_fallback3.png"
    result = plot_track_patch_sign_map(report, out)
    assert result is not None, "Should fall back to main_case when sign_map missing"
    assert out.exists()
