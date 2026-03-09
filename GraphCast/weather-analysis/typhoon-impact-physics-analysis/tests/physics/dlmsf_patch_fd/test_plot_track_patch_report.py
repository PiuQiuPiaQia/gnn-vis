from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from physics.dlmsf_patch_fd.plot_track_patch_report import write_track_patch_figures


def _dummy_report():
    base_map = [
        [1008.0, 1007.0, 1006.5],
        [1007.5, 1006.0, 1005.5],
        [1008.5, 1007.2, 1006.8],
    ]
    ig_map = [
        [0.2, 0.4, 0.1],
        [0.5, float("nan"), 0.3],
        [0.1, 0.2, 0.6],
    ]
    dlmsf_map = [
        [0.1, 0.6, 0.2],
        [0.4, float("nan"), 0.2],
        [0.2, 0.3, 0.5],
    ]
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
                "plot": {
                    "environment_field": "mean_sea_level_pressure",
                    "lat_vals": [10.0, 11.0, 12.0],
                    "lon_vals": [120.0, 121.0, 122.0],
                    "core_mask": [
                        [False, False, False],
                        [False, True, False],
                        [False, False, False],
                    ],
                    "environment_map": base_map,
                    "ig_abs_map": ig_map,
                    "dlmsf_abs_map": dlmsf_map,
                    "ig_topq_mask": [
                        [False, True, False],
                        [True, False, False],
                        [False, False, True],
                    ],
                    "dlmsf_topq_mask": [
                        [False, True, False],
                        [False, False, False],
                        [False, True, True],
                    ],
                    "overlap_mask": [
                        [False, True, False],
                        [False, False, False],
                        [False, False, True],
                    ],
                    "union_mask": [
                        [False, True, False],
                        [True, False, False],
                        [False, True, True],
                    ],
                    "topq_fraction": 0.2,
                    "topq_k": 4,
                },
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


def test_write_track_patch_figures_from_dict(tmp_path):
    outputs = write_track_patch_figures(
        _dummy_report(),
        output_dir=tmp_path,
        prefix="dlmsf_track_patch",
        dpi=100,
    )

    assert [path.name for path in outputs] == [
        "dlmsf_track_patch_main_case.png",
        "dlmsf_track_patch_deletion.png",
    ]
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_track_patch_figures_from_json_path(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_dummy_report()), encoding="utf-8")

    outputs = write_track_patch_figures(report_path, prefix="track_patch_plot", dpi=100)

    assert [path.name for path in outputs] == [
        "track_patch_plot_main_case.png",
        "track_patch_plot_deletion.png",
    ]
