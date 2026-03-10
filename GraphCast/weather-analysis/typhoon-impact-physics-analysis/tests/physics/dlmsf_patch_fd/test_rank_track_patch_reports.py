from __future__ import annotations

import json
from pathlib import Path

from physics.dlmsf_patch_fd.rank_track_patch_reports import (
    collect_track_patch_rows,
    rank_track_patch_rows,
    write_ranked_rows_csv,
)


def _write_report(
    path,
    *,
    pearson: float,
    iou: float,
    spearman: float = 0.0,
    full_scalar: float,
    baseline_scalar: float,
    completeness: float,
    high_aopc: float,
    random_aopc: float,
    low_aopc: float,
    source_pipeline: str = "swe",
) -> None:
    payload = {
        "source_pipeline": source_pipeline,
        "main_case": "along_p3",
        "topq_fraction": 0.2,
        "cases": {
            "along_p3": {
                "track_scalar_full": full_scalar,
                "track_scalar_baseline": baseline_scalar,
                "ig_completeness_rel_err": completeness,
                "metrics": {
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "iou_topq": iou,
                    "topq_fraction": 0.2,
                    "topq_k": 8,
                },
                "deletion": {
                    "high_ig_aopc": high_aopc,
                    "random_mean_aopc": random_aopc,
                    "low_ig_aopc": low_aopc,
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_track_patch_rows_filters_subset_and_loads_metrics(tmp_path):
    swe_path = tmp_path / "swe_case" / "dlmsf_track_patch_alignment.json"
    other_path = tmp_path / "other_case" / "dlmsf_track_patch_alignment.json"
    swe_path.parent.mkdir(parents=True)
    other_path.parent.mkdir(parents=True)
    _write_report(
        swe_path,
        pearson=0.6,
        iou=0.4,
        full_scalar=3.0,
        baseline_scalar=1.0,
        completeness=0.1,
        high_aopc=0.5,
        random_aopc=0.2,
        low_aopc=0.1,
        source_pipeline="swe",
    )
    _write_report(
        other_path,
        pearson=0.9,
        iou=0.8,
        full_scalar=4.0,
        baseline_scalar=1.0,
        completeness=0.1,
        high_aopc=0.7,
        random_aopc=0.2,
        low_aopc=0.1,
        source_pipeline="other",
    )

    rows = collect_track_patch_rows([tmp_path], subset="swe")

    assert len(rows) == 1
    assert Path(rows[0]["report_path"]).parent.name == "swe_case"
    assert rows[0]["pearson_r"] == 0.6
    assert rows[0]["iou_topq"] == 0.4
    assert rows[0]["track_signal_abs"] == 2.0


def test_rank_track_patch_rows_applies_hard_filters_and_strong_steering(tmp_path):
    for name in ["good_1", "good_2", "weak_signal", "bad_completeness"]:
        (tmp_path / name).mkdir(parents=True)
    _write_report(
        tmp_path / "good_1" / "dlmsf_track_patch_alignment.json",
        pearson=0.7,
        iou=0.5,
        spearman=0.6,
        full_scalar=8.0,
        baseline_scalar=1.0,
        completeness=0.1,
        high_aopc=0.9,
        random_aopc=0.2,
        low_aopc=0.1,
    )
    _write_report(
        tmp_path / "good_2" / "dlmsf_track_patch_alignment.json",
        pearson=0.5,
        iou=0.4,
        spearman=0.5,
        full_scalar=6.0,
        baseline_scalar=1.0,
        completeness=0.1,
        high_aopc=0.7,
        random_aopc=0.3,
        low_aopc=0.2,
    )
    _write_report(
        tmp_path / "weak_signal" / "dlmsf_track_patch_alignment.json",
        pearson=0.95,
        iou=0.9,
        spearman=0.9,
        full_scalar=2.0,
        baseline_scalar=1.5,
        completeness=0.1,
        high_aopc=0.8,
        random_aopc=0.2,
        low_aopc=0.1,
    )
    _write_report(
        tmp_path / "bad_completeness" / "dlmsf_track_patch_alignment.json",
        pearson=0.99,
        iou=0.99,
        spearman=0.99,
        full_scalar=10.0,
        baseline_scalar=1.0,
        completeness=0.25,
        high_aopc=0.8,
        random_aopc=0.2,
        low_aopc=0.1,
    )

    rows = collect_track_patch_rows([tmp_path], subset="swe")
    ranked = rank_track_patch_rows(
        rows,
        completeness_threshold=0.2,
        steering_top_fraction=0.5,
        top_n=3,
    )

    assert ranked["metadata"]["count_total"] == 4
    assert ranked["metadata"]["count_hard_filtered"] == 3
    assert ranked["metadata"]["count_strong_steering"] == 2
    assert [Path(row["report_path"]).parent.name for row in ranked["rows"]] == ["good_1", "good_2"]
    assert ranked["rows"][0]["rank"] == 1
    assert ranked["rows"][0]["score"] >= ranked["rows"][1]["score"]


def test_write_ranked_rows_csv_outputs_header_and_rows(tmp_path):
    rows = [
        {
            "rank": 1,
            "report_name": "case_a.json",
            "report_path": "/tmp/case_a.json",
            "score": 1.2,
            "pearson_r": 0.6,
            "spearman_rho": 0.5,
            "iou_topq": 0.4,
            "track_signal_abs": 3.0,
            "ig_completeness_rel_err": 0.1,
            "high_ig_aopc": 0.6,
            "random_mean_aopc": 0.2,
            "low_ig_aopc": 0.1,
            "deletion_advantage": 0.4,
        }
    ]

    output_path = write_ranked_rows_csv(rows, tmp_path / "ranked.csv")

    text = output_path.read_text(encoding="utf-8")
    assert "report_name" in text
    assert "case_a.json" in text


def test_collect_rows_reads_wind_along_signed_metrics(tmp_path):
    """collect_track_patch_rows must extract signed_spearman from wind_along_signed case."""
    report_path = tmp_path / "swe_case" / "dlmsf_track_patch_alignment.json"
    report_path.parent.mkdir(parents=True)
    _write_report(
        report_path,
        pearson=0.6,
        iou=0.4,
        full_scalar=3.0,
        baseline_scalar=1.0,
        completeness=0.1,
        high_aopc=0.5,
        random_aopc=0.2,
        low_aopc=0.1,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["cases"]["wind_along_signed_p3"] = {
        "signed_spearman": 0.72,
        "sign_agreement_at_30": 0.61,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    rows = collect_track_patch_rows([tmp_path], subset="swe")

    assert len(rows) == 1
    assert rows[0]["signed_spearman"] == 0.72
    assert rows[0]["sign_agreement_at_30"] == 0.61
