from __future__ import annotations

import json
from pathlib import Path

from physics.dlmsf_patch_fd import patch_comparison

from physics.dlmsf_patch_fd.rank_track_patch_reports import (
    collect_track_patch_rows,
    rank_track_patch_rows,
    write_ranked_rows_csv,
)


def _write_report(
    path,
    *,
    iou: float,
    spearman: float = 0.0,
    high_aopc: float,
    random_aopc: float,
    low_aopc: float,
    source_pipeline: str = "swe",
) -> None:
    payload = {
        "source_pipeline": source_pipeline,
        "main_case": "along_p3",
        "topk_k": 50,
        "cases": {
            "along_p3": {
                "metrics": {
                    "spearman_rho": spearman,
                    "iou_topk": iou,
                    "topk_k": 50,
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
        iou=0.4,
        spearman=0.6,
        high_aopc=0.5,
        random_aopc=0.2,
        low_aopc=0.1,
        source_pipeline="swe",
    )
    _write_report(
        other_path,
        iou=0.8,
        spearman=0.9,
        high_aopc=0.7,
        random_aopc=0.2,
        low_aopc=0.1,
        source_pipeline="other",
    )

    rows = collect_track_patch_rows([tmp_path], subset="swe")

    assert len(rows) == 1
    assert Path(rows[0]["report_path"]).parent.name == "swe_case"
    assert rows[0]["spearman_rho"] == 0.6
    assert rows[0]["iou_topk"] == 0.4


def test_rank_track_patch_rows_applies_hard_filters(tmp_path):
    for name in ["good_1", "good_2", "weak_signal", "bad_deletion"]:
        (tmp_path / name).mkdir(parents=True)
    _write_report(
        tmp_path / "good_1" / "dlmsf_track_patch_alignment.json",
        iou=0.5,
        spearman=0.6,
        high_aopc=0.9,
        random_aopc=0.2,
        low_aopc=0.1,
    )
    _write_report(
        tmp_path / "good_2" / "dlmsf_track_patch_alignment.json",
        iou=0.4,
        spearman=0.5,
        high_aopc=0.7,
        random_aopc=0.3,
        low_aopc=0.2,
    )
    _write_report(
        tmp_path / "weak_signal" / "dlmsf_track_patch_alignment.json",
        iou=0.9,
        spearman=0.9,
        high_aopc=0.8,
        random_aopc=0.2,
        low_aopc=0.1,
    )
    # bad_deletion: high_aopc < random_aopc → hard filter rejects
    _write_report(
        tmp_path / "bad_deletion" / "dlmsf_track_patch_alignment.json",
        iou=0.99,
        spearman=0.99,
        high_aopc=0.1,
        random_aopc=0.5,
        low_aopc=0.3,
    )

    rows = collect_track_patch_rows([tmp_path], subset="swe")
    ranked = rank_track_patch_rows(
        rows,
        top_n=3,
    )

    assert ranked["metadata"]["count_total"] == 4
    assert ranked["metadata"]["count_hard_filtered"] == 3  # bad_deletion rejected
    assert [Path(row["report_path"]).parent.name for row in ranked["rows"]] == [
        "weak_signal", "good_1", "good_2"
    ]
    assert ranked["rows"][0]["rank"] == 1
    assert ranked["rows"][0]["score"] >= ranked["rows"][1]["score"]


def test_write_ranked_rows_csv_outputs_header_and_rows(tmp_path):
    rows = [
        {
            "rank": 1,
            "report_name": "case_a.json",
            "report_path": "/tmp/case_a.json",
            "score": 1.2,
            "spearman_rho": 0.5,
            "iou_topk": 0.4,
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
    assert "spearman_rho" in text


def test_patch_comparison_source_has_no_removed_metric_tokens():
    source = Path(patch_comparison.__file__).read_text(encoding="utf-8")
    removed_tokens = (
        "pear" + "son_r",
        "pear" + "son_pval",
        "pear" + "son=",
    )
    for token in removed_tokens:
        assert token not in source
