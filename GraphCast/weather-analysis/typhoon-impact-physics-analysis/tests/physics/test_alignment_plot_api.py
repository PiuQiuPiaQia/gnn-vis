# tests/physics/test_alignment_plot_api.py
# -*- coding: utf-8 -*-
"""Unit tests for refactored alignment.py plotting API (pairs-based signatures)."""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from physics.swe.alignment import (
    AlignmentReport,
    GroupMetrics,
    plot_alignment_scatter,
    plot_topk_iou_curves,
    plot_topk_overlap_maps,
)

# pairs 结构说明：
# - plot_topk_overlap_maps / plot_topk_iou_curves：List[Tuple[group_tag, S_map, gnn_key]]
# - plot_alignment_scatter：List[Tuple[group_name, S_map, gnn_key, xlabel, ylabel]]


def _dummy_map(seed: int = 0, shape=(10, 12)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float64)


@pytest.fixture()
def lat_lon():
    return np.linspace(20.0, 30.0, 10), np.linspace(120.0, 132.0, 12)


@pytest.fixture()
def gnn_ig_maps():
    return {"z_500": _dummy_map(1), "uv_500": _dummy_map(2)}


@pytest.fixture()
def simple_report():
    report = AlignmentReport(
        target_time_idx=0, lead_time_h=6,
        patch_radius=2, patch_score_agg="mean", sigma_deg=3.0,
    )
    report.groups.append(GroupMetrics("h", 0.5, 0.01, {50: 0.3}, 100))
    return report


class TestPlotTopkOverlapMapsNewApi:
    """plot_topk_overlap_maps 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_pairs_generates_two_files(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="swe",
        )
        # 分别断言每个 group 的文件存在，避免宽松 glob 误通过
        assert len(list(tmp_path.glob("swe_overlap_h_*_t0.png"))) == 1
        assert len(list(tmp_path.glob("swe_overlap_uv_*_t0.png"))) == 1

    def test_dlmsf_pairs_generates_two_files(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        S_map = _dummy_map(5)
        pairs = [("z", S_map, "z_500"), ("uv", S_map, "uv_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="dlmsf",
        )
        # 分别断言每个 group 的文件存在
        assert len(list(tmp_path.glob("dlmsf_overlap_z_*_t0.png"))) == 1
        assert len(list(tmp_path.glob("dlmsf_overlap_uv_*_t0.png"))) == 1

    def test_missing_gnn_key_skipped(self, tmp_path, lat_lon):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        empty_maps = {}  # 无匹配
        plot_topk_overlap_maps(
            pairs, empty_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="swe",
        )
        assert list(tmp_path.glob("*.png")) == []

    def test_output_prefix_in_filename(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=2, output_dir=tmp_path, output_prefix="swe",
        )
        files = list(tmp_path.glob("swe_overlap_h_*_t2.png"))
        assert len(files) == 1


class TestPlotAlignmentScatterNewApi:
    """plot_alignment_scatter 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_scatter_creates_file(self, tmp_path, gnn_ig_maps, simple_report):
        pairs = [
            ("h",  _dummy_map(0), "z_500",  "SWE $S_h$",    "GNN IG (z₅₀₀)"),
            ("uv", _dummy_map(3), "uv_500", "SWE $S_{uv}$", "GNN IG (uv)"),
        ]
        plot_alignment_scatter(
            pairs, gnn_ig_maps, simple_report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_scatter_t0.png").exists()

    def test_dlmsf_scatter_creates_file(self, tmp_path, gnn_ig_maps):
        """DLMSF scatter 正常生成文件；空 groups 时标题注解部分不渲染，但不应崩溃。"""
        S = _dummy_map(5)
        pairs = [
            ("z",  S, "z_500",  "DLMSF $S$", "GNN IG (z₅₀₀)"),
            ("uv", S, "uv_500", "DLMSF $S$", "GNN IG (uv magnitude)"),
        ]
        report = AlignmentReport(0, 6, 2, "mean", 0.0)  # 空 groups：无 spearman 注解
        plot_alignment_scatter(
            pairs, gnn_ig_maps, report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="dlmsf",
        )
        assert (tmp_path / "dlmsf_scatter_t0.png").exists()

    def test_prefix_controls_filename(self, tmp_path, gnn_ig_maps, simple_report):
        pairs = [("h", _dummy_map(0), "z_500", "X", "Y")]
        plot_alignment_scatter(
            pairs, gnn_ig_maps, simple_report,
            target_time_idx=1, lead_time_h=12,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_scatter_t1.png").exists()
        assert not (tmp_path / "dlmsf_scatter_t1.png").exists()


class TestPlotTopkIouCurvesNewApi:
    """plot_topk_iou_curves 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_iou_creates_file(self, tmp_path, gnn_ig_maps):
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_iou_t0.png").exists()

    def test_dlmsf_iou_creates_file(self, tmp_path, gnn_ig_maps):
        S = _dummy_map(5)
        pairs = [("z", S, "z_500"), ("uv", S, "uv_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="dlmsf",
        )
        assert (tmp_path / "dlmsf_iou_t0.png").exists()

    def test_prefix_controls_filename(self, tmp_path, gnn_ig_maps):
        pairs = [("h", _dummy_map(0), "z_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=3, lead_time_h=24,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_iou_t3.png").exists()
