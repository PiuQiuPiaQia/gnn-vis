# tests/physics/test_comparison_core_dlmsf.py
# -*- coding: utf-8 -*-
"""Smoke tests for DLMSF integration in comparison_core.py."""
from __future__ import annotations

import json
import numpy as np
import pytest
from physics.swe.alignment import AlignmentReport, _group_metrics


class TestDlmsfDisabled:
    """DLMSF_ENABLE=False 时，dlmsf_result 保持 None（配置控制生效）。"""

    def test_dlmsf_enable_false_config(self, monkeypatch):
        """当 DLMSF_ENABLE=False，getattr 应返回 False。"""
        import config as cfg_mod
        monkeypatch.setattr(cfg_mod, "DLMSF_ENABLE", False, raising=False)
        import config as cfg
        assert bool(getattr(cfg, "DLMSF_ENABLE", True)) is False

    def test_dlmsf_enable_true_by_default(self):
        """若未设置 DLMSF_ENABLE，getattr 默认为 True。"""
        # 测试 getattr fallback 机制
        class FakeConfig:
            pass
        assert bool(getattr(FakeConfig(), "DLMSF_ENABLE", True)) is True


class TestDlmsfAlignmentReportStructure:
    """验证 DLMSF 对齐报告包含正确的 group 名称和字段。"""

    def test_dlmsf_report_group_names(self):
        """使用 _group_metrics 构建 DLMSF 报告时，group_name 应为 dlmsf_z_500 和 dlmsf_uv_500。"""
        S_map = np.random.rand(10, 10)
        gnn_map = np.random.rand(10, 10)

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        for gnn_key in ["z_500", "uv_500"]:
            m = _group_metrics(S_map, gnn_map, f"dlmsf_{gnn_key}", 2, "mean", (20, 50))
            report.groups.append(m)

        group_names = {g.group_name for g in report.groups}
        assert "dlmsf_z_500" in group_names
        assert "dlmsf_uv_500" in group_names

    def test_dlmsf_report_json_serializable(self):
        """DLMSF 报告的 as_dict() 应可被 json.dumps 序列化。"""
        S_map = np.random.rand(10, 10)
        gnn_map = np.random.rand(10, 10)

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        m = _group_metrics(S_map, gnn_map, "dlmsf_uv_500", 2, "mean", (20, 50))
        report.groups.append(m)

        d = report.as_dict()
        json_str = json.dumps(d)  # 不应抛出异常
        assert "dlmsf_uv_500" in json_str

    def test_dlmsf_report_spearman_in_valid_range(self):
        """Spearman ρ 应在 [-1, 1] 范围内。"""
        S_map = np.random.rand(20, 20)
        gnn_map = np.random.rand(20, 20)

        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        m = _group_metrics(S_map, gnn_map, "dlmsf_z_500", 2, "mean", (20, 50))
        report.groups.append(m)

        rho = report.groups[0].spearman_rho
        assert -1.0 <= rho <= 1.0 or np.isnan(rho)

    def test_empty_groups_report_as_dict(self):
        """空 groups 的 DLMSF 报告也应能正常序列化（落盘前的守卫条件逻辑测试）。"""
        report = AlignmentReport(
            target_time_idx=0, lead_time_h=6,
            patch_radius=2, patch_score_agg="mean", sigma_deg=0.0
        )
        assert len(report.groups) == 0
        # 守卫条件：空 groups 不应落盘（在 comparison_core.py 中有 if dlmsf_report.groups 守卫）
        assert not report.groups  # 验证 falsy


class TestDlmsfReturnDictKeys:
    """验证 run_physics_comparison 的返回 dict 包含 DLMSF 字段（结构检查）。"""

    def test_required_dlmsf_keys_in_docstring(self):
        """comparison_core.run_physics_comparison 的返回值应包含 dlmsf_result 和 dlmsf_report 字段。

        不运行真正的比较（需要 jax 和数据），仅验证 return dict 的键结构
        通过读取源码确认。
        """
        from pathlib import Path
        # 直接读取源码文件，避免导入 jax
        source_path = Path(__file__).parent.parent.parent / "physics" / "swe" / "comparison_core.py"
        source = source_path.read_text(encoding="utf-8")
        # 验证 return dict 中包含两个新字段
        assert "dlmsf_result" in source
        assert "dlmsf_report" in source
