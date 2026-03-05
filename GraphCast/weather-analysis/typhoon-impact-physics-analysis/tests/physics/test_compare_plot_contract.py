# -*- coding: utf-8 -*-
"""Contract tests for compare plotting artifacts."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


def _called_function_names_from_source(function_name: str) -> set[str]:
    root = Path(__file__).resolve().parents[2]
    src_path = root / "physics" / "swe" / "comparison_core.py"
    source = src_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    target = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target = node
            break
    if target is None:
        raise AssertionError(f"Function not found: {function_name}")

    names: set[str] = set()
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


class ComparePlotContractTest(unittest.TestCase):
    def test_compare_pipeline_does_not_emit_standalone_swe_heatmaps(self):
        """compare 输出应仅保留对比图，不再生成单独 SWE 热力图。"""
        called = _called_function_names_from_source("run_physics_comparison")
        self.assertNotIn("plot_sensitivity_heatmaps", called)

    def test_compare_pipeline_uses_topk_overlap_only_not_rank_panels(self):
        """compare 输出应移除 rank/rank-comparison 面板，仅保留 Top-K 相关图。"""
        called = _called_function_names_from_source("run_physics_comparison")
        self.assertIn("plot_topk_overlap_maps", called)
        self.assertNotIn("plot_comparison_panels", called)


if __name__ == "__main__":
    unittest.main()
