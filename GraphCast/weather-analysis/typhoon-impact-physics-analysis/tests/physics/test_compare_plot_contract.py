# -*- coding: utf-8 -*-
"""Behavior tests for compare plotting artifact guards."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from physics.swe.comparison_core import (
    _build_dlmsf_alignment_inputs,
    _build_swe_alignment_inputs,
    _should_emit_alignment_scatter,
    _should_emit_topk_artifacts,
)


class ComparePlotContractTest(unittest.TestCase):
    def test_swe_supplemental_only_path_keeps_topk_artifacts_but_skips_scatter(self):
        swe_result = SimpleNamespace(
            S_h=np.ones((2, 2), dtype=np.float64),
            S_uv=np.full((2, 2), 2.0, dtype=np.float64),
        )

        inputs = _build_swe_alignment_inputs(
            swe_result,
            signed_gnn_maps={},
            magnitude_gnn_maps={"uv_500": np.full((2, 2), 3.0, dtype=np.float64)},
        )

        self.assertFalse(_should_emit_alignment_scatter(inputs["main_pairs_scatter"]))
        self.assertTrue(_should_emit_topk_artifacts(inputs["overlap_pairs"]))

    def test_dlmsf_supplemental_only_path_keeps_topk_artifacts_but_skips_scatter(self):
        dlmsf_result = SimpleNamespace(
            S_abs_map=np.ones((2, 2), dtype=np.float64),
        )

        inputs = _build_dlmsf_alignment_inputs(
            dlmsf_result,
            signed_gnn_maps={},
            magnitude_gnn_maps={"uv_500": np.full((2, 2), 4.0, dtype=np.float64)},
        )

        self.assertFalse(_should_emit_alignment_scatter(inputs["scatter_pairs"]))
        self.assertTrue(_should_emit_topk_artifacts(inputs["overlap_pairs"]))


if __name__ == "__main__":
    unittest.main()
