from __future__ import annotations

import types
from pathlib import Path

from physics.swe import comparison


def test_run_physics_comparison_v2_batches_dataset_types(monkeypatch, tmp_path):
    calls = []

    def _fake_run_physics_comparison(*, cfg_module, output_dir):
        calls.append((cfg_module.DATASET_TYPE, Path(output_dir)))
        return {"ig_sanity": {"status": "ok", "passed": True}}

    fake_module = types.ModuleType("physics.swe.comparison_core")
    setattr(fake_module, "DEFAULT_RESULTS_DIR", tmp_path / "validation_results")
    setattr(fake_module, "run_physics_comparison", _fake_run_physics_comparison)
    monkeypatch.setitem(__import__("sys").modules, "physics.swe.comparison_core", fake_module)

    cfg_module = types.SimpleNamespace(
        DATASET_TYPES=["case_a", "case_b"],
        DATASET_TYPE="case_a",
        RESULTS_GROUP_K=50,
        TRACK_TOPK_K=50,
    )

    result = comparison.run_physics_comparison_v2(output_dir=tmp_path / "out", cfg_module=cfg_module)

    assert result["dataset_types"] == ["case_a", "case_b"]
    assert result["group_k"] == 50
    assert calls == [
        ("case_a", tmp_path / "out" / "k50" / "case_a"),
        ("case_b", tmp_path / "out" / "k50" / "case_b"),
    ]


def test_run_physics_comparison_v2_keeps_single_dataset_return_shape(monkeypatch, tmp_path):
    def _fake_run_physics_comparison(*, cfg_module, output_dir):
        return {"ig_sanity": {"status": "ok", "passed": True}, "value": cfg_module.DATASET_TYPE}

    fake_module = types.ModuleType("physics.swe.comparison_core")
    setattr(fake_module, "DEFAULT_RESULTS_DIR", tmp_path / "validation_results")
    setattr(fake_module, "run_physics_comparison", _fake_run_physics_comparison)
    monkeypatch.setitem(__import__("sys").modules, "physics.swe.comparison_core", fake_module)

    cfg_module = types.SimpleNamespace(
        DATASET_TYPE="case_single",
        TRACK_TOPK_K=50,
    )

    result = comparison.run_physics_comparison_v2(output_dir=tmp_path / "out", cfg_module=cfg_module)

    assert result["dataset_type"] == "case_single"
    assert result["value"] == "case_single"
    assert result["output_dir"] == str(tmp_path / "out")
