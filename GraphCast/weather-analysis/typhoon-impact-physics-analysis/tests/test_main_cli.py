# -*- coding: utf-8 -*-
from __future__ import annotations

import types

import pytest

import main


def test_main_runs_compare_flow_without_subcommand(monkeypatch):
    called = {"n": 0}

    def _fake_run():
        called["n"] += 1

    fake_module = types.ModuleType("physics.comparison")
    fake_module.run_physics_comparison_v2 = _fake_run
    monkeypatch.setitem(__import__("sys").modules, "physics.comparison", fake_module)

    assert main.main([]) == 0
    assert called["n"] == 1


def test_main_rejects_legacy_compare_subcommand(monkeypatch):
    fake_module = types.ModuleType("physics.comparison")
    fake_module.run_physics_comparison_v2 = lambda: None
    monkeypatch.setitem(__import__("sys").modules, "physics.comparison", fake_module)

    with pytest.raises(SystemExit):
        main.main(["compare"])
