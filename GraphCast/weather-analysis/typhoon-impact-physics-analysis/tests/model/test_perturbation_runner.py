# -*- coding: utf-8 -*-
"""Tests for perturbation runner _resolve_target_spec helper."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from model.perturbation.runner import _resolve_target_spec, TargetSpec


class TestResolveTargetSpec:
    """Tests for _resolve_target_spec helper."""

    def test_supports_single_target_context(self):
        """Should resolve from single-target context (target_var, base_value)."""
        context = SimpleNamespace(
            target_var="mean_sea_level_pressure",
            base_value=1013.25,
        )
        spec = _resolve_target_spec(context)
        
        assert isinstance(spec, TargetSpec)
        assert spec.target_vars == ["mean_sea_level_pressure"]
        assert spec.base_values == {"mean_sea_level_pressure": 1013.25}

    def test_supports_multi_target_context(self):
        """Should resolve from multi-target context (target_vars, base_values)."""
        context = SimpleNamespace(
            target_vars=["var1", "var2"],
            base_values={"var1": 100.0, "var2": 200.0},
        )
        spec = _resolve_target_spec(context)
        
        assert isinstance(spec, TargetSpec)
        assert spec.target_vars == ["var1", "var2"]
        assert spec.base_values == {"var1": 100.0, "var2": 200.0}

    def test_raises_on_missing_target_info(self):
        """Should raise ValueError when target info is missing."""
        context = SimpleNamespace(
            # Missing target_var/base_value and target_vars/base_values
        )
        with pytest.raises(ValueError) as exc_info:
            _resolve_target_spec(context)
        assert "Cannot resolve target specification" in str(exc_info.value)

    def test_multi_target_takes_precedence(self):
        """Multi-target attributes should take precedence if both exist."""
        context = SimpleNamespace(
            target_var="single_var",
            base_value=50.0,
            target_vars=["multi1", "multi2"],
            base_values={"multi1": 10.0, "multi2": 20.0},
        )
        spec = _resolve_target_spec(context)
        
        # Multi-target takes precedence
        assert spec.target_vars == ["multi1", "multi2"]
        assert spec.base_values == {"multi1": 10.0, "multi2": 20.0}
