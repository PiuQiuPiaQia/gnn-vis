# -*- coding: utf-8 -*-
"""Comparison between physics and data-driven methods."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

def run_physics_comparison_v2(output_dir: Path | None = None) -> Dict[str, Any]:
    """Compatibility wrapper for comparison pipeline results."""
    from physics.comparison_core import run_physics_comparison

    result = run_physics_comparison()

    out = dict(result)
    # Pass through ig_sanity payload unchanged
    out["ig_sanity"] = result.get(
        "ig_sanity",
        {"status": "skipped", "reason": "not_available", "passed": None},
    )
    return out
