# -*- coding: utf-8 -*-
"""Gate runner for physics-aware gating."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import json
import numpy as np


@dataclass(frozen=True)
class GateResult:
    passed: bool
    ok_anisotropy: bool
    ok_upstream: bool
    anisotropy_real: float
    upstream_series: list[float]
    anisotropy_threshold: float
    upstream_final_threshold: float


def gate_result_to_payload(result: GateResult) -> dict:
    payload = asdict(result)
    for key, value in list(payload.items()):
        if isinstance(value, float) and not np.isfinite(value):
            payload[key] = None
        elif isinstance(value, list):
            payload[key] = [None if isinstance(x, float) and not np.isfinite(x) else x for x in value]
    return payload


def is_strictly_monotonic(values: Sequence[float], eps: float = 1e-6) -> bool:
    """
    Check if values are strictly monotonic increasing.
    
    Args:
        values: Sequence of numeric values
        eps: Minimum difference required between consecutive values
    
    Returns:
        True if all consecutive differences > eps, False otherwise
    """
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    if len(values) < 2:
        return True

    arr = np.asarray(values, dtype=np.float64)
    diffs = np.diff(arr)

    return bool(np.all(diffs > eps))


def evaluate_gate(
    anisotropy_real: float,
    upstream_series: Sequence[float],
    anisotropy_threshold: float = 1.3,
    upstream_final_threshold: float = 0.95,
    eps: float = 1e-6,
) -> GateResult:
    series = np.asarray(upstream_series, dtype=np.float64)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("upstream_series must be a non-empty 1D sequence")

    ok_anisotropy = bool(np.isfinite(anisotropy_real) and anisotropy_real > anisotropy_threshold)
    finite_series = bool(np.all(np.isfinite(series)))
    sufficient_points = bool(series.size >= 2)
    ok_upstream = bool(
        sufficient_points
        and finite_series
        and is_strictly_monotonic(series.tolist(), eps=eps)
        and float(series[-1]) >= upstream_final_threshold
    )
    return GateResult(
        passed=bool(ok_anisotropy and ok_upstream),
        ok_anisotropy=ok_anisotropy,
        ok_upstream=ok_upstream,
        anisotropy_real=float(anisotropy_real),
        upstream_series=[float(x) for x in series],
        anisotropy_threshold=float(anisotropy_threshold),
        upstream_final_threshold=float(upstream_final_threshold),
    )


def write_gate_report(result: GateResult, output_path: Path) -> None:
    payload = gate_result_to_payload(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
