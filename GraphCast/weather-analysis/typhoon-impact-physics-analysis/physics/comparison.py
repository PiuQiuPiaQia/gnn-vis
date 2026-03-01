# -*- coding: utf-8 -*-
"""Comparison between physics and data-driven methods."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np

from physics.gate_runner import evaluate_gate, gate_result_to_payload, write_gate_report
from physics.metrics import compute_anisotropy_ratio_km


def _extract_upstream_series(result: Dict[str, Any], output_dir: Path, target_time_idx: int) -> list[float]:
    series = result.get("upstream_fraction_series")
    if isinstance(series, (list, tuple)) and len(series) >= 2:
        cleaned = []
        for x in series:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fx):
                cleaned.append(fx)
        if len(cleaned) >= 2:
            return cleaned

    sweep_path = output_dir / f"alignment_ubar_sweep_t{target_time_idx}.json"
    if sweep_path.exists():
        data = json.loads(sweep_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            pairs = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                u = item.get("U_mag", item.get("U_bar"))
                frac = item.get("upstream_fraction")
                if u is None or frac is None:
                    continue
                try:
                    fu = float(u)
                    ff = float(frac)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fu) and np.isfinite(ff):
                    pairs.append((fu, ff))
            pairs.sort(key=lambda p: p[0])
            extracted = [p[1] for p in pairs]
            if len(extracted) >= 2:
                return extracted

    return [float("nan"), float("nan")]


def run_physics_comparison_v2(output_dir: Path | None = None) -> Dict[str, Any]:
    """Compatibility wrapper with v2 hard-gate report output."""
    from physics.comparison_core import run_physics_comparison

    result = run_physics_comparison()
    jax_result = result.get("jax_result")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "validation_results"

    anisotropy_real = float("nan")
    if jax_result is not None:
        if hasattr(jax_result, "dJ_dh_signed"):
            signed = np.asarray(jax_result.dJ_dh_signed, dtype=np.float64)
            neg = np.maximum(-signed, 0.0)
            if np.sum(neg) > 0.0:
                weight_map = neg
            elif hasattr(jax_result, "S_h"):
                weight_map = np.asarray(jax_result.S_h, dtype=np.float64)
            else:
                weight_map = np.abs(signed)
        elif hasattr(jax_result, "S_abs"):
            weight_map = np.asarray(jax_result.S_abs, dtype=np.float64)
        elif hasattr(jax_result, "S_h"):
            weight_map = np.asarray(jax_result.S_h, dtype=np.float64)
        else:
            weight_map = None

        if weight_map is not None:
            anisotropy_real = compute_anisotropy_ratio_km(
                weight_map,
                np.asarray(jax_result.lat_vals, dtype=np.float64),
                np.asarray(jax_result.lon_vals, dtype=np.float64),
            )

    target_time_idx = int(getattr(jax_result, "target_time_idx", 0)) if jax_result is not None else 0
    upstream_series = _extract_upstream_series(result, output_dir, target_time_idx)
    gate = evaluate_gate(anisotropy_real=anisotropy_real, upstream_series=upstream_series)

    gate_path = output_dir / "physics_gate_report.json"
    write_gate_report(gate, gate_path)

    out = dict(result)
    out["gate_result"] = gate_result_to_payload(gate)
    out["gate_report_path"] = str(gate_path)
    # Pass through ig_sanity payload unchanged
    out["ig_sanity"] = result.get(
        "ig_sanity",
        {"status": "skipped", "reason": "not_available", "passed": None},
    )
    return out
