# -*- coding: utf-8 -*-
"""IG perturbation sanity check helpers."""
from __future__ import annotations

import json
import numpy as np
import xarray
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def sample_random_grid_indices(
    n_lat: int,
    n_lon: int,
    k: int,
    excluded: Set[Tuple[int, int]],
    seed: int,
) -> List[Tuple[int, int]]:
    """Sample k unique random grid indices, excluding given set.
    
    Args:
        n_lat: Number of latitude points.
        n_lon: Number of longitude points.
        k: Number of indices to sample.
        excluded: Set of (lat_idx, lon_idx) to exclude.
        seed: Random seed for reproducibility.
    
    Returns:
        List of (lat_idx, lon_idx) tuples, may be fewer than k if insufficient points.
    """
    total_points = n_lat * n_lon
    available = total_points - len(excluded)
    actual_k = min(k, available)
    
    if actual_k <= 0:
        return []
    
    rng = np.random.default_rng(seed)
    
    # Generate candidates efficiently
    result: List[Tuple[int, int]] = []
    max_attempts = total_points * 10  # Safety limit
    attempts = 0
    
    while len(result) < actual_k and attempts < max_attempts:
        lat_idx = int(rng.integers(0, n_lat))
        lon_idx = int(rng.integers(0, n_lon))
        pair = (lat_idx, lon_idx)
        if pair not in excluded and pair not in result:
            result.append(pair)
        attempts += 1
    
    # Fallback: systematic search if random failed
    if len(result) < actual_k:
        for lat_idx in range(n_lat):
            for lon_idx in range(n_lon):
                pair = (lat_idx, lon_idx)
                if pair not in excluded and pair not in result:
                    result.append(pair)
                    if len(result) >= actual_k:
                        break
            if len(result) >= actual_k:
                break
    
    return result


def build_point_score_da(
    ig_maps_by_var: Dict[str, np.ndarray],
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> xarray.DataArray:
    """Build point score DataArray from IG maps.
    
    Args:
        ig_maps_by_var: Dict mapping variable name to IG attribution map (lat, lon).
        lat_vals: Latitude coordinate values.
        lon_vals: Longitude coordinate values.
    
    Returns:
        DataArray with dims (lat, lon), name 'ig_candidate_score'.
    
    Raises:
        ValueError: If any map has mismatched shape (no silent pass).
    """
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    score_map = np.zeros((n_lat, n_lon), dtype=np.float64)
    
    mismatched_vars = []
    for var_name, var_map in ig_maps_by_var.items():
        arr = np.abs(np.asarray(var_map, dtype=np.float64))
        if arr.shape == (n_lat, n_lon):
            score_map += arr
        else:
            mismatched_vars.append(f"{var_name}(expected=({n_lat},{n_lon}), got={arr.shape})")
    
    if mismatched_vars:
        raise ValueError(
            f"IG maps have mismatched shapes: {', '.join(mismatched_vars)}"
        )
    
    return xarray.DataArray(
        score_map,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="ig_candidate_score",
    )


def compute_sanity_summary(
    topk_deltas: List[float],
    rand_deltas: List[float],
    min_lift_ratio: float,
) -> Dict[str, Any]:
    """Compute sanity check summary statistics.
    
    Args:
        topk_deltas: List of delta values for top-k candidates.
        rand_deltas: List of delta values for random candidates.
        min_lift_ratio: Minimum lift ratio threshold to pass.
    
    Returns:
        Dict with topk_mean, rand_mean, lift_ratio, passed, reason (if failed).
        
    Failure reasons:
        - 'empty_inputs': either list is empty
        - 'non_finite_input': any non-finite value in either list
        - 'rand_mean_too_small': rand_mean <= 1e-12
        
    Pass rule: (topk_mean > rand_mean) AND (lift_ratio >= min_lift_ratio)
    """
    # Fail if either list empty
    if not topk_deltas or not rand_deltas:
        return {
            "topk_mean": float("nan"),
            "rand_mean": float("nan"),
            "lift_ratio": float("nan"),
            "passed": False,
            "reason": "empty_inputs",
        }
    
    # Fail if any non-finite in either list
    topk_arr = np.asarray(topk_deltas, dtype=np.float64)
    rand_arr = np.asarray(rand_deltas, dtype=np.float64)
    
    if not np.all(np.isfinite(topk_arr)) or not np.all(np.isfinite(rand_arr)):
        return {
            "topk_mean": float("nan"),
            "rand_mean": float("nan"),
            "lift_ratio": float("nan"),
            "passed": False,
            "reason": "non_finite_input",
        }
    
    # Compute means directly from finite arrays
    topk_mean = float(np.mean(topk_arr))
    rand_mean = float(np.mean(rand_arr))
    
    # Fail if rand_mean too small
    if rand_mean <= 1e-12:
        return {
            "topk_mean": topk_mean,
            "rand_mean": rand_mean,
            "lift_ratio": float("inf") if topk_mean > 0 else float("nan"),
            "passed": False,
            "reason": "rand_mean_too_small",
        }
    
    lift_ratio = abs(topk_mean) / abs(rand_mean)
    
    # Pass rule: topk_mean > rand_mean AND lift_ratio >= min_lift_ratio
    passed = (topk_mean > rand_mean) and (lift_ratio >= min_lift_ratio)
    
    result: Dict[str, Any] = {
        "topk_mean": topk_mean,
        "rand_mean": rand_mean,
        "lift_ratio": lift_ratio,
        "passed": passed,
    }
    if not passed:
        result["reason"] = "lift_ratio_not_met" if lift_ratio < min_lift_ratio else "topk_not_greater"
    
    return result


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize non-finite floats to None for JSON serialization."""
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def write_ig_sanity_report(payload: Dict[str, Any], output_path: Path) -> None:
    """Write IG sanity report to JSON file.
    
    Args:
        payload: Dict to serialize (non-finite floats sanitized to None).
        output_path: Path to write JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(payload)
    output_path.write_text(
        json.dumps(sanitized, allow_nan=False, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_ig_perturb_sanity(
    context,
    runtime_cfg,
    baseline_inputs: xarray.Dataset,
    gnn_ig_raw: Dict[str, np.ndarray],
    patch_radius: int,
    patch_score_agg: str,
) -> Dict[str, Any]:
    """Run IG perturbation sanity check.
    
    High-level helper that:
    - Builds point score DA from gnn_ig_raw
    - Reuses IG runner helpers + perturbation runner helpers
    - Evaluates top-k IG candidates vs random baseline
    - Returns payload with status and stats.
    
    Args:
        context: AnalysisContext with model/data.
        runtime_cfg: AnalysisConfig with settings.
        baseline_inputs: Baseline input dataset.
        gnn_ig_raw: Dict of variable name -> IG attribution map.
        patch_radius: Radius for patch perturbation.
        patch_score_agg: Aggregation method for patch scoring.
    
    Returns:
        Dict with status, reason (if not ok), passed, stats, excerpts.
        Status is one of: 'skipped', 'failed', 'ok'.
    """
    from model.ig.runner import _build_patch_candidate_maps
    from model.perturbation.runner import (
        _select_top_k_candidates,
        _evaluate_candidates_with_perturbation,
    )
    from shared.analysis_pipeline import resolve_spatial_variables
    
    # Check if enabled
    if not getattr(runtime_cfg, "ig_sanity_enable", True):
        return {
            "status": "skipped",
            "reason": "disabled",
            "passed": None,
        }
    
    topk = getattr(runtime_cfg, "ig_sanity_topk", 10)
    random_k = getattr(runtime_cfg, "ig_sanity_random_k", 10)
    seed = getattr(runtime_cfg, "ig_sanity_seed", 42)
    min_lift_ratio = getattr(runtime_cfg, "ig_sanity_min_lift_ratio", 1.1)
    
    lat_vals = np.asarray(context.lat_vals)
    lon_vals = np.asarray(context.lon_vals)
    n_lat, n_lon = len(lat_vals), len(lon_vals)
    
    # Build point score from IG maps
    point_score_da = build_point_score_da(gnn_ig_raw, lat_vals, lon_vals)
    
    # Build patch candidate maps
    patch_result = _build_patch_candidate_maps(
        gnn_ig_raw,
        patch_radius,
        patch_score_agg,
        lat_vals,
        lon_vals,
    )
    candidate_score_da = patch_result["score_da"]
    patch_maps_by_var = patch_result["maps_by_var"]
    
    # Select top-k candidates
    topk_candidates = _select_top_k_candidates(
        candidate_score_da,
        patch_maps_by_var,
        topk,
        point_score_da,
    )
    
    if not topk_candidates:
        return {
            "status": "failed",
            "reason": "no_topk_candidates",
            "passed": False,
        }
    
    # Collect excluded indices for random sampling
    excluded: Set[Tuple[int, int]] = {(c.lat_idx, c.lon_idx) for c in topk_candidates}
    
    # Sample random indices
    random_indices = sample_random_grid_indices(n_lat, n_lon, random_k, excluded, seed)
    
    # Build random candidates
    from model.perturbation.runner import CandidatePoint
    random_candidates: List[CandidatePoint] = []
    for idx, (lat_idx, lon_idx) in enumerate(random_indices):
        random_candidates.append(
            CandidatePoint(
                lat_idx=lat_idx,
                lon_idx=lon_idx,
                lat=float(lat_vals[lat_idx]),
                lon=float(lon_vals[lon_idx]),
                candidate_score=float(candidate_score_da.values[lat_idx, lon_idx]),
                point_ig_score=float(point_score_da.values[lat_idx, lon_idx]),
                ig_rank=idx + 1,
                dominant_var="random",
            )
        )
    
    if not random_candidates:
        return {
            "status": "failed",
            "reason": "no_random_candidates",
            "passed": False,
        }
    
    # Resolve vars to use: limit to keys in gnn_ig_raw AND existing in eval_inputs
    vars_to_use = [
        v for v in resolve_spatial_variables(
            runtime_cfg.perturb_variables,
            context.eval_inputs,
        )
        if v in gnn_ig_raw and v in context.eval_inputs.data_vars
    ]
    
    # Honor include_target_inputs=False by removing target_var
    if not runtime_cfg.include_target_inputs:
        target_var = getattr(context, "target_var", None)
        if target_var and target_var in vars_to_use:
            vars_to_use = [v for v in vars_to_use if v != target_var]
    
    # Use dataclasses.replace to avoid mutating frozen dataclass
    runtime_cfg_perturb = replace(runtime_cfg, perturb_time=runtime_cfg.target_time_idx)
    
    # Evaluate top-k candidates
    topk_rows = _evaluate_candidates_with_perturbation(
        context,
        runtime_cfg_perturb,
        vars_to_use,
        baseline_inputs,
        topk_candidates,
        patch_radius,
    )
    
    # Evaluate random candidates
    rand_rows = _evaluate_candidates_with_perturbation(
        context,
        runtime_cfg_perturb,
        vars_to_use,
        baseline_inputs,
        random_candidates,
        patch_radius,
    )
    
    # Extract delta values
    topk_deltas = [row.get("delta_abs_mean", float("nan")) for row in topk_rows]
    rand_deltas = [row.get("delta_abs_mean", float("nan")) for row in rand_rows]
    
    # Compute summary
    summary = compute_sanity_summary(topk_deltas, rand_deltas, min_lift_ratio)
    
    # Treat malformed-input reasons as execution failure.
    invalid_reasons = {"empty_inputs", "non_finite_input", "rand_mean_too_small"}
    if not summary.get("passed", False) and summary.get("reason") in invalid_reasons:
        return {
            "status": "failed",
            "reason": summary["reason"],
            "passed": False,
            "topk_mean": summary.get("topk_mean"),
            "rand_mean": summary.get("rand_mean"),
            "lift_ratio": summary.get("lift_ratio"),
            "min_lift_ratio": min_lift_ratio,
            "topk_size": len(topk_candidates),
            "random_size": len(random_candidates),
            "seed": seed,
        }
    
    # Build payload
    return {
        "status": "ok",
        "passed": summary["passed"],
        "topk_mean": summary["topk_mean"],
        "rand_mean": summary["rand_mean"],
        "lift_ratio": summary["lift_ratio"],
        "min_lift_ratio": min_lift_ratio,
        "topk_size": len(topk_candidates),
        "random_size": len(random_candidates),
        "seed": seed,
        "topk_excerpts": [
            {
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "delta_abs_mean": row.get("delta_abs_mean"),
            }
            for row in topk_rows[:5]
        ],
        "random_excerpts": [
            {
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "delta_abs_mean": row.get("delta_abs_mean"),
            }
            for row in rand_rows[:5]
        ],
    }
