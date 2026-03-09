from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import xarray

from shared.analysis_pipeline import AnalysisConfig, AnalysisContext, select_target_data
from shared.impact_analysis_utils import build_indexer, resolve_level_sel

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - lightweight tests may import helpers without jax
    def _missing_jax(*args, **kwargs):
        raise ModuleNotFoundError("jax is required for this code path")

    jax = SimpleNamespace(
        random=SimpleNamespace(PRNGKey=lambda seed: seed),
    )


@dataclass(frozen=True)
class CandidatePoint:
    lat_idx: int
    lon_idx: int
    lat: float
    lon: float
    candidate_score: float
    point_ig_score: float
    ig_rank: int
    dominant_var: str


@dataclass
class TargetSpec:
    """Resolved target specification supporting single or multi-target contexts."""
    target_vars: List[str]
    base_values: Dict[str, float]


def _resolve_target_spec(context: AnalysisContext) -> TargetSpec:
    """Resolve target specification from context.
    
    Supports both:
    a) context has target_vars/base_values (multi-target)
    b) single-target context with target_var/base_value
    
    Args:
        context: AnalysisContext with model/data.
    
    Returns:
        TargetSpec with target_vars and base_values.
    """
    # Check for multi-target context
    if hasattr(context, "target_vars") and hasattr(context, "base_values"):
        target_vars = list(context.target_vars)
        base_values = dict(context.base_values)
        return TargetSpec(target_vars=target_vars, base_values=base_values)
    
    # Single-target context
    target_var = getattr(context, "target_var", None)
    base_value = getattr(context, "base_value", None)
    
    if target_var is not None and base_value is not None:
        return TargetSpec(target_vars=[target_var], base_values={target_var: base_value})
    
    # Fallback: try to infer from runtime_cfg if available
    raise ValueError(
        "Cannot resolve target specification from context. "
        "Context must have either (target_vars, base_values) or (target_var, base_value)."
    )


def _select_top_k_candidates(
    candidate_score_da: xarray.DataArray,
    patch_maps_by_var: Dict[str, np.ndarray],
    top_k: int,
    point_ig_score_da: xarray.DataArray,
) -> List[CandidatePoint]:
    # Guard: top_k <= 0 returns empty list
    if top_k <= 0:
        return []
    
    score = np.asarray(candidate_score_da.values)
    point_ig_score = np.asarray(point_ig_score_da.values)
    lat_vals = np.asarray(candidate_score_da.coords["lat"].values)
    lon_vals = np.asarray(candidate_score_da.coords["lon"].values)
    n_lat, n_lon = score.shape

    # Only rank finite candidate scores
    finite_mask = np.isfinite(score)
    if not np.any(finite_mask):
        return []
    
    safe_score = np.where(finite_mask, score, -np.inf)
    actual_k = max(1, min(int(top_k), int(np.sum(finite_mask))))
    flat_idx = np.argsort(safe_score.ravel())[::-1][:actual_k]

    candidates: List[CandidatePoint] = []
    for rank, idx in enumerate(flat_idx, start=1):
        lat_idx = int(idx // n_lon)
        lon_idx = int(idx % n_lon)
        # Handle empty patch_maps_by_var gracefully
        if patch_maps_by_var:
            dominant_var = max(
                patch_maps_by_var,
                key=lambda name: float(patch_maps_by_var[name][lat_idx, lon_idx]),
            )
        else:
            dominant_var = "unknown"
        candidates.append(
            CandidatePoint(
                lat_idx=lat_idx,
                lon_idx=lon_idx,
                lat=float(lat_vals[lat_idx]),
                lon=float(lon_vals[lon_idx]),
                candidate_score=float(score[lat_idx, lon_idx]),
                point_ig_score=float(point_ig_score[lat_idx, lon_idx]),
                ig_rank=rank,
                dominant_var=dominant_var,
            )
        )
    return candidates


def _evaluate_candidates_with_perturbation(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
    baseline_inputs: xarray.Dataset,
    candidates: List[CandidatePoint],
    patch_radius: int,
) -> List[Dict[str, Any]]:
    time_sel = slice(None) if runtime_cfg.perturb_time == "all" else int(runtime_cfg.perturb_time)
    
    # Resolve target specification from context
    target_spec = _resolve_target_spec(context)

    print(f"\n[Perturb] Evaluating {len(candidates)} IG candidates...")
    rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        lat_start = max(candidate.lat_idx - patch_radius, 0)
        lat_end = min(candidate.lat_idx + patch_radius + 1, len(context.lat_vals))
        lon_start = max(candidate.lon_idx - patch_radius, 0)
        lon_end = min(candidate.lon_idx + patch_radius + 1, len(context.lon_vals))
        lat_slice = slice(lat_start, lat_end)
        lon_slice = slice(lon_start, lon_end)

        saved_values = {}
        try:
            for var_name in vars_to_use:
                data_array = context.eval_inputs[var_name]
                level_sel = resolve_level_sel(data_array, runtime_cfg.perturb_levels)
                indexer = build_indexer(data_array, lat_slice, lon_slice, time_sel, level_sel)
                arr = data_array.values
                saved_values[var_name] = (indexer, arr[indexer].copy())
                arr[indexer] = baseline_inputs[var_name].values[indexer]

            outputs = context.run_forward_jitted(
                rng=jax.random.PRNGKey(0),
                inputs=context.eval_inputs,
                targets_template=context.targets_template,
                forcings=context.eval_forcings,
            )

            delta_by_target: Dict[str, float] = {}
            for target_var in target_spec.target_vars:
                out_var = select_target_data(
                    outputs,
                    target_var,
                    target_level=runtime_cfg.target_level,
                )
                new_value = out_var.isel(time=runtime_cfg.target_time_idx).sel(
                    lat=context.center_lat,
                    lon=context.center_lon,
                    method="nearest",
                ).values.item()
                delta_by_target[target_var] = float(new_value - target_spec.base_values[target_var])

            delta_values = np.array(list(delta_by_target.values()), dtype=np.float64)
            rows.append(
                {
                    "lat": candidate.lat,
                    "lon": candidate.lon,
                    "lat_idx": candidate.lat_idx,
                    "lon_idx": candidate.lon_idx,
                    "patch_radius": patch_radius,
                    "candidate_score": candidate.candidate_score,
                    "point_ig_score": candidate.point_ig_score,
                    "ig_rank": candidate.ig_rank,
                    "dominant_var": candidate.dominant_var,
                    "delta_abs_mean": float(np.mean(np.abs(delta_values))),
                    "delta_signed_mean": float(np.mean(delta_values)),
                    "delta_by_target": delta_by_target,
                }
            )
        finally:
            for var_name, (indexer, old_vals) in saved_values.items():
                context.eval_inputs[var_name].values[indexer] = old_vals

        if idx % max(1, len(candidates) // 10) == 0 or idx == len(candidates):
            print(f"  progress: {idx}/{len(candidates)}")

    return sorted(rows, key=lambda item: item["delta_abs_mean"], reverse=True)
