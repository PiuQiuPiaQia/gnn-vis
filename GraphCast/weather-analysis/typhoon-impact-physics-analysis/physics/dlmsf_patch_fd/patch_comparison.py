from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jax
import numpy as np
import scipy.stats
import xarray

from physics.dlmsf_patch_fd.dlmsf_sensitivity import DLMSFSensitivityResult, _extract_uv_levels, _haversine_km, compute_dlmsf_patch_fd
from physics.dlmsf_patch_fd.ig_phys import compute_ig_phys_dlmsf_along
from shared.analysis_pipeline import AnalysisConfig, AnalysisContext, resolve_spatial_variables
from shared.importance_common import collapse_input_attribution_to_latlon
from shared.patch_geometry import CenteredWindow, build_centered_window, build_sliding_patches, patch_scores_to_grid
from shared.track_target import (
    compute_track_scalar_diagnostics,
    resolve_track_reference,
    track_scalar_from_outputs,
)


@dataclass
class PatchAlignmentMetrics:
    direction: str
    patch_size: int
    n_patches: int
    pearson_r: float
    pearson_pval: float
    spearman_rho: float
    spearman_pval: float
    iou_topq: float
    topq_fraction: float
    topq_k: int


@dataclass
class DeletionCurveSummary:
    step_fraction: List[float]
    masked_fraction: List[float]
    high_ig_delta: List[float]
    low_ig_delta: List[float]
    random_mean_delta: List[float]
    high_ig_auc: float
    high_ig_aopc: float
    low_ig_auc: float
    low_ig_aopc: float
    random_mean_auc: float
    random_mean_aopc: float
    random_repeats: int
    seed: int
    # E3: DLMSF-topK deletion ordering (optional; defaults to empty/zero)
    dlmsf_high_delta: List[float] = field(default_factory=list)
    dlmsf_high_auc: float = 0.0
    dlmsf_high_aopc: float = 0.0


def _safe_corr(
    fn,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    if x.size < 2:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    try:
        stat = fn(x, y)
    except Exception:
        return float("nan"), float("nan")
    return float(stat[0]), float(stat[1])


def _topq_patch_indices(
    scores: np.ndarray,
    *,
    finite_mask: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, int]:
    finite_idx = np.flatnonzero(np.asarray(finite_mask, dtype=bool))
    if finite_idx.size == 0:
        return np.array([], dtype=np.int64), 0
    actual_k = max(1, int(math.ceil(float(fraction) * float(finite_idx.size))))
    order = np.argsort(np.asarray(scores, dtype=np.float64)[finite_idx], kind="stable")[::-1]
    return np.asarray(finite_idx[order[:actual_k]], dtype=np.int64), actual_k


def _topq_iou(
    a: np.ndarray,
    b: np.ndarray,
    fraction: float,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    finite = np.isfinite(a) & np.isfinite(b)
    idx_a, actual_k = _topq_patch_indices(a, finite_mask=finite, fraction=fraction)
    idx_b, _ = _topq_patch_indices(b, finite_mask=finite, fraction=fraction)
    if actual_k == 0:
        return 0.0, 0, idx_a, idx_b
    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    union = len(set_a | set_b)
    if union == 0:
        return 0.0, actual_k, idx_a, idx_b
    return float(len(set_a & set_b) / float(union)), actual_k, idx_a, idx_b


def compute_topk_iou_signed(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
    sign: str,
) -> float:
    """IoU of top-k signed patches between two score arrays.

    Parameters
    ----------
    a, b:
        1-D arrays of patch scores (may contain NaN).
    k:
        Number of top entries to consider (by magnitude within the chosen sign).
    sign:
        ``"pos"`` selects the top-k largest positive values;
        ``"neg"`` selects the top-k most-negative (smallest) values.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 when neither array has any entry with the
        requested sign.
    """
    if sign not in ("pos", "neg"):
        raise ValueError("sign must be 'pos' or 'neg'")

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    finite = np.isfinite(a) & np.isfinite(b)

    if sign == "pos":
        mask_a = finite & (a > 0)
        mask_b = finite & (b > 0)
        # top-k by largest value
        def _topk_idx(arr, mask):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return set()
            order = np.argsort(arr[idx])[::-1]
            return set(idx[order[:k]].tolist())
    else:  # neg
        mask_a = finite & (a < 0)
        mask_b = finite & (b < 0)
        # top-k by most-negative (smallest) value
        def _topk_idx(arr, mask):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return set()
            order = np.argsort(arr[idx])  # ascending → most negative first
            return set(idx[order[:k]].tolist())

    set_a = _topk_idx(a, mask_a)
    set_b = _topk_idx(b, mask_b)

    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0

    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(len(set_a & set_b) / float(union))


def compute_sign_agreement(
    a: np.ndarray,
    b: np.ndarray,
    k: int,
) -> float:
    """Fraction of top-k (by absolute value) overlapping patches with matching sign.

    Both arrays are ranked by absolute value.  The top-k sets are intersected;
    within the intersection the fraction of indices where ``sign(a[i]) ==
    sign(b[i])`` is returned.

    Returns ``float("nan")`` when the intersection is empty (no overlap).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    finite = np.isfinite(a) & np.isfinite(b)
    finite_idx = np.flatnonzero(finite)

    if finite_idx.size == 0:
        return float("nan")

    abs_a = np.abs(a[finite_idx])
    abs_b = np.abs(b[finite_idx])

    top_a = set(finite_idx[np.argsort(abs_a)[::-1][:k]].tolist())
    top_b = set(finite_idx[np.argsort(abs_b)[::-1][:k]].tolist())

    overlap = top_a & top_b
    if len(overlap) == 0:
        return float("nan")

    agree = sum(1 for i in overlap if np.sign(a[i]) == np.sign(b[i]))
    return float(agree) / float(len(overlap))


def _patch_scores_from_maps(
    *,
    window: CenteredWindow,
    patch_size: int,
    stride: int,
    signed_cell_map: np.ndarray,
    abs_cell_map: np.ndarray,
) -> Dict[str, Any]:
    patches = build_sliding_patches(window, patch_size=patch_size, stride=stride)
    signed_scores = np.array(
        [float(np.sum(np.asarray(signed_cell_map, dtype=np.float64)[patch.mask])) for patch in patches],
        dtype=np.float64,
    )
    abs_scores = np.array(
        [float(np.sum(np.asarray(abs_cell_map, dtype=np.float64)[patch.mask])) for patch in patches],
        dtype=np.float64,
    )
    return {
        "patches": patches,
        "signed_scores": signed_scores,
        "abs_scores": abs_scores,
        "signed_map": patch_scores_to_grid(signed_scores, patches, window.shape, core_mask=window.core_mask),
        "abs_map": patch_scores_to_grid(abs_scores, patches, window.shape, core_mask=window.core_mask),
    }


def _extract_window_field_map(
    inputs_data: xarray.Dataset,
    *,
    field_name: str,
    window: CenteredWindow,
) -> np.ndarray:
    field = inputs_data[field_name]
    if "batch" in field.dims:
        field = field.isel(batch=0)
    if "time" in field.dims:
        field = field.isel(time=-1)
    if "level" in field.dims:
        field = field.isel(level=0)
    if "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError(f"Field {field_name!r} does not provide lat/lon dimensions for track-patch plotting")
    return np.asarray(
        field.isel(lat=window.lat_indices, lon=window.lon_indices).values,
        dtype=np.float64,
    )


def _patch_indices_to_mask(
    patches,
    *,
    shape: tuple[int, int],
    patch_indices: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for patch_idx in np.asarray(patch_indices, dtype=np.int64).tolist():
        mask |= np.asarray(patches[int(patch_idx)].mask, dtype=bool)
    return mask


def _build_case_plot_payload(
    *,
    window: CenteredWindow,
    patches,
    environment_map: np.ndarray,
    environment_field_name: str,
    ig_abs_map: np.ndarray,
    ig_abs_scores: np.ndarray,
    dlmsf_abs_map: np.ndarray,
    dlmsf_abs_scores: np.ndarray,
    topq_fraction: float,
) -> Dict[str, Any]:
    _, actual_k, ig_top_idx, dlmsf_top_idx = _topq_iou(
        np.asarray(ig_abs_scores, dtype=np.float64),
        np.asarray(dlmsf_abs_scores, dtype=np.float64),
        topq_fraction,
    )
    ig_top_mask = _patch_indices_to_mask(patches, shape=window.shape, patch_indices=ig_top_idx)
    dlmsf_top_mask = _patch_indices_to_mask(patches, shape=window.shape, patch_indices=dlmsf_top_idx)
    overlap_mask = ig_top_mask & dlmsf_top_mask
    union_mask = ig_top_mask | dlmsf_top_mask
    return {
        "environment_field": str(environment_field_name),
        "lat_vals": np.asarray(window.lat_vals, dtype=np.float64).tolist(),
        "lon_vals": np.asarray(window.lon_vals, dtype=np.float64).tolist(),
        "core_mask": np.asarray(window.core_mask, dtype=bool).tolist(),
        "environment_map": np.asarray(environment_map, dtype=np.float64).tolist(),
        "ig_abs_map": np.asarray(ig_abs_map, dtype=np.float64).tolist(),
        "dlmsf_abs_map": np.asarray(dlmsf_abs_map, dtype=np.float64).tolist(),
        "ig_topq_mask": np.asarray(ig_top_mask, dtype=bool).tolist(),
        "dlmsf_topq_mask": np.asarray(dlmsf_top_mask, dtype=bool).tolist(),
        "overlap_mask": np.asarray(overlap_mask, dtype=bool).tolist(),
        "union_mask": np.asarray(union_mask, dtype=bool).tolist(),
        "topq_fraction": float(topq_fraction),
        "topq_k": int(actual_k),
    }


def _classify_patch_sign(ig_score: float, dlmsf_score: float) -> int:
    """Classify a pair of signed patch scores for sign agreement.

    Returns:
        1  same-sign positive (both > 0)
        2  same-sign negative (both < 0)
        3  opposite-sign or non-finite
    """
    if not np.isfinite(ig_score) or not np.isfinite(dlmsf_score):
        return 3
    if ig_score > 0.0 and dlmsf_score > 0.0:
        return 1
    if ig_score < 0.0 and dlmsf_score < 0.0:
        return 2
    return 3


def _build_case_visualization_payload(
    *,
    window: CenteredWindow,
    patches,
    direction: str,
    patch_size: int,
    target_time_idx: int,
    topq_fraction: float,
    ig_abs_map: np.ndarray,
    ig_abs_scores: np.ndarray,
    ig_signed_scores: np.ndarray,
    dlmsf_abs_map: np.ndarray,
    dlmsf_abs_scores: np.ndarray,
    dlmsf_signed_scores: np.ndarray,
) -> Dict[str, Any]:
    """Build the four-figure visualization payload for a single track-patch case.

    All patch-level score arrays must have length == len(patches).

    Payload sections:
        meta        - metadata used for output filenames
        overlap     - |IG|/|DLMSF| heat maps, Top-q overlap mask, Spearman rho, IoU@q
        scatter     - patch-level abs score arrays (x=DLMSF, y=IG) + Spearman rho
        sign_map    - discrete sign-class grid (0 non-overlap, 1 same+, 2 same-, 3 opp)
        deletion    - None by default; wired in by run_track_patch_analysis
    """
    ig_abs = np.asarray(ig_abs_scores, dtype=np.float64)
    ig_signed = np.asarray(ig_signed_scores, dtype=np.float64)
    dlmsf_abs = np.asarray(dlmsf_abs_scores, dtype=np.float64)
    dlmsf_signed = np.asarray(dlmsf_signed_scores, dtype=np.float64)

    # Compute metrics from patch-level arrays
    iou_at_20, actual_k, ig_top_idx, dlmsf_top_idx = _topq_iou(ig_abs, dlmsf_abs, topq_fraction)
    spearman_rho, _ = _safe_corr(scipy.stats.spearmanr, ig_abs, dlmsf_abs)

    # Compute Top-q overlap patch indices (intersection)
    overlap_patch_idx = sorted(set(ig_top_idx.tolist()) & set(dlmsf_top_idx.tolist()))

    # Build overlap cell mask
    overlap_mask = np.zeros(window.shape, dtype=bool)
    for i in overlap_patch_idx:
        overlap_mask |= np.asarray(patches[i].mask, dtype=bool)

    # Build sign_class_map: 0=non-overlap, 1=same+, 2=same-, 3=opposite
    # Each cell is assigned the class of the overlap patch with the highest
    # combined strength (|ig| + |dlmsf|) covering that cell.
    sign_class_map = np.zeros(window.shape, dtype=np.int64)
    cell_strength = np.full(window.shape, -np.inf, dtype=np.float64)

    same_sign_count = 0
    for patch_idx in overlap_patch_idx:
        sign_class = _classify_patch_sign(
            float(ig_signed[patch_idx]),
            float(dlmsf_signed[patch_idx]),
        )
        if sign_class in (1, 2):
            same_sign_count += 1
        strength = float(ig_abs[patch_idx]) + float(dlmsf_abs[patch_idx])
        mask = np.asarray(patches[patch_idx].mask, dtype=bool)
        update = mask & (strength > cell_strength)
        sign_class_map[update] = sign_class
        cell_strength[update] = strength

    overlap_patch_count = len(overlap_patch_idx)
    sign_agreement_at_20 = (
        float(same_sign_count) / float(overlap_patch_count)
        if overlap_patch_count > 0
        else float("nan")
    )

    lat_vals_list = np.asarray(window.lat_vals, dtype=np.float64).tolist()
    lon_vals_list = np.asarray(window.lon_vals, dtype=np.float64).tolist()

    return {
        "meta": {
            "direction": str(direction),
            "patch_size": int(patch_size),
            "target_time_idx": int(target_time_idx),
            "topq_fraction": float(topq_fraction),
        },
        "overlap": {
            "lat_vals": lat_vals_list,
            "lon_vals": lon_vals_list,
            "ig_abs_map": np.asarray(ig_abs_map, dtype=np.float64).tolist(),
            "dlmsf_abs_map": np.asarray(dlmsf_abs_map, dtype=np.float64).tolist(),
            "overlap_mask": overlap_mask.tolist(),
            "spearman_rho": float(spearman_rho),
            "iou_at_20": float(iou_at_20),
        },
        "scatter": {
            "x_patch_abs_scores": dlmsf_abs.tolist(),
            "y_patch_abs_scores": ig_abs.tolist(),
            "spearman_rho": float(spearman_rho),
        },
        "sign_map": {
            "lat_vals": lat_vals_list,
            "lon_vals": lon_vals_list,
            "sign_class_map": sign_class_map.tolist(),
            "sign_agreement_at_20": sign_agreement_at_20,
            "overlap_mask": overlap_mask.tolist(),
        },
        "deletion": None,
    }


def _run_forward_track_scalar(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    inputs_data: xarray.Dataset,
    *,
    center_field_name: str,
    window: CenteredWindow,
    direction_mode: str,
    softmin_temperature: float,
) -> tuple[float, Any]:
    track_ref = resolve_track_reference(runtime_cfg.target_time_idx)
    outputs = context.run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=inputs_data,
        targets_template=context.targets_template,
        forcings=context.eval_forcings,
    )
    scalar = track_scalar_from_outputs(
        outputs,
        center_field_name=center_field_name,
        target_time_idx=runtime_cfg.target_time_idx,
        track_ref=track_ref,
        window=window,
        direction_mode=direction_mode,
        softmin_temperature=softmin_temperature,
    )
    return float(np.array(scalar)), outputs


def _compute_track_ig_cell_maps(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
    *,
    vars_to_use: Sequence[str],
    center_field_name: str,
    window: CenteredWindow,
    direction_mode: str,
    softmin_temperature: float,
) -> Dict[str, Any]:
    track_ref = resolve_track_reference(runtime_cfg.target_time_idx)

    def _loss(inputs_data):
        outputs = context.run_forward_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=inputs_data,
            targets_template=context.targets_template,
            forcings=context.eval_forcings,
        )
        return track_scalar_from_outputs(
            outputs,
            center_field_name=center_field_name,
            target_time_idx=runtime_cfg.target_time_idx,
            track_ref=track_ref,
            window=window,
            direction_mode=direction_mode,
            softmin_temperature=softmin_temperature,
        )

    grad_fn = jax.grad(_loss)
    ig_steps = int(runtime_cfg.gradient_steps)
    if ig_steps <= 0:
        raise ValueError(f"gradient_steps must be positive, got {ig_steps}")

    diff_arrays = {
        var_name: np.asarray(context.eval_inputs[var_name].values) - np.asarray(baseline_inputs[var_name].values)
        for var_name in vars_to_use
    }
    ig_sum = {
        var_name: np.zeros_like(diff_arrays[var_name], dtype=np.float64)
        for var_name in vars_to_use
    }

    print(
        f"[Track-IG] direction={direction_mode}  vars={len(vars_to_use)}  steps={ig_steps}"
    )
    for step in range(ig_steps):
        alpha = (step + 0.5) / float(ig_steps)
        interp = context.eval_inputs.copy(deep=False)
        for var_name in vars_to_use:
            interp[var_name] = xarray.DataArray(
                np.asarray(baseline_inputs[var_name].values) + alpha * diff_arrays[var_name],
                dims=context.eval_inputs[var_name].dims,
                coords=context.eval_inputs[var_name].coords,
                attrs=context.eval_inputs[var_name].attrs,
            )
        grads = grad_fn(interp)
        for var_name in vars_to_use:
            ig_sum[var_name] += diff_arrays[var_name] * np.asarray(grads[var_name].values)
        if (step + 1) % 10 == 0 or (step + 1) == ig_steps:
            print(f"  IG steps: {step + 1}/{ig_steps}")

    signed_cell_map = np.zeros(window.shape, dtype=np.float64)
    abs_cell_map = np.zeros(window.shape, dtype=np.float64)
    raw_ig_per_var: Dict[str, np.ndarray] = {}
    lhs = 0.0
    for var_name in vars_to_use:
        ig_full = ig_sum[var_name] / float(ig_steps)
        raw_ig_per_var[var_name] = ig_full
        lhs += float(np.sum(ig_full))
        signed_da = collapse_input_attribution_to_latlon(
            ig_full,
            context.eval_inputs[var_name],
            abs_before_sum=False,
        )
        abs_da = collapse_input_attribution_to_latlon(
            ig_full,
            context.eval_inputs[var_name],
            abs_before_sum=True,
        )
        signed_cell_map += np.asarray(
            signed_da.isel(lat=window.lat_indices, lon=window.lon_indices).values,
            dtype=np.float64,
        )
        abs_cell_map += np.asarray(
            abs_da.isel(lat=window.lat_indices, lon=window.lon_indices).values,
            dtype=np.float64,
        )

    full_scalar, _ = _run_forward_track_scalar(
        context,
        runtime_cfg,
        context.eval_inputs,
        center_field_name=center_field_name,
        window=window,
        direction_mode=direction_mode,
        softmin_temperature=softmin_temperature,
    )
    base_scalar, _ = _run_forward_track_scalar(
        context,
        runtime_cfg,
        baseline_inputs,
        center_field_name=center_field_name,
        window=window,
        direction_mode=direction_mode,
        softmin_temperature=softmin_temperature,
    )
    rhs = float(full_scalar - base_scalar)
    rel_err = abs(lhs - rhs) / (abs(rhs) + 1e-8)
    print(
        "[Track-IG] Completeness check: "
        f"lhs={lhs:.6e}, rhs={rhs:.6e}, rel_err={rel_err:.6%}"
    )

    return {
        "signed_cell_map": signed_cell_map,
        "abs_cell_map": abs_cell_map,
        "raw_ig_per_var": raw_ig_per_var,
        "full_scalar": full_scalar,
        "baseline_scalar": base_scalar,
        "lhs": lhs,
        "rhs": rhs,
        "rel_err": rel_err,
    }


def _project_wind_ig_along_track(
    *,
    ig_u_full: np.ndarray,
    ig_v_full: np.ndarray,
    u_da: xarray.DataArray,
    v_da: xarray.DataArray,
    window: "CenteredWindow",
    d_hat: tuple,
    levels_bottom_hpa: float = 925.0,
    levels_top_hpa: float = 300.0,
) -> np.ndarray:
    """Project wind-component IG onto the along-track direction within a level band.

    Computes a signed lat×lon cell map for the analysis window as::

        signed_cell[row, col] = Σ_{other dims, l ∈ band}
                                 (ig_u[...,l,row,col] * d̂_u
                                  + ig_v[...,l,row,col] * d̂_v)

    Parameters
    ----------
    ig_u_full, ig_v_full:
        Raw IG arrays for ``u_component_of_wind`` and ``v_component_of_wind``,
        same shape as the respective DataArrays.
    u_da, v_da:
        Original DataArrays (used for dimension/coordinate labels).
    window:
        Spatial analysis window; output shape matches ``window.shape``.
    d_hat:
        ``(d_u, d_v)`` along-track unit direction vector.
    levels_bottom_hpa, levels_top_hpa:
        Inclusive pressure-level range.  Only levels satisfying
        ``levels_top_hpa ≤ level ≤ levels_bottom_hpa`` are retained.

    Returns
    -------
    np.ndarray of shape ``window.shape``.
    """
    d_u, d_v = float(d_hat[0]), float(d_hat[1])

    ig_u_da = xarray.DataArray(
        np.asarray(ig_u_full, dtype=np.float64),
        dims=u_da.dims,
        coords=u_da.coords,
    )
    ig_v_da = xarray.DataArray(
        np.asarray(ig_v_full, dtype=np.float64),
        dims=v_da.dims,
        coords=v_da.coords,
    )

    # Drop batch dim if present
    if "batch" in ig_u_da.dims:
        ig_u_da = ig_u_da.isel(batch=0)
    if "batch" in ig_v_da.dims:
        ig_v_da = ig_v_da.isel(batch=0)

    # Filter level band
    if "level" in ig_u_da.dims:
        lo = min(levels_top_hpa, levels_bottom_hpa)
        hi = max(levels_top_hpa, levels_bottom_hpa)
        level_vals = np.asarray(ig_u_da.coords["level"].values, dtype=np.float64)
        level_mask = (level_vals >= lo) & (level_vals <= hi)
        ig_u_da = ig_u_da.isel(level=level_mask)
        ig_v_da = ig_v_da.isel(level=level_mask)

    # Project onto d_hat
    ig_along_da = ig_u_da * d_u + ig_v_da * d_v

    # Sum over all non-spatial dimensions
    reduce_dims = [dim for dim in ig_along_da.dims if dim not in {"lat", "lon"}]
    if reduce_dims:
        ig_along_da = ig_along_da.sum(dim=reduce_dims)
    ig_along_latlon = ig_along_da.transpose("lat", "lon")

    # Extract window subgrid using numpy outer-product indexing
    ig_np = np.asarray(ig_along_latlon.values, dtype=np.float64)
    signed_cell_map = ig_np[
        window.lat_indices[:, None],
        window.lon_indices[None, :],
    ]
    return signed_cell_map


def _compute_alignment_metrics(
    *,
    direction_mode: str,
    patch_size: int,
    ig_abs_scores: np.ndarray,
    dlmsf_parallel_scores: np.ndarray,
    topk_fraction: float,
) -> PatchAlignmentMetrics:
    finite = np.isfinite(ig_abs_scores) & np.isfinite(dlmsf_parallel_scores)
    x = np.asarray(ig_abs_scores, dtype=np.float64)[finite]
    y = np.abs(np.asarray(dlmsf_parallel_scores, dtype=np.float64)[finite]
    )

    pearson_r, pearson_pval = _safe_corr(scipy.stats.pearsonr, x, y)
    spearman_rho, spearman_pval = _safe_corr(scipy.stats.spearmanr, x, y)
    iou_topq, topq_k, _, _ = _topq_iou(
        np.asarray(ig_abs_scores, dtype=np.float64),
        np.abs(np.asarray(dlmsf_parallel_scores, dtype=np.float64)),
        topk_fraction,
    )

    return PatchAlignmentMetrics(
        direction=str(direction_mode),
        patch_size=int(patch_size),
        n_patches=int(x.size),
        pearson_r=pearson_r,
        pearson_pval=pearson_pval,
        spearman_rho=spearman_rho,
        spearman_pval=spearman_pval,
        iou_topq=iou_topq,
        topq_fraction=float(topk_fraction),
        topq_k=topq_k,
    )


def _compute_physical_aopc(
    *,
    dlmsf_signed_scores: np.ndarray,
    ig_abs_scores: np.ndarray,
    seed: int = 42,
    random_repeats: int = 8,
) -> Dict[str, Any]:
    """Compute AOPC curves on the physical scalar J_along analytically (E4).

    Uses the pre-computed DLMSF per-patch ΔJ values to build cumulative-delta
    curves for three deletion orderings — DLMSF-topK, IG-topK, and random —
    without any new forward passes.

    The cumulative delta at step k is the running sum of
    ``dlmsf_signed_scores`` in the chosen patch order.

    Parameters
    ----------
    dlmsf_signed_scores:
        1-D array of per-patch ΔJ values (signed along-track DLMSF scores).
    ig_abs_scores:
        1-D array of per-patch absolute IG scores (used for IG-topK ordering).
    seed:
        RNG seed for random orderings.
    random_repeats:
        Number of random shuffles to average.

    Returns
    -------
    dict with keys:
        ``"high_dlmsf_cumulative"``   — cumulative ΔJ in DLMSF-topK order.
        ``"ig_cumulative"``           — cumulative ΔJ in IG-topK order.
        ``"random_mean_cumulative"``  — mean cumulative ΔJ over random orders.
        ``"aopc_dlmsf"``              — AOPC for DLMSF ordering.
        ``"aopc_ig"``                 — AOPC for IG ordering.
        ``"aopc_random_mean"``        — mean AOPC over random orderings.
        ``"n_patches"``               — number of patches.
    """
    s = np.asarray(dlmsf_signed_scores, dtype=np.float64).ravel()
    ig = np.asarray(ig_abs_scores, dtype=np.float64).ravel()
    n = len(s)

    def _cumulative(order: np.ndarray) -> np.ndarray:
        return np.cumsum(s[order])

    # DLMSF-topK: sort by |dlmsf_signed| descending
    dlmsf_order = np.argsort(np.abs(s), kind="stable")[::-1]
    dlmsf_cumul = _cumulative(dlmsf_order).tolist()

    # IG-topK: sort by ig_abs descending
    ig_order = np.argsort(ig, kind="stable")[::-1]
    ig_cumul = _cumulative(ig_order).tolist()

    # Random orderings
    rng = np.random.default_rng(int(seed))
    random_cumuls: List[np.ndarray] = []
    for _ in range(max(1, int(random_repeats))):
        rand_order = rng.permutation(n)
        random_cumuls.append(_cumulative(rand_order))

    random_mean_cumul = (
        np.mean(np.stack(random_cumuls, axis=0), axis=0) if random_cumuls else np.zeros(n)
    )

    aopc = lambda cumul: float(np.mean(cumul)) if len(cumul) > 0 else 0.0

    return {
        "high_dlmsf_cumulative": dlmsf_cumul,
        "ig_cumulative": ig_cumul,
        "random_mean_cumulative": random_mean_cumul.tolist(),
        "aopc_dlmsf": aopc(dlmsf_cumul),
        "aopc_ig": aopc(ig_cumul),
        "aopc_random_mean": aopc(random_mean_cumul),
        "n_patches": n,
    }


def classify_patch_roles(
    ig_abs_scores: np.ndarray,
    dlmsf_abs_scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Classify each patch into one of four roles based on top-k membership.

    Roles:
        0 — **neither**: not in top-k of IG or DLMSF.
        1 — **model_only**: in top-k of IG, but not DLMSF.
        2 — **physics_only**: in top-k of DLMSF, but not IG.
        3 — **consensus**: in top-k of both IG and DLMSF.

    NaN entries are excluded from the top-k sets and receive role 0.

    Parameters
    ----------
    ig_abs_scores:
        1-D array of absolute IG patch scores.
    dlmsf_abs_scores:
        1-D array of absolute DLMSF patch scores.
    k:
        Number of top patches to consider.  Capped at the number of finite
        entries in each array.

    Returns
    -------
    np.ndarray of dtype int, shape ``(n_patches,)``.
    """
    ig = np.asarray(ig_abs_scores, dtype=np.float64).ravel()
    dlmsf = np.asarray(dlmsf_abs_scores, dtype=np.float64).ravel()
    n = len(ig)

    def _topk_set(scores: np.ndarray) -> set:
        finite_idx = np.flatnonzero(np.isfinite(scores))
        actual_k = min(k, len(finite_idx))
        if actual_k == 0:
            return set()
        order = np.argsort(scores[finite_idx])[::-1]
        return set(finite_idx[order[:actual_k]].tolist())

    set_ig = _topk_set(ig)
    set_dlmsf = _topk_set(dlmsf)

    roles = np.zeros(n, dtype=np.int64)
    for i in range(n):
        in_ig = i in set_ig
        in_dlmsf = i in set_dlmsf
        if in_ig and in_dlmsf:
            roles[i] = 3
        elif in_ig:
            roles[i] = 1
        elif in_dlmsf:
            roles[i] = 2
        # else: 0 (already set)
    return roles


def _compute_dlmsf_env_mask(
    *,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    core_radius_deg: float,
    annulus_inner_km: float,
    annulus_outer_km: float,
) -> np.ndarray:
    """Compute the boolean annular environment mask used by DLMSF.

    Replicates the mask logic inside ``compute_dlmsf_925_300``:
    a cell is included if its great-circle distance from the center is
    strictly outside the core radius, within the annulus inner/outer bounds.

    Parameters
    ----------
    lat_vals, lon_vals:
        1-D coordinate arrays for the spatial grid.
    center_lat, center_lon:
        Typhoon center coordinates.
    core_radius_deg:
        Core exclusion radius in degrees (converted to km as deg × 111).
    annulus_inner_km, annulus_outer_km:
        Annulus inner and outer bounds in km (inclusive).

    Returns
    -------
    np.ndarray of dtype bool, shape ``(n_lat, n_lon)``.
    """
    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    lon_vals = np.asarray(lon_vals, dtype=np.float64)
    nlat, nlon = len(lat_vals), len(lon_vals)
    dist_km = np.zeros((nlat, nlon), dtype=np.float64)
    for row in range(nlat):
        for col in range(nlon):
            dist_km[row, col] = _haversine_km(
                float(center_lat), float(center_lon),
                float(lat_vals[row]), float(lon_vals[col]),
            )
    core_km = float(core_radius_deg) * 111.0
    env_mask = (
        (dist_km > core_km)
        & (dist_km >= float(annulus_inner_km))
        & (dist_km <= float(annulus_outer_km))
    )
    return env_mask


def _mask_inputs_by_window_mask(
    eval_inputs: xarray.Dataset,
    baseline_inputs: xarray.Dataset,
    vars_to_use: Sequence[str],
    window: CenteredWindow,
    union_mask: np.ndarray,
) -> xarray.Dataset:
    masked = eval_inputs.copy(deep=False)
    rows, cols = np.nonzero(np.asarray(union_mask, dtype=bool))

    for var_name in vars_to_use:
        original_da = eval_inputs[var_name]
        base_da = baseline_inputs[var_name]
        values = np.asarray(original_da.values).copy()
        base_values = np.asarray(base_da.values)
        lat_axis = original_da.dims.index("lat")
        lon_axis = original_da.dims.index("lon")

        for row, col in zip(rows.tolist(), cols.tolist()):
            slicer = [slice(None)] * values.ndim
            slicer[lat_axis] = int(window.lat_indices[row])
            slicer[lon_axis] = int(window.lon_indices[col])
            values[tuple(slicer)] = base_values[tuple(slicer)]

        masked[var_name] = xarray.DataArray(
            values,
            dims=original_da.dims,
            coords=original_da.coords,
            attrs=original_da.attrs,
        )
    return masked


def _compute_single_deletion_curve(
    *,
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
    vars_to_use: Sequence[str],
    window: CenteredWindow,
    patches,
    order: np.ndarray,
    center_field_name: str,
    direction_mode: str,
    softmin_temperature: float,
    base_scalar: float,
) -> Dict[str, Any]:
    union_mask = np.zeros(window.shape, dtype=bool)
    step_fraction: List[float] = []
    masked_fraction: List[float] = []
    deltas: List[float] = []
    valid_env_cells = int((~window.core_mask).sum())

    for step, patch_idx in enumerate(order.tolist(), start=1):
        union_mask |= np.asarray(patches[int(patch_idx)].mask, dtype=bool)
        masked_inputs = _mask_inputs_by_window_mask(
            context.eval_inputs,
            baseline_inputs,
            vars_to_use,
            window,
            union_mask,
        )
        new_scalar, _ = _run_forward_track_scalar(
            context,
            runtime_cfg,
            masked_inputs,
            center_field_name=center_field_name,
            window=window,
            direction_mode=direction_mode,
            softmin_temperature=softmin_temperature,
        )
        deltas.append(float(base_scalar - new_scalar))
        step_fraction.append(float(step) / float(len(order)))
        masked_fraction.append(float(union_mask.sum()) / float(max(valid_env_cells, 1)))

    if deltas:
        auc = float(np.trapz(np.asarray(deltas, dtype=np.float64), np.asarray(masked_fraction, dtype=np.float64)))
        aopc = float(np.mean(deltas))
    else:
        auc = 0.0
        aopc = 0.0
    return {
        "step_fraction": step_fraction,
        "masked_fraction": masked_fraction,
        "deltas": deltas,
        "auc": auc,
        "aopc": aopc,
    }


def _run_deletion_validation(
    *,
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
    vars_to_use: Sequence[str],
    window: CenteredWindow,
    patches,
    ig_abs_scores: np.ndarray,
    center_field_name: str,
    softmin_temperature: float,
    seed: int,
    random_repeats: int,
    dlmsf_signed_scores: "np.ndarray | None" = None,
) -> DeletionCurveSummary:
    base_scalar, _ = _run_forward_track_scalar(
        context,
        runtime_cfg,
        context.eval_inputs,
        center_field_name=center_field_name,
        window=window,
        direction_mode="along",
        softmin_temperature=softmin_temperature,
    )
    score_arr = np.asarray(ig_abs_scores, dtype=np.float64)
    high_order = np.argsort(score_arr, kind="stable")[::-1]
    low_order = np.argsort(score_arr, kind="stable")

    high_curve = _compute_single_deletion_curve(
        context=context,
        runtime_cfg=runtime_cfg,
        baseline_inputs=baseline_inputs,
        vars_to_use=vars_to_use,
        window=window,
        patches=patches,
        order=high_order,
        center_field_name=center_field_name,
        direction_mode="along",
        softmin_temperature=softmin_temperature,
        base_scalar=base_scalar,
    )
    low_curve = _compute_single_deletion_curve(
        context=context,
        runtime_cfg=runtime_cfg,
        baseline_inputs=baseline_inputs,
        vars_to_use=vars_to_use,
        window=window,
        patches=patches,
        order=low_order,
        center_field_name=center_field_name,
        direction_mode="along",
        softmin_temperature=softmin_temperature,
        base_scalar=base_scalar,
    )

    rng = np.random.default_rng(seed)
    random_deltas: List[np.ndarray] = []
    random_auc: List[float] = []
    random_aopc: List[float] = []
    for _ in range(max(1, int(random_repeats))):
        order = rng.permutation(len(patches))
        curve = _compute_single_deletion_curve(
            context=context,
            runtime_cfg=runtime_cfg,
            baseline_inputs=baseline_inputs,
            vars_to_use=vars_to_use,
            window=window,
            patches=patches,
            order=order,
            center_field_name=center_field_name,
            direction_mode="along",
            softmin_temperature=softmin_temperature,
            base_scalar=base_scalar,
        )
        random_deltas.append(np.asarray(curve["deltas"], dtype=np.float64))
        random_auc.append(float(curve["auc"]))
        random_aopc.append(float(curve["aopc"]))

    random_mean_delta = np.mean(np.stack(random_deltas, axis=0), axis=0) if random_deltas else np.array([], dtype=np.float64)

    # E3: DLMSF-topK ordering (sort patches by |dlmsf_signed_score| descending)
    dlmsf_high_curve = None
    if dlmsf_signed_scores is not None:
        dlmsf_abs_order = np.argsort(
            np.abs(np.asarray(dlmsf_signed_scores, dtype=np.float64)), kind="stable"
        )[::-1]
        dlmsf_high_curve = _compute_single_deletion_curve(
            context=context,
            runtime_cfg=runtime_cfg,
            baseline_inputs=baseline_inputs,
            vars_to_use=vars_to_use,
            window=window,
            patches=patches,
            order=dlmsf_abs_order,
            center_field_name=center_field_name,
            direction_mode="along",
            softmin_temperature=softmin_temperature,
            base_scalar=base_scalar,
        )

    return DeletionCurveSummary(
        step_fraction=list(high_curve["step_fraction"]),
        masked_fraction=list(high_curve["masked_fraction"]),
        high_ig_delta=list(high_curve["deltas"]),
        low_ig_delta=list(low_curve["deltas"]),
        random_mean_delta=random_mean_delta.tolist(),
        high_ig_auc=float(high_curve["auc"]),
        high_ig_aopc=float(high_curve["aopc"]),
        low_ig_auc=float(low_curve["auc"]),
        low_ig_aopc=float(low_curve["aopc"]),
        random_mean_auc=float(np.mean(random_auc)) if random_auc else 0.0,
        random_mean_aopc=float(np.mean(random_aopc)) if random_aopc else 0.0,
        random_repeats=max(1, int(random_repeats)),
        seed=int(seed),
        dlmsf_high_delta=list(dlmsf_high_curve["deltas"]) if dlmsf_high_curve else [],
        dlmsf_high_auc=float(dlmsf_high_curve["auc"]) if dlmsf_high_curve else 0.0,
        dlmsf_high_aopc=float(dlmsf_high_curve["aopc"]) if dlmsf_high_curve else 0.0,
    )


def _case_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    visualization = dict(case["visualization"])
    # Wire deletion display fields into visualization["deletion"]
    if case.get("deletion") is not None:
        d = case["deletion"]
        visualization["deletion"] = {
            "masked_fraction": list(d.masked_fraction),
            "high_ig_delta": list(d.high_ig_delta),
            "random_mean_delta": list(d.random_mean_delta),
            "low_ig_delta": list(d.low_ig_delta),
            "aopc_high": float(d.high_ig_aopc),
            "aopc_random": float(d.random_mean_aopc),
            "aopc_low": float(d.low_ig_aopc),
        }

    summary = {
        "direction": case["direction"],
        "patch_size": int(case["patch_size"]),
        "window_size": int(case["window_size"]),
        "core_size": int(case["core_size"]),
        "stride": int(case["stride"]),
        "track_scalar_full": float(case["ig"]["full_scalar"]),
        "track_scalar_baseline": float(case["ig"]["baseline_scalar"]),
        "ig_completeness_lhs": float(case["ig"]["lhs"]),
        "ig_completeness_rhs": float(case["ig"]["rhs"]),
        "ig_completeness_rel_err": float(case["ig"]["rel_err"]),
        "metrics": asdict(case["metrics"]),
        "track_diagnostics": dict(case["track_diagnostics"]),
        "plot": dict(case["plot"]),
        "visualization": visualization,
    }
    if case.get("deletion") is not None:
        summary["deletion"] = asdict(case["deletion"])
    if case.get("physical_aopc") is not None:
        summary["physical_aopc"] = dict(case["physical_aopc"])
    return summary


def write_patch_analysis_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_track_patch_analysis(
    *,
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
    cfg_module,
) -> Dict[str, Any]:
    center_field_name = str(getattr(cfg_module, "TRACK_CENTER_FIELD", "mean_sea_level_pressure"))
    window_size = int(getattr(cfg_module, "TRACK_WINDOW_SIZE", 21))
    core_size = int(getattr(cfg_module, "TRACK_CORE_SIZE", 3))
    stride = int(getattr(cfg_module, "TRACK_PATCH_STRIDE", 2))
    main_patch_size = int(getattr(cfg_module, "TRACK_PATCH_SIZE", 3))
    direction = str(getattr(cfg_module, "TRACK_DIRECTION", "along")).lower().strip()
    if direction != "along":
        raise ValueError(f"Main track-patch analysis only supports direction='along', got {direction!r}")
    topk_fraction = float(getattr(cfg_module, "TRACK_TOPK_FRACTION", 0.2))
    softmin_temperature = float(getattr(cfg_module, "TRACK_SOFTMIN_TEMPERATURE", 1.0))
    deletion_enable = bool(getattr(cfg_module, "TRACK_DELETION_ENABLE", True))
    deletion_seed = int(getattr(cfg_module, "TRACK_DELETION_SEED", 42))
    random_repeats = int(getattr(cfg_module, "TRACK_DELETION_RANDOM_REPEATS", 8))

    full_lat = np.asarray(context.eval_inputs.coords["lat"].values, dtype=np.float64)
    full_lon = np.asarray(context.eval_inputs.coords["lon"].values, dtype=np.float64)
    track_ref = resolve_track_reference(runtime_cfg.target_time_idx)
    window = build_centered_window(
        full_lat,
        full_lon,
        track_ref.init_lat,
        track_ref.init_lon,
        window_size=window_size,
        core_size=core_size,
    )

    vars_all = resolve_spatial_variables(runtime_cfg.perturb_variables, context.eval_inputs)
    if not runtime_cfg.include_target_inputs:
        vars_all = [v for v in vars_all if v != runtime_cfg.target_variable]
    if not vars_all:
        raise ValueError("No spatial variables available for track-patch IG analysis")

    cases: Dict[str, Any] = {}
    main_case_key = f"{direction}_p{main_patch_size}"
    environment_map = _extract_window_field_map(
        context.eval_inputs,
        field_name=center_field_name,
        window=window,
    )

    _, full_outputs = _run_forward_track_scalar(
        context,
        runtime_cfg,
        context.eval_inputs,
        center_field_name=center_field_name,
        window=window,
        direction_mode=direction,
        softmin_temperature=softmin_temperature,
    )
    full_track_diagnostics = compute_track_scalar_diagnostics(
        full_outputs,
        center_field_name=center_field_name,
        target_time_idx=runtime_cfg.target_time_idx,
        track_ref=track_ref,
        window=window,
        direction_mode=direction,
        softmin_temperature=softmin_temperature,
    )
    ig_maps = _compute_track_ig_cell_maps(
        context,
        runtime_cfg,
        baseline_inputs,
        vars_to_use=vars_all,
        center_field_name=center_field_name,
        window=window,
        direction_mode=direction,
        softmin_temperature=softmin_temperature,
    )
    ig_patch = _patch_scores_from_maps(
        window=window,
        patch_size=main_patch_size,
        stride=stride,
        signed_cell_map=ig_maps["signed_cell_map"],
        abs_cell_map=ig_maps["abs_cell_map"],
    )
    dlmsf_result = compute_dlmsf_patch_fd(
        eval_inputs=context.eval_inputs,
        baseline_inputs=baseline_inputs,
        window=window,
        center_lat=track_ref.init_lat,
        center_lon=track_ref.init_lon,
        d_hat=track_ref.along_hat,
        target_time_idx=runtime_cfg.target_time_idx,
        patch_size=main_patch_size,
        stride=stride,
        direction_mode=direction,
        core_radius_deg=float(getattr(cfg_module, "SWE_CORE_RADIUS_DEG", 3.0)),
        annulus_inner_km=float(getattr(cfg_module, "SWE_STEERING_ANNULUS_INNER_KM", 300.0)),
        annulus_outer_km=float(getattr(cfg_module, "SWE_STEERING_ANNULUS_OUTER_KM", 900.0)),
        levels_bottom_hpa=float(getattr(cfg_module, "DLMSF_LEVELS_BOTTOM_HPA", 925.0)),
        levels_top_hpa=float(getattr(cfg_module, "DLMSF_LEVELS_TOP_HPA", 300.0)),
    )
    metrics = _compute_alignment_metrics(
        direction_mode=direction,
        patch_size=main_patch_size,
        ig_abs_scores=ig_patch["abs_scores"],
        dlmsf_parallel_scores=dlmsf_result.patch_parallel_scores,
        topk_fraction=topk_fraction,
    )
    deletion = None
    if deletion_enable:
        deletion = _run_deletion_validation(
            context=context,
            runtime_cfg=runtime_cfg,
            baseline_inputs=baseline_inputs,
            vars_to_use=vars_all,
            window=window,
            patches=ig_patch["patches"],
            ig_abs_scores=ig_patch["abs_scores"],
            center_field_name=center_field_name,
            softmin_temperature=softmin_temperature,
            seed=deletion_seed,
            random_repeats=random_repeats,
            dlmsf_signed_scores=dlmsf_result.patch_parallel_scores,
        )

    # E4: Physical AOPC (analytical, no forward passes needed)
    physical_aopc = _compute_physical_aopc(
        dlmsf_signed_scores=dlmsf_result.patch_parallel_scores,
        ig_abs_scores=ig_patch["abs_scores"],
        seed=deletion_seed,
        random_repeats=random_repeats,
    )

    cases[main_case_key] = {
        "direction": direction,
        "patch_size": main_patch_size,
        "window_size": window_size,
        "core_size": core_size,
        "stride": stride,
        "window": window,
        "plot": _build_case_plot_payload(
            window=window,
            patches=ig_patch["patches"],
            environment_map=environment_map,
            environment_field_name=center_field_name,
            ig_abs_map=ig_patch["abs_map"],
            ig_abs_scores=ig_patch["abs_scores"],
            dlmsf_abs_map=dlmsf_result.S_abs_map,
            dlmsf_abs_scores=np.abs(dlmsf_result.patch_parallel_scores),
            topq_fraction=topk_fraction,
        ),
        "visualization": _build_case_visualization_payload(
            window=window,
            patches=ig_patch["patches"],
            direction=direction,
            patch_size=main_patch_size,
            target_time_idx=runtime_cfg.target_time_idx,
            topq_fraction=topk_fraction,
            ig_abs_map=ig_patch["abs_map"],
            ig_abs_scores=ig_patch["abs_scores"],
            ig_signed_scores=ig_patch["signed_scores"],
            dlmsf_abs_map=dlmsf_result.S_abs_map,
            dlmsf_abs_scores=np.abs(dlmsf_result.patch_parallel_scores),
            dlmsf_signed_scores=dlmsf_result.patch_parallel_scores,
        ),
        "ig": {
            **ig_maps,
            "patches": ig_patch["patches"],
            "patch_signed_scores": ig_patch["signed_scores"],
            "patch_abs_scores": ig_patch["abs_scores"],
            "patch_signed_map": ig_patch["signed_map"],
            "patch_abs_map": ig_patch["abs_map"],
        },
        "dlmsf_result": dlmsf_result,
        "metrics": metrics,
        "deletion": deletion,
        "physical_aopc": physical_aopc,
        "track_diagnostics": {
            **asdict(full_track_diagnostics),
            "reference_init_lat": track_ref.init_lat,
            "reference_init_lon": track_ref.init_lon,
            "reference_target_lat": track_ref.target_lat,
            "reference_target_lon": track_ref.target_lon,
        },
    }
    print(
        f"[Track-Patch] {main_case_key}: "
        f"pearson={metrics.pearson_r:+.3f}  spearman={metrics.spearman_rho:+.3f}  "
        f"iou@{int(round(100.0 * metrics.topq_fraction))}%={metrics.iou_topq:.3f}"
    )

    # -----------------------------------------------------------------------
    # E2: wind-along signed IG vs signed DLMSF (925-300 hPa wind projection)
    # -----------------------------------------------------------------------
    wind_u_var = "u_component_of_wind"
    wind_v_var = "v_component_of_wind"
    levels_bottom = float(getattr(cfg_module, "DLMSF_LEVELS_BOTTOM_HPA", 925.0))
    levels_top = float(getattr(cfg_module, "DLMSF_LEVELS_TOP_HPA", 300.0))
    raw_ig = ig_maps.get("raw_ig_per_var", {})
    if wind_u_var in raw_ig and wind_v_var in raw_ig:
        wind_along_key = f"wind_along_signed_p{main_patch_size}"
        wind_signed_cell_map = _project_wind_ig_along_track(
            ig_u_full=raw_ig[wind_u_var],
            ig_v_full=raw_ig[wind_v_var],
            u_da=context.eval_inputs[wind_u_var],
            v_da=context.eval_inputs[wind_v_var],
            window=window,
            d_hat=track_ref.along_hat,
            levels_bottom_hpa=levels_bottom,
            levels_top_hpa=levels_top,
        )
        wind_abs_cell_map = np.abs(wind_signed_cell_map)
        wind_patch = _patch_scores_from_maps(
            window=window,
            patch_size=main_patch_size,
            stride=stride,
            signed_cell_map=wind_signed_cell_map,
            abs_cell_map=wind_abs_cell_map,
        )
        dlmsf_signed = dlmsf_result.patch_parallel_scores
        n_patches_total = len(wind_patch["patches"])
        k20 = max(1, int(math.ceil(0.20 * n_patches_total)))
        wind_sign_agreement = compute_sign_agreement(
            wind_patch["signed_scores"], dlmsf_signed, k=k20
        )
        wind_iou_pos = compute_topk_iou_signed(
            wind_patch["signed_scores"], dlmsf_signed, k=k20, sign="pos"
        )
        wind_iou_neg = compute_topk_iou_signed(
            wind_patch["signed_scores"], dlmsf_signed, k=k20, sign="neg"
        )
        cases[wind_along_key] = {
            "direction": direction,
            "patch_size": main_patch_size,
            "window_size": window_size,
            "core_size": core_size,
            "stride": stride,
            "window": window,
            "wind_along_signed_cell_map": wind_signed_cell_map,
            "wind_patch_signed_scores": wind_patch["signed_scores"],
            "wind_patch_abs_scores": wind_patch["abs_scores"],
            "dlmsf_signed_scores": dlmsf_signed,
            "sign_agreement_at_20": float(wind_sign_agreement) if wind_sign_agreement == wind_sign_agreement else None,
            "iou_pos_at_20": float(wind_iou_pos),
            "iou_neg_at_20": float(wind_iou_neg),
            "levels_bottom_hpa": levels_bottom,
            "levels_top_hpa": levels_top,
        }
        print(
            f"[Track-Patch] {wind_along_key}: "
            f"sign_agr@20%={wind_sign_agreement:.3f}  "
            f"iou_pos@20%={wind_iou_pos:.3f}  iou_neg@20%={wind_iou_neg:.3f}"
        )

    # -----------------------------------------------------------------------
    # E1: ig_phys_vs_dlmsf_fd — analytical IG on J_along vs FD
    # -----------------------------------------------------------------------
    core_radius_deg_e1 = float(getattr(cfg_module, "SWE_CORE_RADIUS_DEG", 3.0))
    annulus_inner_km_e1 = float(getattr(cfg_module, "SWE_STEERING_ANNULUS_INNER_KM", 300.0))
    annulus_outer_km_e1 = float(getattr(cfg_module, "SWE_STEERING_ANNULUS_OUTER_KM", 900.0))
    if "u_component_of_wind" in context.eval_inputs and "v_component_of_wind" in context.eval_inputs:
        try:
            u_eval, v_eval, levels_eval = _extract_uv_levels(
                context.eval_inputs, window.lat_vals, window.lon_vals
            )
            u_base, v_base, _ = _extract_uv_levels(
                baseline_inputs, window.lat_vals, window.lon_vals
            )
            u_anom = (u_eval - u_base).astype(np.float64)
            v_anom = (v_eval - v_base).astype(np.float64)
            env_mask_e1 = _compute_dlmsf_env_mask(
                lat_vals=window.lat_vals,
                lon_vals=window.lon_vals,
                center_lat=track_ref.init_lat,
                center_lon=track_ref.init_lon,
                core_radius_deg=core_radius_deg_e1,
                annulus_inner_km=annulus_inner_km_e1,
                annulus_outer_km=annulus_outer_km_e1,
            )
            ig_phys_result = compute_ig_phys_dlmsf_along(
                u=u_anom,
                v=v_anom,
                levels_hpa=levels_eval.astype(np.float64),
                lat_vals=window.lat_vals.astype(np.float64),
                lon_vals=window.lon_vals.astype(np.float64),
                center_lat=float(track_ref.init_lat),
                center_lon=float(track_ref.init_lon),
                d_hat=track_ref.along_hat,
                env_mask=env_mask_e1,
            )
            # IG_phys along-track cell map = ig_u + ig_v (both already projected)
            ig_phys_cell_map = (
                np.asarray(ig_phys_result["ig_u_latlon"], dtype=np.float64)
                + np.asarray(ig_phys_result["ig_v_latlon"], dtype=np.float64)
            )
            ig_phys_abs_cell_map = np.abs(ig_phys_cell_map)
            ig_phys_patch = _patch_scores_from_maps(
                window=window,
                patch_size=main_patch_size,
                stride=stride,
                signed_cell_map=ig_phys_cell_map,
                abs_cell_map=ig_phys_abs_cell_map,
            )
            ig_phys_scores = ig_phys_patch["signed_scores"]
            dlmsf_fd_scores = dlmsf_result.patch_parallel_scores
            _pearson_e1, _ = _safe_corr(scipy.stats.pearsonr, ig_phys_scores, dlmsf_fd_scores)
            _spearman_e1, _ = _safe_corr(scipy.stats.spearmanr, ig_phys_scores, dlmsf_fd_scores)
            iou_e1, _, _, _ = _topq_iou(
                np.abs(ig_phys_scores), np.abs(dlmsf_fd_scores), topk_fraction
            )
            e1_case_key = f"ig_phys_vs_dlmsf_fd_p{main_patch_size}"
            cases[e1_case_key] = {
                "direction": direction,
                "patch_size": main_patch_size,
                "window_size": window_size,
                "core_size": core_size,
                "stride": stride,
                "window": window,
                "ig_phys_cell_map": ig_phys_cell_map,
                "ig_phys_patch_scores": ig_phys_scores,
                "dlmsf_fd_scores": dlmsf_fd_scores,
                "j_along_analytical": float(ig_phys_result["j_along"]),
                "j_phys_baseline": float(dlmsf_result.J_phys_baseline),
                "pearson_r": float(_pearson_e1),
                "spearman_rho": float(_spearman_e1),
                "iou_topq": float(iou_e1),
                "topq_fraction": float(topk_fraction),
            }
            print(
                f"[Track-Patch] {e1_case_key}: "
                f"pearson={_pearson_e1:+.3f}  spearman={_spearman_e1:+.3f}  "
                f"iou@{int(round(100.0 * topk_fraction))}%={iou_e1:.3f}  "
                f"j_analytical={ig_phys_result['j_along']:.4f}  j_fd={dlmsf_result.J_phys_baseline:.4f}"
            )
        except Exception as _e1_exc:
            print(f"[Track-Patch] E1 skipped: {_e1_exc}")

    if main_case_key not in cases:
        raise ValueError(f"Main case {main_case_key!r} was not produced")

    summary = {
        "source_pipeline": "swe",
        "main_case": main_case_key,
        "window_size": window_size,
        "core_size": core_size,
        "stride": stride,
        "track_center_field": center_field_name,
        "softmin_temperature": softmin_temperature,
        "topq_fraction": topk_fraction,
        "cases": {key: _case_summary(case) for key, case in cases.items()},
    }
    return {
        "main_case": main_case_key,
        "window": window,
        "cases": cases,
        "summary": summary,
        "main_result": cases[main_case_key]["dlmsf_result"],
        "main_metrics": cases[main_case_key]["metrics"],
        "main_deletion": cases[main_case_key]["deletion"],
    }
