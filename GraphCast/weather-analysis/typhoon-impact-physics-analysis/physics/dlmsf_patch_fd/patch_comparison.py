from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jax
import numpy as np
import scipy.stats
import xarray

from physics.dlmsf_patch_fd.dlmsf_sensitivity import DLMSFSensitivityResult, compute_dlmsf_patch_fd
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
    lhs = 0.0
    for var_name in vars_to_use:
        ig_full = ig_sum[var_name] / float(ig_steps)
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
        "full_scalar": full_scalar,
        "baseline_scalar": base_scalar,
        "lhs": lhs,
        "rhs": rhs,
        "rel_err": rel_err,
    }


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
    )


def _case_summary(case: Dict[str, Any]) -> Dict[str, Any]:
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
    }
    if case.get("deletion") is not None:
        summary["deletion"] = asdict(case["deletion"])
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
