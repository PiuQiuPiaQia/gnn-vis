#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 IG 候选筛选 + 扰动验证 的台风关键网格点排名。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np
import xarray

import config as cfg
from analysis_pipeline import AnalysisConfig, build_analysis_context, select_target_data
from heatmap_utils import plot_importance_heatmap
from impact_analysis_utils import build_indexer, resolve_level_sel
from importance_common import reduce_input_attribution_to_latlon, target_scalar
from model_utils import load_normalization_stats


ROOT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


@dataclass(frozen=True)
class CandidatePoint:
    """候选网格点：由 IG 分数筛选得到。"""

    lat_idx: int
    lon_idx: int
    lat: float
    lon: float
    ig_score: float
    ig_rank: int
    dominant_var: str


def _resolve_spatial_variables(
    eval_inputs: xarray.Dataset,
    perturb_variables: Optional[List[str]],
    target_vars: List[str],
    include_target_inputs: bool,
) -> List[str]:
    spatial_vars = [
        var_name
        for var_name, data_array in eval_inputs.data_vars.items()
        if ("lat" in data_array.dims and "lon" in data_array.dims)
    ]
    if perturb_variables is None:
        selected = spatial_vars
    else:
        perturb_set = set(perturb_variables)
        selected = [var_name for var_name in spatial_vars if var_name in perturb_set]

    if include_target_inputs:
        return selected
    target_set = set(target_vars)
    return [var_name for var_name in selected if var_name not in target_set]


def _build_climatology_baseline_da(
    original_da: xarray.DataArray,
    climatology_da: xarray.DataArray,
) -> xarray.DataArray:
    original_vals = np.asarray(original_da.values)

    if "level" in original_da.dims and "level" in climatology_da.dims:
        clim_levels = np.asarray(climatology_da.coords["level"].values)
        clim_values = np.asarray(climatology_da.values)
        level_lookup = {
            float(level): float(value)
            for level, value in zip(clim_levels, clim_values)
        }
        fallback = float(np.mean(clim_values))
        current_levels = np.asarray(original_da.coords["level"].values)
        selected_levels = np.array(
            [level_lookup.get(float(level), fallback) for level in current_levels],
            dtype=original_vals.dtype,
        )
        shape = [1] * original_vals.ndim
        shape[original_da.dims.index("level")] = selected_levels.shape[0]
        baseline_vals = np.broadcast_to(selected_levels.reshape(shape), original_vals.shape)
    else:
        scalar = float(np.asarray(climatology_da.values).mean())
        baseline_vals = np.full(original_vals.shape, scalar, dtype=original_vals.dtype)

    return xarray.DataArray(
        baseline_vals,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )


def _build_climatology_baseline_inputs(
    eval_inputs: xarray.Dataset,
    vars_to_use: List[str],
    mean_by_level: xarray.Dataset,
) -> xarray.Dataset:
    baseline_inputs = eval_inputs.copy(deep=False)
    for var_name in vars_to_use:
        original_da = eval_inputs[var_name]
        if var_name in mean_by_level:
            baseline_da = _build_climatology_baseline_da(original_da, mean_by_level[var_name])
        else:
            # 极少数统计缺失变量时，回退到样本中位数避免流程中断。
            fallback = float(np.median(np.asarray(original_da.values)))
            baseline_vals = np.full_like(np.asarray(original_da.values), fallback)
            baseline_da = xarray.DataArray(
                baseline_vals,
                dims=original_da.dims,
                coords=original_da.coords,
                attrs=original_da.attrs,
            )
            print(f"[warn] {var_name} not found in climatology stats, fallback=sample median")
        baseline_inputs[var_name] = baseline_da
    return baseline_inputs


def _to_data_array_with_same_meta(
    original_da: xarray.DataArray,
    values: np.ndarray,
) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=original_da.dims,
        coords=original_da.coords,
        attrs=original_da.attrs,
    )


def _combined_target_scalar(context, runtime_cfg: AnalysisConfig, inputs_data: xarray.Dataset):
    scalars = [
        target_scalar(context, runtime_cfg, inputs_data, target_var)
        for target_var in context.target_vars
    ]
    if len(scalars) == 1:
        return scalars[0]
    return sum(scalars) / float(len(scalars))


def _compute_ig_candidate_score_map(
    context,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
    baseline_inputs: xarray.Dataset,
    ig_steps: int,
) -> Dict[str, Any]:
    def _loss(inputs_data):
        return _combined_target_scalar(context, runtime_cfg, inputs_data)

    grad_fn = jax.grad(_loss)

    diff_arrays = {
        var_name: np.asarray(context.eval_inputs[var_name].values)
        - np.asarray(baseline_inputs[var_name].values)
        for var_name in vars_to_use
    }
    ig_full_sum = {
        var_name: np.zeros_like(diff_arrays[var_name], dtype=np.float64)
        for var_name in vars_to_use
    }

    print("\n[IG] Computing global attribution maps...")
    for step in range(ig_steps):
        alpha = (step + 0.5) / float(ig_steps)
        interpolated_inputs = context.eval_inputs.copy(deep=False)
        for var_name in vars_to_use:
            interp_values = np.asarray(baseline_inputs[var_name].values) + alpha * diff_arrays[var_name]
            interpolated_inputs[var_name] = _to_data_array_with_same_meta(
                context.eval_inputs[var_name],
                interp_values,
            )

        grads = grad_fn(interpolated_inputs)
        for var_name in vars_to_use:
            ig_full_sum[var_name] += diff_arrays[var_name] * np.asarray(grads[var_name].values)

        if (step + 1) % 10 == 0 or (step + 1) == ig_steps:
            print(f"  IG steps: {step + 1}/{ig_steps}")

    ig_maps_by_var: Dict[str, np.ndarray] = {}
    signed_contrib_sum = 0.0
    for var_name in vars_to_use:
        ig_full = ig_full_sum[var_name] / float(ig_steps)
        signed_contrib_sum += float(np.sum(ig_full))
        ig_map_da = reduce_input_attribution_to_latlon(
            attribution=ig_full,
            original_da=context.eval_inputs[var_name],
            runtime_cfg=runtime_cfg,
        )
        ig_maps_by_var[var_name] = np.asarray(ig_map_da.values, dtype=np.float64)

    lat_vals = np.asarray(context.eval_inputs.coords["lat"].values)
    lon_vals = np.asarray(context.eval_inputs.coords["lon"].values)
    score_map = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)
    for var_name in vars_to_use:
        score_map += np.abs(ig_maps_by_var[var_name])

    rhs = float(
        np.array(
            _combined_target_scalar(context, runtime_cfg, context.eval_inputs)
            - _combined_target_scalar(context, runtime_cfg, baseline_inputs)
        )
    )
    rel_err = abs(signed_contrib_sum - rhs) / (abs(rhs) + 1e-8)
    print(
        "[IG] Completeness check: "
        f"lhs={signed_contrib_sum:.6e}, rhs={rhs:.6e}, rel_err={rel_err:.6%}"
    )

    score_da = xarray.DataArray(
        score_map,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="ig_candidate_score",
    )
    return {
        "score_da": score_da,
        "maps_by_var": ig_maps_by_var,
        "lhs": signed_contrib_sum,
        "rhs": rhs,
        "rel_err": rel_err,
    }


def _select_top_k_candidates(
    ig_score_da: xarray.DataArray,
    ig_maps_by_var: Dict[str, np.ndarray],
    top_k: int,
) -> List[CandidatePoint]:
    score = np.asarray(ig_score_da.values)
    lat_vals = np.asarray(ig_score_da.coords["lat"].values)
    lon_vals = np.asarray(ig_score_da.coords["lon"].values)
    n_lat, n_lon = score.shape

    safe_score = np.where(np.isfinite(score), score, -np.inf)
    actual_k = max(1, min(int(top_k), safe_score.size))
    flat_idx = np.argsort(safe_score.ravel())[::-1][:actual_k]

    candidates: List[CandidatePoint] = []
    for rank, idx in enumerate(flat_idx, start=1):
        lat_idx = int(idx // n_lon)
        lon_idx = int(idx % n_lon)
        dominant_var = max(
            ig_maps_by_var,
            key=lambda name: abs(float(ig_maps_by_var[name][lat_idx, lon_idx])),
        )
        candidates.append(
            CandidatePoint(
                lat_idx=lat_idx,
                lon_idx=lon_idx,
                lat=float(lat_vals[lat_idx]),
                lon=float(lon_vals[lon_idx]),
                ig_score=float(score[lat_idx, lon_idx]),
                ig_rank=rank,
                dominant_var=dominant_var,
            )
        )
    return candidates


def _evaluate_candidates_with_perturbation(
    context,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
    baseline_inputs: xarray.Dataset,
    candidates: List[CandidatePoint],
) -> List[Dict[str, Any]]:
    time_sel = slice(None) if runtime_cfg.perturb_time == "all" else int(runtime_cfg.perturb_time)

    print(f"\n[Perturb] Evaluating {len(candidates)} IG candidates...")
    rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        lat_start = max(candidate.lat_idx - runtime_cfg.patch_radius, 0)
        lat_end = min(candidate.lat_idx + runtime_cfg.patch_radius + 1, len(context.lat_vals))
        lon_start = max(candidate.lon_idx - runtime_cfg.patch_radius, 0)
        lon_end = min(candidate.lon_idx + runtime_cfg.patch_radius + 1, len(context.lon_vals))
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
            for target_var in context.target_vars:
                out_var = select_target_data(
                    outputs,
                    target_var,
                    target_levels=runtime_cfg.target_levels,
                    target_level=runtime_cfg.target_level,
                )
                new_value = out_var.isel(time=runtime_cfg.target_time_idx).sel(
                    lat=context.center_lat,
                    lon=context.center_lon,
                    method="nearest",
                ).values.item()
                delta_by_target[target_var] = float(new_value - context.base_values[target_var])

            delta_values = np.array(list(delta_by_target.values()), dtype=np.float64)
            rows.append(
                {
                    "lat": candidate.lat,
                    "lon": candidate.lon,
                    "lat_idx": candidate.lat_idx,
                    "lon_idx": candidate.lon_idx,
                    "ig_score": candidate.ig_score,
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


def _compute_erf_explanation_map(
    context,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
) -> xarray.DataArray:
    def _loss(inputs_data):
        return _combined_target_scalar(context, runtime_cfg, inputs_data)

    grad_fn = jax.grad(_loss)
    grads = grad_fn(context.eval_inputs)

    lat_vals = np.asarray(context.eval_inputs.coords["lat"].values)
    lon_vals = np.asarray(context.eval_inputs.coords["lon"].values)
    erf_score = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)

    for var_name in vars_to_use:
        grad_values = np.asarray(grads[var_name].values)
        if runtime_cfg.erf_abs:
            grad_values = np.abs(grad_values)
        grad_map_da = reduce_input_attribution_to_latlon(
            attribution=grad_values,
            original_da=context.eval_inputs[var_name],
            runtime_cfg=runtime_cfg,
        )
        erf_score += np.abs(np.asarray(grad_map_da.values, dtype=np.float64))

    return xarray.DataArray(
        erf_score,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="erf_explanation_score",
    )


def _save_ranking_csv(
    rows: List[Dict[str, Any]],
    target_vars: List[str],
    output_csv: str,
) -> Path:
    csv_path = ROOT_DIR / output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "lat",
                "lon",
                "lat_idx",
                "lon_idx",
                "delta_abs_mean",
                "delta_signed_mean",
                "ig_score",
                "ig_rank",
                "dominant_var",
                *[f"delta_{target_var}" for target_var in target_vars],
            ]
        )
        for rank, item in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    item["lat"],
                    item["lon"],
                    item["lat_idx"],
                    item["lon_idx"],
                    item["delta_abs_mean"],
                    item["delta_signed_mean"],
                    item["ig_score"],
                    item["ig_rank"],
                    item["dominant_var"],
                    *[item["delta_by_target"].get(target_var, np.nan) for target_var in target_vars],
                ]
            )
    return csv_path


def _save_heatmaps(
    runtime_cfg: AnalysisConfig,
    center_lat: float,
    center_lon: float,
    ig_score_da: xarray.DataArray,
    erf_score_da: xarray.DataArray,
    output_ig_png: Optional[str],
    output_erf_png: Optional[str],
) -> None:
    if output_ig_png:
        ig_path = ROOT_DIR / output_ig_png
        plot_importance_heatmap(
            importance_da=ig_score_da,
            center_lat=center_lat,
            center_lon=center_lon,
            output_path=ig_path,
            title="IG candidate score map",
            cmap=runtime_cfg.gradient_cmap,
            dpi=runtime_cfg.heatmap_dpi,
            vmax_quantile=runtime_cfg.gradient_vmax_quantile,
            diverging=False,
            cbar_label="sum_v |IG_v(lat,lon)|",
            center_window_deg=runtime_cfg.gradient_center_window_deg,
            center_s_quantile=runtime_cfg.gradient_center_scale_quantile,
            alpha_quantile=runtime_cfg.gradient_alpha_quantile,
        )
        print(f"Saved IG score map: {ig_path}")

    if output_erf_png:
        erf_path = ROOT_DIR / output_erf_png
        plot_importance_heatmap(
            importance_da=erf_score_da,
            center_lat=center_lat,
            center_lon=center_lon,
            output_path=erf_path,
            title="ERF explanation map",
            cmap=runtime_cfg.erf_cmap,
            dpi=runtime_cfg.heatmap_dpi,
            vmax_quantile=runtime_cfg.erf_vmax_quantile,
            diverging=False,
            cbar_label="sum_v |d target / d input_v|",
            center_window_deg=runtime_cfg.erf_center_window_deg,
            center_s_quantile=runtime_cfg.erf_center_scale_quantile,
            alpha_quantile=runtime_cfg.erf_alpha_quantile,
        )
        print(f"Saved ERF explanation map: {erf_path}")


def run_gridpoint_importance_ranking(
) -> Dict[str, Any]:
    runtime_cfg = AnalysisConfig.from_module(cfg)
    context = build_analysis_context(runtime_cfg)

    top_k = max(1, int(runtime_cfg.top_k_candidates))
    top_n = max(1, int(runtime_cfg.top_n_report))
    include_target_inputs = bool(runtime_cfg.include_target_inputs)
    output_csv = runtime_cfg.output_csv
    output_ig_png = runtime_cfg.output_ig_png
    output_erf_png = runtime_cfg.output_erf_png

    vars_to_use = _resolve_spatial_variables(
        eval_inputs=context.eval_inputs,
        perturb_variables=runtime_cfg.perturb_variables,
        target_vars=context.target_vars,
        include_target_inputs=include_target_inputs,
    )
    if not vars_to_use:
        raise ValueError("No candidate spatial variables found")

    _, mean_by_level, _ = load_normalization_stats(runtime_cfg.dir_path_stats)
    baseline_inputs = _build_climatology_baseline_inputs(
        eval_inputs=context.eval_inputs,
        vars_to_use=vars_to_use,
        mean_by_level=mean_by_level,
    )

    actual_ig_steps = int(runtime_cfg.gradient_steps)
    if actual_ig_steps <= 0:
        raise ValueError("ig_steps must be positive")

    print("\n=== Typhoon Gridpoint Importance Ranking ===")
    print(f"target variables: {', '.join(context.target_vars)}")
    print(f"input variables used: {len(vars_to_use)}")
    print(f"top_k candidates from IG: {top_k}")
    print(f"patch_radius for perturbation: {runtime_cfg.patch_radius}")

    ig_result = _compute_ig_candidate_score_map(
        context=context,
        runtime_cfg=runtime_cfg,
        vars_to_use=vars_to_use,
        baseline_inputs=baseline_inputs,
        ig_steps=actual_ig_steps,
    )
    ig_score_da = ig_result["score_da"]
    candidates = _select_top_k_candidates(
        ig_score_da=ig_score_da,
        ig_maps_by_var=ig_result["maps_by_var"],
        top_k=top_k,
    )

    ranked_rows = _evaluate_candidates_with_perturbation(
        context=context,
        runtime_cfg=runtime_cfg,
        vars_to_use=vars_to_use,
        baseline_inputs=baseline_inputs,
        candidates=candidates,
    )

    print(f"\nTop-{min(top_n, len(ranked_rows))} grid points by |Δ|:")
    for rank, item in enumerate(ranked_rows[:top_n], start=1):
        print(
            f"{rank:02d}. lat={item['lat']:.2f}, lon={item['lon']:.2f} "
            f"|Δ|={item['delta_abs_mean']:.6e} "
            f"Δ={item['delta_signed_mean']:+.6e} "
            f"IG_rank={item['ig_rank']} "
            f"dom_var={item['dominant_var']}"
        )

    csv_path = _save_ranking_csv(
        rows=ranked_rows,
        target_vars=context.target_vars,
        output_csv=output_csv,
    )
    print(f"Saved ranking CSV: {csv_path}")

    print("\n[ERF] Computing explanation map...")
    erf_score_da = _compute_erf_explanation_map(
        context=context,
        runtime_cfg=runtime_cfg,
        vars_to_use=vars_to_use,
    )
    _save_heatmaps(
        runtime_cfg=runtime_cfg,
        center_lat=context.center_lat,
        center_lon=context.center_lon,
        ig_score_da=ig_score_da,
        erf_score_da=erf_score_da,
        output_ig_png=output_ig_png,
        output_erf_png=output_erf_png,
    )

    return {
        "target_vars": context.target_vars,
        "vars_to_use": vars_to_use,
        "top_k": top_k,
        "top_n": top_n,
        "ig_steps": actual_ig_steps,
        "candidates": candidates,
        "ranked_rows": ranked_rows,
        "ig_completeness": {
            "lhs": ig_result["lhs"],
            "rhs": ig_result["rhs"],
            "rel_err": ig_result["rel_err"],
        },
        "output_csv": str(csv_path),
    }


def main() -> int:
    run_gridpoint_importance_ranking()
    print("done")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        raise SystemExit(exit_code)
