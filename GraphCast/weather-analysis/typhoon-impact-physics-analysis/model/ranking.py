#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""台风关键网格点排名主流程编排（IG → 候选筛选 → 排名）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray

import config as cfg
from shared.analysis_pipeline import AnalysisConfig, build_analysis_context
from shared.baseline import _build_climatology_baseline_inputs
from shared.heatmap_utils import plot_importance_heatmap
from shared.model_utils import load_normalization_stats
from model.ig.runner import _build_patch_candidate_maps, _compute_ig_candidate_score_map
from model.perturbation.runner import _select_top_k_candidates


ROOT_DIR = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()


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


def _save_ranking_csv(
    rows: List[Dict[str, Any]],
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
                "patch_radius",
                "candidate_score",
                "point_ig_score",
                "ig_rank",
                "dominant_var",
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
                    item["patch_radius"],
                    item["candidate_score"],
                    item["point_ig_score"],
                    item["ig_rank"],
                    item["dominant_var"],
                ]
            )
    return csv_path


def run_gridpoint_importance_ranking() -> Dict[str, Any]:
    runtime_cfg = AnalysisConfig.from_module(cfg)
    context = build_analysis_context(runtime_cfg)

    top_k = max(1, int(runtime_cfg.top_k_candidates))
    top_n = max(1, int(runtime_cfg.top_n_report))
    include_target_inputs = bool(runtime_cfg.include_target_inputs)
    patch_score_agg = str(runtime_cfg.patch_score_agg).lower().strip()
    patch_radius = int(runtime_cfg.patch_radius)
    output_csv = runtime_cfg.output_csv
    output_ig_png = runtime_cfg.output_ig_png

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
    print(f"patch_radius: {patch_radius}")
    print(f"patch score aggregation: {patch_score_agg}")

    ig_result = _compute_ig_candidate_score_map(
        context=context,
        runtime_cfg=runtime_cfg,
        vars_to_use=vars_to_use,
        baseline_inputs=baseline_inputs,
        ig_steps=actual_ig_steps,
    )
    point_ig_score_da = ig_result["score_da"]
    lat_vals = np.asarray(point_ig_score_da.coords["lat"].values)
    lon_vals = np.asarray(point_ig_score_da.coords["lon"].values)

    patch_result = _build_patch_candidate_maps(
        ig_maps_by_var=ig_result["maps_by_var"],
        patch_radius=patch_radius,
        patch_score_agg=patch_score_agg,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
    )
    candidates = _select_top_k_candidates(
        candidate_score_da=patch_result["score_da"],
        patch_maps_by_var=patch_result["maps_by_var"],
        top_k=top_k,
        point_ig_score_da=point_ig_score_da,
    )

    # Convert candidates to ranking rows (ranking-only, no perturbation evaluation)
    ranked_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        ranked_rows.append({
            "lat": float(cand.lat),
            "lon": float(cand.lon),
            "lat_idx": int(cand.lat_idx),
            "lon_idx": int(cand.lon_idx),
            "patch_radius": patch_radius,
            "candidate_score": float(cand.candidate_score),
            "point_ig_score": float(cand.point_ig_score),
            "ig_rank": int(cand.ig_rank),
            "dominant_var": str(cand.dominant_var),
        })

    ig_score_da = patch_result["score_da"]

    print(f"\nTop-{min(top_n, len(ranked_rows))} grid points by candidate score:")
    for rank, item in enumerate(ranked_rows[:top_n], start=1):
        print(
            f"{rank:02d}. lat={item['lat']:.2f}, lon={item['lon']:.2f} "
            f"score={item['candidate_score']:.6e} "
            f"IG_rank={item['ig_rank']} "
            f"dom_var={item['dominant_var']}"
        )

    csv_path = _save_ranking_csv(
        rows=ranked_rows,
        output_csv=output_csv,
    )
    print(f"Saved ranking CSV: {csv_path}")

    # Save IG heatmap only (ranking-only workflow)
    if output_ig_png:
        ig_path = ROOT_DIR / output_ig_png
        plot_importance_heatmap(
            importance_da=ig_score_da,
            center_lat=context.center_lat,
            center_lon=context.center_lon,
            output_path=ig_path,
            title=f"Patch IG candidate score map (r={patch_radius}, agg={patch_score_agg})",
            cmap=runtime_cfg.gradient_cmap,
            dpi=runtime_cfg.heatmap_dpi,
            vmax_quantile=runtime_cfg.gradient_vmax_quantile,
            diverging=False,
            cbar_label=f"sum_v patch_{patch_score_agg}(|IG_v|, r={patch_radius})",
            center_window_deg=runtime_cfg.gradient_center_window_deg,
            center_s_quantile=runtime_cfg.gradient_center_scale_quantile,
            alpha_quantile=runtime_cfg.gradient_alpha_quantile,
        )
        print(f"Saved IG score map: {ig_path}")

    return {
        "target_vars": context.target_vars,
        "vars_to_use": vars_to_use,
        "top_k": top_k,
        "top_n": top_n,
        "ig_steps": actual_ig_steps,
        "patch_radius": patch_radius,
        "patch_score_agg": patch_score_agg,
        "candidates": candidates,
        "ranked_rows": ranked_rows,
        "ig_completeness": {
            "lhs": ig_result["lhs"],
            "rhs": ig_result["rhs"],
            "rel_err": ig_result["rel_err"],
        },
        "output_csv": str(csv_path),
    }
