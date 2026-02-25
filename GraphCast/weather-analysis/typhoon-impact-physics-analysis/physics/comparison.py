from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np
import xarray

import config as cfg
from shared.analysis_pipeline import AnalysisConfig, build_analysis_context
from shared.importance_common import _combined_target_scalar, reduce_input_attribution_to_latlon
from shared.model_utils import load_normalization_stats
from shared.baseline import _build_climatology_baseline_inputs
from physics.alignment import (
    compute_alignment_report,
    plot_alignment_scatter,
    plot_comparison_panels,
    plot_sensitivity_heatmaps,
    plot_topk_iou_curves,
    save_report_json,
)
from physics.sensitivity import (
    compute_sensitivity_jax,
    extract_swe_initial_conditions,
)

ROOT_DIR = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = ROOT_DIR / "validation_results"

_SWE_COMPARABLE_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]


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


def _compute_gnn_ig_for_swe_vars(
    context,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
) -> Dict[str, np.ndarray]:
    vars_to_use = [v for v in _SWE_COMPARABLE_VARS if v in context.eval_inputs.data_vars]
    if not vars_to_use:
        raise ValueError("SWE 可比变量（geopotential/u/v）均不在 eval_inputs 中")

    def _loss(inputs_data):
        return _combined_target_scalar(context, runtime_cfg, inputs_data)

    grad_fn = jax.grad(_loss)

    diff_arrays = {
        v: np.asarray(context.eval_inputs[v].values) - np.asarray(baseline_inputs[v].values)
        for v in vars_to_use
    }
    ig_sum = {v: np.zeros_like(diff_arrays[v], dtype=np.float64) for v in vars_to_use}

    ig_steps = runtime_cfg.gradient_steps
    print(f"[GNN-IG] Computing IG for {len(vars_to_use)} SWE-comparable vars ({ig_steps} steps)...")
    for step in range(ig_steps):
        alpha = (step + 0.5) / float(ig_steps)
        interp = context.eval_inputs.copy(deep=False)
        for v in vars_to_use:
            interp[v] = xarray.DataArray(
                np.asarray(baseline_inputs[v].values) + alpha * diff_arrays[v],
                dims=context.eval_inputs[v].dims,
                coords=context.eval_inputs[v].coords,
            )
        grads = grad_fn(interp)
        for v in vars_to_use:
            ig_sum[v] += diff_arrays[v] * np.asarray(grads[v].values)
        if (step + 1) % 10 == 0 or (step + 1) == ig_steps:
            print(f"  IG steps: {step+1}/{ig_steps}")

    ig_maps: Dict[str, np.ndarray] = {}
    for v in vars_to_use:
        ig_full = ig_sum[v] / float(ig_steps)
        ig_da = reduce_input_attribution_to_latlon(
            attribution=ig_full,
            original_da=context.eval_inputs[v],
            runtime_cfg=runtime_cfg,
        )
        ig_maps[v] = np.abs(np.asarray(ig_da.values, dtype=np.float64))

    return ig_maps


def _crop_gnn_to_swe_domain(
    gnn_map_full: np.ndarray,
    full_lat: np.ndarray,
    full_lon: np.ndarray,
    swe_lat: np.ndarray,
    swe_lon: np.ndarray,
) -> np.ndarray:
    lat_idx = np.array([np.argmin(np.abs(full_lat - la)) for la in swe_lat])
    lon_idx = np.array([np.argmin(np.abs(full_lon - lo)) for lo in swe_lon])
    return gnn_map_full[np.ix_(lat_idx, lon_idx)]


def _build_gnn_group_maps(
    gnn_ig_raw: Dict[str, np.ndarray],
    full_lat: np.ndarray,
    full_lon: np.ndarray,
    swe_lat: np.ndarray,
    swe_lon: np.ndarray,
) -> Dict[str, np.ndarray]:
    def _crop(var: str) -> Optional[np.ndarray]:
        if var not in gnn_ig_raw:
            return None
        return _crop_gnn_to_swe_domain(gnn_ig_raw[var], full_lat, full_lon, swe_lat, swe_lon)

    z_map  = _crop("geopotential")
    u_map  = _crop("u_component_of_wind")
    v_map  = _crop("v_component_of_wind")

    out: Dict[str, np.ndarray] = {}
    if z_map is not None:
        out["z_500"] = z_map
    if u_map is not None and v_map is not None:
        out["uv_500"] = u_map + v_map
    elif u_map is not None:
        out["uv_500"] = u_map
    elif v_map is not None:
        out["uv_500"] = v_map

    parts = [x for x in [z_map, u_map, v_map] if x is not None]
    if parts:
        out["total"] = sum(parts)

    return out


def run_physics_comparison() -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")

    t_start = time.perf_counter()

    print("\n=== Physics (SWE) vs GNN IG Comparison ===")
    runtime_cfg = AnalysisConfig.from_module(cfg)
    context = build_analysis_context(runtime_cfg)

    vars_all = _resolve_spatial_variables(
        eval_inputs=context.eval_inputs,
        perturb_variables=runtime_cfg.perturb_variables,
        target_vars=context.target_vars,
        include_target_inputs=runtime_cfg.include_target_inputs,
    )
    _, mean_by_level, _ = load_normalization_stats(runtime_cfg.dir_path_stats)
    baseline_inputs = _build_climatology_baseline_inputs(
        eval_inputs=context.eval_inputs,
        vars_to_use=vars_all,
        mean_by_level=mean_by_level,
    )

    full_lat = np.asarray(context.eval_inputs.coords["lat"].values)
    full_lon = np.asarray(context.eval_inputs.coords["lon"].values)
    t_idx = runtime_cfg.target_time_idx
    lead_h = (t_idx + 1) * 6
    print(f"\nTarget: +{lead_h}h (target_time_idx={t_idx})")

    print("\n[Phase 1] SWE Physical Sensitivity (Method A: jax.grad)")
    domain_half = getattr(cfg, "SWE_DOMAIN_HALF_DEG", 20.0)
    sigma_deg   = getattr(cfg, "SWE_SIGMA_DEG", 3.0)
    swe_dt      = getattr(cfg, "SWE_DT", 300.0)
    constraint_mode = getattr(cfg, "SWE_CONSTRAINT_MODE", "none")
    print(f"  Constraint mode: {constraint_mode}")

    h0, u0, v0, swe_lat, swe_lon = extract_swe_initial_conditions(
        context.eval_inputs, context.center_lat, context.center_lon,
        domain_half_deg=domain_half,
    )
    print(f"  Sub-domain: {h0.shape}  lat∈[{swe_lat.min():.1f},{swe_lat.max():.1f}]  "
          f"lon∈[{swe_lon.min():.1f},{swe_lon.max():.1f}]")
    print(f"  h0 range: {h0.min():.0f}–{h0.max():.0f} m  "
          f"u0 range: {u0.min():.1f}–{u0.max():.1f} m/s")

    jax_result = compute_sensitivity_jax(
        h0, u0, v0, swe_lat, swe_lon,
        context.center_lat, context.center_lon,
        t_idx, sigma_deg=sigma_deg, dt=swe_dt,
        constraint_mode=constraint_mode,
    )

    print("\n[Phase 2] GNN IG for SWE-comparable vars (geopotential, u, v @ 500hPa)")
    gnn_ig_raw = _compute_gnn_ig_for_swe_vars(context, runtime_cfg, baseline_inputs)
    print(f"  GNN IG computed for: {list(gnn_ig_raw.keys())}")

    gnn_ig_maps = _build_gnn_group_maps(gnn_ig_raw, full_lat, full_lon, swe_lat, swe_lon)
    print(f"  Grouped maps: {list(gnn_ig_maps.keys())}")

    print("\n[Phase 3] Alignment Metrics")
    patch_radius = getattr(cfg, "PHYSICS_PATCH_RADIUS", runtime_cfg.patch_radius)
    patch_agg    = getattr(cfg, "PHYSICS_PATCH_SCORE_AGG", runtime_cfg.patch_score_agg)
    k_values     = tuple(getattr(cfg, "PHYSICS_TOPK_VALUES", [20, 50, 100, 200]))

    report = compute_alignment_report(
        swe_result=jax_result,
        gnn_ig_maps=gnn_ig_maps,
        patch_radius=patch_radius,
        patch_score_agg=patch_agg,
        sigma_deg=sigma_deg,
        k_values=k_values,
    )

    print("\n[Phase 4] Saving Visualizations")
    dpi = getattr(cfg, "PHYSICS_HEATMAP_DPI", runtime_cfg.heatmap_dpi)

    plot_sensitivity_heatmaps(jax_result, RESULTS_DIR, dpi=dpi)
    plot_comparison_panels(jax_result, gnn_ig_maps, RESULTS_DIR, dpi=dpi)
    plot_alignment_scatter(jax_result, gnn_ig_maps, report, RESULTS_DIR,
                           patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi)
    plot_topk_iou_curves(jax_result, gnn_ig_maps, RESULTS_DIR,
                         k_values=k_values, patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi)

    json_path = RESULTS_DIR / "physics_alignment_metrics.json"
    save_report_json(report, json_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Results saved to: {RESULTS_DIR}/")

    return {
        "jax_result": jax_result,
        "gnn_ig_maps": gnn_ig_maps,
        "report": report,
        "elapsed_sec": elapsed,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SWE physical sensitivity vs GNN IG alignment (single lead time)"
    )
    return p


def main(argv=None) -> int:
    _build_parser().parse_args(argv)
    run_physics_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
