from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import numpy as np
import xarray

import config as cfg
from shared.analysis_pipeline import AnalysisConfig, build_analysis_context
from shared.importance_common import reduce_input_attribution_to_latlon, target_scalar
from shared.model_utils import load_normalization_stats
from shared.baseline import _build_climatology_baseline_inputs
from physics.swe.alignment import (
    compute_alignment_report,
    plot_alignment_scatter,
    plot_topk_overlap_maps,
    plot_topk_iou_curves,
    save_report_json,
)
from physics.swe.swe_sensitivity import (
    compute_sensitivity_jax,
    extract_swe_initial_conditions,
)
from physics.swe.metrics import compute_anisotropy_ratio_km, compute_upstream_fraction
from physics.swe.steering import compute_deep_layer_environmental_steering

ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
RESULTS_DIR = ROOT_DIR / "validation_results"

_SWE_COMPARABLE_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
]


def _sweep_weight_map(swe_result) -> np.ndarray:
    signed = np.asarray(swe_result.dJ_dh_signed, dtype=np.float64)
    neg = np.maximum(-signed, 0.0)
    if np.sum(neg) > 0.0:
        return neg
    return np.asarray(swe_result.S_h, dtype=np.float64)


def _compute_upstream_and_anisotropy(swe_result, lead_h: float) -> Dict[str, float]:
    weights = _sweep_weight_map(swe_result)
    upstream_fraction = compute_upstream_fraction(
        weights=weights,
        lat_vals=swe_result.lat_vals,
        lon_vals=swe_result.lon_vals,
        center_lat=swe_result.center_lat,
        center_lon=swe_result.center_lon,
        U_bar=swe_result.physics_cfg.U_bar,
        V_bar=swe_result.physics_cfg.V_bar,
        lead_h=lead_h,
        core_radius_deg=float(getattr(cfg, "SWE_UPSTREAM_CORE_EXCLUDE_DEG", 0.0)),
    )
    anisotropy_ratio = compute_anisotropy_ratio_km(weights, swe_result.lat_vals, swe_result.lon_vals)
    return {
        "upstream_fraction": float(upstream_fraction),
        "anisotropy_ratio": float(anisotropy_ratio),
    }


def _run_steering_sweep(
    h0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    swe_lat: np.ndarray,
    swe_lon: np.ndarray,
    center_lat: float,
    center_lon: float,
    t_idx: int,
    lead_h: float,
    sigma_deg: float,
    swe_dt: float,
    core_radius_deg: float,
    constraint_mode: str,
    base_u: float,
    base_v: float,
    H_eq: float,
    rayleigh_momentum_h: float,
    rayleigh_height_h: float,
    diffusion_coeff: float,
    sponge_width: int,
    sponge_efold_h: float,
) -> List[Dict[str, float]]:
    mags = list(getattr(cfg, "SWE_UBAR_SWEEP_MAGS", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]))
    if not mags:
        return []

    vec_norm = float(np.hypot(base_u, base_v))
    if vec_norm <= 1e-8:
        u_hat, v_hat = 1.0, 0.0
    else:
        u_hat, v_hat = base_u / vec_norm, base_v / vec_norm

    rows: List[Dict[str, float]] = []
    for mag in mags:
        forced_u = float(mag) * float(u_hat)
        forced_v = float(mag) * float(v_hat)
        sweep_result = compute_sensitivity_jax(
            h0,
            u0,
            v0,
            swe_lat,
            swe_lon,
            center_lat,
            center_lon,
            t_idx,
            sigma_deg=sigma_deg,
            dt=swe_dt,
            core_radius_deg=core_radius_deg,
            forced_U_bar=forced_u,
            forced_V_bar=forced_v,
            lead_hours_override=lead_h,
            constraint_mode=constraint_mode,
            H_eq=H_eq,
            rayleigh_momentum_h=rayleigh_momentum_h,
            rayleigh_height_h=rayleigh_height_h,
            diffusion_coeff=diffusion_coeff,
            sponge_width=sponge_width,
            sponge_efold_h=sponge_efold_h,
        )
        metrics = _compute_upstream_and_anisotropy(sweep_result, lead_h)
        rows.append(
            {
                "U_mag": float(mag),
                "U_bar": float(forced_u),
                "V_bar": float(forced_v),
                "upstream_fraction": metrics["upstream_fraction"],
                "anisotropy_ratio": metrics["anisotropy_ratio"],
            }
        )

    rows.sort(key=lambda r: r["U_mag"])
    return rows


def _compute_deep_layer_steering_from_eval_inputs(
    eval_inputs: xarray.Dataset,
    swe_lat: np.ndarray,
    swe_lon: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> Optional[Dict[str, float]]:
    try:
        u_da = eval_inputs["u_component_of_wind"]
        v_da = eval_inputs["v_component_of_wind"]
        if "batch" in u_da.dims:
            u_da = u_da.isel(batch=0)
            v_da = v_da.isel(batch=0)
        if "time" in u_da.dims:
            u_da = u_da.isel(time=1)
            v_da = v_da.isel(time=1)
        if "level" not in u_da.dims:
            return None

        u_da = u_da.sel(lat=swe_lat, method="nearest").sel(lon=swe_lon, method="nearest").transpose("level", "lat", "lon")
        v_da = v_da.sel(lat=swe_lat, method="nearest").sel(lon=swe_lon, method="nearest").transpose("level", "lat", "lon")
        levels = np.asarray(u_da.coords["level"].values, dtype=np.float32)

        res = compute_deep_layer_environmental_steering(
            u_levels=np.asarray(u_da.values, dtype=np.float32),
            v_levels=np.asarray(v_da.values, dtype=np.float32),
            levels_hpa=levels,
            lat_vals=np.asarray(swe_lat, dtype=np.float32),
            lon_vals=np.asarray(swe_lon, dtype=np.float32),
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            core_radius_deg=float(getattr(cfg, "SWE_CORE_RADIUS_DEG", 3.0)),
            annulus_inner_km=float(getattr(cfg, "SWE_STEERING_ANNULUS_INNER_KM", 300.0)),
            annulus_outer_km=float(getattr(cfg, "SWE_STEERING_ANNULUS_OUTER_KM", 900.0)),
            min_env_points=int(getattr(cfg, "SWE_STEERING_MIN_ENV_POINTS", 30)),
        )
        return {
            "U_bar": float(res.U_bar),
            "V_bar": float(res.V_bar),
            "n_env": float(res.n_env),
            "n_total": float(res.n_total),
        }
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] Deep-layer steering unavailable, fallback to single-level steering: {exc}")
        return None


def _resolve_spatial_variables(
    eval_inputs: xarray.Dataset,
    perturb_variables: Optional[List[str]],
    target_var: str,
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
    return [var_name for var_name in selected if var_name != target_var]


def _compute_gnn_ig_for_swe_vars(
    context,
    runtime_cfg: AnalysisConfig,
    baseline_inputs: xarray.Dataset,
) -> Dict[str, np.ndarray]:
    vars_to_use = [v for v in _SWE_COMPARABLE_VARS if v in context.eval_inputs.data_vars]
    if not vars_to_use:
        raise ValueError("SWE 可比变量（geopotential/u/v）均不在 eval_inputs 中")

    def _loss(inputs_data):
        return target_scalar(context, runtime_cfg, inputs_data, context.target_var)

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
        out["uv_500"] = np.sqrt(u_map ** 2 + v_map ** 2)
    elif u_map is not None:
        out["uv_500"] = u_map
    elif v_map is not None:
        out["uv_500"] = v_map

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
        target_var=context.target_var,
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
    core_radius_deg = float(getattr(cfg, "SWE_CORE_RADIUS_DEG", 3.0))
    constraint_mode = getattr(cfg, "SWE_CONSTRAINT_MODE", "none")
    H_eq = float(getattr(cfg, "SWE_EQ_DEPTH_M", 120.0))
    rayleigh_momentum_h = float(getattr(cfg, "SWE_RAYLEIGH_MOMENTUM_H", 12.0))
    rayleigh_height_h = float(getattr(cfg, "SWE_RAYLEIGH_HEIGHT_H", 24.0))
    diffusion_coeff = float(getattr(cfg, "SWE_DIFFUSION_COEFF", 5e4))
    sponge_width = int(getattr(cfg, "SWE_SPONGE_WIDTH", 6))
    sponge_efold_h = float(getattr(cfg, "SWE_SPONGE_EFOLD_H", 2.0))
    print(f"  Constraint mode: {constraint_mode}")
    print(f"  Core-mask radius: {core_radius_deg:.2f}°")

    h0, u0, v0, swe_lat, swe_lon = extract_swe_initial_conditions(
        context.eval_inputs, context.center_lat, context.center_lon,
        domain_half_deg=domain_half,
    )
    print(f"  Sub-domain: {h0.shape}  lat∈[{swe_lat.min():.1f},{swe_lat.max():.1f}]  "
          f"lon∈[{swe_lon.min():.1f},{swe_lon.max():.1f}]")
    print(f"  h0 range: {h0.min():.0f}–{h0.max():.0f} m  "
          f"u0 range: {u0.min():.1f}–{u0.max():.1f} m/s")

    forced_U_bar = None
    forced_V_bar = None
    if bool(getattr(cfg, "SWE_USE_DEEP_LAYER_STEERING", True)):
        deep = _compute_deep_layer_steering_from_eval_inputs(
            eval_inputs=context.eval_inputs,
            swe_lat=swe_lat,
            swe_lon=swe_lon,
            center_lat=context.center_lat,
            center_lon=context.center_lon,
        )
        if deep is not None:
            forced_U_bar = float(deep["U_bar"])
            forced_V_bar = float(deep["V_bar"])
            print(
                f"  Deep-layer steering override: U_bar={forced_U_bar:+.2f} m/s  "
                f"V_bar={forced_V_bar:+.2f} m/s  env={int(deep['n_env'])}/{int(deep['n_total'])}"
            )

    jax_result = compute_sensitivity_jax(
        h0, u0, v0, swe_lat, swe_lon,
        context.center_lat, context.center_lon,
        t_idx, sigma_deg=sigma_deg, dt=swe_dt,
        core_radius_deg=core_radius_deg,
        forced_U_bar=forced_U_bar,
        forced_V_bar=forced_V_bar,
        constraint_mode=constraint_mode,
        H_eq=H_eq,
        rayleigh_momentum_h=rayleigh_momentum_h,
        rayleigh_height_h=rayleigh_height_h,
        diffusion_coeff=diffusion_coeff,
        sponge_width=sponge_width,
        sponge_efold_h=sponge_efold_h,
    )

    baseline_metrics = _compute_upstream_and_anisotropy(jax_result, lead_h)
    print(f"  [Physics] anisotropy_ratio={baseline_metrics['anisotropy_ratio']:.3f}  upstream_fraction={baseline_metrics['upstream_fraction']:.3f}")

    print("\n[Phase 1b] Forced steering sweep (advection monotonicity)")
    sweep_rows = _run_steering_sweep(
        h0=h0,
        u0=u0,
        v0=v0,
        swe_lat=swe_lat,
        swe_lon=swe_lon,
        center_lat=context.center_lat,
        center_lon=context.center_lon,
        t_idx=t_idx,
        lead_h=float(lead_h),
        sigma_deg=sigma_deg,
        swe_dt=swe_dt,
        core_radius_deg=core_radius_deg,
        constraint_mode=constraint_mode,
        base_u=float(jax_result.physics_cfg.U_bar),
        base_v=float(jax_result.physics_cfg.V_bar),
        H_eq=H_eq,
        rayleigh_momentum_h=rayleigh_momentum_h,
        rayleigh_height_h=rayleigh_height_h,
        diffusion_coeff=diffusion_coeff,
        sponge_width=sponge_width,
        sponge_efold_h=sponge_efold_h,
    )
    if sweep_rows:
        sweep_path = RESULTS_DIR / f"alignment_ubar_sweep_t{t_idx}.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(
            json.dumps(sweep_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved sweep: {sweep_path}")
        print("  Sweep upstream_fraction:", [round(float(r["upstream_fraction"]), 3) for r in sweep_rows])

    print("\n[Phase 2] GNN IG for SWE-comparable vars (geopotential, u, v @ 500hPa)")
    gnn_ig_raw = _compute_gnn_ig_for_swe_vars(context, runtime_cfg, baseline_inputs)
    print(f"  GNN IG computed for: {list(gnn_ig_raw.keys())}")

    # Resolve patch parameters early for IG sanity check
    patch_radius = getattr(cfg, "PHYSICS_PATCH_RADIUS", runtime_cfg.patch_radius)
    patch_score_agg = getattr(cfg, "PHYSICS_PATCH_SCORE_AGG", runtime_cfg.patch_score_agg)

    # Run IG sanity check if enabled
    ig_sanity_payload: Dict[str, Any] = {"status": "skipped", "reason": "disabled", "passed": None}
    sanity_path = RESULTS_DIR / "ig_sanity_metrics.json"
    if runtime_cfg.ig_sanity_enable:
        print("\n[Phase 2b] IG Perturbation Sanity Check")
        from physics.swe.ig_sanity import run_ig_perturb_sanity, write_ig_sanity_report
        ig_sanity_payload = run_ig_perturb_sanity(
            context=context,
            runtime_cfg=runtime_cfg,
            baseline_inputs=baseline_inputs,
            gnn_ig_raw=gnn_ig_raw,
            patch_radius=patch_radius,
            patch_score_agg=patch_score_agg,
        )
        # Write sanity report
        write_ig_sanity_report(ig_sanity_payload, sanity_path)
        print(f"  IG sanity status: {ig_sanity_payload.get('status')}")
        if ig_sanity_payload.get("status") == "ok":
            print(f"  IG sanity passed: {ig_sanity_payload.get('passed')}")
            print(f"  Lift ratio: {ig_sanity_payload.get('lift_ratio', float('nan')):.3f}")
        elif ig_sanity_payload.get("status") == "failed":
            print(f"  IG sanity reason: {ig_sanity_payload.get('reason')}")
    else:
        # Write skipped report for consistent artifact
        from physics.swe.ig_sanity import write_ig_sanity_report
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        write_ig_sanity_report(ig_sanity_payload, sanity_path)

    gnn_ig_maps = _build_gnn_group_maps(gnn_ig_raw, full_lat, full_lon, swe_lat, swe_lon)
    print(f"  Grouped maps: {list(gnn_ig_maps.keys())}")

    print("\n[Phase 3] Alignment Metrics")
    patch_agg = patch_score_agg  # alias for consistency
    k_values = tuple(getattr(cfg, "PHYSICS_TOPK_VALUES", [20, 50, 100, 200]))

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
    panel_topk_overlap_k = int(getattr(cfg, "SWE_PANEL_TOPK_OVERLAP_K", 50))
    plot_topk_overlap_maps(
        jax_result,
        gnn_ig_maps,
        RESULTS_DIR,
        dpi=dpi,
        patch_radius=patch_radius,
        patch_score_agg=patch_agg,
        topk_overlap_k=panel_topk_overlap_k,
    )
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
        "upstream_fraction_series": [float(r["upstream_fraction"]) for r in sweep_rows if np.isfinite(float(r["upstream_fraction"]))],
        "ig_sanity": ig_sanity_payload,
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
