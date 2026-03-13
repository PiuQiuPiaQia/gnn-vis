from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import numpy as np
import xarray

import config as cfg
from shared.analysis_pipeline import AnalysisConfig, build_analysis_context
from shared.importance_common import reduce_input_attribution_to_latlon, target_scalar
from shared.baseline import _build_climatology_baseline_inputs
from physics.swe.alignment import (
    compute_alignment_report,
    plot_alignment_scatter,
    plot_topk_overlap_maps,
    plot_topk_iou_curves,
    save_report_json,
    _group_metrics,
    AlignmentReport,
)
from physics.swe.metrics import compute_anisotropy_ratio_km, compute_upstream_fraction
from physics.swe.steering import compute_deep_layer_environmental_steering

try:
    import jax
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight unit tests
    def _missing_jax(*args, **kwargs):
        raise ModuleNotFoundError("jax is required for this code path")

    jax = SimpleNamespace(
        grad=_missing_jax,
        random=SimpleNamespace(PRNGKey=lambda seed: seed),
    )

ROOT_DIR = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
DEFAULT_RESULTS_DIR = ROOT_DIR / "validation_results"

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


def _compute_upstream_and_anisotropy(
    swe_result,
    lead_h: float,
    *,
    cfg_module=cfg,
) -> Dict[str, float]:
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
        core_radius_deg=float(getattr(cfg_module, "SWE_UPSTREAM_CORE_EXCLUDE_DEG", 0.0)),
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
    cfg_module=cfg,
) -> List[Dict[str, float]]:
    from physics.swe.swe_sensitivity import compute_sensitivity_jax

    mags = list(getattr(cfg_module, "SWE_UBAR_SWEEP_MAGS", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]))
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
        metrics = _compute_upstream_and_anisotropy(
            sweep_result,
            lead_h,
            cfg_module=cfg_module,
        )
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
    *,
    cfg_module=cfg,
) -> Optional[Dict[str, float]]:
    if "u_component_of_wind" in eval_inputs:
        u_time_da = eval_inputs["u_component_of_wind"]
        if "time" in u_time_da.dims and int(u_time_da.sizes.get("time", 0)) < 2:
            print("  [warn] Deep-layer steering unavailable: time dim has fewer than 2 steps; falling back to single-level steering.")
            return None

    try:
        u_da = eval_inputs["u_component_of_wind"]
        v_da = eval_inputs["v_component_of_wind"]
        if "batch" in u_da.dims:
            u_da = u_da.isel(batch=0)
            v_da = v_da.isel(batch=0)
        if "time" in u_da.dims:
            if int(u_da.sizes.get("time", 0)) < 2:
                raise ValueError("Deep-layer steering requires at least 2 time steps when time dim is present")
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
            core_radius_deg=float(getattr(cfg_module, "SWE_CORE_RADIUS_DEG", 3.0)),
            annulus_inner_km=float(getattr(cfg_module, "SWE_STEERING_ANNULUS_INNER_KM", 300.0)),
            annulus_outer_km=float(getattr(cfg_module, "SWE_STEERING_ANNULUS_OUTER_KM", 900.0)),
            min_env_points=int(getattr(cfg_module, "SWE_STEERING_MIN_ENV_POINTS", 30)),
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
    allowed_variables: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    vars_to_use = [v for v in _SWE_COMPARABLE_VARS if v in context.eval_inputs.data_vars]
    if allowed_variables is not None:
        vars_to_use = [v for v in vars_to_use if v in allowed_variables]
    if not vars_to_use:
        if allowed_variables is not None:
            return {}
        raise ValueError("SWE 可比变量（geopotential/u/v）均不在 eval_inputs 中")

    def _loss(inputs_data):
        return target_scalar(context, runtime_cfg, inputs_data, context.target_var)

    ig_steps = int(runtime_cfg.gradient_steps)
    if ig_steps <= 0:
        raise ValueError(f"gradient_steps must be positive, got {ig_steps}")

    grad_fn = jax.grad(_loss)

    diff_arrays = {
        v: np.asarray(context.eval_inputs[v].values) - np.asarray(baseline_inputs[v].values)
        for v in vars_to_use
    }
    ig_sum = {v: np.zeros_like(diff_arrays[v], dtype=np.float64) for v in vars_to_use}

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
        ig_maps[v] = np.asarray(ig_da.values, dtype=np.float64)

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


def _build_gnn_group_maps_split(
    gnn_ig_raw: Dict[str, np.ndarray],
    full_lat: np.ndarray,
    full_lon: np.ndarray,
    swe_lat: np.ndarray,
    swe_lon: np.ndarray,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    def _crop(var: str) -> Optional[np.ndarray]:
        if var not in gnn_ig_raw:
            return None
        return _crop_gnn_to_swe_domain(gnn_ig_raw[var], full_lat, full_lon, swe_lat, swe_lon)

    z_map  = _crop("geopotential")
    u_map  = _crop("u_component_of_wind")
    v_map  = _crop("v_component_of_wind")

    signed_out: Dict[str, np.ndarray] = {}
    magnitude_out: Dict[str, np.ndarray] = {}
    if z_map is not None:
        signed_out["z_500"] = z_map
        magnitude_out["z_500"] = np.abs(z_map)
    if u_map is not None and v_map is not None:
        magnitude_out["uv_500"] = np.sqrt(u_map ** 2 + v_map ** 2)

    return signed_out, magnitude_out


def _resolve_allowed_swe_ig_variables(
    eval_inputs: xarray.Dataset,
    runtime_cfg: AnalysisConfig,
    target_var: str,
) -> set[str]:
    return set(
        _resolve_spatial_variables(
            eval_inputs=eval_inputs,
            perturb_variables=runtime_cfg.perturb_variables,
            target_var=target_var,
            include_target_inputs=runtime_cfg.include_target_inputs,
        )
    )


def _resolve_allowed_swe_comparable_ig_variables(
    eval_inputs: xarray.Dataset,
    runtime_cfg: AnalysisConfig,
    target_var: str,
) -> set[str]:
    return {
        var_name
        for var_name in _resolve_allowed_swe_ig_variables(eval_inputs, runtime_cfg, target_var)
        if var_name in _SWE_COMPARABLE_VARS and var_name in eval_inputs.data_vars
    }


def _build_gnn_group_map_payload(
    signed_gnn_maps: Dict[str, np.ndarray],
    magnitude_gnn_maps: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    # `gnn_ig_maps` remains the backward-compatible all-magnitude grouped view.
    return {
        "gnn_main_maps": signed_gnn_maps,
        "gnn_ig_maps": magnitude_gnn_maps,
        "gnn_ig_magnitude_maps": magnitude_gnn_maps,
    }


def _build_comparison_result_payload(
    *,
    jax_result: Any,
    signed_gnn_maps: Dict[str, np.ndarray],
    magnitude_gnn_maps: Dict[str, np.ndarray],
    report: AlignmentReport,
    dlmsf_result: Any,
    dlmsf_report: Any,
    track_patch_analysis: Any,
    sweep_rows: List[Dict[str, float]],
    ig_sanity_payload: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    result = {
        "jax_result": jax_result,
        "report": report,
        "dlmsf_result": dlmsf_result,
        "dlmsf_report": dlmsf_report,
        "track_patch_analysis": track_patch_analysis,
        "upstream_fraction_series": [
            float(r["upstream_fraction"])
            for r in sweep_rows
            if np.isfinite(float(r["upstream_fraction"]))
        ],
        "ig_sanity": ig_sanity_payload,
        "elapsed_sec": elapsed,
    }
    result.update(_build_gnn_group_map_payload(signed_gnn_maps, magnitude_gnn_maps))
    return result


def _should_emit_alignment_scatter(
    scatter_pairs: List[tuple[str, np.ndarray, str, str, str]],
) -> bool:
    return bool(scatter_pairs)


def _should_emit_topk_artifacts(
    pairs: List[tuple[str, np.ndarray, str]],
) -> bool:
    return bool(pairs)


def _build_dlmsf_alignment_inputs(
    dlmsf_result: Any,
    signed_gnn_maps: Dict[str, np.ndarray],
    magnitude_gnn_maps: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    warnings: List[str] = []

    if "uv_500" in magnitude_gnn_maps and "uv_500" not in signed_gnn_maps:
        warnings.append(
            "DLMSF main comparison for uv_500 uses magnitude GNN uv_500 only."
        )

    overlap_pairs = [
        ("z", dlmsf_result.S_abs_map, "z_500")
        for _ in [0]
        if "z_500" in magnitude_gnn_maps
    ]
    if "uv_500" in magnitude_gnn_maps:
        overlap_pairs.append(("uv", dlmsf_result.S_abs_map, "uv_500"))

    return {
        "main_specs": [],
        "scatter_pairs": [],
        "overlap_pairs": overlap_pairs,
        "warnings": warnings,
    }


def _build_swe_alignment_inputs(
    swe_result: Any,
    signed_gnn_maps: Dict[str, np.ndarray],
    magnitude_gnn_maps: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    warnings: List[str] = []
    main_pairs_metrics: List[tuple[str, np.ndarray, str]] = []
    main_pairs_scatter: List[tuple[str, np.ndarray, str, str, str]] = []

    if "z_500" in magnitude_gnn_maps:
        main_pairs_metrics.append(("h", swe_result.S_h, "z_500"))
        main_pairs_scatter.append(("h", swe_result.S_h, "z_500", "|SWE $S_h$|", "|GNN IG (z₅₀₀)|"))

    if "uv_500" in magnitude_gnn_maps:
        main_pairs_metrics.append(("uv", swe_result.S_uv, "uv_500"))
        main_pairs_scatter.append(("uv", swe_result.S_uv, "uv_500", "|SWE $S_{uv}$|", "|GNN IG (uv₅₀₀)|"))

    overlap_pairs: List[tuple[str, np.ndarray, str]] = []
    if "z_500" in magnitude_gnn_maps:
        overlap_pairs.append(("h", swe_result.S_h, "z_500"))
    if "uv_500" in magnitude_gnn_maps:
        overlap_pairs.append(("uv", swe_result.S_uv, "uv_500"))

    return {
        "main_gnn_maps": magnitude_gnn_maps,
        "supplemental_gnn_maps": magnitude_gnn_maps,
        "main_pairs_metrics": main_pairs_metrics,
        "main_pairs_scatter": main_pairs_scatter,
        "overlap_pairs": overlap_pairs,
        "warnings": warnings,
    }


def run_physics_comparison(
    *,
    cfg_module=cfg,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")

    t_start = time.perf_counter()
    results_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Physics (SWE) vs GNN IG Comparison ===")
    runtime_cfg = AnalysisConfig.from_module(cfg_module)
    context = build_analysis_context(runtime_cfg)

    vars_all = _resolve_allowed_swe_ig_variables(
        eval_inputs=context.eval_inputs,
        runtime_cfg=runtime_cfg,
        target_var=context.target_var,
    )
    allowed_swe_comparable_ig_vars = _resolve_allowed_swe_comparable_ig_variables(
        eval_inputs=context.eval_inputs,
        runtime_cfg=runtime_cfg,
        target_var=context.target_var,
    )
    from shared.model_utils import load_normalization_stats
    _, mean_by_level, _ = load_normalization_stats(runtime_cfg.dir_path_stats)
    baseline_inputs = _build_climatology_baseline_inputs(
        eval_inputs=context.eval_inputs,
        vars_to_use=sorted(vars_all),
        mean_by_level=mean_by_level,
    )

    full_lat = np.asarray(context.eval_inputs.coords["lat"].values)
    full_lon = np.asarray(context.eval_inputs.coords["lon"].values)
    t_idx = runtime_cfg.target_time_idx
    lead_h = (t_idx + 1) * 6
    print(f"\nTarget: +{lead_h}h (target_time_idx={t_idx})")

    print("\n[Phase 1] SWE Physical Sensitivity (Method A: jax.grad)")
    domain_half = getattr(cfg_module, "SWE_DOMAIN_HALF_DEG", 20.0)
    sigma_deg   = getattr(cfg_module, "SWE_SIGMA_DEG", 3.0)
    swe_dt      = getattr(cfg_module, "SWE_DT", 300.0)
    core_radius_deg = float(getattr(cfg_module, "SWE_CORE_RADIUS_DEG", 3.0))
    constraint_mode = getattr(cfg_module, "SWE_CONSTRAINT_MODE", "none")
    H_eq = float(getattr(cfg_module, "SWE_EQ_DEPTH_M", 120.0))
    rayleigh_momentum_h = float(getattr(cfg_module, "SWE_RAYLEIGH_MOMENTUM_H", 12.0))
    rayleigh_height_h = float(getattr(cfg_module, "SWE_RAYLEIGH_HEIGHT_H", 24.0))
    diffusion_coeff = float(getattr(cfg_module, "SWE_DIFFUSION_COEFF", 5e4))
    sponge_width = int(getattr(cfg_module, "SWE_SPONGE_WIDTH", 6))
    sponge_efold_h = float(getattr(cfg_module, "SWE_SPONGE_EFOLD_H", 2.0))
    print(f"  Constraint mode: {constraint_mode}")
    print(f"  Core-mask radius: {core_radius_deg:.2f}°")

    from physics.swe.swe_sensitivity import (
        compute_sensitivity_jax,
        extract_swe_initial_conditions,
    )
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
    if bool(getattr(cfg_module, "SWE_USE_DEEP_LAYER_STEERING", True)):
        deep = _compute_deep_layer_steering_from_eval_inputs(
            eval_inputs=context.eval_inputs,
            swe_lat=swe_lat,
            swe_lon=swe_lon,
            center_lat=context.center_lat,
            center_lon=context.center_lon,
            cfg_module=cfg_module,
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

    baseline_metrics = _compute_upstream_and_anisotropy(
        jax_result,
        lead_h,
        cfg_module=cfg_module,
    )
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
        cfg_module=cfg_module,
    )
    if sweep_rows:
        sweep_path = results_dir / f"alignment_ubar_sweep_t{t_idx}.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(
            json.dumps(sweep_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved sweep: {sweep_path}")
        print("  Sweep upstream_fraction:", [round(float(r["upstream_fraction"]), 3) for r in sweep_rows])

    print("\n[Phase 1c] DLMSF track-patch comparison")
    dlmsf_result = None
    dlmsf_report = None
    track_patch_analysis = None
    if bool(getattr(cfg_module, "DLMSF_ENABLE", True)):
        print("  Track-patch DLMSF vs IG will run after SWE alignment.")
    else:
        print("  DLMSF_ENABLE=False, skipped.")

    print("\n[Phase 2] GNN IG for SWE-comparable vars (geopotential, u, v @ 500hPa)")
    if allowed_swe_comparable_ig_vars:
        gnn_ig_raw = _compute_gnn_ig_for_swe_vars(
            context,
            runtime_cfg,
            baseline_inputs,
            allowed_variables=allowed_swe_comparable_ig_vars,
        )
        print(f"  GNN IG computed for: {list(gnn_ig_raw.keys())}")
    else:
        gnn_ig_raw = {}
        print("  [warn] No allowed SWE-comparable vars after filtering; skipping GNN IG and alignment tracks.")

    # Resolve patch parameters early for IG sanity check
    patch_radius = getattr(cfg_module, "PHYSICS_PATCH_RADIUS", runtime_cfg.patch_radius)
    patch_score_agg = getattr(cfg_module, "PHYSICS_PATCH_SCORE_AGG", runtime_cfg.patch_score_agg)

    # Run IG sanity check if enabled
    ig_sanity_payload: Dict[str, Any] = {"status": "skipped", "reason": "disabled", "passed": None}
    sanity_path = results_dir / "ig_sanity_metrics.json"
    if not gnn_ig_raw:
        from physics.swe.ig_sanity import write_ig_sanity_report

        ig_sanity_payload = {
            "status": "skipped",
            "reason": "no_swe_comparable_vars",
            "passed": None,
        }
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        write_ig_sanity_report(ig_sanity_payload, sanity_path)
    elif runtime_cfg.ig_sanity_enable:
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

    gnn_signed_maps, gnn_ig_maps = _build_gnn_group_maps_split(
        gnn_ig_raw, full_lat, full_lon, swe_lat, swe_lon,
    )
    print(f"  Grouped maps: {list(gnn_ig_maps.keys())}")

    print("\n[Phase 3] Alignment Metrics")
    patch_agg = patch_score_agg  # alias for consistency
    k_values = tuple(getattr(cfg_module, "PHYSICS_TOPK_VALUES", [20, 50, 100, 200]))

    swe_alignment_inputs = _build_swe_alignment_inputs(jax_result, gnn_signed_maps, gnn_ig_maps)
    for msg in swe_alignment_inputs["warnings"]:
        print(f"  [warn] {msg}")

    report = AlignmentReport(
        target_time_idx=t_idx,
        lead_time_h=lead_h,
        patch_radius=patch_radius,
        patch_score_agg=patch_agg,
        sigma_deg=sigma_deg,
    )
    for group_name, swe_map, gnn_key in swe_alignment_inputs["main_pairs_metrics"]:
        m = _group_metrics(
            swe_map,
            swe_alignment_inputs["main_gnn_maps"][gnn_key],
            group_name=group_name,
            patch_radius=patch_radius,
            patch_score_agg=patch_agg,
            k_values=k_values,
        )
        report.groups.append(m)
        print(f"  [Align] {group_name:6s}: ρ={m.spearman_rho:+.3f}  "
              f"IoU@50={m.topk_iou.get(50, float('nan')):.3f}  n={m.n_valid}")

    print("\n[Phase 4] Saving Visualizations")
    dpi = getattr(cfg_module, "PHYSICS_HEATMAP_DPI", runtime_cfg.heatmap_dpi)
    panel_topk_overlap_k = int(getattr(cfg_module, "SWE_PANEL_TOPK_OVERLAP_K", 50))

    swe_pairs = swe_alignment_inputs["overlap_pairs"]
    if _should_emit_topk_artifacts(swe_pairs):
        plot_topk_overlap_maps(
            swe_pairs, gnn_ig_maps,
            np.asarray(jax_result.lat_vals, dtype=np.float64),
            np.asarray(jax_result.lon_vals, dtype=np.float64),
            float(jax_result.center_lat), float(jax_result.center_lon),
            target_time_idx=t_idx,
            output_dir=results_dir,
            output_prefix="swe",
            dpi=dpi,
            patch_radius=patch_radius,
            patch_score_agg=patch_agg,
            topk_overlap_k=panel_topk_overlap_k,
        )
    else:
        print("  [warn] SWE magnitude overlap/IoU track is empty, skipping top-k artifacts.")

    if _should_emit_alignment_scatter(swe_alignment_inputs["main_pairs_scatter"]):
        plot_alignment_scatter(
            swe_alignment_inputs["main_pairs_scatter"], swe_alignment_inputs["main_gnn_maps"], report,
            target_time_idx=t_idx, lead_time_h=lead_h,
            output_dir=results_dir, output_prefix="swe",
            patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
            abs_gnn_for_display=True,
        )
    else:
        print("  [warn] SWE magnitude main comparison is empty, skipping scatter artifact.")

    if _should_emit_topk_artifacts(swe_pairs):
        plot_topk_iou_curves(
            swe_pairs, gnn_ig_maps,
            target_time_idx=t_idx, lead_time_h=lead_h,
            output_dir=results_dir, output_prefix="swe",
            k_values=k_values, patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
        )

    json_path = results_dir / "physics_alignment_metrics.json"
    save_report_json(report, json_path)

    print("\n[Phase 3b] DLMSF vs IG Track-Patch Comparison")
    if bool(getattr(cfg_module, "DLMSF_ENABLE", True)):
        from physics.dlmsf_patch_fd.patch_comparison import (
            run_track_patch_analysis,
            write_patch_analysis_report,
        )
        from physics.dlmsf_patch_fd.plot_track_patch_report import write_track_patch_figures

        track_patch_analysis = run_track_patch_analysis(
            context=context,
            runtime_cfg=runtime_cfg,
            baseline_inputs=baseline_inputs,
            cfg_module=cfg_module,
        )
        dlmsf_result = track_patch_analysis["main_result"]
        dlmsf_report = track_patch_analysis["summary"]["cases"][track_patch_analysis["main_case"]]
        track_patch_json_path = results_dir / "dlmsf_track_patch_alignment.json"
        write_patch_analysis_report(track_patch_analysis["summary"], track_patch_json_path)
        write_track_patch_figures(
            track_patch_analysis["summary"],
            output_dir=results_dir,
            prefix="dlmsf_track_patch",
            dpi=dpi,
        )

        main_metrics = track_patch_analysis["main_metrics"]
        print(
            "  [Track-Patch] main="
            f"{track_patch_analysis['main_case']}  "
            f"spearman={main_metrics.spearman_rho:+.3f}  "
            f"iou@{main_metrics.topk_k}={main_metrics.iou_topk:.3f}"
        )
        if track_patch_analysis["main_deletion"] is not None:
            deletion = track_patch_analysis["main_deletion"]
            print(
                "  [Deletion] "
                f"AOPC(high_ig)={deletion.high_ig_aopc:.4f}  "
                f"AOPC(low_ig)={deletion.low_ig_aopc:.4f}  "
                f"AOPC(random)={deletion.random_mean_aopc:.4f}"
            )
    else:
        print("  DLMSF result unavailable, skipped.")

    elapsed = time.perf_counter() - t_start
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"Results saved to: {results_dir}/")

    return _build_comparison_result_payload(
        jax_result=jax_result,
        signed_gnn_maps=gnn_signed_maps,
        magnitude_gnn_maps=gnn_ig_maps,
        report=report,
        dlmsf_result=dlmsf_result,
        dlmsf_report=dlmsf_report,
        track_patch_analysis=track_patch_analysis,
        sweep_rows=sweep_rows,
        ig_sanity_payload=ig_sanity_payload,
        elapsed=elapsed,
    )


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
