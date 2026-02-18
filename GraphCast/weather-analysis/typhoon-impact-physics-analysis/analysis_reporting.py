# -*- coding: utf-8 -*-
"""台风影响分析输出的报告辅助模块。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from analysis_pipeline import AnalysisConfig


def build_importance_map_specs(
    importance_mode: str,
    target_vars: Sequence[str],
    importance_maps: Dict[str, np.ndarray],
    compare_maps: Optional[Dict[str, Dict[str, np.ndarray]]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """构建用于绘图和排名辅助函数的命名图谱规范。"""
    map_specs: Dict[str, Dict[str, np.ndarray]] = {}

    if importance_mode == "compare":
        if compare_maps is None:
            raise ValueError("compare_maps is required when IMPORTANCE_MODE='compare'")
        if len(target_vars) != 1:
            raise ValueError("IMPORTANCE_MODE='compare' requires exactly one target variable")
        var = target_vars[0]
        # 明确保持面板顺序以对齐比较图的布局。
        map_specs["perturbation"] = {
            "name": "perturbation_importance",
            "values": compare_maps["perturbation"][var],
        }
        map_specs["input_gradient"] = {
            "name": "gradient_importance",
            "values": compare_maps["input_gradient"][var],
        }
        map_specs["erf"] = {
            "name": "erf_importance",
            "values": compare_maps["erf"][var],
        }
        return map_specs

    for var in target_vars:
        name = "importance" if len(target_vars) == 1 else f"importance_{var}"
        map_specs[name] = {
            "name": name,
            "values": importance_maps[var],
        }
    return map_specs


def build_importance_dataarrays(
    map_specs: Dict[str, Dict[str, np.ndarray]],
    lat_sel_vals: np.ndarray,
    lon_sel_vals: np.ndarray,
) -> Dict[str, Any]:
    """将原始图谱值转换为带有经纬度坐标的 xarray.DataArray。"""
    import xarray

    importance_das: Dict[str, Any] = {}
    for key, spec in map_specs.items():
        importance_das[key] = xarray.DataArray(
            spec["values"],
            dims=("lat", "lon"),
            coords={"lat": lat_sel_vals, "lon": lon_sel_vals},
            name=spec["name"],
        )
    return importance_das


def save_importance_plots_with_center(
    runtime_cfg: AnalysisConfig,
    root_dir: Path,
    target_vars: Sequence[str],
    importance_das: Dict[str, Any],
    center_lat: float,
    center_lon: float,
) -> None:
    from heatmap_utils import (
        plot_importance_heatmap,
        plot_importance_heatmap_cartopy,
        plot_importance_heatmap_dual,
        plot_importance_heatmap_panels,
    )

    if runtime_cfg.importance_mode == "compare":
        var = target_vars[0]
        if runtime_cfg.output_png_method_compare:
            compare_path = root_dir / runtime_cfg.output_png_method_compare
            time_label = f"out_t={runtime_cfg.target_time_idx}, in_t={runtime_cfg.perturb_time}"
            # 三个对齐面板：针对单个目标变量的 perturbation / IG / ERF。
            plot_importance_heatmap_panels(
                [
                    importance_das["perturbation"],
                    importance_das["input_gradient"],
                    importance_das["erf"],
                ],
                [
                    f"Perturbation ({var}, {time_label})",
                    f"IG ({var}, {time_label})",
                    f"ERF ({var}, {time_label})",
                ],
                center_lat,
                center_lon,
                compare_path,
                cmap=[
                    runtime_cfg.heatmap_cmap,
                    runtime_cfg.gradient_cmap,
                    runtime_cfg.erf_cmap,
                ],
                dpi=runtime_cfg.heatmap_dpi,
                vmax_quantile=[
                    runtime_cfg.heatmap_vmax_quantile,
                    runtime_cfg.gradient_vmax_quantile,
                    runtime_cfg.erf_vmax_quantile,
                ],
                diverging=[
                    runtime_cfg.heatmap_diverging,
                    runtime_cfg.gradient_diverging,
                    runtime_cfg.erf_diverging,
                ],
                cbar_label=[
                    "Δoutput (perturbed - baseline)",
                    "IG attribution",
                    "ERF sensitivity",
                ],
                center_window_deg=[
                    0.0,
                    runtime_cfg.gradient_center_window_deg,
                    runtime_cfg.erf_center_window_deg,
                ],
                center_s_quantile=[
                    runtime_cfg.heatmap_vmax_quantile,
                    runtime_cfg.gradient_center_scale_quantile,
                    runtime_cfg.erf_center_scale_quantile,
                ],
                alpha_quantile=[
                    None,
                    runtime_cfg.gradient_alpha_quantile,
                    runtime_cfg.erf_alpha_quantile,
                ],
            )
            print(f"Saved method-compare heatmap: {compare_path}")
        else:
            print("Skip method-compare heatmap: OUTPUT_PNG_METHOD_COMPARE is None")
        return

    if len(target_vars) == 1 and runtime_cfg.output_png:
        var = target_vars[0]
        if runtime_cfg.importance_mode == "input_gradient":
            title = f"IG Importance (t={runtime_cfg.target_time_idx}, var={var})"
            cbar_label = "IG attribution"
            vmax_quantile = runtime_cfg.gradient_vmax_quantile
            cmap = runtime_cfg.gradient_cmap
            diverging = runtime_cfg.gradient_diverging
            center_window_deg = runtime_cfg.gradient_center_window_deg
            center_s_quantile = runtime_cfg.gradient_center_scale_quantile
            alpha_quantile = runtime_cfg.gradient_alpha_quantile
        elif runtime_cfg.importance_mode == "erf":
            title = f"ERF Importance (t={runtime_cfg.target_time_idx}, var={var})"
            cbar_label = "ERF sensitivity"
            vmax_quantile = runtime_cfg.erf_vmax_quantile
            cmap = runtime_cfg.erf_cmap
            diverging = runtime_cfg.erf_diverging
            center_window_deg = runtime_cfg.erf_center_window_deg
            center_s_quantile = runtime_cfg.erf_center_scale_quantile
            alpha_quantile = runtime_cfg.erf_alpha_quantile
        else:
            title = f"Perturbation Importance (t={runtime_cfg.target_time_idx}, var={var})"
            cbar_label = None
            vmax_quantile = runtime_cfg.heatmap_vmax_quantile
            cmap = runtime_cfg.heatmap_cmap
            diverging = runtime_cfg.heatmap_diverging
            center_window_deg = 0.0
            center_s_quantile = runtime_cfg.heatmap_vmax_quantile
            alpha_quantile = None

        png_path = root_dir / runtime_cfg.output_png
        plot_importance_heatmap(
            importance_das["importance"],
            center_lat,
            center_lon,
            png_path,
            title,
            cmap=cmap,
            dpi=runtime_cfg.heatmap_dpi,
            vmax_quantile=vmax_quantile,
            diverging=diverging,
            cbar_label=cbar_label,
            center_window_deg=center_window_deg,
            center_s_quantile=center_s_quantile,
            alpha_quantile=alpha_quantile,
        )
        print(f"Saved heatmap: {png_path}")

    if len(target_vars) == 2 and runtime_cfg.output_png_combined:
        if runtime_cfg.importance_mode == "input_gradient":
            cbar_label = "IG attribution"
            vmax_quantile = runtime_cfg.gradient_vmax_quantile
            cmap = runtime_cfg.gradient_cmap
            diverging = runtime_cfg.gradient_diverging
            center_window_deg = runtime_cfg.gradient_center_window_deg
            center_s_quantile = runtime_cfg.gradient_center_scale_quantile
            alpha_quantile = runtime_cfg.gradient_alpha_quantile
        elif runtime_cfg.importance_mode == "erf":
            cbar_label = "ERF sensitivity"
            vmax_quantile = runtime_cfg.erf_vmax_quantile
            cmap = runtime_cfg.erf_cmap
            diverging = runtime_cfg.erf_diverging
            center_window_deg = runtime_cfg.erf_center_window_deg
            center_s_quantile = runtime_cfg.erf_center_scale_quantile
            alpha_quantile = runtime_cfg.erf_alpha_quantile
        else:
            cbar_label = None
            vmax_quantile = runtime_cfg.heatmap_vmax_quantile
            cmap = runtime_cfg.heatmap_cmap
            diverging = runtime_cfg.heatmap_diverging
            center_window_deg = 0.0
            center_s_quantile = runtime_cfg.heatmap_vmax_quantile
            alpha_quantile = None

        combined_path = root_dir / runtime_cfg.output_png_combined
        plot_importance_heatmap_dual(
            [
                importance_das[f"importance_{target_vars[0]}"],
                importance_das[f"importance_{target_vars[1]}"],
            ],
            [f"{target_vars[0]} (t={runtime_cfg.target_time_idx})", f"{target_vars[1]} (t={runtime_cfg.target_time_idx})"],
            center_lat,
            center_lon,
            combined_path,
            cmap=cmap,
            dpi=runtime_cfg.heatmap_dpi,
            vmax_quantile=vmax_quantile,
            diverging=diverging,
            cbar_label=cbar_label,
            center_window_deg=center_window_deg,
            center_s_quantile=center_s_quantile,
            alpha_quantile=alpha_quantile,
        )
        print(f"Saved combined heatmap: {combined_path}")

    if runtime_cfg.output_png_cartopy and len(target_vars) == 1:
        if runtime_cfg.importance_mode == "input_gradient":
            title = f"IG Importance Map (t={runtime_cfg.target_time_idx}, var={target_vars[0]})"
            cbar_label = "IG attribution"
            vmax_quantile = runtime_cfg.gradient_vmax_quantile
            cmap = runtime_cfg.gradient_cmap
            diverging = runtime_cfg.gradient_diverging
            center_window_deg = runtime_cfg.gradient_center_window_deg
            center_s_quantile = runtime_cfg.gradient_center_scale_quantile
            alpha_quantile = runtime_cfg.gradient_alpha_quantile
        elif runtime_cfg.importance_mode == "erf":
            title = f"ERF Importance Map (t={runtime_cfg.target_time_idx}, var={target_vars[0]})"
            cbar_label = "ERF sensitivity"
            vmax_quantile = runtime_cfg.erf_vmax_quantile
            cmap = runtime_cfg.erf_cmap
            diverging = runtime_cfg.erf_diverging
            center_window_deg = runtime_cfg.erf_center_window_deg
            center_s_quantile = runtime_cfg.erf_center_scale_quantile
            alpha_quantile = runtime_cfg.erf_alpha_quantile
        else:
            title = f"Perturbation Importance Map (t={runtime_cfg.target_time_idx}, var={target_vars[0]})"
            cbar_label = None
            vmax_quantile = runtime_cfg.heatmap_vmax_quantile
            cmap = runtime_cfg.heatmap_cmap
            diverging = runtime_cfg.heatmap_diverging
            center_window_deg = 0.0
            center_s_quantile = runtime_cfg.heatmap_vmax_quantile
            alpha_quantile = None
        map_path = root_dir / runtime_cfg.output_png_cartopy
        try:
            plot_importance_heatmap_cartopy(
                importance_das["importance"],
                center_lat,
                center_lon,
                map_path,
                title,
                cmap=cmap,
                dpi=runtime_cfg.heatmap_dpi,
                vmax_quantile=vmax_quantile,
                diverging=diverging,
                cbar_label=cbar_label,
                center_window_deg=center_window_deg,
                center_s_quantile=center_s_quantile,
                alpha_quantile=alpha_quantile,
            )
            print(f"Saved map heatmap: {map_path}")
        except RuntimeError as exc:
            print(f"Skip cartopy map: {exc}")


def print_top_n(
    runtime_cfg: AnalysisConfig,
    target_vars: Sequence[str],
    lat_sel_vals: np.ndarray,
    lon_sel_vals: np.ndarray,
    importance_maps: Dict[str, np.ndarray],
    compare_maps: Optional[Dict[str, Dict[str, np.ndarray]]],
) -> None:
    """按绝对分值打印每个激活方法的前 N 个最具影响力的网格点。"""
    if runtime_cfg.importance_mode == "compare":
        if compare_maps is None:
            raise ValueError("compare_maps is required when IMPORTANCE_MODE='compare'")
        var = target_vars[0]
        for mode_name, mode_maps in (
            ("Perturbation", compare_maps["perturbation"]),
            ("IG", compare_maps["input_gradient"]),
            ("ERF", compare_maps["erf"]),
        ):
            flat_idx = np.argsort(np.abs(mode_maps[var]).ravel())[::-1][: runtime_cfg.top_n]
            print(f"\nTop-{runtime_cfg.top_n} influential grid points for {var} ({mode_name}):")
            for rank, idx in enumerate(flat_idx, start=1):
                row = idx // mode_maps[var].shape[1]
                col = idx % mode_maps[var].shape[1]
                lat = float(lat_sel_vals[row])
                lon = float(lon_sel_vals[col])
                score = float(mode_maps[var][row, col])
                print(f"{rank:02d}. lat={lat:.2f}, lon={lon:.2f}, score={score:.6f}")
        return

    for var in target_vars:
        flat_idx = np.argsort(np.abs(importance_maps[var]).ravel())[::-1][: runtime_cfg.top_n]
        print(f"\nTop-{runtime_cfg.top_n} influential grid points for {var}:")
        for rank, idx in enumerate(flat_idx, start=1):
            row = idx // importance_maps[var].shape[1]
            col = idx % importance_maps[var].shape[1]
            lat = float(lat_sel_vals[row])
            lon = float(lon_sel_vals[col])
            score = float(importance_maps[var][row, col])
            print(f"{rank:02d}. lat={lat:.2f}, lon={lon:.2f}, score={score:.6f}")
