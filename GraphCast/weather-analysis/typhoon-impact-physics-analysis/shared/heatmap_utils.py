# -*- coding: utf-8 -*-
"""台风影响重要性的热图可视化模块。"""

from pathlib import Path
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
import matplotlib.pyplot as plt
import xarray


def _expand_param(value, n: int, name: str):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} length must be {n}, got {len(value)}")
        return list(value)
    return [value] * n


def _finite_values(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if np.ma.isMaskedArray(arr):
        vals = np.ma.compressed(arr)
    else:
        vals = arr.ravel()
    if vals.size == 0:
        return vals
    return vals[np.isfinite(vals)]


def _extract_center_window(
    data: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    window_deg: float,
) -> np.ndarray:
    """返回以 (center_lat, center_lon) 为中心的经纬度窗口内的数据。"""
    if window_deg <= 0:
        return data

    lat_mask = (lat_vals >= (center_lat - window_deg)) & (lat_vals <= (center_lat + window_deg))
    dlon = ((lon_vals - center_lon + 180.0) % 360.0) - 180.0
    lon_mask = np.abs(dlon) <= window_deg

    if not np.any(lat_mask) or not np.any(lon_mask):
        return data

    return data[np.ix_(lat_mask, lon_mask)]


def _safe_abs_quantile(x: np.ndarray, q: float) -> float:
    vals = _finite_values(x)
    if vals.size == 0:
        return 0.0
    val = float(np.quantile(np.abs(vals), q))
    if not np.isfinite(val):
        return 0.0
    return val


def _finite_positive_values(data: np.ndarray) -> np.ndarray:
    vals = _finite_values(data)
    if vals.size == 0:
        return vals
    return vals[vals > 0.0]


def _safe_nonnegative_quantile(data: np.ndarray, q: float) -> float:
    vals = _finite_values(np.maximum(np.asarray(data, dtype=np.float64), 0.0))
    if vals.size == 0:
        return 0.0
    return float(np.quantile(vals, q))


def prepare_shared_nonnegative_display(
    swe_map: np.ndarray,
    ig_map: np.ndarray,
    *,
    norm_quantile: float = 0.99,
    transform: str = "asinh",
    asinh_scale_quantile: float = 0.90,
    log_eps_quantile: float = 0.01,
) -> dict:
    """Build shared-scale display maps for non-negative SWE/IG comparisons.

    Returns transformed maps plus shared color limits so SWE and IG panels can
    use exactly the same color norm per channel.
    """
    q = float(np.clip(norm_quantile, 0.0, 1.0))
    swe = np.maximum(np.asarray(swe_map, dtype=np.float64), 0.0)
    ig = np.maximum(np.asarray(ig_map, dtype=np.float64), 0.0)

    swe_q = _safe_nonnegative_quantile(swe, q)
    ig_q = _safe_nonnegative_quantile(ig, q)
    shared_raw_vmax = max(swe_q, ig_q, 1e-12)

    transform_key = str(transform).lower().strip()
    if transform_key not in {"linear", "asinh", "log"}:
        raise ValueError(f"unsupported display transform: {transform}")

    pos = np.concatenate([_finite_positive_values(swe), _finite_positive_values(ig)])
    if transform_key == "asinh":
        if pos.size == 0:
            scale = 1.0
        else:
            q_scale = float(np.clip(asinh_scale_quantile, 0.0, 1.0))
            scale = float(np.quantile(pos, q_scale))
            if (not np.isfinite(scale)) or scale <= 0.0:
                scale = float(np.max(pos))
            if (not np.isfinite(scale)) or scale <= 0.0:
                scale = 1.0
        swe_show = np.arcsinh(swe / scale)
        ig_show = np.arcsinh(ig / scale)
        shared_vmin = 0.0
        shared_vmax = float(np.arcsinh(shared_raw_vmax / scale))
        transform_label = f"asinh(x/{scale:.3g})"
    elif transform_key == "log":
        if pos.size == 0:
            eps = 1e-12
        else:
            q_eps = float(np.clip(log_eps_quantile, 0.0, 1.0))
            eps = float(np.quantile(pos, q_eps))
            if (not np.isfinite(eps)) or eps <= 0.0:
                eps = max(float(np.min(pos)), 1e-12)
        swe_show = np.log10(swe + eps)
        ig_show = np.log10(ig + eps)
        shared_vmin = float(np.log10(eps))
        shared_vmax = float(np.log10(shared_raw_vmax + eps))
        transform_label = f"log10(x+{eps:.3g})"
    else:
        swe_show = swe
        ig_show = ig
        shared_vmin = 0.0
        shared_vmax = float(shared_raw_vmax)
        transform_label = "linear"

    if shared_vmax <= shared_vmin:
        shared_vmax = shared_vmin + 1e-12

    denom = shared_raw_vmax + 1e-12
    swe_norm = np.clip(swe / denom, 0.0, 1.0)
    ig_norm = np.clip(ig / denom, 0.0, 1.0)

    return {
        "swe_show": swe_show,
        "ig_show": ig_show,
        "swe_norm": swe_norm,
        "ig_norm": ig_norm,
        "shared_raw_vmax": float(shared_raw_vmax),
        "shared_vmin": float(shared_vmin),
        "shared_vmax": float(shared_vmax),
        "transform_label": transform_label,
    }


def _compute_norm(
    data: np.ndarray,
    vmax_quantile: Optional[float],
    diverging: bool,
    *,
    center_window: Optional[np.ndarray] = None,
    center_s_quantile: float = 0.99,
):
    """计算 (vmin, vmax, norm)。对于发散型图，使用以 0 为中心的对称范围。"""
    if diverging:
        ref = center_window if center_window is not None else data
        s = _safe_abs_quantile(ref, center_s_quantile) if vmax_quantile is not None else float(np.max(np.abs(ref)))
        if s <= 0:
            s = float(np.max(np.abs(data)))
        if s <= 0:
            s = 1e-12
        vmin = -s
        vmax = +s
        from matplotlib.colors import TwoSlopeNorm

        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        vals = _finite_values(data)
        if vals.size == 0:
            vals = np.array([0.0])
        vmax = float(np.quantile(vals, vmax_quantile)) if vmax_quantile is not None else float(np.max(vals))
        vmin = float(np.min(vals))
        norm = None
    return vmin, vmax, norm


def _build_legend_handles(cmap_obj, diverging: bool, alpha_quantile: Optional[float]):
    from matplotlib.artist import Artist
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles: list[Artist] = [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markeredgecolor="#00e5ff",
            markeredgewidth=2,
            markersize=8,
            label="Cyclone center",
        )
    ]

    if diverging:
        handles.extend(
            [
                Patch(facecolor=cmap_obj(0.90), edgecolor="none", label="Positive impact"),
                Patch(facecolor=cmap_obj(0.10), edgecolor="none", label="Negative impact"),
            ]
        )
    else:
        handles.extend(
            [
                Patch(facecolor=cmap_obj(0.90), edgecolor="none", label="Higher impact"),
                Patch(facecolor=cmap_obj(0.20), edgecolor="none", label="Lower impact"),
            ]
        )

    if alpha_quantile is not None:
        handles.append(
            Patch(
                facecolor="#d9d9d9",
                edgecolor="none",
                label=f"Transparent low-signal (< q={alpha_quantile:.2f})",
            )
        )
    return handles


def _build_explanation_text(
    *,
    diverging: bool,
    vmax_quantile: Optional[float],
    center_window_deg: float,
    center_s_quantile: float,
    alpha_quantile: Optional[float],
) -> str:
    lines = []
    if vmax_quantile is None:
        lines.append("Color scale: full finite range.")
    else:
        lines.append(f"Color scale clipped at q={vmax_quantile:.3f}.")

    if diverging:
        if center_window_deg > 0:
            lines.append(
                "Zero-centered limits from cyclone "
                f"+/-{center_window_deg:g} deg window (|value| q={center_s_quantile:.2f})."
            )
        else:
            lines.append("Zero-centered symmetric color scale.")

    if alpha_quantile is not None:
        lines.append(f"Cells below |value| q={alpha_quantile:.2f} are transparent.")

    return "\n".join(lines)


def _add_legend_and_explanation(
    ax,
    *,
    cmap_obj,
    diverging: bool,
    vmax_quantile: Optional[float],
    center_window_deg: float,
    center_s_quantile: float,
    alpha_quantile: Optional[float],
):
    ax.legend(
        handles=_build_legend_handles(cmap_obj, diverging, alpha_quantile),
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    explanation = _build_explanation_text(
        diverging=diverging,
        vmax_quantile=vmax_quantile,
        center_window_deg=center_window_deg,
        center_s_quantile=center_s_quantile,
        alpha_quantile=alpha_quantile,
    )
    if explanation:
        ax.text(
            0.01,
            0.01,
            explanation,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )


def _apply_transparency_mask(
    data: np.ndarray,
    *,
    window_ref: np.ndarray,
    alpha_quantile: Optional[float],
) -> np.ndarray:
    """将 |data| 较小的值遮蔽为透明（通过 NaN/掩码数组）。"""
    if alpha_quantile is None:
        return data
    thr = _safe_abs_quantile(window_ref, alpha_quantile)
    if thr <= 0:
        return data
    return np.ma.masked_where(np.abs(data) < thr, data)


def plot_importance_heatmap(
    importance_da: xarray.DataArray,
    center_lat: float,
    center_lon: float,
    output_path: Path,
    title: str,
    cmap: str = "magma",
    dpi: int = 200,
    vmax_quantile: Optional[float] = 0.995,
    diverging: bool = False,
    cbar_label: Optional[str] = None,
    center_window_deg: float = 10.0,
    center_s_quantile: float = 0.99,
    alpha_quantile: Optional[float] = None,
) -> None:
    lat_vals = importance_da.coords["lat"].values
    lon_vals = importance_da.coords["lon"].values
    data = importance_da.values

    if data.size == 0:
        raise ValueError("importance map is empty")

    window = _extract_center_window(data, lat_vals, lon_vals, center_lat, center_lon, center_window_deg)
    data = _apply_transparency_mask(data, window_ref=window, alpha_quantile=alpha_quantile)
    vmin, vmax, norm = _compute_norm(
        np.asarray(data),
        vmax_quantile,
        diverging,
        center_window=window,
        center_s_quantile=center_s_quantile,
    )

    lat_asc = lat_vals[0] < lat_vals[-1]
    origin = "lower" if lat_asc else "upper"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
    im = ax.imshow(
        data,
        extent=(lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()),
        origin=origin,
        cmap=cmap_obj,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        aspect="auto",
    )
    ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    _add_legend_and_explanation(
        ax,
        cmap_obj=cmap_obj,
        diverging=diverging,
        vmax_quantile=vmax_quantile,
        center_window_deg=center_window_deg,
        center_s_quantile=center_s_quantile,
        alpha_quantile=alpha_quantile,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    label = cbar_label
    if label is None:
        label = "Δoutput (perturbed - baseline)" if diverging else "Importance |Δoutput|"
    cbar.set_label(label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_importance_heatmap_panels(
    importance_list: list,
    titles: list,
    center_lat: float,
    center_lon: float,
    output_path: Path,
    cmap: Union[str, Sequence[str]] = "coolwarm",
    dpi: int = 200,
    vmax_quantile: Union[Optional[float], Sequence[Optional[float]]] = 0.995,
    diverging: Union[bool, Sequence[bool]] = False,
    cbar_label: Union[Optional[str], Sequence[Optional[str]]] = None,
    center_window_deg: Union[float, Sequence[float]] = 10.0,
    center_s_quantile: Union[float, Sequence[float]] = 0.99,
    alpha_quantile: Union[Optional[float], Sequence[Optional[float]]] = None,
    fixed_vmin: Union[Optional[float], Sequence[Optional[float]]] = None,
    fixed_vmax: Union[Optional[float], Sequence[Optional[float]]] = None,
) -> None:
    """渲染一个或多个并排重要性面板，每个面板独立样式。"""
    if not importance_list:
        raise ValueError("plot_importance_heatmap_panels expects at least one map")
    if len(titles) != len(importance_list):
        raise ValueError("titles length must match number of maps")

    n_panel = len(importance_list)
    vmax_quantile_list = _expand_param(vmax_quantile, n_panel, "vmax_quantile")
    diverging_list = _expand_param(diverging, n_panel, "diverging")
    cbar_label_list = _expand_param(cbar_label, n_panel, "cbar_label")
    cmap_list = _expand_param(cmap, n_panel, "cmap")
    center_window_deg_list = _expand_param(center_window_deg, n_panel, "center_window_deg")
    center_s_quantile_list = _expand_param(center_s_quantile, n_panel, "center_s_quantile")
    alpha_quantile_list = _expand_param(alpha_quantile, n_panel, "alpha_quantile")
    fixed_vmin_list = _expand_param(fixed_vmin, n_panel, "fixed_vmin")
    fixed_vmax_list = _expand_param(fixed_vmax, n_panel, "fixed_vmax")

    fig, axes = plt.subplots(1, n_panel, figsize=(7 * n_panel, 6), dpi=dpi)
    if n_panel == 1:
        axes = [axes]

    for i, (ax, importance_da, title) in enumerate(zip(axes, importance_list, titles)):
        lat_vals = importance_da.coords["lat"].values
        lon_vals = importance_da.coords["lon"].values
        data = importance_da.values
        if data.size == 0:
            raise ValueError("importance map is empty")

        is_diverging = bool(diverging_list[i])
        vq_item = cast(Any, vmax_quantile_list[i])
        vq = None if vq_item is None else float(vq_item)
        panel_window_deg = float(cast(Any, center_window_deg_list[i]))
        panel_s_quantile = float(cast(Any, center_s_quantile_list[i]))
        panel_alpha_item = cast(Any, alpha_quantile_list[i])
        panel_alpha = None if panel_alpha_item is None else float(panel_alpha_item)
        panel_cmap = str(cast(Any, cmap_list[i]))
        fixed_vmin_item = cast(Any, fixed_vmin_list[i])
        fixed_vmax_item = cast(Any, fixed_vmax_list[i])
        panel_fixed_vmin = None if fixed_vmin_item is None else float(fixed_vmin_item)
        panel_fixed_vmax = None if fixed_vmax_item is None else float(fixed_vmax_item)

        # 逐面板归一化确保混合方法（如 perturb/IG）的可读性。
        window = _extract_center_window(data, lat_vals, lon_vals, center_lat, center_lon, panel_window_deg)
        data = _apply_transparency_mask(data, window_ref=window, alpha_quantile=panel_alpha)
        if panel_fixed_vmin is not None and panel_fixed_vmax is not None:
            vmin = panel_fixed_vmin
            vmax = panel_fixed_vmax
            if is_diverging:
                from matplotlib.colors import TwoSlopeNorm

                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = None
        else:
            vmin, vmax, norm = _compute_norm(
                np.asarray(data),
                vq,
                is_diverging,
                center_window=window,
                center_s_quantile=panel_s_quantile,
            )
        lat_asc = lat_vals[0] < lat_vals[-1]
        origin = "lower" if lat_asc else "upper"

        cmap_obj = plt.get_cmap(panel_cmap).copy()
        cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
        im = ax.imshow(
            data,
            extent=(lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()),
            origin=origin,
            cmap=cmap_obj,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            aspect="auto",
        )
        ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        _add_legend_and_explanation(
            ax,
            cmap_obj=cmap_obj,
            diverging=is_diverging,
            vmax_quantile=vq,
            center_window_deg=panel_window_deg,
            center_s_quantile=panel_s_quantile,
            alpha_quantile=panel_alpha,
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        label = cbar_label_list[i]
        if label is None:
            label = "Δoutput (perturbed - baseline)" if is_diverging else "Importance Δoutput"
        cbar.set_label(str(label))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
