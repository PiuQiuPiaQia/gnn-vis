# -*- coding: utf-8 -*-
"""Heatmap visualization for perturbation importance."""

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
    """Return data in a lat/lon window around (center_lat, center_lon)."""
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


def _compute_norm(
    data: np.ndarray,
    vmax_quantile: Optional[float],
    diverging: bool,
    *,
    center_window: Optional[np.ndarray] = None,
    center_s_quantile: float = 0.99,
):
    """Compute (vmin, vmax, norm). For diverging plots, uses symmetric limits around 0."""
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
    """Mask small |data| values to transparent (via NaN/masked array)."""
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
    vmax_quantile: float = 0.995,
    diverging: bool = False,
    cbar_label: Optional[str] = None,
    center_window_deg: float = 10.0,
    center_s_quantile: float = 0.99,
    alpha_quantile: Optional[float] = None,
):
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


def plot_importance_heatmap_cartopy(
    importance_da: xarray.DataArray,
    center_lat: float,
    center_lon: float,
    output_path: Path,
    title: str,
    cmap: str = "magma",
    dpi: int = 200,
    vmax_quantile: float = 0.995,
    diverging: bool = False,
    cbar_label: Optional[str] = None,
    center_window_deg: float = 10.0,
    center_s_quantile: float = 0.99,
    alpha_quantile: Optional[float] = None,
):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise RuntimeError("cartopy is required for map visualization") from exc

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

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = cast(Any, plt.axes(projection=ccrs.PlateCarree()))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
    mesh = ax.pcolormesh(
        lon_vals,
        lat_vals,
        data,
        cmap=cmap_obj,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        [center_lon],
        [center_lat],
        c="#00e5ff",
        marker="x",
        s=70,
        linewidths=2,
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent([lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

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
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    label = cbar_label
    if label is None:
        label = "Δoutput (perturbed - baseline)" if diverging else "Importance |Δoutput|"
    cbar.set_label(label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_importance_heatmap_dual(
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
):
    if len(importance_list) != 2:
        raise ValueError("plot_importance_heatmap_dual expects exactly 2 maps")

    n_panel = len(importance_list)
    vmax_quantile_list = _expand_param(vmax_quantile, n_panel, "vmax_quantile")
    diverging_list = _expand_param(diverging, n_panel, "diverging")
    cbar_label_list = _expand_param(cbar_label, n_panel, "cbar_label")
    cmap_list = _expand_param(cmap, n_panel, "cmap")
    center_window_deg_list = _expand_param(center_window_deg, n_panel, "center_window_deg")
    center_s_quantile_list = _expand_param(center_s_quantile, n_panel, "center_s_quantile")
    alpha_quantile_list = _expand_param(alpha_quantile, n_panel, "alpha_quantile")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
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

        window = _extract_center_window(data, lat_vals, lon_vals, center_lat, center_lon, panel_window_deg)
        data = _apply_transparency_mask(data, window_ref=window, alpha_quantile=panel_alpha)
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
