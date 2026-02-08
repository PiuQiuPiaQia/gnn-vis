# -*- coding: utf-8 -*-
"""Heatmap visualization for perturbation importance."""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import xarray


def _expand_param(value, n: int, name: str):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} length must be {n}, got {len(value)}")
        return list(value)
    return [value] * n


def _compute_norm(data: np.ndarray, vmax_quantile: Optional[float], diverging: bool):
    if diverging:
        max_abs = float(np.quantile(np.abs(data), vmax_quantile)) if vmax_quantile is not None else float(np.max(np.abs(data)))
        vmax = max_abs
        vmin = -max_abs
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        vmax = float(np.quantile(data, vmax_quantile)) if vmax_quantile is not None else float(np.max(data))
        vmin = float(np.min(data))
        norm = None
    return vmin, vmax, norm


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
):
    lat_vals = importance_da.coords["lat"].values
    lon_vals = importance_da.coords["lon"].values
    data = importance_da.values

    if data.size == 0:
        raise ValueError("importance map is empty")

    vmin, vmax, norm = _compute_norm(data, vmax_quantile, diverging)

    lat_asc = lat_vals[0] < lat_vals[-1]
    origin = "lower" if lat_asc else "upper"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    im = ax.imshow(
        data,
        extent=(lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()),
        origin=origin,
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        aspect="auto",
    )
    ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2, label="Cyclone")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    label = cbar_label
    if label is None:
        label = "Δoutput (perturbed - baseline)" if diverging else "Importance |Δoutput|"
    cbar.set_label(label)
    if diverging:
        ax.text(
            0.01,
            0.99,
            "red: positive, blue: negative",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

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

    vmin, vmax, norm = _compute_norm(data, vmax_quantile, diverging)

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon_vals,
        lat_vals,
        data,
        cmap=cmap,
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
        label="Cyclone",
    )
    ax.set_extent([lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    label = cbar_label
    if label is None:
        label = "Δoutput (perturbed - baseline)" if diverging else "Importance |Δoutput|"
    cbar.set_label(label)
    if diverging:
        ax.text(
            0.01,
            0.99,
            "red: positive, blue: negative",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )

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
    cmap: str = "coolwarm",
    dpi: int = 200,
    vmax_quantile: Union[float, Sequence[float]] = 0.995,
    diverging: Union[bool, Sequence[bool]] = False,
    cbar_label: Union[Optional[str], Sequence[Optional[str]]] = None,
):
    if len(importance_list) != 2:
        raise ValueError("plot_importance_heatmap_dual expects exactly 2 maps")

    n_panel = len(importance_list)
    vmax_quantile_list = _expand_param(vmax_quantile, n_panel, "vmax_quantile")
    diverging_list = _expand_param(diverging, n_panel, "diverging")
    cbar_label_list = _expand_param(cbar_label, n_panel, "cbar_label")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    for i, (ax, importance_da, title) in enumerate(zip(axes, importance_list, titles)):
        lat_vals = importance_da.coords["lat"].values
        lon_vals = importance_da.coords["lon"].values
        data = importance_da.values
        if data.size == 0:
            raise ValueError("importance map is empty")

        is_diverging = bool(diverging_list[i])
        vq = vmax_quantile_list[i]
        vmin, vmax, norm = _compute_norm(data, vq, is_diverging)
        lat_asc = lat_vals[0] < lat_vals[-1]
        origin = "lower" if lat_asc else "upper"

        im = ax.imshow(
            data,
            extent=(lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()),
            origin=origin,
            cmap=cmap,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            norm=norm,
            aspect="auto",
        )
        ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2, label="Cyclone")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.legend(loc="upper right")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        label = cbar_label_list[i]
        if label is None:
            label = "Δoutput (perturbed - baseline)" if is_diverging else "Importance Δoutput"
        cbar.set_label(label)
        if is_diverging:
            ax.text(
                0.01,
                0.99,
                "red: positive, blue: negative",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
