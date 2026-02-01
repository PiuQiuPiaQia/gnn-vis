# -*- coding: utf-8 -*-
"""Heatmap visualization for perturbation importance."""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray


def plot_importance_heatmap(
    importance_da: xarray.DataArray,
    center_lat: float,
    center_lon: float,
    output_path: Path,
    title: str,
    cmap: str = "magma",
    dpi: int = 200,
    vmax_quantile: float = 0.995,
):
    lat_vals = importance_da.coords["lat"].values
    lon_vals = importance_da.coords["lon"].values
    data = importance_da.values

    if data.size == 0:
        raise ValueError("importance map is empty")

    vmax = float(np.quantile(data, vmax_quantile)) if vmax_quantile is not None else float(np.max(data))
    vmin = float(np.min(data))

    lat_asc = lat_vals[0] < lat_vals[-1]
    origin = "lower" if lat_asc else "upper"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    im = ax.imshow(
        data,
        extent=(lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()),
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2, label="Cyclone")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Importance |Δoutput|")

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

    vmax = float(np.quantile(data, vmax_quantile)) if vmax_quantile is not None else float(np.max(data))
    vmin = float(np.min(data))

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon_vals,
        lat_vals,
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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
    cbar.set_label("Importance |Δoutput|")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
