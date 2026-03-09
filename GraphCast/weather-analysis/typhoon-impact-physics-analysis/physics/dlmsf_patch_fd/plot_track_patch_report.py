from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def _load_report(report_or_path) -> Dict:
    if isinstance(report_or_path, (str, Path)):
        return json.loads(Path(report_or_path).read_text(encoding="utf-8"))
    return dict(report_or_path)


def _as_float_array(values, *, ndim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got shape {arr.shape}")
    return arr


def _as_bool_array(values, *, ndim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.ndim != ndim:
        raise ValueError(f"Expected {ndim}D bool array, got shape {arr.shape}")
    return arr


def _prep_lat_oriented_field(
    field: np.ndarray,
    lat_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field)
    lat = np.asarray(lat_vals, dtype=np.float64)
    if lat.size >= 2 and lat[0] > lat[-1]:
        return np.flipud(arr), lat[::-1]
    return arr, lat


def _plot_field(
    ax,
    field: np.ndarray,
    *,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    title: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    plot_field, plot_lat = _prep_lat_oriented_field(field, lat_vals)
    extent = [
        float(np.min(lon_vals)),
        float(np.max(lon_vals)),
        float(np.min(plot_lat)),
        float(np.max(plot_lat)),
    ]
    image = ax.imshow(
        np.ma.masked_invalid(plot_field),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def plot_track_patch_main_case(report_or_path, output_path: str | Path, dpi: int = 200) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    plot_payload = case.get("plot")
    if not isinstance(plot_payload, dict):
        return None

    lat_vals = _as_float_array(plot_payload.get("lat_vals", []), ndim=1)
    lon_vals = _as_float_array(plot_payload.get("lon_vals", []), ndim=1)
    core_mask = _as_bool_array(plot_payload.get("core_mask", []), ndim=2)
    environment_map = _as_float_array(plot_payload.get("environment_map", []), ndim=2)
    ig_abs_map = _as_float_array(plot_payload.get("ig_abs_map", []), ndim=2)
    dlmsf_abs_map = _as_float_array(plot_payload.get("dlmsf_abs_map", []), ndim=2)
    ig_topq_mask = _as_bool_array(plot_payload.get("ig_topq_mask", []), ndim=2)
    dlmsf_topq_mask = _as_bool_array(plot_payload.get("dlmsf_topq_mask", []), ndim=2)

    overlap_code = np.zeros(environment_map.shape, dtype=np.float64)
    overlap_code[ig_topq_mask] = 1.0
    overlap_code[dlmsf_topq_mask] = 2.0
    overlap_code[ig_topq_mask & dlmsf_topq_mask] = 3.0
    overlap_code[core_mask] = 4.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), dpi=dpi, constrained_layout=True)
    env_name = str(plot_payload.get("environment_field", report.get("track_center_field", "environment")))
    _plot_field(
        axes[0],
        environment_map,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        title=f"Environment ({env_name})",
        cmap="cividis",
        colorbar_label=env_name,
    )
    _plot_field(
        axes[1],
        ig_abs_map,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        title="|IG| Patch Score",
        cmap="YlOrRd",
        colorbar_label="|IG|",
    )
    _plot_field(
        axes[2],
        dlmsf_abs_map,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        title="|DLMSF_parallel| Patch Score",
        cmap="YlGnBu",
        colorbar_label="|DLMSF_parallel|",
    )

    overlap_field, overlap_lat = _prep_lat_oriented_field(overlap_code, lat_vals)
    overlap_extent = [
        float(np.min(lon_vals)),
        float(np.max(lon_vals)),
        float(np.min(overlap_lat)),
        float(np.max(overlap_lat)),
    ]
    overlap_cmap = ListedColormap(["#ffffff", "#d95f02", "#1b9e77", "#2c3e50", "#d9d9d9"])
    overlap_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], overlap_cmap.N)
    axes[3].imshow(
        overlap_field,
        origin="lower",
        extent=overlap_extent,
        aspect="auto",
        cmap=overlap_cmap,
        norm=overlap_norm,
    )
    axes[3].set_title(f"Top-{int(round(100.0 * float(plot_payload.get('topq_fraction', 0.2))))}% Overlap")
    axes[3].set_xlabel("Longitude")
    axes[3].set_ylabel("Latitude")
    axes[3].legend(
        handles=[
            Patch(facecolor="#d95f02", edgecolor="none", label="IG Top-q"),
            Patch(facecolor="#1b9e77", edgecolor="none", label="DLMSF Top-q"),
            Patch(facecolor="#2c3e50", edgecolor="none", label="Overlap"),
            Patch(facecolor="#d9d9d9", edgecolor="none", label="Core"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )

    metrics = case.get("metrics", {})
    annotation = "\n".join(
        [
            f"Pearson: {float(metrics.get('pearson_r', np.nan)):+.3f}",
            f"Spearman: {float(metrics.get('spearman_rho', np.nan)):+.3f}",
            f"IoU@{int(round(100.0 * float(metrics.get('topq_fraction', plot_payload.get('topq_fraction', 0.2)))))}%: {float(metrics.get('iou_topq', np.nan)):.3f}",
        ]
    )
    axes[3].text(
        0.98,
        0.98,
        annotation,
        transform=axes[3].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.suptitle(
        f"DLMSF vs IG Track-Patch Main Case ({main_case})\n"
        f"window={report.get('window_size')}  core={report.get('core_size')}  stride={report.get('stride')}",
        fontsize=12,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_track_patch_deletion(report_or_path, output_path: str | Path, dpi: int = 200) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    deletion = case.get("deletion")
    if not isinstance(deletion, dict):
        return None

    masked_fraction = np.asarray(deletion.get("masked_fraction", []), dtype=np.float64)
    if masked_fraction.size == 0:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi, constrained_layout=True)
    ax.plot(
        masked_fraction,
        np.asarray(deletion.get("high_ig_delta", []), dtype=np.float64),
        color="#c0392b",
        linewidth=2.5,
        label=(
            "Delete High-IG "
            f"(AOPC={float(deletion.get('high_ig_aopc', np.nan)):.3f}, "
            f"AUC={float(deletion.get('high_ig_auc', np.nan)):.3f})"
        ),
    )
    ax.plot(
        masked_fraction,
        np.asarray(deletion.get("random_mean_delta", []), dtype=np.float64),
        color="#7f8c8d",
        linewidth=2.0,
        linestyle=":",
        label=(
            "Delete Random "
            f"(AOPC={float(deletion.get('random_mean_aopc', np.nan)):.3f}, "
            f"AUC={float(deletion.get('random_mean_auc', np.nan)):.3f})"
        ),
    )
    ax.plot(
        masked_fraction,
        np.asarray(deletion.get("low_ig_delta", []), dtype=np.float64),
        color="#2980b9",
        linewidth=2.0,
        linestyle="--",
        label=(
            "Delete Low-IG "
            f"(AOPC={float(deletion.get('low_ig_aopc', np.nan)):.3f}, "
            f"AUC={float(deletion.get('low_ig_auc', np.nan)):.3f})"
        ),
    )
    ax.set_xlabel("Masked Fraction of Environment Window")
    ax.set_ylabel("Delta s_parallel")
    ax.set_title(f"Deletion Validation ({main_case})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_track_patch_figures(
    report_or_path,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "dlmsf_track_patch",
    dpi: int = 200,
) -> List[Path]:
    report = _load_report(report_or_path)
    if output_dir is None:
        if isinstance(report_or_path, (str, Path)):
            output_dir = Path(report_or_path).resolve().parent
        else:
            output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    main_case_path = output_dir / f"{prefix}_main_case.png"
    main_case_written = plot_track_patch_main_case(report, main_case_path, dpi=dpi)
    if main_case_written is not None:
        outputs.append(main_case_written)

    deletion_path = output_dir / f"{prefix}_deletion.png"
    deletion_written = plot_track_patch_deletion(report, deletion_path, dpi=dpi)
    if deletion_written is not None:
        outputs.append(deletion_written)
    return outputs
