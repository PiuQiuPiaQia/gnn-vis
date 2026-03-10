from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Annotation helpers — exposed for unit-test verification
# ---------------------------------------------------------------------------


def _format_overlap_annotation(*, spearman_rho: float, iou_at_20: float) -> str:
    """Return annotation string for the overlap figure. No Pearson."""
    return (
        f"Spearman \u03c1: {float(spearman_rho):+.3f}\n"
        f"IoU@20%: {float(iou_at_20):.3f}"
    )


def _format_scatter_annotation(*, spearman_rho: float) -> str:
    """Return annotation string for the scatter figure. No Pearson."""
    return f"Spearman \u03c1: {float(spearman_rho):+.3f}"


def _format_deletion_annotation(
    *, aopc_high: float, aopc_random: float, aopc_low: float
) -> str:
    """Return annotation string for the deletion figure. No AUC."""
    return (
        f"AOPC_high: {float(aopc_high):.3f}\n"
        f"AOPC_random: {float(aopc_random):.3f}\n"
        f"AOPC_low: {float(aopc_low):.3f}"
    )


def _format_sign_map_annotation(*, sign_agreement_at_20: float) -> str:
    """Return annotation string for the sign map figure. No Pearson."""
    return f"Sign agreement@20%: {float(sign_agreement_at_20):.2%}"


# ---------------------------------------------------------------------------
# Figure 1: |IG| + |DLMSF| heatmaps + Top-q overlap binary map
# ---------------------------------------------------------------------------


def plot_track_patch_overlap_q20(
    report_or_path, output_path: str | Path, dpi: int = 200
) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    viz = case.get("visualization")
    if not isinstance(viz, dict):
        return None
    overlap = viz.get("overlap")
    if not isinstance(overlap, dict):
        return None

    lat_vals = _as_float_array(overlap["lat_vals"], ndim=1)
    lon_vals = _as_float_array(overlap["lon_vals"], ndim=1)
    ig_abs_map = _as_float_array(overlap["ig_abs_map"], ndim=2)
    dlmsf_abs_map = _as_float_array(overlap["dlmsf_abs_map"], ndim=2)
    overlap_mask = _as_bool_array(overlap["overlap_mask"], ndim=2)
    spearman_rho = float(overlap["spearman_rho"])
    iou_at_20 = float(overlap["iou_at_20"])

    meta = viz.get("meta", {})
    direction = str(meta.get("direction", "along"))
    topq_pct = int(round(float(meta.get("topq_fraction", 0.2)) * 100))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi, constrained_layout=True)

    _plot_field(
        axes[0],
        ig_abs_map,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        title="|IG| Patch Score",
        cmap="YlOrRd",
        colorbar_label="|IG|",
    )
    _plot_field(
        axes[1],
        dlmsf_abs_map,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        title=f"|DLMSF_{direction}| Patch Score",
        cmap="YlGnBu",
        colorbar_label=f"|DLMSF_{direction}|",
    )

    # Binary overlap map
    overlap_float = overlap_mask.astype(np.float64)
    overlap_field, overlap_lat = _prep_lat_oriented_field(overlap_float, lat_vals)
    extent = [
        float(np.min(lon_vals)),
        float(np.max(lon_vals)),
        float(np.min(overlap_lat)),
        float(np.max(overlap_lat)),
    ]
    overlap_cmap = ListedColormap(["#f0f0f0", "#2c3e50"])
    overlap_norm = BoundaryNorm([-0.5, 0.5, 1.5], overlap_cmap.N)
    axes[2].imshow(
        overlap_field,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=overlap_cmap,
        norm=overlap_norm,
    )
    axes[2].set_title(f"Top-{topq_pct}% Overlap")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].legend(
        handles=[
            mpatches.Patch(facecolor="#f0f0f0", edgecolor="#aaaaaa", label="Not in overlap"),
            mpatches.Patch(facecolor="#2c3e50", edgecolor="none", label="Overlap"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )

    annotation = _format_overlap_annotation(spearman_rho=spearman_rho, iou_at_20=iou_at_20)
    axes[2].text(
        0.98,
        0.98,
        annotation,
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 2: Patch-level scatter (rank agreement)
# ---------------------------------------------------------------------------


def plot_track_patch_scatter(
    report_or_path, output_path: str | Path, dpi: int = 200
) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    viz = case.get("visualization")
    if not isinstance(viz, dict):
        return None
    scatter = viz.get("scatter")
    if not isinstance(scatter, dict):
        return None

    x = _as_float_array(scatter["x_patch_abs_scores"], ndim=1)
    y = _as_float_array(scatter["y_patch_abs_scores"], ndim=1)
    spearman_rho = float(scatter["spearman_rho"])

    meta = viz.get("meta", {})
    direction = str(meta.get("direction", "along"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi, constrained_layout=True)
    ax.scatter(x, y, alpha=0.7, s=40, color="#2980b9")
    ax.set_xlabel("|IG| patch score")
    ax.set_ylabel(f"|DLMSF_{direction}| patch score")
    ax.set_title("Patch-Level Rank Agreement")
    ax.grid(alpha=0.25)

    annotation = _format_scatter_annotation(spearman_rho=spearman_rho)
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 3: Deletion validation (3 curves, AOPC only — no AUC)
# ---------------------------------------------------------------------------


def plot_track_patch_deletion_validation(
    report_or_path, output_path: str | Path, dpi: int = 200
) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    viz = case.get("visualization")
    if not isinstance(viz, dict):
        return None
    deletion = viz.get("deletion")
    if not isinstance(deletion, dict):
        return None

    masked_fraction = np.asarray(deletion.get("masked_fraction", []), dtype=np.float64)
    if masked_fraction.size == 0:
        return None

    high_ig_delta = np.asarray(deletion.get("high_ig_delta", []), dtype=np.float64)
    low_ig_delta = np.asarray(deletion.get("low_ig_delta", []), dtype=np.float64)
    random_mean_delta = np.asarray(deletion.get("random_mean_delta", []), dtype=np.float64)
    aopc_high = float(deletion.get("aopc_high", float("nan")))
    aopc_random = float(deletion.get("aopc_random", float("nan")))
    aopc_low = float(deletion.get("aopc_low", float("nan")))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi, constrained_layout=True)
    ax.plot(
        masked_fraction,
        high_ig_delta,
        color="#c0392b",
        linewidth=2.5,
        label=f"Delete High-IG (AOPC={aopc_high:.3f})",
    )
    ax.plot(
        masked_fraction,
        random_mean_delta,
        color="#7f8c8d",
        linewidth=2.0,
        linestyle=":",
        label=f"Delete Random (AOPC={aopc_random:.3f})",
    )
    ax.plot(
        masked_fraction,
        low_ig_delta,
        color="#2980b9",
        linewidth=2.0,
        linestyle="--",
        label=f"Delete Low-IG (AOPC={aopc_low:.3f})",
    )
    ax.set_xlabel("Masked Fraction of Environment Window")
    ax.set_ylabel("Delta score")
    ax.set_title(f"Deletion Validation ({main_case})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    annotation = _format_deletion_annotation(
        aopc_high=aopc_high, aopc_random=aopc_random, aopc_low=aopc_low
    )
    ax.text(
        0.98,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 4: Sign agreement map (Top-q overlap patches)
# ---------------------------------------------------------------------------

# sign_class codes:
#   0 = outside overlap
#   1 = ++ (both positive, agree)
#   2 = -- (both negative, agree)
#   3 = opposite or non-finite (disagree)

_SIGN_CMAP = ListedColormap(["#f0f0f0", "#27ae60", "#2980b9", "#c0392b"])
_SIGN_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _SIGN_CMAP.N)


def plot_track_patch_sign_map(
    report_or_path, output_path: str | Path, dpi: int = 200
) -> Path | None:
    report = _load_report(report_or_path)
    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    viz = case.get("visualization")
    if not isinstance(viz, dict):
        return None
    sign_map_data = viz.get("sign_map")
    if not isinstance(sign_map_data, dict):
        return None

    lat_vals = _as_float_array(sign_map_data["lat_vals"], ndim=1)
    lon_vals = _as_float_array(sign_map_data["lon_vals"], ndim=1)
    sign_class_map = np.asarray(sign_map_data["sign_class_map"], dtype=np.int32)
    sign_agreement_at_20 = float(sign_map_data.get("sign_agreement_at_20", float("nan")))

    meta = viz.get("meta", {})
    topq_pct = int(round(float(meta.get("topq_fraction", 0.2)) * 100))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sign_field, plot_lat = _prep_lat_oriented_field(
        sign_class_map.astype(np.float64), lat_vals
    )
    extent = [
        float(np.min(lon_vals)),
        float(np.max(lon_vals)),
        float(np.min(plot_lat)),
        float(np.max(plot_lat)),
    ]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi, constrained_layout=True)
    ax.imshow(
        sign_field,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=_SIGN_CMAP,
        norm=_SIGN_NORM,
    )
    ax.set_title(f"Sign Agreement Map (Top-{topq_pct}% Overlap)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#f0f0f0", edgecolor="#aaaaaa", label="Outside overlap"),
            mpatches.Patch(facecolor="#27ae60", edgecolor="none", label="++ agree"),
            mpatches.Patch(facecolor="#2980b9", edgecolor="none", label="-- agree"),
            mpatches.Patch(facecolor="#c0392b", edgecolor="none", label="Opposite / invalid"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )

    annotation = _format_sign_map_annotation(sign_agreement_at_20=sign_agreement_at_20)
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#444444", "alpha": 0.9, "boxstyle": "round,pad=0.3"},
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_track_patch_figures(
    report_or_path,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "dlmsf_track_patch",
    dpi: int = 200,
) -> List[Path]:
    """Write the four fixed DLMSF validation figures.

    The ``prefix`` argument is kept for call-site compatibility but is
    intentionally ignored — filenames are derived from visualization metadata.
    """
    _ = prefix  # deprecated, no-op

    report = _load_report(report_or_path)
    if output_dir is None:
        if isinstance(report_or_path, (str, Path)):
            output_dir = Path(report_or_path).resolve().parent
        else:
            output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_case = str(report.get("main_case", ""))
    case = report.get("cases", {}).get(main_case, {})
    viz = case.get("visualization", {})
    meta = viz.get("meta", {})

    direction = str(meta.get("direction", "along"))
    patch_size = int(meta.get("patch_size", 3))
    target_time_idx = int(meta.get("target_time_idx", 0))
    topq_pct = int(round(float(meta.get("topq_fraction", 0.2)) * 100))

    p1 = output_dir / f"dlmsf_{direction}_overlap_q{topq_pct}_t{target_time_idx}.png"
    p2 = output_dir / f"dlmsf_{direction}_scatter_t{target_time_idx}.png"
    p3 = output_dir / f"deletion_validation_{direction}_p{patch_size}.png"
    p4 = output_dir / f"dlmsf_{direction}_sign_map_t{target_time_idx}.png"

    outputs: List[Path] = []

    result = plot_track_patch_overlap_q20(report, p1, dpi=dpi)
    if result is not None:
        outputs.append(result)

    result = plot_track_patch_scatter(report, p2, dpi=dpi)
    if result is not None:
        outputs.append(result)

    result = plot_track_patch_deletion_validation(report, p3, dpi=dpi)
    if result is not None:
        outputs.append(result)

    result = plot_track_patch_sign_map(report, p4, dpi=dpi)
    if result is not None:
        outputs.append(result)

    return outputs
