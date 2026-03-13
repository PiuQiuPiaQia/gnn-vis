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

from physics.swe.alignment import _patch_magnitude, _safe_finite_pair


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


def _positive_linthresh(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.abs(arr[np.isfinite(arr)])
    positive = finite[finite > 0]
    if positive.size == 0:
        return 1e-8
    return float(np.quantile(positive, 0.01))


# ---------------------------------------------------------------------------
# Annotation helpers — exposed for unit-test verification
# ---------------------------------------------------------------------------


def _format_overlap_annotation(*, spearman_rho: float, iou_at_50: float) -> str:
    """Return annotation string for the overlap figure."""
    return (
        f"Spearman \u03c1: {float(spearman_rho):+.3f}\n"
        f"IoU@50: {float(iou_at_50):.3f}"
    )


def _format_scatter_annotation(*, spearman_rho: float) -> str:
    """Return annotation string for the scatter figure."""
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


# ---------------------------------------------------------------------------
# Figure 1: Top-K overlap binary map
# ---------------------------------------------------------------------------


def plot_track_patch_overlap_k50(
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
    overlap_mask = _as_bool_array(overlap["overlap_mask"], ndim=2)
    spearman_rho = float(overlap["spearman_rho"])
    iou_at_50 = float(overlap["iou_at_50"])

    meta = viz.get("meta", {})
    topk_k = int(meta.get("topk_k", 50))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=dpi, constrained_layout=True)

    # Binary overlap map
    overlap_float = overlap_mask.astype(np.float64)
    overlap_field, overlap_lat = _prep_lat_oriented_field(overlap_float, lat_vals)
    extent = (
        float(np.min(lon_vals)),
        float(np.max(lon_vals)),
        float(np.min(overlap_lat)),
        float(np.max(overlap_lat)),
    )
    overlap_cmap = ListedColormap(["#f0f0f0", "#2c3e50"])
    overlap_norm = BoundaryNorm([-0.5, 0.5, 1.5], overlap_cmap.N)
    ax.imshow(
        overlap_field,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=overlap_cmap,
        norm=overlap_norm,
    )
    ax.set_title(f"Top-{topk_k} Overlap")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#f0f0f0", edgecolor="#aaaaaa", label="Not in overlap"),
            mpatches.Patch(facecolor="#2c3e50", edgecolor="none", label="Overlap"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )

    annotation = _format_overlap_annotation(spearman_rho=spearman_rho, iou_at_50=iou_at_50)
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
# Figure 2: Abs grid-level scatter (hotspot agreement)
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

    x_abs_map = _as_float_array(scatter["x_abs_map"], ndim=2)
    y_abs_map = _as_float_array(scatter["y_abs_map"], ndim=2)
    patch_radius = int(scatter.get("patch_radius", 0))
    patch_score_agg = str(scatter.get("patch_score_agg", "mean"))
    x, y = _safe_finite_pair(
        _patch_magnitude(x_abs_map, patch_radius, patch_score_agg),
        _patch_magnitude(y_abs_map, patch_radius, patch_score_agg),
    )
    spearman_rho = float(scatter["spearman_rho"])

    meta = viz.get("meta", {})
    direction = str(meta.get("direction", "along"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi, constrained_layout=True)
    ax.scatter(x, y, alpha=0.7, s=40, color="#2980b9")
    ax.set_xscale("symlog", linthresh=_positive_linthresh(x))
    ax.set_yscale("symlog", linthresh=_positive_linthresh(y))
    ax.set_xlabel(f"|DLMSF_{direction}| patch score")
    ax.set_ylabel("|IG| patch score")
    ax.set_title("Patch-Level Hotspot Agreement")
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
# Main entry point
# ---------------------------------------------------------------------------


def write_track_patch_figures(
    report_or_path,
    *,
    output_dir: str | Path | None = None,
    prefix: str = "dlmsf_track_patch",
    dpi: int = 200,
) -> List[Path]:
    """Write the fixed DLMSF validation figures.

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
    topk_k = int(meta.get("topk_k", 50))

    p1 = output_dir / f"dlmsf_{direction}_overlap_k{topk_k}_t{target_time_idx}.png"
    p2 = output_dir / f"dlmsf_{direction}_scatter_t{target_time_idx}.png"
    p3 = output_dir / f"deletion_validation_{direction}_p{patch_size}.png"
    outputs: List[Path] = []

    result = plot_track_patch_overlap_k50(report, p1, dpi=dpi)
    if result is not None:
        outputs.append(result)

    result = plot_track_patch_scatter(report, p2, dpi=dpi)
    if result is not None:
        outputs.append(result)

    result = plot_track_patch_deletion_validation(report, p3, dpi=dpi)
    if result is not None:
        outputs.append(result)

    return outputs
