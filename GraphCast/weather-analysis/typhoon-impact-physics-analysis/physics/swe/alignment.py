from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import xarray

from shared.patch_scoring_utils import window_reduce_2d

if TYPE_CHECKING:
    from physics.swe.swe_sensitivity import SWESensitivityResult


@dataclass
class GroupMetrics:
    group_name: str
    spearman_rho: float
    spearman_pval: float
    topk_iou: Dict[int, float]
    n_valid: int


@dataclass
class AlignmentReport:
    target_time_idx: int
    lead_time_h: int
    patch_radius: int
    patch_score_agg: str
    sigma_deg: float
    groups: List[GroupMetrics] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "_comment": "SWE 物理敏感度与 GNN IG 对齐指标。",
            "_field_notes": {
                "target_time_idx": "预报时效索引。0 表示 +6h，1 表示 +12h，以此类推。",
                "lead_time_h": "预报时效（小时）。",
                "patch_radius": "计算指标前进行局部 patch 聚合时使用的半径。",
                "patch_score_agg": "patch 聚合方式（如 mean）。",
                "sigma_deg": "敏感度目标函数中高斯权重的宽度（度）。",
                "groups": "按变量组统计的对齐指标（h 与 uv）。",
                "spearman_rho": "Spearman 秩相关系数，越接近 1 表示排序一致性越高。",
                "spearman_pval": "Spearman 检验 p 值，越小表示相关性越显著。",
                "topk_iou": "Top-K 热点集合的交并比（IoU），越大表示热点重叠越高。",
                "n_valid": "参与统计的有效样本数量（有限值网格点数）。",
            },
            "target_time_idx": self.target_time_idx,
            "lead_time_h": self.lead_time_h,
            "patch_radius": self.patch_radius,
            "patch_score_agg": self.patch_score_agg,
            "sigma_deg": self.sigma_deg,
            "groups": {
                g.group_name: {
                    "_comment": "SWE 敏感度图与 GNN IG 图在排序与热点重叠上的一致性。",
                    "spearman_rho": round(g.spearman_rho, 4),
                    "spearman_pval": float(f"{g.spearman_pval:.2e}"),
                    "topk_iou": {f"k{k}": round(v, 4) for k, v in g.topk_iou.items()},
                    "_topk_iou_note": "SWE 与 GNN 两张图在 Top-K 热点上的 IoU（交并比）。",
                    "n_valid": g.n_valid,
                }
                for g in self.groups
            },
        }


def _patch_signed(arr: np.ndarray, radius: int, agg: str) -> np.ndarray:
    """Patch aggregation that preserves sign (for correlation/scatter)."""
    return window_reduce_2d(arr.astype(np.float64), radius, agg)


def _patch_magnitude(arr: np.ndarray, radius: int, agg: str) -> np.ndarray:
    """Patch aggregation using magnitude |arr| (for top-k/IoU hotspot detection)."""
    return window_reduce_2d(np.abs(arr.astype(np.float64)), radius, agg)


def _safe_finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask].ravel(), b[mask].ravel()


def _topk_mask_from_rank(rank_map: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    rank = np.asarray(rank_map, dtype=np.float64)
    flat = rank.ravel()
    finite_idx = np.flatnonzero(np.isfinite(flat))
    if finite_idx.size == 0:
        return np.zeros(rank.shape, dtype=bool), 0
    actual_k = max(1, min(int(k), int(finite_idx.size)))
    vals = flat[finite_idx]
    top_local = np.argpartition(vals, -actual_k)[-actual_k:]
    top_idx = finite_idx[top_local]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[top_idx] = True
    return mask.reshape(rank.shape), actual_k


def _topk_index_set(values: np.ndarray, k: int) -> Tuple[set[int], int]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.size)
    actual_k = min(int(k), n)
    if actual_k <= 0:
        return set(), 0
    return set(np.argsort(arr, kind="stable")[::-1][:actual_k].tolist()), actual_k


def _topk_overlap_code(swe_score: np.ndarray, ig_score: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    """Encode overlap map: 0 none, 1 SWE-only, 2 IG-only, 3 overlap.
    
    Uses jointly-finite population consistent with compute_topk_iou.
    Cells that are NaN in either map are excluded from top-k selection.
    """
    joint_finite = np.isfinite(swe_score) & np.isfinite(ig_score)
    code = np.zeros(swe_score.shape, dtype=np.int8)
    if not joint_finite.any():
        return code, 0

    joint_flat = joint_finite.ravel()
    swe_flat = swe_score.ravel()
    ig_flat = ig_score.ravel()
    joint_idx = np.flatnonzero(joint_flat)
    swe_joint = swe_flat[joint_idx]
    ig_joint = ig_flat[joint_idx]

    top_swe_local, actual_k = _topk_index_set(swe_joint, k)
    top_ig_local, _ = _topk_index_set(ig_joint, k)
    top_swe_idx = joint_idx[list(top_swe_local)] if top_swe_local else np.array([], dtype=np.int64)
    top_ig_idx = joint_idx[list(top_ig_local)] if top_ig_local else np.array([], dtype=np.int64)

    swe_mask = np.zeros(swe_flat.shape, dtype=bool)
    ig_mask = np.zeros(ig_flat.shape, dtype=bool)
    swe_mask[top_swe_idx] = True
    ig_mask[top_ig_idx] = True
    swe_mask = swe_mask.reshape(swe_score.shape)
    ig_mask = ig_mask.reshape(ig_score.shape)

    code[swe_mask & (~ig_mask)] = 1
    code[(~swe_mask) & ig_mask] = 2
    code[swe_mask & ig_mask] = 3

    return code, actual_k


def compute_spearman(
    swe_map: np.ndarray,
    gnn_map: np.ndarray,
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
) -> Tuple[float, float]:
    """Compute Spearman correlation with signed-preserving preprocessing."""
    s, g = _safe_finite_pair(
        _patch_signed(swe_map, patch_radius, patch_score_agg),
        _patch_signed(gnn_map, patch_radius, patch_score_agg),
    )
    if len(s) < 5:
        return np.nan, np.nan
    sr: Any = scipy.stats.spearmanr(s, g)
    return float(sr[0]), float(sr[1])


def compute_topk_iou(
    swe_map: np.ndarray,
    gnn_map: np.ndarray,
    k_values: Tuple[int, ...] = (20, 50, 100, 200),
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
) -> Dict[int, float]:
    """Compute Top-K IoU with magnitude preprocessing for hotspot detection."""
    s = _patch_magnitude(swe_map, patch_radius, patch_score_agg)
    g = _patch_magnitude(gnn_map, patch_radius, patch_score_agg)
    s_flat = s.ravel()
    g_flat = g.ravel()
    finite = np.isfinite(s_flat) & np.isfinite(g_flat)
    s_valid = s_flat[finite]
    g_valid = g_flat[finite]
    n = int(s_valid.size)
    result: Dict[int, float] = {}
    if n == 0:
        for k in k_values:
            result[k] = 0.0
        return result

    for k in k_values:
        top_s, actual_k = _topk_index_set(s_valid, k)
        top_g, _ = _topk_index_set(g_valid, k)
        if actual_k <= 0:
            result[k] = 0.0
            continue
        inter = len(top_s & top_g)
        union = len(top_s | top_g)
        result[k] = inter / union if union > 0 else 0.0
    return result


def _group_metrics(
    swe_map: np.ndarray,
    gnn_map: np.ndarray,
    group_name: str,
    patch_radius: int,
    patch_score_agg: str,
    k_values: Tuple[int, ...] = (20, 50, 100, 200),
) -> GroupMetrics:
    rho, rpval = compute_spearman(swe_map, gnn_map, patch_radius, patch_score_agg)
    iou = compute_topk_iou(swe_map, gnn_map, k_values, patch_radius, patch_score_agg)
    s, _ = _safe_finite_pair(
        _patch_signed(swe_map, patch_radius, patch_score_agg),
        _patch_signed(gnn_map, patch_radius, patch_score_agg),
    )
    return GroupMetrics(
        group_name=group_name,
        spearman_rho=rho,
        spearman_pval=rpval,
        topk_iou=iou,
        n_valid=len(s),
    )


def compute_alignment_report(
    swe_result: SWESensitivityResult,
    gnn_ig_maps: Dict[str, np.ndarray],
    gnn_main_maps: Optional[Dict[str, np.ndarray]] = None,
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    sigma_deg: float = 3.0,
    k_values: Tuple[int, ...] = (20, 50, 100, 200),
) -> AlignmentReport:
    lead_h = (swe_result.target_time_idx + 1) * 6
    report = AlignmentReport(
        target_time_idx=swe_result.target_time_idx,
        lead_time_h=lead_h,
        patch_radius=patch_radius,
        patch_score_agg=patch_score_agg,
        sigma_deg=sigma_deg,
    )

    if gnn_main_maps is None and gnn_ig_maps:
        raise ValueError(
            "compute_alignment_report requires gnn_main_maps for signed alignment; "
            "magnitude-only gnn_ig_maps are not sufficient"
        )

    signed_maps = gnn_main_maps or {}
    if "z_500" in gnn_ig_maps and "z_500" not in signed_maps:
        raise ValueError(
            "compute_alignment_report requires signed z_500 in gnn_main_maps"
        )
    pairs = [
        ("h", swe_result.S_h, "z_500", signed_maps, "signed z_500"),
        ("uv", swe_result.S_uv, "uv_500", signed_maps, "signed uv_500"),
    ]
    for group_name, swe_map, gnn_key, source_maps, required_desc in pairs:
        if gnn_key not in source_maps:
            if gnn_key in gnn_ig_maps:
                print(f"  [Align] skip {group_name}: requires {required_desc}, got magnitude-only map")
            continue
        m = _group_metrics(swe_map, source_maps[gnn_key],
                           group_name, patch_radius, patch_score_agg, k_values)
        report.groups.append(m)
        print(f"  [Align] {group_name:6s}: ρ={m.spearman_rho:+.3f}  "
              f"IoU@50={m.topk_iou.get(50, float('nan')):.3f}  n={m.n_valid}")

    return report


def plot_topk_overlap_maps(
    pairs: List[Tuple[str, np.ndarray, str]],
    gnn_ig_maps: Dict[str, np.ndarray],
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    center_lat: float,
    center_lon: float,
    target_time_idx: int,
    output_dir: Path,
    output_prefix: str = "swe",
    dpi: int = 200,
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    topk_overlap_k: int = 50,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    output_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(lat_vals, dtype=np.float64)
    lon = np.asarray(lon_vals, dtype=np.float64)
    lead_h = (target_time_idx + 1) * 6
    lat_asc = lat[0] < lat[-1]
    origin = "lower" if lat_asc else "upper"

    for group_tag, score_map, gnn_key in pairs:
        if gnn_key not in gnn_ig_maps:
            continue
        # Use magnitude preprocessing for top-k overlap detection
        swe_arr = _patch_magnitude(score_map, patch_radius, patch_score_agg)
        gnn_arr = _patch_magnitude(gnn_ig_maps[gnn_key], patch_radius, patch_score_agg)
        overlap_code, actual_k = _topk_overlap_code(swe_arr, gnn_arr, topk_overlap_k)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
        cmap = ListedColormap(["#f2f2f2", "#ff7f0e", "#1f77b4", "#2ca02c"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(
            overlap_code,
            extent=(lon.min(), lon.max(), lat.min(), lat.max()),
            origin=origin,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        ax.scatter([center_lon], [center_lat], c="#00e5ff", marker="x", s=80, linewidths=2)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Top-{actual_k} overlap ({group_tag}) — +{lead_h}h")
        ax.legend(
            handles=[
                Patch(facecolor="#2ca02c", edgecolor="none", label="Overlap"),
                Patch(facecolor="#ff7f0e", edgecolor="none", label=f"{output_prefix.upper()} only"),
                Patch(facecolor="#1f77b4", edgecolor="none", label="IG only"),
            ],
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )
        fig.tight_layout()
        out = output_dir / f"{output_prefix}_overlap_{group_tag}_k{actual_k}_t{target_time_idx}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved: {out}")


def plot_sensitivity_heatmaps(
    swe_result: SWESensitivityResult,
    output_dir: Path,
    dpi: int = 200,
    name_suffix: str = "",
    log_scale: bool = False,
    log_eps: float = 1e-10,
    alpha_quantile: Optional[float] = None,
    vmax_quantile: Optional[float] = None,
) -> None:
    from shared.heatmap_utils import plot_importance_heatmap

    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = swe_result.lat_vals, swe_result.lon_vals
    lead_h = (swe_result.target_time_idx + 1) * 6
    t = swe_result.target_time_idx
    suffix = f"_{name_suffix}" if name_suffix else ""
    title_suffix = f" ({name_suffix})" if name_suffix else ""

    for field_tag, arr, cbar_label in [
        ("h",     swe_result.S_h,     "|∂J/∂h₀|"),
        ("uv",    swe_result.S_uv,    "√(|∂J/∂u₀|²+|∂J/∂v₀|²)"),
    ]:
        if log_scale:
            arr_plot = np.log10(np.maximum(arr, 0.0) + float(log_eps))
            cbar_label_plot = f"log10({cbar_label} + {log_eps:g})"
            title_field = f"log10(S_{{{field_tag}}}+{log_eps:g})"
        else:
            arr_plot = arr
            cbar_label_plot = cbar_label
            title_field = f"S_{{{field_tag}}}"

        da = xarray.DataArray(arr_plot, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        out = output_dir / f"swe_sensitivity_{field_tag}{suffix}_t{t}.png"
        plot_importance_heatmap(
            importance_da=da,
            center_lat=swe_result.center_lat,
            center_lon=swe_result.center_lon,
            output_path=out,
            title=f"SWE Physical Sensitivity ${title_field}$ — +{lead_h}h{title_suffix}",
            cmap="magma",
            dpi=dpi,
            vmax_quantile=vmax_quantile,
            diverging=False,
            cbar_label=cbar_label_plot,
            center_window_deg=10.0,
            center_s_quantile=0.99,
            alpha_quantile=alpha_quantile,
        )
        print(f"Saved: {out}")


def plot_alignment_scatter(
    pairs: List[Tuple[str, np.ndarray, str, str, str]],
    gnn_ig_maps: Dict[str, np.ndarray],
    report: AlignmentReport,
    target_time_idx: int,
    lead_time_h: int,
    output_dir: Path,
    output_prefix: str = "swe",
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    dpi: int = 200,
    abs_gnn_for_display: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    t = target_time_idx
    lead_h = lead_time_h

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), dpi=dpi)
    axes_list = np.atleast_1d(axes)
    for ax, (gname, score_map, gnn_key, xlabel, ylabel) in zip(axes_list, pairs):
        if gnn_key not in gnn_ig_maps:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gname)
            continue

        s = _patch_signed(score_map, patch_radius, patch_score_agg)
        g = _patch_signed(gnn_ig_maps[gnn_key], patch_radius, patch_score_agg)
        a, b = _safe_finite_pair(s, g)
        if len(a) < 3:
            continue

        g_display = np.abs(b) if abs_gnn_for_display else b

        ax.scatter(a, g_display, s=4, alpha=0.35, rasterized=True, color="steelblue")
        ax.set_xscale("symlog", linthresh=np.quantile(a[a > 0], 0.01) if (a > 0).any() else 1e-8)
        ax.set_yscale(
            "symlog",
            linthresh=np.quantile(g_display[g_display > 0], 0.01) if (g_display > 0).any() else 1e-8,
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        gm = next((gm for gm in report.groups if gm.group_name == gname), None)
        if gm is not None:
            ax.set_title(
                f"{gname} | ρ={gm.spearman_rho:+.3f} | IoU@50={gm.topk_iou.get(50, float('nan')):.3f}",
                fontsize=10,
            )

    fig.suptitle(f"{output_prefix.upper()} vs GNN IG Alignment — +{lead_h}h", fontsize=13)
    fig.tight_layout()
    out = output_dir / f"{output_prefix}_scatter_t{t}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_topk_iou_curves(
    pairs: List[Tuple[str, np.ndarray, str]],
    gnn_ig_maps: Dict[str, np.ndarray],
    target_time_idx: int,
    lead_time_h: int,
    output_dir: Path,
    output_prefix: str = "swe",
    k_values: Tuple[int, ...] = (10, 20, 50, 100, 150, 200, 300),
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lead_h = lead_time_h
    t = target_time_idx
    colors = ["royalblue", "tomato", "mediumseagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    for i, (gname, score_map, gnn_key) in enumerate(pairs):
        if gnn_key not in gnn_ig_maps:
            continue
        color = colors[i % len(colors)]
        iou_vals = [
            compute_topk_iou(score_map, gnn_ig_maps[gnn_key], (k,), patch_radius, patch_score_agg)[k]
            for k in k_values
        ]
        ax.plot(k_values, iou_vals, marker=".", label=gname, color=color, linewidth=2)

    ax.set_xlabel("K (Top-K threshold)")
    ax.set_ylabel("IoU")
    ax.set_title(f"Top-K IoU Robustness — +{lead_h}h", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    out = output_dir / f"{output_prefix}_iou_t{t}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def save_report_json(report: AlignmentReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")
