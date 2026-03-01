from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import xarray

from shared.patch_scoring_utils import window_reduce_2d
from physics.swe_sensitivity import SWESensitivityResult


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


def _patch(arr: np.ndarray, radius: int, agg: str) -> np.ndarray:
    return window_reduce_2d(np.abs(arr.astype(np.float64)), radius, agg)


def _safe_finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask].ravel(), b[mask].ravel()


def compute_spearman(
    swe_map: np.ndarray,
    gnn_map: np.ndarray,
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
) -> Tuple[float, float]:
    s, g = _safe_finite_pair(
        _patch(swe_map, patch_radius, patch_score_agg),
        _patch(gnn_map, patch_radius, patch_score_agg),
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
    s = _patch(swe_map, patch_radius, patch_score_agg)
    g = _patch(gnn_map, patch_radius, patch_score_agg)
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
        actual_k = min(int(k), n)
        if actual_k <= 0:
            result[k] = 0.0
            continue
        top_s = set(np.argsort(s_valid)[::-1][:actual_k].tolist())
        top_g = set(np.argsort(g_valid)[::-1][:actual_k].tolist())
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
        _patch(swe_map, patch_radius, patch_score_agg),
        _patch(gnn_map, patch_radius, patch_score_agg),
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

    pairs = [
        ("h",     swe_result.S_h,     "z_500"),
        ("uv",    swe_result.S_uv,    "uv_500"),
    ]
    for group_name, swe_map, gnn_key in pairs:
        if gnn_key not in gnn_ig_maps:
            continue
        m = _group_metrics(swe_map, gnn_ig_maps[gnn_key],
                           group_name, patch_radius, patch_score_agg, k_values)
        report.groups.append(m)
        print(f"  [Align] {group_name:6s}: ρ={m.spearman_rho:+.3f}  "
              f"IoU@50={m.topk_iou.get(50, float('nan')):.3f}  n={m.n_valid}")

    return report


def plot_comparison_panels(
    swe_result: SWESensitivityResult,
    gnn_ig_maps: Dict[str, np.ndarray],
    output_dir: Path,
    dpi: int = 200,
    alpha_quantile: Optional[float] = None,
    vmax_quantile: Optional[float] = None,
) -> None:
    from shared.heatmap_utils import plot_importance_heatmap_panels

    output_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = swe_result.lat_vals, swe_result.lon_vals
    lead_h = (swe_result.target_time_idx + 1) * 6

    groups = [
        ("h",     swe_result.S_h,     "z_500",  "SWE $S_h$",    "GNN IG (z₅₀₀)",    "|∂J/∂h₀|"),
        ("uv",    swe_result.S_uv,    "uv_500", "SWE $S_{uv}$", "GNN IG (uv magnitude)", "|∂J/∂(u,v)₀|"),
    ]

    def _norm_q(x: np.ndarray) -> np.ndarray:
        fin = x[np.isfinite(x)]
        vmax = float(np.max(fin)) if fin.size > 0 else 1.0
        return np.clip(x / (vmax + 1e-12), 0.0, 1.0)

    for group_tag, swe_arr, gnn_key, swe_title, gnn_title, cbar_swe in groups:
        if gnn_key not in gnn_ig_maps:
            continue
        gnn_arr = gnn_ig_maps[gnn_key]
        diff = _norm_q(swe_arr) - _norm_q(gnn_arr)

        maps = [
            xarray.DataArray(swe_arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
            xarray.DataArray(gnn_arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
            xarray.DataArray(diff,    dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
        ]
        out = output_dir / f"physics_gnn_comparison_{group_tag}_t{swe_result.target_time_idx}.png"
        plot_importance_heatmap_panels(
            importance_list=maps,
            titles=[swe_title, gnn_title, f"SWE − GNN (norm.) +{lead_h}h"],
            center_lat=swe_result.center_lat,
            center_lon=swe_result.center_lon,
            output_path=out,
            cmap=["magma", "Blues", "RdBu_r"],
            dpi=dpi,
            vmax_quantile=[vmax_quantile, vmax_quantile, None],
            diverging=[False, False, True],
            cbar_label=[cbar_swe, "GNN IG score", "Δ (normalized)"],
            alpha_quantile=[alpha_quantile, alpha_quantile, None],
        )
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
    swe_result: SWESensitivityResult,
    gnn_ig_maps: Dict[str, np.ndarray],
    report: AlignmentReport,
    output_dir: Path,
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lead_h = (swe_result.target_time_idx + 1) * 6
    t = swe_result.target_time_idx

    pairs = [
        ("h",     swe_result.S_h,     "z_500",  "SWE $S_h$",    "GNN IG (z₅₀₀)"),
        ("uv",    swe_result.S_uv,    "uv_500", "SWE $S_{uv}$", "GNN IG (uv magnitude)"),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), dpi=dpi)
    axes_list = np.atleast_1d(axes)
    for ax, (gname, swe_arr, gnn_key, xlabel, ylabel) in zip(axes_list, pairs):
        if gnn_key not in gnn_ig_maps:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gname)
            continue

        s = _patch(swe_arr, patch_radius, patch_score_agg)
        g = _patch(gnn_ig_maps[gnn_key], patch_radius, patch_score_agg)
        a, b = _safe_finite_pair(s, g)
        if len(a) < 3:
            continue

        ax.scatter(a, b, s=4, alpha=0.35, rasterized=True, color="steelblue")
        ax.set_xscale("symlog", linthresh=np.quantile(a[a > 0], 0.01) if (a > 0).any() else 1e-8)
        ax.set_yscale("symlog", linthresh=np.quantile(b[b > 0], 0.01) if (b > 0).any() else 1e-8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        gm = next((gm for gm in report.groups if gm.group_name == gname), None)
        if gm is not None:
            ax.set_title(
                f"{gname} | ρ={gm.spearman_rho:+.3f} | IoU@50={gm.topk_iou.get(50, float('nan')):.3f}",
                fontsize=10,
            )

    fig.suptitle(f"SWE vs GNN IG Alignment — +{lead_h}h", fontsize=13)
    fig.tight_layout()
    out = output_dir / f"physics_alignment_scatter_t{t}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_topk_iou_curves(
    swe_result: SWESensitivityResult,
    gnn_ig_maps: Dict[str, np.ndarray],
    output_dir: Path,
    k_values: Tuple[int, ...] = (10, 20, 50, 100, 150, 200, 300),
    patch_radius: int = 2,
    patch_score_agg: str = "mean",
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    lead_h = (swe_result.target_time_idx + 1) * 6
    t = swe_result.target_time_idx

    pairs = [
        ("h",     swe_result.S_h,     "z_500"),
        ("uv",    swe_result.S_uv,    "uv_500"),
    ]
    colors = {"h": "royalblue", "uv": "tomato"}

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    for gname, swe_arr, gnn_key in pairs:
        if gnn_key not in gnn_ig_maps:
            continue
        iou_vals = [
            compute_topk_iou(swe_arr, gnn_ig_maps[gnn_key], (k,), patch_radius, patch_score_agg)[k]
            for k in k_values
        ]
        ax.plot(k_values, iou_vals, marker=".", label=gname, color=colors[gname], linewidth=2)

    ax.set_xlabel("K (Top-K threshold)")
    ax.set_ylabel("IoU")
    ax.set_title(f"Top-K IoU Robustness — +{lead_h}h", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    out = output_dir / f"physics_alignment_topk_t{t}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def save_report_json(report: AlignmentReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")
