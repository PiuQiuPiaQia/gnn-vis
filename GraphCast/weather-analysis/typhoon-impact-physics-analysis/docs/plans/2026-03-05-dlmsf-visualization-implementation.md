# DLMSF 对比可视化实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构 `alignment.py` 三个绘图函数签名（去除对 `SWESensitivityResult` 的强依赖），并在 Phase 3b 中为 DLMSF 生成对称的三类对比图。

**Architecture:** 方案 B——将 `swe_result` 参数拆散为 `pairs`（含 S_map）、`lat_vals`、`lon_vals`、`center_lat`、`center_lon`、`target_time_idx`，新增 `output_prefix` 控制文件名前缀。SWE 和 DLMSF 共用同一套函数。

**Tech Stack:** Python 3.10+, numpy, matplotlib, pytest

---

## 文件命名规则（P2 方案）

| 图类型 | SWE | DLMSF |
|--------|-----|-------|
| Top-K 重叠图 | `swe_overlap_{group}_k{k}_t{t}.png` | `dlmsf_overlap_{group}_k{k}_t{t}.png` |
| 散点对齐图 | `swe_scatter_t{t}.png` | `dlmsf_scatter_t{t}.png` |
| IoU 曲线图 | `swe_iou_t{t}.png` | `dlmsf_iou_t{t}.png` |

- SWE pairs 的 group_tag：`h`、`uv`
- DLMSF pairs 的 group_tag：`z`、`uv`

---

## Task 1: 新建绘图 API 单元测试（TDD——先写失败测试）

**Files:**
- Create: `tests/physics/test_alignment_plot_api.py`

**Step 1: 新建测试文件**

```python
# tests/physics/test_alignment_plot_api.py
# -*- coding: utf-8 -*-
"""Unit tests for refactored alignment.py plotting API (pairs-based signatures)."""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from physics.swe.alignment import (
    AlignmentReport,
    GroupMetrics,
    plot_alignment_scatter,
    plot_topk_iou_curves,
    plot_topk_overlap_maps,
)


def _dummy_map(seed: int = 0, shape=(10, 12)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float64)


@pytest.fixture()
def lat_lon():
    return np.linspace(20.0, 30.0, 10), np.linspace(120.0, 132.0, 12)


@pytest.fixture()
def gnn_ig_maps():
    return {"z_500": _dummy_map(1), "uv_500": _dummy_map(2)}


@pytest.fixture()
def simple_report():
    report = AlignmentReport(
        target_time_idx=0, lead_time_h=6,
        patch_radius=2, patch_score_agg="mean", sigma_deg=3.0,
    )
    report.groups.append(GroupMetrics("h", 0.5, 0.01, {50: 0.3}, 100))
    return report


class TestPlotTopkOverlapMapsNewApi:
    """plot_topk_overlap_maps 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_pairs_generates_two_files(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="swe",
        )
        files = sorted(tmp_path.glob("swe_overlap_*_t0.png"))
        assert len(files) == 2

    def test_dlmsf_pairs_generates_two_files(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        S_map = _dummy_map(5)
        pairs = [("z", S_map, "z_500"), ("uv", S_map, "uv_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="dlmsf",
        )
        files = sorted(tmp_path.glob("dlmsf_overlap_*_t0.png"))
        assert len(files) == 2

    def test_missing_gnn_key_skipped(self, tmp_path, lat_lon):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        empty_maps = {}  # 无匹配
        plot_topk_overlap_maps(
            pairs, empty_maps, lat, lon, 25.0, 126.0,
            target_time_idx=0, output_dir=tmp_path, output_prefix="swe",
        )
        assert list(tmp_path.glob("*.png")) == []

    def test_output_prefix_in_filename(self, tmp_path, lat_lon, gnn_ig_maps):
        lat, lon = lat_lon
        pairs = [("h", _dummy_map(0), "z_500")]
        plot_topk_overlap_maps(
            pairs, gnn_ig_maps, lat, lon, 25.0, 126.0,
            target_time_idx=2, output_dir=tmp_path, output_prefix="swe",
        )
        files = list(tmp_path.glob("swe_overlap_h_*_t2.png"))
        assert len(files) == 1


class TestPlotAlignmentScatterNewApi:
    """plot_alignment_scatter 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_scatter_creates_file(self, tmp_path, gnn_ig_maps, simple_report):
        pairs = [
            ("h",  _dummy_map(0), "z_500",  "SWE $S_h$",    "GNN IG (z₅₀₀)"),
            ("uv", _dummy_map(3), "uv_500", "SWE $S_{uv}$", "GNN IG (uv)"),
        ]
        plot_alignment_scatter(
            pairs, gnn_ig_maps, simple_report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_scatter_t0.png").exists()

    def test_dlmsf_scatter_creates_file(self, tmp_path, gnn_ig_maps):
        S = _dummy_map(5)
        pairs = [
            ("z",  S, "z_500",  "DLMSF $S$", "GNN IG (z₅₀₀)"),
            ("uv", S, "uv_500", "DLMSF $S$", "GNN IG (uv magnitude)"),
        ]
        report = AlignmentReport(0, 6, 2, "mean", 0.0)
        plot_alignment_scatter(
            pairs, gnn_ig_maps, report,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="dlmsf",
        )
        assert (tmp_path / "dlmsf_scatter_t0.png").exists()

    def test_prefix_controls_filename(self, tmp_path, gnn_ig_maps, simple_report):
        pairs = [("h", _dummy_map(0), "z_500", "X", "Y")]
        plot_alignment_scatter(
            pairs, gnn_ig_maps, simple_report,
            target_time_idx=1, lead_time_h=12,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_scatter_t1.png").exists()
        assert not (tmp_path / "dlmsf_scatter_t1.png").exists()


class TestPlotTopkIouCurvesNewApi:
    """plot_topk_iou_curves 新签名：接受 pairs 而非 swe_result。"""

    def test_swe_iou_creates_file(self, tmp_path, gnn_ig_maps):
        pairs = [("h", _dummy_map(0), "z_500"), ("uv", _dummy_map(3), "uv_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_iou_t0.png").exists()

    def test_dlmsf_iou_creates_file(self, tmp_path, gnn_ig_maps):
        S = _dummy_map(5)
        pairs = [("z", S, "z_500"), ("uv", S, "uv_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=0, lead_time_h=6,
            output_dir=tmp_path, output_prefix="dlmsf",
        )
        assert (tmp_path / "dlmsf_iou_t0.png").exists()

    def test_prefix_controls_filename(self, tmp_path, gnn_ig_maps):
        pairs = [("h", _dummy_map(0), "z_500")]
        plot_topk_iou_curves(
            pairs, gnn_ig_maps,
            target_time_idx=3, lead_time_h=24,
            output_dir=tmp_path, output_prefix="swe",
        )
        assert (tmp_path / "swe_iou_t3.png").exists()
```

**Step 2: 运行测试，确认失败**

```
pytest tests/physics/test_alignment_plot_api.py -v
```

预期：全部 FAIL（`plot_topk_overlap_maps` 等函数签名不匹配）。

**Step 3: 提交测试文件**

```bash
git add tests/physics/test_alignment_plot_api.py
git commit -m "test: add failing tests for alignment.py new pairs-based plotting API"
```

---

## Task 2: 重构 `alignment.py` 三个绘图函数

**Files:**
- Modify: `physics/swe/alignment.py`

### `plot_topk_overlap_maps` 新实现

将原函数体中所有 `swe_result.*` 引用替换为对应显式参数，`groups` 循环替换为 `for group_tag, S_map, gnn_key in pairs:`，文件名模板改为 `{output_prefix}_overlap_{group_tag}_k{actual_k}_t{target_time_idx}.png`，图例中 `"SWE only"` 改为 `f"{output_prefix.upper()} only"`。

**Step 1: 替换函数签名及实现**

```python
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

    for group_tag, S_map, gnn_key in pairs:
        if gnn_key not in gnn_ig_maps:
            continue
        gnn_arr = _patch(gnn_ig_maps[gnn_key], patch_radius, patch_score_agg)
        overlap_code, actual_k = _topk_overlap_code(S_map, gnn_arr, topk_overlap_k)
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
```

### `plot_alignment_scatter` 新实现

```python
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
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    t = target_time_idx
    lead_h = lead_time_h

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), dpi=dpi)
    axes_list = np.atleast_1d(axes)
    for ax, (gname, S_map, gnn_key, xlabel, ylabel) in zip(axes_list, pairs):
        if gnn_key not in gnn_ig_maps:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gname)
            continue

        s = _patch(S_map, patch_radius, patch_score_agg)
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

    fig.suptitle(f"{output_prefix.upper()} vs GNN IG Alignment — +{lead_h}h", fontsize=13)
    fig.tight_layout()
    out = output_dir / f"{output_prefix}_scatter_t{t}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")
```

### `plot_topk_iou_curves` 新实现

```python
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
    for (gname, S_map, gnn_key), color in zip(pairs, colors):
        if gnn_key not in gnn_ig_maps:
            continue
        iou_vals = [
            compute_topk_iou(S_map, gnn_ig_maps[gnn_key], (k,), patch_radius, patch_score_agg)[k]
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
```

**Step 2: 同时移除头部 `TYPE_CHECKING` 下对 `SWESensitivityResult` 的导入（如不再被其他地方引用）**

检查：`SWESensitivityResult` 在 `alignment.py` 中只出现在 `TYPE_CHECKING` 块，且只有旧签名类型注解使用它。重构后可以安全删除该 import 块：

```python
# 删除这两行：
if TYPE_CHECKING:
    from physics.swe.swe_sensitivity import SWESensitivityResult
```

也可删除顶部导入中不再用到的 `TYPE_CHECKING`。

**Step 3: 运行新测试，确认通过**

```
pytest tests/physics/test_alignment_plot_api.py -v
```

预期：所有 12 个测试 PASS。

**Step 4: 运行全量测试，确认无回归**

```
pytest tests/ -v --ignore=tests/model -x
```

注意：`test_compare_plot_contract.py` 只检查函数名字符串，不检查签名，不受影响。

**Step 5: 提交**

```bash
git add physics/swe/alignment.py
git commit -m "refactor: replace swe_result param with pairs+explicit args in plotting fns, add output_prefix"
```

---

## Task 3: 更新 `comparison_core.py` Phase 4（SWE 调用侧）

**Files:**
- Modify: `physics/swe/comparison_core.py:531-546`

Phase 4 当前三处调用需更新为新签名。

**Step 1: 替换 Phase 4 中三处绘图调用**

```python
print("\n[Phase 4] Saving Visualizations")
dpi = getattr(cfg, "PHYSICS_HEATMAP_DPI", runtime_cfg.heatmap_dpi)
panel_topk_overlap_k = int(getattr(cfg, "SWE_PANEL_TOPK_OVERLAP_K", 50))

swe_pairs_overlap = [("h", jax_result.S_h, "z_500"), ("uv", jax_result.S_uv, "uv_500")]
plot_topk_overlap_maps(
    swe_pairs_overlap, gnn_ig_maps,
    np.asarray(jax_result.lat_vals, dtype=np.float64),
    np.asarray(jax_result.lon_vals, dtype=np.float64),
    float(jax_result.center_lat), float(jax_result.center_lon),
    target_time_idx=t_idx,
    output_dir=RESULTS_DIR,
    output_prefix="swe",
    dpi=dpi,
    patch_radius=patch_radius,
    patch_score_agg=patch_agg,
    topk_overlap_k=panel_topk_overlap_k,
)

swe_pairs_scatter = [
    ("h",  jax_result.S_h,  "z_500",  "SWE $S_h$",    "GNN IG (z₅₀₀)"),
    ("uv", jax_result.S_uv, "uv_500", "SWE $S_{uv}$", "GNN IG (uv magnitude)"),
]
plot_alignment_scatter(
    swe_pairs_scatter, gnn_ig_maps, report,
    target_time_idx=t_idx, lead_time_h=lead_h,
    output_dir=RESULTS_DIR, output_prefix="swe",
    patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
)

swe_pairs_iou = [("h", jax_result.S_h, "z_500"), ("uv", jax_result.S_uv, "uv_500")]
plot_topk_iou_curves(
    swe_pairs_iou, gnn_ig_maps,
    target_time_idx=t_idx, lead_time_h=lead_h,
    output_dir=RESULTS_DIR, output_prefix="swe",
    k_values=k_values, patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
)
```

**Step 2: 运行 contract 测试确认未破坏**

```
pytest tests/physics/test_compare_plot_contract.py -v
```

预期：2 个测试 PASS（函数名未变）。

**Step 3: 提交**

```bash
git add physics/swe/comparison_core.py
git commit -m "feat: update Phase 4 SWE plotting calls to new pairs-based API"
```

---

## Task 4: Phase 3b 新增 DLMSF 绘图调用

**Files:**
- Modify: `physics/swe/comparison_core.py`（Phase 3b 区块，约 551–580 行）

在 `save_report_json(dlmsf_report, dlmsf_json_path)` 之后，`if dlmsf_report.groups:` 块内新增三处绘图调用。

**Step 1: 在 Phase 3b 的 `if dlmsf_report.groups:` 块内追加绘图**

```python
        dlmsf_json_path = RESULTS_DIR / "dlmsf_alignment_metrics.json"
        if dlmsf_report.groups:
            save_report_json(dlmsf_report, dlmsf_json_path)

            # --- DLMSF 对比可视化 ---
            dlmsf_pairs_overlap = [
                ("z",  dlmsf_result.S_map, "z_500"),
                ("uv", dlmsf_result.S_map, "uv_500"),
            ]
            plot_topk_overlap_maps(
                dlmsf_pairs_overlap, gnn_ig_maps,
                swe_lat, swe_lon,
                float(context.center_lat), float(context.center_lon),
                target_time_idx=t_idx,
                output_dir=RESULTS_DIR,
                output_prefix="dlmsf",
                dpi=dpi,
                patch_radius=patch_radius,
                patch_score_agg=patch_agg,
                topk_overlap_k=panel_topk_overlap_k,
            )

            dlmsf_pairs_scatter = [
                ("z",  dlmsf_result.S_map, "z_500",  "DLMSF $S$", "GNN IG (z₅₀₀)"),
                ("uv", dlmsf_result.S_map, "uv_500", "DLMSF $S$", "GNN IG (uv magnitude)"),
            ]
            plot_alignment_scatter(
                dlmsf_pairs_scatter, gnn_ig_maps, dlmsf_report,
                target_time_idx=t_idx, lead_time_h=lead_h,
                output_dir=RESULTS_DIR, output_prefix="dlmsf",
                patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
            )

            dlmsf_pairs_iou = [
                ("z",  dlmsf_result.S_map, "z_500"),
                ("uv", dlmsf_result.S_map, "uv_500"),
            ]
            plot_topk_iou_curves(
                dlmsf_pairs_iou, gnn_ig_maps,
                target_time_idx=t_idx, lead_time_h=lead_h,
                output_dir=RESULTS_DIR, output_prefix="dlmsf",
                k_values=k_values, patch_radius=patch_radius, patch_score_agg=patch_agg, dpi=dpi,
            )
        else:
            print("  [warn] DLMSF 对齐组为空（gnn_ig_maps 中无匹配 key），跳过写出报告")
```

注意：`dpi` 和 `panel_topk_overlap_k` 在 Phase 4 中已赋值，Phase 3b 位于 Phase 4 之后，可直接引用这两个变量。

**Step 2: 提交**

```bash
git add physics/swe/comparison_core.py
git commit -m "feat: add DLMSF visualization plots in Phase 3b (overlap/scatter/iou)"
```

---

## Task 5: 扩展 contract 测试，覆盖 DLMSF 绘图调用

**Files:**
- Modify: `tests/physics/test_comparison_core_dlmsf.py`

**Step 1: 新增一个类，检查 Phase 3b 调用了绘图函数且使用了 `output_prefix="dlmsf"`**

```python
class TestDlmsfVisualizationContract:
    """验证 comparison_core.py 中 DLMSF 绘图调用已正确写入源码。"""

    def _read_source(self) -> str:
        from pathlib import Path
        return (
            Path(__file__).parent.parent.parent / "physics" / "swe" / "comparison_core.py"
        ).read_text(encoding="utf-8")

    def test_dlmsf_overlap_plot_called(self):
        """Phase 3b 应调用 plot_topk_overlap_maps 并传入 output_prefix='dlmsf'。"""
        source = self._read_source()
        assert 'output_prefix="dlmsf"' in source or "output_prefix='dlmsf'" in source

    def test_dlmsf_scatter_plot_called(self):
        """Phase 3b 应调用 plot_alignment_scatter（通过 output_prefix 区分）。"""
        source = self._read_source()
        assert "dlmsf_pairs_scatter" in source

    def test_dlmsf_iou_plot_called(self):
        """Phase 3b 应调用 plot_topk_iou_curves（通过 output_prefix 区分）。"""
        source = self._read_source()
        assert "dlmsf_pairs_iou" in source
```

**Step 2: 运行全量测试**

```
pytest tests/ -v --ignore=tests/model -x
```

预期：全部 PASS（新增 contract 测试 + 之前所有测试）。

**Step 3: 提交**

```bash
git add tests/physics/test_comparison_core_dlmsf.py
git commit -m "test: add contract tests for DLMSF visualization calls in Phase 3b"
```

---

## 最终验证

```
pytest tests/ -v --ignore=tests/model
```

新增测试数：12（`test_alignment_plot_api.py`）+ 3（`test_comparison_core_dlmsf.py` 扩展）= 15 个。
原有测试无回归。
