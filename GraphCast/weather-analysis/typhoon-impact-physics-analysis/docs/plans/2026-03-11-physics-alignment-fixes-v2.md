# Physics Alignment Fixes v2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 DLMSF-IG 对比实验中剩余的 6 类对齐问题（time_idx 硬编码、env_mask 静默退化、IG 未限制到 annulus、patch score 无归一化、scatter 轴标签反向、残留 sign 逻辑）。

**Architecture:** 全部变更集中在 `physics/dlmsf_patch_fd/` 目录的三个文件（`dlmsf_sensitivity.py`、`patch_comparison.py`、`plot_track_patch_report.py`）。每个 task 遵循 TDD：先写失败测试，再最小实现，再跑绿。

**Tech Stack:** Python 3.10+, numpy, xarray, scipy, pytest

---

## Task 1：time_idx 传播

**Files:**
- Modify: `physics/dlmsf_patch_fd/dlmsf_sensitivity.py:239-255`
- Modify: `physics/dlmsf_patch_fd/patch_comparison.py:1285-1350` (env_mask 预计算 + E1 块)
- Test: `tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py`

**背景：**
`compute_dlmsf_patch_fd()` 的 `_extract_uv_levels()` 调用写死了 `time_idx=1`，而 `env_mask` 预计算和 E1 块也没传 `time_idx`。`target_time_idx` 已经作为参数传入了这些函数，只是没有被转发下去。

**Step 1: 写失败测试**

在 `tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py` 末尾添加：

```python
class TestDlmsfTimeIdxPropagation:
    """time_idx 必须被传播到 _extract_uv_levels，不能写死成 1。"""

    def test_different_target_time_idx_gives_different_j_phys(self):
        """用 t=0 和 t=1 放不同风速，确认 J_phys_baseline 不同。"""
        lev = np.asarray([925, 500, 300], dtype=np.float32)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(115, 125, 11)
        # t=0: u=2, t=1: u=8
        u_data = np.zeros((1, 2, 3, 11, 11), dtype=np.float32)
        v_data = np.zeros((1, 2, 3, 11, 11), dtype=np.float32)
        u_data[:, 0, :, :, :] = 2.0
        u_data[:, 1, :, :, :] = 8.0
        ds = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray(
                    u_data, dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
                "v_component_of_wind": xr.DataArray(
                    v_data, dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
            }
        )
        baseline = xr.Dataset(
            {
                "u_component_of_wind": xr.DataArray(
                    np.zeros_like(u_data), dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
                "v_component_of_wind": xr.DataArray(
                    np.zeros_like(v_data), dims=("batch", "time", "level", "lat", "lon"),
                    coords={"level": lev, "lat": lat, "lon": lon},
                ),
            }
        )
        window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                       window_size=11, core_size=3)

        result0 = compute_dlmsf_patch_fd(
            eval_inputs=ds, baseline_inputs=baseline, window=window,
            center_lat=0.0, center_lon=120.0, d_hat=(1.0, 0.0),
            target_time_idx=0, patch_size=3, stride=2,
        )
        result1 = compute_dlmsf_patch_fd(
            eval_inputs=ds, baseline_inputs=baseline, window=window,
            center_lat=0.0, center_lon=120.0, d_hat=(1.0, 0.0),
            target_time_idx=1, patch_size=3, stride=2,
        )

        assert result0.J_phys_baseline != result1.J_phys_baseline, (
            "J_phys_baseline 应随 target_time_idx 变化；若相同说明 time_idx 仍被写死"
        )
```

**Step 2: 跑测试，确认 FAIL**

```bash
pytest tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py::TestDlmsfTimeIdxPropagation -v
```
预期：FAIL（两个 result 的 J_phys_baseline 相同，因为都读了 time_idx=1）

**Step 3: 修复 dlmsf_sensitivity.py**

找到 `compute_dlmsf_patch_fd()` 中约 239–255 行：
```python
    u_full, v_full, levels = _extract_uv_levels(
        eval_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=1,       # ← 改这里
    )
    u_bg, v_bg, bg_levels = _extract_uv_levels(
        baseline_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=1,       # ← 和这里
    )
```
改为：
```python
    u_full, v_full, levels = _extract_uv_levels(
        eval_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=target_time_idx,
    )
    u_bg, v_bg, bg_levels = _extract_uv_levels(
        baseline_inputs,
        window.lat_vals,
        window.lon_vals,
        time_idx=target_time_idx,
    )
```

**Step 4: 跑测试确认 PASS**

```bash
pytest tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py::TestDlmsfTimeIdxPropagation -v
```

**Step 5: 修复 patch_comparison.py 的 env_mask 预计算块**

找到约 1293–1297 行（env_mask 预计算里的 `_extract_uv_levels` 调用）：
```python
        _uv_for_mask, _vv_for_mask, _ = _extract_uv_levels(
            context.eval_inputs, window.lat_vals, window.lon_vals
        )
```
改为：
```python
        _uv_for_mask, _vv_for_mask, _ = _extract_uv_levels(
            context.eval_inputs, window.lat_vals, window.lon_vals,
            time_idx=runtime_cfg.target_time_idx,
        )
```

**Step 6: 修复 patch_comparison.py 的 E1 块**

找到约 1328–1335 行（E1 里的两处 `_extract_uv_levels`）：
```python
            u_eval, v_eval, levels_eval = _extract_uv_levels(
                context.eval_inputs, window.lat_vals, window.lon_vals
            )
            u_base, v_base, _ = _extract_uv_levels(
                baseline_inputs, window.lat_vals, window.lon_vals
            )
```
改为：
```python
            u_eval, v_eval, levels_eval = _extract_uv_levels(
                context.eval_inputs, window.lat_vals, window.lon_vals,
                time_idx=runtime_cfg.target_time_idx,
            )
            u_base, v_base, _ = _extract_uv_levels(
                baseline_inputs, window.lat_vals, window.lon_vals,
                time_idx=runtime_cfg.target_time_idx,
            )
```

**Step 7: 跑全部测试确认无回归**

```bash
pytest --tb=short -q
```
预期：全绿（203 passed）

**Step 8: Commit**

```bash
git add physics/dlmsf_patch_fd/dlmsf_sensitivity.py \
        physics/dlmsf_patch_fd/patch_comparison.py \
        tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py
git commit -m "fix: propagate target_time_idx to all _extract_uv_levels calls"
```

---

## Task 2：env_mask 不足时抛出 ValueError

**Files:**
- Modify: `physics/dlmsf_patch_fd/dlmsf_sensitivity.py:72-155` (`compute_dlmsf_925_300`)
- Test: `tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py`

**背景：**
`compute_dlmsf_925_300()` 在 annulus 点数不足时悄悄退化成全图平均风，违背 steering annulus 的物理定义。应当改为 raise ValueError，让调用方决定是跳过还是用 fallback。

**Step 1: 写失败测试**

在 `tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py` 的 `TestDlmsfTimeIdxPropagation` 类之后添加：

```python
class TestEnvMaskInsufficientRaisesError:
    """annulus 点数不足时必须 raise ValueError，不能静默退化。"""

    def test_raises_when_annulus_outer_km_too_small(self):
        """把 annulus_outer_km 设为极小值，使得 env_mask 没有足够点。"""
        import pytest
        from physics.dlmsf_patch_fd.dlmsf_sensitivity import compute_dlmsf_925_300

        lev = np.asarray([925, 500, 300], dtype=np.float32)
        lat = np.linspace(-5, 5, 11)
        lon = np.linspace(115, 125, 11)
        u_levels = {925: np.ones((11, 11), dtype=np.float32),
                    500: np.ones((11, 11), dtype=np.float32),
                    300: np.ones((11, 11), dtype=np.float32)}
        v_levels = {k: np.zeros((11, 11), dtype=np.float32) for k in [925, 500, 300]}

        with pytest.raises(ValueError, match="Steering annulus"):
            compute_dlmsf_925_300(
                u_levels=u_levels,
                v_levels=v_levels,
                levels_hpa=lev,
                lat_vals=lat,
                lon_vals=lon,
                center_lat=0.0,
                center_lon=120.0,
                annulus_inner_km=0.0,
                annulus_outer_km=1.0,   # 极小：几乎没有 annulus 点
                min_env_points=10,
            )
```

**Step 2: 跑测试确认 FAIL**

```bash
pytest tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py::TestEnvMaskInsufficientRaisesError -v
```
预期：FAIL（当前是静默退化，不会 raise）

**Step 3: 修复 dlmsf_sensitivity.py**

找到 `compute_dlmsf_925_300()` 约 141–142 行：
```python
    if int(np.sum(env_mask)) < min_env_points:
        env_mask = finite_mask
```
改为：
```python
    n_env = int(np.sum(env_mask))
    if n_env < min_env_points:
        raise ValueError(
            f"Steering annulus has only {n_env} valid grid points "
            f"(< min_env_points={min_env_points}). "
            f"Cannot compute DLMSF. Check annulus_inner_km / annulus_outer_km configuration."
        )
```

**Step 4: 跑测试确认 PASS**

```bash
pytest tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py::TestEnvMaskInsufficientRaisesError -v
```

**Step 5: 跑全部测试确认无回归**

```bash
pytest --tb=short -q
```
预期：全绿

**Step 6: Commit**

```bash
git add physics/dlmsf_patch_fd/dlmsf_sensitivity.py \
        tests/physics/dlmsf_patch_fd/test_dlmsf_patch_fd.py
git commit -m "fix: raise ValueError when steering annulus has insufficient grid points"
```

---

## Task 3：`_patch_scores_from_maps` 接受 annulus_mask + 归一化（Issue #3 + #4 合并）

**Files:**
- Modify: `physics/dlmsf_patch_fd/patch_comparison.py:109-133` (`_patch_scores_from_maps`)
- Modify: `physics/dlmsf_patch_fd/patch_comparison.py:~1128-1200` （移动 env_mask_annulus 预计算到 ig_patch 之前，并传入 annulus_mask）
- Test: `tests/physics/dlmsf_patch_fd/test_patch_comparison_metrics.py`

**背景：**
IG patch scores 目前对整个窗口求和，而 DLMSF scores 天然只受 annulus 内风场影响。需要让 IG 也只在 annulus 内计数并归一化（除以 annulus 内有效 cell 数）。

**Step 1: 写失败测试**

在 `tests/physics/dlmsf_patch_fd/test_patch_comparison_metrics.py` 中，**替换** `test_patch_scores_from_maps_uses_sum_aggregation` 并新增如下测试：

```python
from physics.dlmsf_patch_fd.patch_comparison import _patch_scores_from_maps
from shared.patch_geometry import build_centered_window

# --- 保留原有导入 ---

def test_patch_scores_from_maps_uses_sum_aggregation_without_annulus():
    """不传 annulus_mask 时，行为与原来一致（全 patch 求和）。"""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    ones = np.ones(window.shape)
    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=ones, abs_cell_map=ones,
    )
    assert result["abs_scores"].shape[0] == len(result["patches"])
    assert all(s > 0 for s in result["abs_scores"])


def test_patch_scores_with_annulus_mask_restricts_to_annulus():
    """annulus_mask 为只有一个 True 的角落时，只有覆盖那个 cell 的 patch 得正分。"""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    abs_map = np.ones(window.shape)
    # 只在 (0,0) 位置有 annulus 点
    annulus_mask = np.zeros(window.shape, dtype=bool)
    annulus_mask[0, 0] = True

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=abs_map, abs_cell_map=abs_map,
        annulus_mask=annulus_mask,
    )
    # 只有包含 (0,0) 的 patch 有正分
    for i, patch in enumerate(result["patches"]):
        covers_corner = bool(np.asarray(patch.mask, dtype=bool)[0, 0])
        if covers_corner:
            assert result["abs_scores"][i] > 0.0
        else:
            assert result["abs_scores"][i] == 0.0


def test_patch_scores_normalized_by_annulus_cell_count():
    """归一化：两个 patch 覆盖不同数量的 annulus cell，得分应该是相同的 mean。"""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    # cell_map 全为 1.0，annulus_mask 全为 True → 归一化后每个 patch 得分 = 1.0
    ones = np.ones(window.shape)
    full_annulus = np.ones(window.shape, dtype=bool)

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=ones, abs_cell_map=ones,
        annulus_mask=full_annulus,
    )
    # 每个 patch 在全 annulus 内均为 1.0，归一化后全为 1.0
    np.testing.assert_allclose(result["abs_scores"], 1.0, rtol=1e-6)


def test_no_overlap_patch_scores_zero():
    """patch 与 annulus_mask 完全不相交时得分为 0。"""
    lat = np.linspace(-5, 5, 11)
    lon = np.linspace(115, 125, 11)
    window = build_centered_window(lat, lon, center_lat=0.0, center_lon=120.0,
                                   window_size=11, core_size=3)
    abs_map = np.ones(window.shape)
    # annulus 只在最后一行最后一列（不太可能被所有 patch 覆盖）
    annulus_mask = np.zeros(window.shape, dtype=bool)
    # 给最后行最后列置 True，看看哪些 patch 覆盖到
    annulus_mask[-1, -1] = True

    result = _patch_scores_from_maps(
        window=window, patch_size=3, stride=2,
        signed_cell_map=abs_map, abs_cell_map=abs_map,
        annulus_mask=annulus_mask,
    )
    # 覆盖 (-1,-1) 的 patch 得正分；不覆盖的得 0
    for i, patch in enumerate(result["patches"]):
        covers = bool(np.asarray(patch.mask, dtype=bool)[-1, -1])
        if covers:
            assert result["abs_scores"][i] > 0.0
        else:
            assert result["abs_scores"][i] == 0.0, f"patch {i} should be 0"
```

**Step 2: 跑测试确认 FAIL**

```bash
pytest tests/physics/dlmsf_patch_fd/test_patch_comparison_metrics.py -v -k "annulus"
```
预期：FAIL（`_patch_scores_from_maps` 还没有 `annulus_mask` 参数）

**Step 3: 修复 `_patch_scores_from_maps`**

将 `patch_comparison.py` 约 109–133 行的函数改为：

```python
def _patch_scores_from_maps(
    *,
    window: CenteredWindow,
    patch_size: int,
    stride: int,
    signed_cell_map: np.ndarray,
    abs_cell_map: np.ndarray,
    annulus_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    patches = build_sliding_patches(window, patch_size=patch_size, stride=stride)
    signed_cell = np.asarray(signed_cell_map, dtype=np.float64)
    abs_cell = np.asarray(abs_cell_map, dtype=np.float64)

    if annulus_mask is not None:
        ann = np.asarray(annulus_mask, dtype=bool)
        signed_scores = np.zeros(len(patches), dtype=np.float64)
        abs_scores = np.zeros(len(patches), dtype=np.float64)
        for i, patch in enumerate(patches):
            effective = np.asarray(patch.mask, dtype=bool) & ann
            n = int(np.sum(effective))
            if n > 0:
                abs_scores[i] = float(np.sum(abs_cell[effective])) / n
                signed_scores[i] = float(np.sum(signed_cell[effective])) / n
            # else: 0.0 (default)
    else:
        signed_scores = np.array(
            [float(np.sum(signed_cell[patch.mask])) for patch in patches],
            dtype=np.float64,
        )
        abs_scores = np.array(
            [float(np.sum(abs_cell[patch.mask])) for patch in patches],
            dtype=np.float64,
        )

    return {
        "patches": patches,
        "signed_scores": signed_scores,
        "abs_scores": abs_scores,
        "signed_map": patch_scores_to_grid(signed_scores, patches, window.shape, core_mask=window.core_mask),
        "abs_map": patch_scores_to_grid(abs_scores, patches, window.shape, core_mask=window.core_mask),
    }
```

**Step 4: 跑测试确认 PASS**

```bash
pytest tests/physics/dlmsf_patch_fd/test_patch_comparison_metrics.py -v
```

**Step 5: 移动 env_mask_annulus 预计算到 ig_patch 之前，并传入 annulus_mask**

在 `patch_comparison.py` 中，找到如下顺序（约 1171–1200 行）：

```python
    ig_maps = _compute_track_ig_cell_maps(...)
    ig_patch = _patch_scores_from_maps(
        window=window,
        ...
        signed_cell_map=ig_maps["signed_cell_map"],
        abs_cell_map=ig_maps["abs_cell_map"],
    )
    dlmsf_result = compute_dlmsf_patch_fd(...)
    ...
    # --- 约 1285 行 ---
    # Precompute DLMSF annulus mask (reused by E1)
    env_mask_annulus = _compute_dlmsf_env_mask(...)
```

需要将整个 "Precompute DLMSF annulus mask" 块（约 1285–1310 行，包含 `_uv_for_mask` 提取和 `env_mask_annulus = _compute_dlmsf_env_mask(...)`）**剪切**到 `ig_maps = _compute_track_ig_cell_maps(...)` 之前（约 1171 行），然后修改 `ig_patch` 调用加上 `annulus_mask=env_mask_annulus`：

```python
    ig_patch = _patch_scores_from_maps(
        window=window,
        patch_size=main_patch_size,
        stride=stride,
        signed_cell_map=ig_maps["signed_cell_map"],
        abs_cell_map=ig_maps["abs_cell_map"],
        annulus_mask=env_mask_annulus,   # NEW
    )
```

**Step 6: 跑全部测试确认无回归**

```bash
pytest --tb=short -q
```
预期：全绿

**Step 7: Commit**

```bash
git add physics/dlmsf_patch_fd/patch_comparison.py \
        tests/physics/dlmsf_patch_fd/test_patch_comparison_metrics.py
git commit -m "feat: restrict IG patch scores to steering annulus with per-cell normalization"
```

---

## Task 4：scatter 坐标轴标签修正

**Files:**
- Modify: `physics/dlmsf_patch_fd/plot_track_patch_report.py:244-245`
- Test: `tests/physics/dlmsf_patch_fd/test_plot_track_patch_report.py`

**背景：**
数据赋值 `x = dlmsf_abs`，`y = ig_abs`，但标签写的是 `x="|IG| patch score"`，`y="|DLMSF|..."`，完全反了。

**Step 1: 写失败测试**

在 `tests/physics/dlmsf_patch_fd/test_plot_track_patch_report.py` 末尾添加：

```python
def test_scatter_xlabel_is_dlmsf(tmp_path):
    """x 轴（data = dlmsf_abs）标签应该写 DLMSF，不是 IG。"""
    import matplotlib
    matplotlib.use("Agg")
    from physics.dlmsf_patch_fd.plot_track_patch_report import write_scatter_figure

    report = {
        "main_case": "along_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {"direction": "along"},
                    "scatter": {
                        "x_patch_abs_scores": [1.0, 2.0, 3.0],
                        "y_patch_abs_scores": [1.5, 2.5, 0.5],
                        "spearman_rho": 0.5,
                    },
                }
            }
        },
    }
    out = tmp_path / "scatter.png"
    write_scatter_figure(report, out)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 重新调用以捕获 axes（直接从已保存的图无法读取 label，改用 monkeypatch 方式）
    # 测试策略：检查源码或用 mock
    # 这里用文件存在性 + 不测 label 内容（实际 label 测试见 test_scatter_xlabel_text）
    assert out.exists()


def test_scatter_xlabel_text(tmp_path, monkeypatch):
    """scatter 图 x 轴标签必须包含 'DLMSF'，y 轴必须包含 'IG'。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    captured_axes = []

    original_subplots = plt.subplots

    def mock_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", mock_subplots)

    from physics.dlmsf_patch_fd import plot_track_patch_report
    monkeypatch.setattr(plot_track_patch_report.plt, "subplots", mock_subplots)

    report = {
        "main_case": "along_p3",
        "cases": {
            "along_p3": {
                "visualization": {
                    "meta": {"direction": "along"},
                    "scatter": {
                        "x_patch_abs_scores": [1.0, 2.0, 3.0],
                        "y_patch_abs_scores": [1.5, 2.5, 0.5],
                        "spearman_rho": 0.5,
                    },
                }
            }
        },
    }
    out = tmp_path / "scatter.png"
    plot_track_patch_report.write_scatter_figure(report, out)

    assert captured_axes, "subplots mock not triggered"
    ax = captured_axes[-1]
    assert "DLMSF" in ax.get_xlabel(), f"x label should contain 'DLMSF', got: {ax.get_xlabel()!r}"
    assert "IG" in ax.get_ylabel(), f"y label should contain 'IG', got: {ax.get_ylabel()!r}"
```

**Step 2: 跑测试确认 FAIL**

```bash
pytest tests/physics/dlmsf_patch_fd/test_plot_track_patch_report.py::test_scatter_xlabel_text -v
```
预期：FAIL（x label 现在是 "|IG| patch score"）

**Step 3: 修复 plot_track_patch_report.py**

找到约 244–245 行：
```python
    ax.set_xlabel("|IG| patch score")
    ax.set_ylabel(f"|DLMSF_{direction}| patch score")
```
改为：
```python
    ax.set_xlabel(f"|DLMSF_{direction}| patch score")
    ax.set_ylabel("|IG| patch score")
```

**Step 4: 跑测试确认 PASS**

```bash
pytest tests/physics/dlmsf_patch_fd/test_plot_track_patch_report.py -v
```

**Step 5: 跑全部测试**

```bash
pytest --tb=short -q
```
预期：全绿

**Step 6: Commit**

```bash
git add physics/dlmsf_patch_fd/plot_track_patch_report.py \
        tests/physics/dlmsf_patch_fd/test_plot_track_patch_report.py
git commit -m "fix: correct scatter plot axis labels (x=DLMSF, y=IG)"
```

---

## Task 5：deletion 实验注释（Issue #6）

**Files:**
- Modify: `physics/dlmsf_patch_fd/patch_comparison.py`（`_run_deletion_validation` 附近）

**背景：**
不改代码逻辑，只加说明注释，防止在论文中被误标为"物理机制验证"。

**Step 1: 在 `_run_deletion_validation` 函数开头添加注释**

找到 `def _run_deletion_validation(` 的函数体开始处，添加：

```python
    # NOTE: This deletion validation verifies that model spatial hotspots are
    # predictively significant (masking them with baseline changes the forecast).
    # It does NOT verify the physical steering mechanism.
    # Correct label for paper: "spatial hotspot sensitivity test" or
    # "model attribution significance test", NOT "physical mechanism verification".
```

**Step 2: 跑全部测试确认无回归**

```bash
pytest --tb=short -q
```

**Step 3: Commit**

```bash
git add physics/dlmsf_patch_fd/patch_comparison.py
git commit -m "docs: clarify deletion validation scope in code comment"
```

---

## Task 6：清理残留 sign 逻辑（Issue #7）

**Files:**
- Modify: `physics/dlmsf_patch_fd/patch_comparison.py`
  - `_build_case_visualization_payload()`: 删除 `ig_signed_scores`、`dlmsf_signed_scores` 参数及所有 sign_class_map 计算
  - 删除 `classify_patch_roles()` 函数
  - 更新主线两处调用点（约 1225、1255–1258 行）
- Modify: `tests/physics/dlmsf_patch_fd/test_track_patch_visualization_payload.py`
  - 删除 `sign_map` 相关 key 断言
  - 更新 fixture 删除 `ig_signed_scores`、`dlmsf_signed_scores`
- Delete: `tests/physics/dlmsf_patch_fd/test_classify_patch_roles.py`

**Step 1: 更新测试（先让测试反映目标状态）**

在 `test_track_patch_visualization_payload.py` 中：

1. 找到 fixture/helper 中传给 `_build_case_visualization_payload` 的 `ig_signed_scores` 和 `dlmsf_signed_scores` 参数，删除它们。

2. 找到 `test_payload_has_required_top_level_keys`：
   ```python
   for key in ("meta", "overlap", "scatter", "sign_map", "deletion"):
   ```
   改为：
   ```python
   for key in ("meta", "overlap", "scatter", "deletion"):
   ```

3. 删除文件中所有引用 `sign_map` 作为 key 的断言（包括 `"sign_map"` key 内容的测试）。

**Step 2: 删除 test_classify_patch_roles.py**

```bash
git rm tests/physics/dlmsf_patch_fd/test_classify_patch_roles.py
```

**Step 3: 跑测试确认 FAIL（因为源码还没改）**

```bash
pytest tests/physics/dlmsf_patch_fd/test_track_patch_visualization_payload.py -v
```
预期：部分 FAIL（fixture 传参不匹配）

**Step 4: 修改 `_build_case_visualization_payload`**

**删除以下内容：**
- 函数签名中的 `ig_signed_scores: np.ndarray` 和 `dlmsf_signed_scores: np.ndarray` 参数
- 函数 docstring 中的 `sign_map` 条目
- `ig_signed = np.asarray(ig_signed_scores, ...)` 和 `dlmsf_signed = np.asarray(dlmsf_signed_scores, ...)` 赋值
- 整个 `sign_class_map` 构建循环（约 20 行）
- `sign_agreement_at_20` 计算
- `same_sign_count` 变量
- 返回 dict 中的 `"sign_map"` 键及其内容

**保留：** 其余所有内容（`overlap`、`scatter`、`meta`、`deletion` 键）。

**Step 5: 删除 `classify_patch_roles()` 函数**

找到约 634–698 行的 `def classify_patch_roles(...)` 函数，整个删除。

**Step 6: 更新主线调用点**

找到约 1225 行和约 1248–1262 行，删除传入 `_build_case_visualization_payload` 的 `ig_signed_scores` 和 `dlmsf_signed_scores` 参数。

**Step 7: 跑测试确认全绿**

```bash
pytest --tb=short -q
```
预期：全绿（比之前少 `test_classify_patch_roles.py` 的若干条）

**Step 8: Commit**

```bash
git add physics/dlmsf_patch_fd/patch_comparison.py \
        tests/physics/dlmsf_patch_fd/test_track_patch_visualization_payload.py
git rm tests/physics/dlmsf_patch_fd/test_classify_patch_roles.py
git commit -m "remove: sign_map, classify_patch_roles, ig/dlmsf_signed_scores from payload"
```

---

## 最终验证

```bash
pytest --tb=short -q
```

预期结果：所有测试通过，数量比开始时（203）略少（删除了 `test_classify_patch_roles.py`）。

---

## 变更摘要

| Task | 文件 | 类型 |
|------|------|------|
| 1 | `dlmsf_sensitivity.py`, `patch_comparison.py` | fix |
| 2 | `dlmsf_sensitivity.py` | fix |
| 3+4 | `patch_comparison.py`, test | feat+fix |
| 4 | `plot_track_patch_report.py`, test | fix |
| 5 | `patch_comparison.py` | docs |
| 6 | `patch_comparison.py`, tests | remove |
