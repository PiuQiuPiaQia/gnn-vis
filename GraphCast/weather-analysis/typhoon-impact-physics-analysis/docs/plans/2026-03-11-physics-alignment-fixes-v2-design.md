# Physics Alignment Fixes v2 — Design

## Goal

修复 DLMSF-IG 对比实验中仍然存在的 7 个对齐问题，其中 5 个关键问题直接破坏实验结论，2 个中等问题引入不必要的复杂度。

## Background

前一批修复（commit a750293）删除了 E2/E4/cross 方向，并清理了若干无用字段。但以下问题仍然存在：

1. `time_idx=1` 写死：物理计算实际上没有跟随 `target_time_idx`
2. `env_mask` 不足时 silent 退化成全图平均，破坏 steering annulus 定义
3. 主比较用全窗口 IG vs annulus DLMSF，两者不一致
4. patch score 无归一化，边界 patch 得分系统偏低
5. scatter 图坐标标签与数据赋值相反
6. deletion 实验缺乏说明注释（中等）
7. 残留的 sign/direction 逻辑未彻底清理（中等）

## Design Decisions

### Issue 1：time_idx 传播

**位置：**
- `dlmsf_sensitivity.py` `compute_dlmsf_patch_fd()` 第 243、249 行：`_extract_uv_levels(..., time_idx=1)`
- `patch_comparison.py` env_mask 预计算块（~1297 行）：`_extract_uv_levels(...)` 无 time_idx 参数
- `patch_comparison.py` E1 块（~1328、1330 行）：同上

**修复：** 所有调用改为 `time_idx=target_time_idx`（或 `runtime_cfg.target_time_idx`）。

### Issue 2：env_mask 不足抛出 ValueError

**位置：** `dlmsf_sensitivity.py` `compute_dlmsf_925_300()` 第 141–142 行

**修复：**
```python
# 删除：
if int(np.sum(env_mask)) < min_env_points:
    env_mask = finite_mask

# 改为：
n_env = int(np.sum(env_mask))
if n_env < min_env_points:
    raise ValueError(
        f"Steering annulus has only {n_env} valid grid points "
        f"(< min_env_points={min_env_points}); cannot compute DLMSF. "
        f"Check annulus_inner_km / annulus_outer_km configuration."
    )
```

调用方（`compute_dlmsf_patch_fd`、`patch_comparison.py`）的 try/except 捕获并跳过该 case，记录 WARNING。

### Issue 3 + 4：IG 限制到 annulus + 归一化（合并实现）

**`_patch_scores_from_maps()` 新增参数：**
```python
def _patch_scores_from_maps(
    *,
    window: CenteredWindow,
    patch_size: int,
    stride: int,
    signed_cell_map: np.ndarray,
    abs_cell_map: np.ndarray,
    annulus_mask: np.ndarray | None = None,  # NEW
) -> Dict[str, Any]:
```

**计分逻辑（有 annulus_mask 时）：**
```python
effective = patch.mask & annulus_mask
n = int(np.sum(effective))
abs_scores[i] = float(np.sum(abs_cell[effective])) / n if n > 0 else 0.0
signed_scores[i] = float(np.sum(signed_cell[effective])) / n if n > 0 else 0.0
```

无 annulus_mask 时保持原有行为（向后兼容）。

**主比较调用：**
```python
ig_patch = _patch_scores_from_maps(
    window=window,
    patch_size=main_patch_size,
    stride=stride,
    signed_cell_map=ig_maps["signed_cell_map"],
    abs_cell_map=ig_maps["abs_cell_map"],
    annulus_mask=env_mask_annulus,  # NEW
)
```

注意：`env_mask_annulus` 必须在 `ig_patch` 计算之前已经准备好（目前它是在主比较之后才预计算的，顺序需要调整）。

### Issue 5：scatter 坐标标签

**位置：** `plot_track_patch_report.py` 第 244–245 行

**数据约定（保持不变）：** `x = dlmsf_abs`（物理参考），`y = ig_abs`（模型输出）

**修复标签：**
```python
ax.set_xlabel("|DLMSF| patch score")   # 原："|IG| patch score"
ax.set_ylabel("|IG| patch score")      # 原：f"|DLMSF_{direction}| patch score"
```

### Issue 6：deletion 注释（中等）

在 `_run_deletion_validation()` 附近加注释：
```
# NOTE: Deletion validation verifies that model spatial hotspots are predictively
# significant (removing them changes the forecast). This is NOT equivalent to
# verifying physical mechanism — it cannot confirm the model uses steering flow.
# Label accordingly in paper: "spatial hotspot sensitivity test".
```

### Issue 7：残留 sign 逻辑清理（中等）

删除以下内容：
- `_build_case_visualization_payload()` 返回值中的 `sign_map` 键
- `classify_patch_roles()` 函数（或其调用，视是否有测试覆盖）
- `ig_signed_scores`、`dlmsf_signed_scores` 在主线 payload 的使用

## 执行顺序注意

Issue 3 要求 `env_mask_annulus` 在主比较 `ig_patch` 之前准备好。
目前代码顺序：`ig_patch` → `dlmsf_result` → `env_mask_annulus`。
需要调整为：`env_mask_annulus` → `ig_patch`（含 annulus_mask）→ `dlmsf_result`。

## 不做的事

- 不改动 deletion 的实际计算逻辑（全变量全层全时间替换是合理的，只加注释）
- 不改动 DLMSF `patch_parallel_scores` 的计算（它已经是 annulus-aware 的）
- 不归一化 DLMSF scores（delta J 是物理量，无需归一化）
