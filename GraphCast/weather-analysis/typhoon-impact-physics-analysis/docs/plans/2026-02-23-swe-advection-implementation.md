# SWE Background Advection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add background DLM steering wind `(Ū, V̄)` to the SWE tendency equations so sensitivity maps shift upstream for non-zero lead times.

**Architecture:** Three-layer change: (1) `SWEPhysicsConfig` gains `U_bar`/`V_bar` fields and `_swe_tendency` gains advection terms; (2) new `compute_dlm_background_wind` extracts ERA5 DLM wind in an annulus; (3) `compute_sensitivity_jax` wires the extraction into config construction, and config.py exposes tuning knobs. `U_bar=V_bar=0` is the default and preserves all existing behaviour.

**Tech Stack:** Python, JAX/jnp, xarray, numpy, built-in unittest

---

### Task 1: Add advection tests (RED)

**Files:**
- Create: `tests/test_swe_advection.py`
- Modify: none

**Step 1: Write failing tests**

Create `tests/test_swe_advection.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for SWE background advection terms and DLM wind extraction."""

import unittest
import numpy as np
import jax.numpy as jnp
import xarray

from physics.swe_model import SWEPhysicsConfig, _swe_tendency
from physics.sensitivity import compute_dlm_background_wind


class TestSWETendencyAdvection(unittest.TestCase):
    """Tests that _swe_tendency includes background advection when U_bar != 0."""

    def _cfg(self, U_bar=0.0, V_bar=0.0):
        return SWEPhysicsConfig(
            H=5500.0, f0=1e-4, g=9.81,
            dx=50000.0, dy=50000.0, dt=300.0,
            n_lat=6, n_lon=6,
            U_bar=U_bar, V_bar=V_bar,
        )

    def test_zero_background_wind_unchanged(self):
        """With U_bar=V_bar=0, tendency should match original formula."""
        cfg0 = self._cfg(U_bar=0.0, V_bar=0.0)
        n = 6
        h = jnp.ones((n, n)) * 10.0
        u = jnp.zeros((n, n))
        v = jnp.zeros((n, n))
        dh, du, dv = _swe_tendency(h, u, v, cfg0)
        # All zero: uniform field, no gradients
        np.testing.assert_allclose(dh, 0.0, atol=1e-10)
        np.testing.assert_allclose(du, 0.0, atol=1e-10)
        np.testing.assert_allclose(dv, 0.0, atol=1e-10)

    def test_nonzero_ubar_changes_tendency(self):
        """With U_bar != 0 and a non-uniform h, dh/dt must differ from U_bar=0 case."""
        cfg_no_adv = self._cfg(U_bar=0.0)
        cfg_adv    = self._cfg(U_bar=10.0)
        n = 6
        # Linear ramp in x so dh/dx != 0
        h = jnp.tile(jnp.arange(n, dtype=float), (n, 1))
        u = jnp.zeros((n, n))
        v = jnp.zeros((n, n))
        dh_no, _, _ = _swe_tendency(h, u, v, cfg_no_adv)
        dh_ad, _, _ = _swe_tendency(h, u, v, cfg_adv)
        self.assertFalse(
            jnp.allclose(dh_no, dh_ad),
            msg="Non-zero U_bar must change dh/dt for a non-uniform h field",
        )


class TestComputeDLMBackgroundWind(unittest.TestCase):
    """Tests for compute_dlm_background_wind."""

    def _make_eval_inputs(self, u_val=5.0, v_val=-3.0, n_lat=41, n_lon=41):
        """Build a minimal xarray Dataset with uniform u/v at two pressure levels."""
        lat = np.linspace(-42, -2, n_lat)
        lon = np.linspace(138, 178, n_lon)
        levels = np.array([850, 500, 300], dtype=float)
        data = np.ones((1, 2, len(levels), n_lat, n_lon), dtype=np.float32)
        u_data = data * u_val
        v_data = data * v_val
        coords = {"batch": [0], "time": [0, 1], "level": levels, "lat": lat, "lon": lon}
        return xarray.Dataset({
            "u_component_of_wind": xarray.DataArray(u_data, dims=["batch","time","level","lat","lon"], coords=coords),
            "v_component_of_wind": xarray.DataArray(v_data, dims=["batch","time","level","lat","lon"], coords=coords),
        })

    def test_uniform_field_returns_input_wind(self):
        """Uniform u/v field → DLM result equals that uniform value."""
        ds = self._make_eval_inputs(u_val=5.0, v_val=-3.0)
        U_bar, V_bar = compute_dlm_background_wind(
            ds, center_lat=-22.0, center_lon=158.0,
            inner_radius_km=300.0, outer_radius_km=800.0,
            p_bot_hpa=850.0, p_top_hpa=300.0,
        )
        self.assertAlmostEqual(U_bar, 5.0, places=1)
        self.assertAlmostEqual(V_bar, -3.0, places=1)

    def test_returns_floats(self):
        """Return types must be Python floats."""
        ds = self._make_eval_inputs()
        U_bar, V_bar = compute_dlm_background_wind(
            ds, center_lat=-22.0, center_lon=158.0,
        )
        self.assertIsInstance(U_bar, float)
        self.assertIsInstance(V_bar, float)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run to verify RED**

```
python -m unittest tests/test_swe_advection.py -v
```

Expected: FAIL with `ImportError` (functions not yet exist).

---

### Task 2: Add `U_bar`/`V_bar` to `SWEPhysicsConfig` and advection to `_swe_tendency` (GREEN)

**Files:**
- Modify: `physics/swe_model.py`
- Test: `tests/test_swe_advection.py::TestSWETendencyAdvection`

**Step 1: Extend `SWEPhysicsConfig`**

Add two fields at the end of the dataclass (before the closing):
```python
U_bar: float = 0.0   # 背景纬向引导风 (m/s)
V_bar: float = 0.0   # 背景经向引导风 (m/s)
```
`SWEPhysicsConfig` is `frozen=True` so default values require Python ≥ 3.10 dataclass field ordering — place them last.

**Step 2: Update `_swe_tendency`**

Replace the three tendency lines (`swe_model.py:112-114`) with:
```python
dh_dt = -cfg.U_bar * dh_dx - cfg.V_bar * dh_dy - cfg.H * (du_dx + dv_dy)
du_dt = -cfg.U_bar * _centered_diff_x(u, cfg.dx) - cfg.V_bar * _centered_diff_y(u, cfg.dy) - cfg.g * dh_dx + cfg.f0 * v
dv_dt = -cfg.U_bar * _centered_diff_x(v, cfg.dx) - cfg.V_bar * _centered_diff_y(v, cfg.dy) - cfg.g * dh_dy - cfg.f0 * u
```

Note: `dh_dx` and `dh_dy` are already computed above — reuse them. Compute `du_dx_adv`/`dv_dx_adv` inline (no new named variables needed).

**Step 3: Run tendency tests**

```
python -m unittest tests/test_swe_advection.TestSWETendencyAdvection -v
```

Expected: both `TestSWETendencyAdvection` tests PASS.

**Step 4: Run all existing tests to confirm no regression**

```
python -m unittest tests/test_geostrophic_mapping.py -v
```

Expected: 3 PASS.

**Step 5: Commit**

```
git add physics/swe_model.py tests/test_swe_advection.py
git commit -m "feat: add U_bar/V_bar advection terms to SWEPhysicsConfig and _swe_tendency"
```

---

### Task 3: Implement `compute_dlm_background_wind` (GREEN)

**Files:**
- Modify: `physics/sensitivity.py`
- Test: `tests/test_swe_advection.py::TestComputeDLMBackgroundWind`

**Step 1: Add function after `_n_steps_for`**

Insert after line 88 (`return int(round(...))`):

```python
def compute_dlm_background_wind(
    eval_inputs: xarray.Dataset,
    center_lat: float,
    center_lon: float,
    inner_radius_km: float = 300.0,
    outer_radius_km: float = 800.0,
    p_bot_hpa: float = 850.0,
    p_top_hpa: float = 300.0,
    time_idx: int = 1,
) -> tuple[float, float]:
    """深层平均引导风 (DLM Steering) — 环形平均 + 质量加权垂直积分。

    Args:
        eval_inputs:      ERA5 xarray Dataset，含 u/v_component_of_wind。
        center_lat/lon:   台风中心坐标（度）。
        inner_radius_km:  环形内半径 (km)。
        outer_radius_km:  环形外半径 (km)。
        p_bot_hpa:        垂直积分下界压力层 (hPa)。
        p_top_hpa:        垂直积分上界压力层 (hPa)。
        time_idx:         时间步索引（0 = 分析时刻，1 = +6h）。

    Returns:
        (U_bar, V_bar) in m/s — Python floats.
    """
    DEG_TO_M = 111320.0

    lat_all = eval_inputs.coords["lat"].values.astype(float)
    lon_all = eval_inputs.coords["lon"].values.astype(float)

    # 计算每个格点到台风中心的距离 (km)
    dlat_m = (lat_all - center_lat) * DEG_TO_M
    dlon_deg = ((lon_all - center_lon + 180.0) % 360.0) - 180.0
    cos_lat = np.cos(np.deg2rad(center_lat))
    dlon_m = dlon_deg * DEG_TO_M * cos_lat

    lat2d, lon2d = np.meshgrid(dlat_m, dlon_m, indexing="ij")
    dist_km = np.sqrt(lat2d ** 2 + lon2d ** 2) / 1000.0
    annulus_mask = (dist_km >= inner_radius_km) & (dist_km <= outer_radius_km)

    if not np.any(annulus_mask):
        return 0.0, 0.0

    def _extract_var(var_name: str) -> np.ndarray:
        da = eval_inputs[var_name]
        if "batch" in da.dims:
            da = da.isel(batch=0)
        if "time" in da.dims:
            da = da.isel(time=time_idx)
        return da  # shape (level, lat, lon)

    u_da = _extract_var(_ERA5_U_VAR)
    v_da = _extract_var(_ERA5_V_VAR)
    levels = u_da.coords["level"].values.astype(float)

    # 选择 p_top ≤ level ≤ p_bot 的层
    lev_mask = (levels >= p_top_hpa) & (levels <= p_bot_hpa)
    if not np.any(lev_mask):
        return 0.0, 0.0
    selected_levels = levels[lev_mask]

    u_arr = np.asarray(u_da.values)[lev_mask]  # (n_lev, n_lat, n_lon)
    v_arr = np.asarray(v_da.values)[lev_mask]

    # 环形空间平均 → (n_lev,)
    u_prof = np.array([u_arr[k][annulus_mask].mean() for k in range(u_arr.shape[0])])
    v_prof = np.array([v_arr[k][annulus_mask].mean() for k in range(v_arr.shape[0])])

    # 质量加权垂直积分（梯形法）
    dp = np.abs(np.diff(selected_levels))
    u_bar = float(np.sum((u_prof[:-1] + u_prof[1:]) / 2.0 * dp) / dp.sum())
    v_bar = float(np.sum((v_prof[:-1] + v_prof[1:]) / 2.0 * dp) / dp.sum())
    return u_bar, v_bar
```

**Step 2: Run DLM tests**

```
python -m unittest tests/test_swe_advection.TestComputeDLMBackgroundWind -v
```

Expected: 2 PASS.

**Step 3: Run all tests**

```
python -m unittest tests/ -v
```

Expected: 5 PASS total.

**Step 4: Commit**

```
git add physics/sensitivity.py
git commit -m "feat: add compute_dlm_background_wind for ERA5 DLM steering extraction"
```

---

### Task 4: Wire DLM wind into `compute_sensitivity_jax` and config

**Files:**
- Modify: `physics/sensitivity.py`
- Modify: `config.py`
- Test: smoke run

**Step 1: Add config knobs to `config.py`** (after `SWE_CONSTRAINT_MODE`):

```python
SWE_DLM_INNER_KM  = 300.0   # 环形内半径 (km)
SWE_DLM_OUTER_KM  = 800.0   # 环形外半径 (km)
SWE_DLM_P_BOT_HPA = 850.0   # DLM 积分下界 (hPa)
SWE_DLM_P_TOP_HPA = 300.0   # DLM 积分上界 (hPa)
```

**Step 2: Extend `compute_sensitivity_jax` signature**

Add optional parameter:
```python
eval_inputs: "xarray.Dataset | None" = None,
```

**Step 3: DLM extraction block**

In both the `"none"` and `"geostrophic_hard"` branches, **before** calling `make_physics_config`, insert:

```python
U_bar, V_bar = 0.0, 0.0
if eval_inputs is not None:
    U_bar, V_bar = compute_dlm_background_wind(
        eval_inputs, center_lat, center_lon,
        inner_radius_km=getattr(cfg_module, "SWE_DLM_INNER_KM", 300.0),  # noqa
        outer_radius_km=getattr(cfg_module, "SWE_DLM_OUTER_KM", 800.0),
        p_bot_hpa=getattr(cfg_module, "SWE_DLM_P_BOT_HPA", 850.0),
        p_top_hpa=getattr(cfg_module, "SWE_DLM_P_TOP_HPA", 300.0),
    )
    print(f"  DLM background wind: Ū={U_bar:+.2f} m/s  V̄={V_bar:+.2f} m/s")
```

Then pass `U_bar=U_bar, V_bar=V_bar` to `make_physics_config`.

> Note: `cfg_module` is not available inside `compute_sensitivity_jax`. Instead, pass the four scalar DLM params as direct kwargs to `compute_sensitivity_jax` with the same defaults as config, and read from config only in `physics/comparison.py`.

**Revised approach — cleaner API:**

Extend `compute_sensitivity_jax`:
```python
def compute_sensitivity_jax(
    ...existing params...,
    eval_inputs=None,
    dlm_inner_km: float = 300.0,
    dlm_outer_km: float = 800.0,
    dlm_p_bot_hpa: float = 850.0,
    dlm_p_top_hpa: float = 300.0,
) -> SWESensitivityResult:
```

In `physics/comparison.py` pass these from config:
```python
jax_result = compute_sensitivity_jax(
    ...,
    eval_inputs=context.eval_inputs,
    dlm_inner_km=getattr(cfg, "SWE_DLM_INNER_KM", 300.0),
    dlm_outer_km=getattr(cfg, "SWE_DLM_OUTER_KM", 800.0),
    dlm_p_bot_hpa=getattr(cfg, "SWE_DLM_P_BOT_HPA", 850.0),
    dlm_p_top_hpa=getattr(cfg, "SWE_DLM_P_TOP_HPA", 300.0),
)
```

**Step 4: Smoke run**

```
python main.py compare 2>&1 | grep -E "DLM|Constraint|Done in"
```

Expected lines like:
```
Constraint mode: geostrophic_hard
DLM background wind: Ū=... m/s  V̄=... m/s
Done in ...s
```

**Step 5: Commit**

```
git add physics/sensitivity.py physics/comparison.py config.py
git commit -m "feat: wire DLM background wind into compute_sensitivity_jax and compare pipeline"
```

---

### Task 5: Verify upstream shift and full regression

**Files:** none (verification only)

**Step 1: Run all unit tests**

```
python -m unittest tests/ -v
```

Expected: 5 PASS.

**Step 2: Confirm upstream shift**

```python
python - <<'PY'
import numpy as np, config as cfg
from physics.sensitivity import compute_sensitivity_jax
from physics.sensitivity import extract_swe_initial_conditions
from shared.analysis_pipeline import AnalysisConfig, build_analysis_context

runtime_cfg = AnalysisConfig.from_module(cfg)
ctx = build_analysis_context(runtime_cfg)
domain_half = getattr(cfg, "SWE_DOMAIN_HALF_DEG", 20.0)
h0,u0,v0,lat,lon = extract_swe_initial_conditions(ctx.eval_inputs, ctx.center_lat, ctx.center_lon, domain_half)

# With DLM
r_dlm = compute_sensitivity_jax(h0,u0,v0,lat,lon,ctx.center_lat,ctx.center_lon,0,
                                 eval_inputs=ctx.eval_inputs,
                                 constraint_mode="geostrophic_hard")
peak_idx = np.unravel_index(r_dlm.S_h.argmax(), r_dlm.S_h.shape)
peak_lat = r_dlm.lat_vals[peak_idx[0]]
peak_lon = r_dlm.lon_vals[peak_idx[1]]
print(f"TC center: ({ctx.center_lat:.1f}, {ctx.center_lon:.1f})")
print(f"S_h peak (with DLM): ({peak_lat:.1f}, {peak_lon:.1f})")
print(f"Ū={r_dlm.physics_cfg.U_bar:+.2f} m/s  V̄={r_dlm.physics_cfg.V_bar:+.2f} m/s")
PY
```

Expected: peak lat/lon differs from TC center in the upstream direction.

**Step 3: Run compare to regenerate full output**

```
python main.py compare
```

Confirm new PNGs in `validation_results/` and `physics_alignment_metrics.json` updated.
