# SWE Background Advection Design

## Goal

Introduce background steering wind `(Ū, V̄)` into the linearized SWE tendency equations so that sensitivity regions shift upstream (physically realistic for 6–24 h lead times) instead of remaining fixed at the typhoon center.

## Problem

Current equations (`physics/swe_model.py:112-114`) have no advection term:
```
∂h/∂t = -H(∂u/∂x + ∂v/∂y)
∂u/∂t = -g ∂h/∂x + f₀v
∂v/∂t = -g ∂h/∂y - f₀u
```
The typhoon "stays in place" and sensitivity maps are centered at the current position regardless of lead time.

## Approved Design (Option A: DLM annulus extraction)

### Equations after fix

```
∂h/∂t = -Ū ∂h/∂x - V̄ ∂h/∂y - H(∂u/∂x + ∂v/∂y)
∂u/∂t = -Ū ∂u/∂x - V̄ ∂u/∂y - g ∂h/∂x + f₀v
∂v/∂t = -Ū ∂v/∂x - V̄ ∂v/∂y - g ∂h/∂y - f₀u
```

Setting `Ū = V̄ = 0` recovers current behaviour exactly.

### Background wind extraction

- **Source**: ERA5 `u/v` fields already available in `eval_inputs`
- **Method**: DLM annulus average matching the standard steering-flow formula
  - Annulus: inner radius 300 km, outer radius 800 km around TC center
  - Vertical layers: 850–300 hPa, mass-weighted (trapezoid rule)
- **Output**: scalar pair `(Ū, V̄)` in m/s
- **Location**: new function `compute_dlm_background_wind(eval_inputs, lat_vals, lon_vals, center_lat, center_lon, ...)` in `physics/sensitivity.py`

### Config switches

```python
# config.py
SWE_DLM_INNER_KM  = 300.0   # annulus inner radius (km)
SWE_DLM_OUTER_KM  = 800.0   # annulus outer radius (km)
SWE_DLM_P_BOT_HPA = 850.0   # DLM bottom pressure level (hPa)
SWE_DLM_P_TOP_HPA = 300.0   # DLM top pressure level (hPa)
```

### Model changes

- `SWEPhysicsConfig` gains two new fields: `U_bar: float = 0.0`, `V_bar: float = 0.0`
- `make_physics_config` gains optional `U_bar / V_bar` kwargs (default 0)
- `_swe_tendency` adds background advection terms using existing `_centered_diff_x/y`
- `compute_sensitivity_jax` extracts DLM wind and passes it to `make_physics_config`

### Compatibility

- `U_bar = V_bar = 0` → identical to current output; existing unit tests pass unchanged
- `geostrophic_hard` constraint mode works on top of this change with no extra modification

### Validation criterion

With non-zero background wind, the peak of `S_h` (and `S_total`) should shift in the upstream direction relative to `(center_lat, center_lon)` compared to the `Ū=V̄=0` baseline.
