# -*- coding: utf-8 -*-
"""SWE model core implementation."""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # Gravitational constant (m/s^2)


@dataclass
class V2PhysicsConfig:
    """Configuration for V2 SWE physics model.

    Attributes:
        dx: Grid spacing in x direction (meters)
        dy: Grid spacing in y direction (meters)
        H_eq: Equivalent depth (meters)
        U_bar: Background zonal flow (m/s)
        V_bar: Background meridional flow (m/s)
        dt: Time step (seconds), optional with default
    """
    dx: float
    dy: float
    H_eq: float
    U_bar: float
    V_bar: float
    dt: float = 1.0


def enforce_cfl_dt(cfg: V2PhysicsConfig, cfl: float = 0.35) -> float:
    """Enforce CFL condition on time step.

    Computes the maximum stable time step based on the CFL condition
    for the shallow water equations and returns the minimum of the
    current dt and the CFL limit.

    Formula: dt_max = cfl * min(dx, dy) / (c + max(|U_bar|, |V_bar|))
    where c = sqrt(g * H_eq) is the gravity wave phase speed.

    Args:
        cfg: Physics configuration containing grid spacing, depth, and flow
        cfl: CFL number, default 0.35 for stability margin

    Returns:
        Time step that satisfies CFL condition (<= cfg.dt)

    Raises:
        ValueError: If cfl, dx, dy, dt, or H_eq is non-positive
    """
    # Input validation
    if cfl <= 0:
        raise ValueError(f"cfl must be positive, got {cfl}")
    if cfg.dx <= 0:
        raise ValueError(f"dx must be positive, got {cfg.dx}")
    if cfg.dy <= 0:
        raise ValueError(f"dy must be positive, got {cfg.dy}")
    if cfg.dt <= 0:
        raise ValueError(f"dt must be positive, got {cfg.dt}")
    if cfg.H_eq <= 0:
        raise ValueError(f"H_eq must be positive, got {cfg.H_eq}")

    # Gravity wave phase speed
    c = math.sqrt(G * cfg.H_eq)

    # Maximum background flow speed
    max_flow = max(abs(cfg.U_bar), abs(cfg.V_bar))

    # CFL-limited time step
    dt_max = cfl * min(cfg.dx, cfg.dy) / (c + max_flow)

    # Return minimum of current dt and CFL limit
    return min(cfg.dt, dt_max)
