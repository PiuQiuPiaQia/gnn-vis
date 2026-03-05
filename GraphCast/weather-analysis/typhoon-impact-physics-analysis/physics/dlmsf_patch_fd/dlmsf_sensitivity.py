from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray


@dataclass
class DLMSFSensitivityResult:
    S_map: np.ndarray            # 2D (n_lat, n_lon)
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    center_lat: float
    center_lon: float
    target_time_idx: int
    d_hat: Tuple[float, float]   # (d_u, d_v)
    J_phys_baseline: float
    U_dlmsf: float
    V_dlmsf: float
    n_patches: int
    elapsed_sec: float


def compute_d_hat(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> Tuple[float, float]:
    """计算台风移动方向的单位向量 (d_u, d_v)。

    d_u 对应经向位移（东向为正），d_v 对应纬向位移（北向为正）。
    若台风静止（lat0==lat1 且 lon0==lon1），返回 (0.0, 0.0)。
    """
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    # 经度差折叠到 [-180, 180]
    dlon = ((dlon + 180.0) % 360.0) - 180.0
    mag = math.hypot(dlon, dlat)
    if mag < 1e-10:
        return 0.0, 0.0
    return float(dlon / mag), float(dlat / mag)
