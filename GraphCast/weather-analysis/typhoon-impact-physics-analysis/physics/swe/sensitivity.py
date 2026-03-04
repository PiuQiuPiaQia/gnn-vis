# -*- coding: utf-8 -*-
"""Physics-based sensitivity analysis for V2 model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class V2SensitivityResult:
    """Container for signed-first sensitivity analysis results.
    
    Attributes:
        dJ_dh_signed: Signed gradient values (float64).
        S_abs: Absolute values of the gradient (float64).
    """
    dJ_dh_signed: np.ndarray
    S_abs: np.ndarray


def pack_signed_sensitivity(dJ_dh_signed: np.ndarray) -> V2SensitivityResult:
    """Pack signed gradient into a V2SensitivityResult.
    
    Converts the signed gradient to float64, stores it, and computes
    the absolute values.
    
    Args:
        dJ_dh_signed: Signed gradient array of any shape.
        
    Returns:
        V2SensitivityResult containing the signed gradient and its absolute values,
        both as float64 arrays with the same shape as input.
    """
    # Convert to float64 to ensure numerical stability
    dJ_dh_signed_f64 = np.asarray(dJ_dh_signed, dtype=np.float64)
    
    # Compute absolute values
    S_abs = np.abs(dJ_dh_signed_f64)
    
    return V2SensitivityResult(
        dJ_dh_signed=dJ_dh_signed_f64,
        S_abs=S_abs
    )
