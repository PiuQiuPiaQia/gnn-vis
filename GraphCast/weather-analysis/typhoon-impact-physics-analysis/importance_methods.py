# -*- coding: utf-8 -*-
"""兼容性垫片，重新导出各方法的实现。

保持旧版导入（`from importance_methods import ...`）在方法拆分到独立文件后仍可正常工作。
"""

from __future__ import annotations

from erf_importance import compute_erf_importance
from gradient_importance import compute_gradient_importance
from perturbation_importance import compute_perturbation_importance

__all__ = [
    "compute_erf_importance",
    "compute_gradient_importance",
    "compute_perturbation_importance",
]
