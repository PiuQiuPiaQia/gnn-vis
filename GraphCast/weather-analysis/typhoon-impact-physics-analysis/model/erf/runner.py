from __future__ import annotations

from typing import List

import jax
import numpy as np
import xarray

from shared.analysis_pipeline import AnalysisConfig, AnalysisContext
from shared.importance_common import _combined_target_scalar, reduce_input_attribution_to_latlon


def _compute_erf_explanation_map(
    context: AnalysisContext,
    runtime_cfg: AnalysisConfig,
    vars_to_use: List[str],
) -> xarray.DataArray:
    def _loss(inputs_data):
        return _combined_target_scalar(context, runtime_cfg, inputs_data)

    grad_fn = jax.grad(_loss)
    grads = grad_fn(context.eval_inputs)

    lat_vals = np.asarray(context.eval_inputs.coords["lat"].values)
    lon_vals = np.asarray(context.eval_inputs.coords["lon"].values)
    erf_score = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.float64)

    for var_name in vars_to_use:
        grad_values = np.asarray(grads[var_name].values)
        if runtime_cfg.erf_abs:
            grad_values = np.abs(grad_values)
        grad_map_da = reduce_input_attribution_to_latlon(
            attribution=grad_values,
            original_da=context.eval_inputs[var_name],
            runtime_cfg=runtime_cfg,
        )
        erf_score += np.abs(np.asarray(grad_map_da.values, dtype=np.float64))

    return xarray.DataArray(
        erf_score,
        dims=("lat", "lon"),
        coords={"lat": lat_vals, "lon": lon_vals},
        name="erf_explanation_score",
    )
