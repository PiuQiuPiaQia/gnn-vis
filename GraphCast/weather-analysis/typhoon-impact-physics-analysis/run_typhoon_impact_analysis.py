#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for typhoon impact physics analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import config as cfg

from analysis_pipeline import AnalysisConfig, build_analysis_context
from analysis_reporting import (
    build_importance_dataarrays,
    build_importance_map_specs,
    print_top_n,
    save_importance_plots_with_center,
)


ROOT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()


def run_typhoon_impact_analysis() -> Dict[str, Any]:
    # Local import: keep module importable in environments without heavy deps.
    from importance_methods import (
        compute_gradient_importance,
        compute_perturbation_importance,
    )

    # 1) Build runtime config and initialize model/data context.
    runtime_cfg = AnalysisConfig.from_module(cfg)
    context = build_analysis_context(runtime_cfg)

    # 2) Compute importance maps based on selected mode.
    compare_maps: Optional[Dict[str, Dict[str, Any]]] = None
    if runtime_cfg.importance_mode == "perturbation":
        importance_maps = compute_perturbation_importance(context, runtime_cfg)
    elif runtime_cfg.importance_mode == "input_gradient":
        importance_maps = compute_gradient_importance(context, runtime_cfg)
    elif runtime_cfg.importance_mode == "compare":
        if len(context.target_vars) != 1:
            raise ValueError("IMPORTANCE_MODE='compare' requires exactly one target variable")
        perturb_maps = compute_perturbation_importance(context, runtime_cfg)
        gradient_maps = compute_gradient_importance(context, runtime_cfg)
        compare_maps = {
            "perturbation": perturb_maps,
            "input_gradient": gradient_maps,
        }
        importance_maps = perturb_maps
    else:
        raise ValueError(f"Unknown IMPORTANCE_MODE: {runtime_cfg.importance_mode}")

    # 3) Convert numeric maps to structured outputs for plotting/reporting.
    lat_sel_vals = context.lat_vals[context.lat_indices]
    lon_sel_vals = context.lon_vals[context.lon_indices]

    map_specs = build_importance_map_specs(
        importance_mode=runtime_cfg.importance_mode,
        target_vars=context.target_vars,
        importance_maps=importance_maps,
        compare_maps=compare_maps,
    )
    importance_das = build_importance_dataarrays(
        map_specs=map_specs,
        lat_sel_vals=lat_sel_vals,
        lon_sel_vals=lon_sel_vals,
    )

    # 4) Persist figures and print ranked influential grid points.
    save_importance_plots_with_center(
        runtime_cfg=runtime_cfg,
        root_dir=ROOT_DIR,
        target_vars=context.target_vars,
        importance_das=importance_das,
        center_lat=context.center_lat,
        center_lon=context.center_lon,
    )

    print_top_n(
        runtime_cfg=runtime_cfg,
        target_vars=context.target_vars,
        lat_sel_vals=lat_sel_vals,
        lon_sel_vals=lon_sel_vals,
        importance_maps=importance_maps,
        compare_maps=compare_maps,
    )

    # 5) Return artifacts for notebook/debug reuse.
    return {
        "runtime_cfg": runtime_cfg,
        "context": context,
        "importance_maps": importance_maps,
        "compare_maps": compare_maps,
        "importance_das": importance_das,
    }


def main() -> int:
    run_typhoon_impact_analysis()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
