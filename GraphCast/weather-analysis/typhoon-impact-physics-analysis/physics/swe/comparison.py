# -*- coding: utf-8 -*-
"""Comparison between physics and data-driven methods."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import config as cfg


def _resolve_dataset_types(cfg_module) -> list[str]:
    dataset_types = getattr(cfg_module, "DATASET_TYPES", None)
    if dataset_types is None:
        dataset_types = getattr(cfg_module, "DATASET_TYPE", None)

    if dataset_types is None:
        raise ValueError(
            "No dataset types configured. Set DATASET_TYPES to a list for batch runs "
            "or DATASET_TYPE to a single dataset key."
        )

    if isinstance(dataset_types, str):
        resolved = [dataset_types]
    else:
        resolved = [str(item) for item in dataset_types]

    if not resolved:
        raise ValueError(
            "Dataset type list is empty. Set DATASET_TYPES to at least one dataset key."
        )
    return resolved


def _resolve_results_group_k(cfg_module) -> int | None:
    explicit = getattr(cfg_module, "RESULTS_GROUP_K", None)
    if explicit is not None:
        return int(explicit)

    if hasattr(cfg_module, "TRACK_TOPK_K"):
        return int(getattr(cfg_module, "TRACK_TOPK_K"))
    if hasattr(cfg_module, "SWE_PANEL_TOPK_OVERLAP_K"):
        return int(getattr(cfg_module, "SWE_PANEL_TOPK_OVERLAP_K"))

    k_values = getattr(cfg_module, "PHYSICS_TOPK_VALUES", None)
    if isinstance(k_values, (list, tuple)) and k_values:
        return int(k_values[0])
    return None


def _clone_cfg_with_dataset_type(cfg_module, dataset_type: str):
    payload = dict(vars(cfg_module))
    payload["DATASET_TYPE"] = dataset_type
    return SimpleNamespace(**payload)


def _resolve_run_output_dir(
    *,
    cfg_module,
    dataset_type: str,
    output_dir: Path | None,
    batch_mode: bool,
) -> Path:
    from physics.swe.comparison_core import DEFAULT_RESULTS_DIR

    root_dir = Path(output_dir) if output_dir is not None else DEFAULT_RESULTS_DIR
    if not batch_mode:
        return root_dir

    group_k = _resolve_results_group_k(cfg_module)
    k_dir = f"k{group_k}" if group_k is not None else "k_unspecified"
    return root_dir / k_dir / dataset_type


def run_physics_comparison_v2(
    output_dir: Path | None = None,
    *,
    cfg_module=cfg,
) -> Dict[str, Any]:
    """Run the comparison pipeline for one or more configured dataset types."""
    from physics.swe.comparison_core import run_physics_comparison

    dataset_types = _resolve_dataset_types(cfg_module)
    explicit_batch = getattr(cfg_module, "DATASET_TYPES", None) is not None
    batch_mode = explicit_batch or len(dataset_types) > 1

    runs: Dict[str, Dict[str, Any]] = {}
    for dataset_type in dataset_types:
        run_cfg = _clone_cfg_with_dataset_type(cfg_module, dataset_type)
        run_output_dir = _resolve_run_output_dir(
            cfg_module=run_cfg,
            dataset_type=dataset_type,
            output_dir=output_dir,
            batch_mode=batch_mode,
        )
        result = run_physics_comparison(cfg_module=run_cfg, output_dir=run_output_dir)

        out = dict(result)
        out["dataset_type"] = dataset_type
        out["output_dir"] = str(run_output_dir)
        out["ig_sanity"] = result.get(
            "ig_sanity",
            {"status": "skipped", "reason": "not_available", "passed": None},
        )
        runs[dataset_type] = out

    if not batch_mode:
        return runs[dataset_types[0]]

    root_dir = _resolve_run_output_dir(
        cfg_module=cfg_module,
        dataset_type=dataset_types[0],
        output_dir=output_dir,
        batch_mode=False,
    )
    return {
        "dataset_types": dataset_types,
        "output_root": str(root_dir),
        "group_k": _resolve_results_group_k(cfg_module),
        "runs": runs,
    }
