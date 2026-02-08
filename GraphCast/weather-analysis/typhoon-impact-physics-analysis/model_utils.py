# -*- coding: utf-8 -*-
"""GraphCast model and data utilities for typhoon impact analysis."""

import sys
from pathlib import Path
import dataclasses
import functools

import xarray
import jax
import haiku as hk

# Add project paths before importing graphcast
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
PREPROCESS_DIR = PROJECT_DIR / "graphcast-preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization


def load_checkpoint(params_path: str):
    with open(params_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return ckpt


def load_dataset(dataset_path: str):
    with open(dataset_path, "rb") as f:
        ds = xarray.load_dataset(f).compute()
    return ds


def extract_eval_data(example_batch: xarray.Dataset, task_config, eval_steps: int = 4):
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config),
    )
    return eval_inputs, eval_targets, eval_forcings


def load_normalization_stats(stats_dir: str):
    with open(f"{stats_dir}/stats-diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(f"{stats_dir}/stats-mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(f"{stats_dir}/stats-stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level


def build_run_forward(model_config, task_config, params, state, diffs_stddev_by_level, mean_by_level, stddev_by_level):
    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))
    return run_forward_jitted
