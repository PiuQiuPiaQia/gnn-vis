from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from shared.analysis_pipeline import (
    AnalysisConfig,
    load_configured_dataset,
    resolve_dataset_file_value,
)


def test_resolve_dataset_file_value_passes_dataset_start_time_to_callable():
    calls = []

    def dataset_file(start_time):
        calls.append(start_time)
        return "long-window.nc"

    resolved = resolve_dataset_file_value(dataset_file, "2021-05-16 00Z")

    assert resolved == "long-window.nc"
    assert calls == ["2021-05-16 00Z"]


def test_analysis_config_reads_dataset_window_fields():
    cfg_module = SimpleNamespace(
        DATASET_CONFIGS={"low_res": {"dataset_file": "sample.nc"}},
        DATASET_TYPE="low_res",
        DATASET_START_TIME="2021-05-16 00Z",
        EVAL_STEPS=6,
        TARGET_TIME_IDX=0,
        TARGET_VARIABLE="msl",
        TARGET_LEVEL=None,
        PATCH_RADIUS=2,
        PATCH_SCORE_AGG="mean",
        PERTURB_TIME="all",
        PERTURB_VARIABLES=None,
        PERTURB_LEVELS=None,
        BASELINE_MODE="local_annulus_median",
        LOCAL_BASELINE_INNER_DEG=5.0,
        LOCAL_BASELINE_OUTER_DEG=12.0,
        LOCAL_BASELINE_MIN_POINTS=120,
        HEATMAP_DPI=200,
        IG_STEPS=50,
        INCLUDE_TARGET_INPUTS=False,
        GRADIENT_VMAX_QUANTILE=0.9,
        GRADIENT_CMAP="RdBu_r",
        GRADIENT_CENTER_WINDOW_DEG=10.0,
        GRADIENT_CENTER_SCALE_QUANTILE=0.99,
        GRADIENT_ALPHA_QUANTILE=0.9,
        GRADIENT_TIME_AGG="single",
        DIR_PATH_PARAMS="/params",
        DIR_PATH_DATASET="/dataset",
        DIR_PATH_STATS="/stats",
    )

    config = AnalysisConfig.from_module(cfg_module)

    assert config.dataset_start_time == "2021-05-16 00Z"
    assert config.eval_steps == 6


def test_load_configured_dataset_uses_dataset_config_start_time_when_global_is_none():
    rel_time = np.arange(0, 10 * 6, 6, dtype=np.int64)
    rel_time = rel_time.astype("timedelta64[h]").astype("timedelta64[ns]")
    abs_time = pd.date_range("2021-05-13 06:00:00", periods=10, freq="6h")
    dataset = xr.Dataset(
        data_vars={
            "mean_sea_level_pressure": xr.Variable(
                ("batch", "time", "lat", "lon"),
                np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
            )
        },
        coords={
            "batch": ("batch", [0]),
            "time": ("time", rel_time),
            "datetime": xr.Variable(("batch", "time"), abs_time.values[None, :]),
            "lat": ("lat", [0.0]),
            "lon": ("lon", [0.0]),
        },
    )

    runtime_cfg = SimpleNamespace(
        dataset_start_time=None,
        eval_steps=4,
        dir_path_dataset="/dataset",
    )
    dataset_config = {
        "dataset_file": lambda start_time: dataset,
        "dataset_start_time": "2021-05-14 06:00:00",
    }

    loaded = load_configured_dataset(runtime_cfg, dataset_config)

    expected_time = np.array([0, 6, 12, 18, 24, 30], dtype=np.int64)
    expected_time = expected_time.astype("timedelta64[h]").astype("timedelta64[ns]")
    expected_values = np.arange(4, 10, dtype=np.float32)[None, :]

    assert np.array_equal(loaded["time"].values, expected_time)
    assert np.array_equal(
        loaded["mean_sea_level_pressure"].values[:, :, 0, 0],
        expected_values,
    )
