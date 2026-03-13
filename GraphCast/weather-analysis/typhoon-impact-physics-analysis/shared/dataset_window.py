from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def normalize_start_time(value: str | pd.Timestamp | np.datetime64) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _extract_datetime_index(dataset: xr.Dataset) -> pd.DatetimeIndex:
    if "datetime" not in dataset.coords and "datetime" not in dataset.variables:
        raise ValueError("Dataset window slicing requires a 'datetime' coordinate or variable.")

    datetime_da = dataset["datetime"]
    if "time" not in datetime_da.dims:
        raise ValueError("Dataset 'datetime' must include a 'time' dimension.")

    if "batch" in datetime_da.dims:
        batch_size = int(dataset.sizes.get("batch", datetime_da.sizes["batch"]))
        if batch_size != 1:
            raise ValueError(
                f"Dataset window slicing only supports batch=1, got batch={batch_size}."
            )
        datetime_da = datetime_da.isel(batch=0)

    return pd.DatetimeIndex(np.asarray(datetime_da.values))


def _format_datetime(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def slice_graphcast_dataset_window(
    dataset: xr.Dataset,
    start_time: str | pd.Timestamp | np.datetime64 | None,
    *,
    window_size: int,
) -> xr.Dataset:
    if start_time is None:
        return dataset
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if "time" not in dataset.dims:
        raise ValueError("Dataset window slicing requires a 'time' dimension.")

    datetime_index = _extract_datetime_index(dataset)
    start_timestamp = normalize_start_time(start_time)
    matches = np.flatnonzero(datetime_index.values == start_timestamp.to_datetime64())
    if len(matches) == 0:
        available_start = _format_datetime(datetime_index[0])
        available_end = _format_datetime(datetime_index[-1])
        raise ValueError(
            "Requested DATASET_START_TIME was not found in dataset datetime coordinate: "
            f"{start_timestamp.strftime('%Y-%m-%d %H:%M')}. "
            f"Available range: [{available_start}, {available_end}]"
        )

    start_idx = int(matches[0])
    stop_idx = start_idx + window_size
    total_steps = int(dataset.sizes["time"])
    if stop_idx > total_steps:
        raise ValueError(
            "Requested dataset window exceeds available time range: "
            f"start_idx={start_idx}, window_size={window_size}, total_time={total_steps}"
        )

    sliced = dataset.isel(time=slice(start_idx, stop_idx))
    rebased_time = np.asarray(sliced["time"].values) - np.asarray(sliced["time"].values)[0]
    return sliced.assign_coords(time=("time", rebased_time))
