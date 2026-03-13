import numpy as np
import pandas as pd
import xarray as xr

from shared.dataset_window import slice_graphcast_dataset_window


def _make_graphcast_style_dataset(time_steps: int = 10) -> xr.Dataset:
    rel_time = np.arange(0, time_steps * 6, 6, dtype=np.int64)
    rel_time = rel_time.astype("timedelta64[h]").astype("timedelta64[ns]")
    abs_time = pd.date_range("2021-05-13 06:00:00", periods=time_steps, freq="6h")
    values = np.arange(time_steps, dtype=np.float32).reshape(1, time_steps, 1, 1)
    return xr.Dataset(
        data_vars={
            "mean_sea_level_pressure": xr.Variable(
                ("batch", "time", "lat", "lon"),
                values,
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


def test_slice_graphcast_dataset_window_rebases_relative_time():
    dataset = _make_graphcast_style_dataset(time_steps=10)

    sliced = slice_graphcast_dataset_window(
        dataset,
        "2021-05-13 18:00:00",
        window_size=6,
    )

    expected_time = np.array([0, 6, 12, 18, 24, 30], dtype=np.int64)
    expected_time = expected_time.astype("timedelta64[h]").astype("timedelta64[ns]")
    expected_datetime = pd.date_range("2021-05-13 18:00:00", periods=6, freq="6h")

    assert np.array_equal(sliced["time"].values, expected_time)
    assert np.array_equal(sliced["datetime"].values[0], expected_datetime.values)
    assert np.array_equal(
        sliced["mean_sea_level_pressure"].values[:, :, 0, 0],
        np.arange(2, 8, dtype=np.float32)[None, :],
    )


def test_slice_graphcast_dataset_window_raises_for_missing_start_time():
    dataset = _make_graphcast_style_dataset(time_steps=10)

    try:
        slice_graphcast_dataset_window(dataset, "2021-05-20 00:00:00", window_size=6)
    except ValueError as exc:
        assert "DATASET_START_TIME" in str(exc)
    else:
        raise AssertionError("Expected missing DATASET_START_TIME to raise ValueError")
