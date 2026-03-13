from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from cyclone_points import CYCLONE_TAUKTAE_CENTERS
from preprocess.wb2_graphcast_export import (
    build_explicit_window,
    build_relative_times,
    build_track_window,
    default_output_path,
    load_with_progress,
)


def test_build_track_window_extends_tauktae_to_steps_04_layout():
    window = build_track_window(CYCLONE_TAUKTAE_CENTERS, steps=4)

    assert len(window) == 6
    assert str(window[0]) == "2021-05-13 06:00:00"
    assert str(window[-1]) == "2021-05-14 12:00:00"


def test_build_relative_times_matches_graphcast_steps_04_layout():
    rel_times = build_relative_times(steps=4)
    expected = np.array([0, 6, 12, 18, 24, 30], dtype=np.int64)
    expected = expected.astype("timedelta64[h]").astype("timedelta64[ns]")

    assert np.array_equal(rel_times, expected)


def test_default_output_path_matches_requested_graphcast_filename():
    output_path = default_output_path(base_dir="/tmp/dataset", steps=4)

    assert output_path == Path(
        "/tmp/dataset/dataset-source-era5_date-2021-05-13_res-1.0_levels-13_steps-04.nc"
    )


def test_build_explicit_window_accepts_month_first_user_input():
    window = build_explicit_window("05/13/2021 06Z", "05/19/2021 06Z")

    assert len(window) == 25
    assert str(window[0]) == "2021-05-13 06:00:00"
    assert str(window[-1]) == "2021-05-19 06:00:00"


def test_default_output_path_keeps_graphcast_style_for_non_midnight_start():
    window = build_explicit_window("05/13/2021 06Z", "05/19/2021 06Z")

    output_path = default_output_path(base_dir="/tmp/dataset", abs_times=window)

    assert output_path == Path(
        "/tmp/dataset/dataset-source-era5_date-2021-05-13_res-1.0_levels-13_steps-23.nc"
    )


def test_load_with_progress_loads_xarray_objects_without_progress_output():
    ds = xr.Dataset(
        {
            "foo": xr.DataArray(
                da.from_array(np.arange(6, dtype=np.float32), chunks=2),
                dims=("time",),
            )
        }
    )

    loaded = load_with_progress(ds, "test load", show_progress=False)

    assert isinstance(loaded, xr.Dataset)
    assert np.array_equal(loaded["foo"].values, np.arange(6, dtype=np.float32))
