from pathlib import Path

import numpy as np

from cyclone_points import CYCLONE_TAUKTAE_CENTERS
from preprocess.wb2_graphcast_export import (
    build_relative_times,
    build_track_window,
    default_output_path,
)


def test_build_track_window_extends_tauktae_to_steps_04_layout():
    window = build_track_window(CYCLONE_TAUKTAE_CENTERS, steps=4)

    assert len(window) == 6
    assert str(window[0]) == "2021-05-16 00:00:00"
    assert str(window[-1]) == "2021-05-17 06:00:00"


def test_build_relative_times_matches_graphcast_steps_04_layout():
    rel_times = build_relative_times(steps=4)
    expected = np.array([0, 6, 12, 18, 24, 30], dtype=np.int64)
    expected = expected.astype("timedelta64[h]").astype("timedelta64[ns]")

    assert np.array_equal(rel_times, expected)


def test_default_output_path_matches_requested_graphcast_filename():
    output_path = default_output_path(base_dir="/tmp/dataset", steps=4)

    assert output_path == Path(
        "/tmp/dataset/dataset-source-era5_date-2021-05-16_res-1.0_levels-13_steps-04.nc"
    )
