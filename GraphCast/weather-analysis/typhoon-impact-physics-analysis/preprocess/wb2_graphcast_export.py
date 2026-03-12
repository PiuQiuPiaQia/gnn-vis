from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from cyclone_points import CYCLONE_TAUKTAE_CENTERS


GRAPHCAST_LOW_RES_STORE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr"
)
SOLAR_STORE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2"
)
GRAPHCAST_13_LEVELS = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)
STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)
SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
)
ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
WB2_LOW_RES_VARS = STATIC_VARS + tuple(v for v in SURFACE_VARS if v != "toa_incident_solar_radiation") + ATMOSPHERIC_VARS
EXPECTED_DATA_VARS = STATIC_VARS + SURFACE_VARS + ATMOSPHERIC_VARS


def ensure_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("dask", "gcsfs", "netCDF4", "zarr"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        deps = " ".join(missing)
        raise ModuleNotFoundError(
            "Missing runtime dependencies for WeatherBench2 export: "
            f"{', '.join(missing)}. Install them with: pip install {deps}"
        )


def parse_track_times(track_points: Sequence[dict]) -> pd.DatetimeIndex:
    times = pd.to_datetime(
        [point["time"] for point in track_points],
        format="%Y-%m-%d %HZ",
    )
    expected = pd.date_range(start=times[0], periods=len(times), freq="6h")
    if not times.equals(expected):
        raise ValueError(
            "Cyclone track times must be contiguous 6-hour steps starting at the "
            f"first point, got {list(times.astype(str))}"
        )
    return times


def build_track_window(track_points: Sequence[dict], steps: int = 4) -> pd.DatetimeIndex:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    track_times = parse_track_times(track_points)
    output_times = pd.date_range(start=track_times[0], periods=steps + 2, freq="6h")
    if len(track_times) > len(output_times):
        raise ValueError(
            "Requested export window is shorter than the provided cyclone track: "
            f"{len(track_times)} track points vs {len(output_times)} output times"
        )

    if not track_times.equals(output_times[: len(track_times)]):
        raise ValueError(
            "Cyclone track does not align with the requested export window. "
            f"track={list(track_times.astype(str))}, window={list(output_times.astype(str))}"
        )
    return output_times


def build_relative_times(steps: int = 4) -> np.ndarray:
    hours = np.arange(0, 6 * (steps + 2), 6, dtype=np.int64)
    return hours.astype("timedelta64[h]").astype("timedelta64[ns]")


def default_output_path(base_dir: str | Path = "/root/autodl-tmp/dataset", steps: int = 4) -> Path:
    start_time = build_track_window(CYCLONE_TAUKTAE_CENTERS, steps=steps)[0]
    file_name = (
        f"dataset-source-era5_date-{start_time.strftime('%Y-%m-%d')}"
        f"_res-1.0_levels-13_steps-{steps:02d}.nc"
    )
    return Path(base_dir) / file_name


def normalize_lat_lon_names(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    return ds


def load_low_res_fields(
    abs_times: pd.DatetimeIndex,
    *,
    low_res_store: str = GRAPHCAST_LOW_RES_STORE,
    solar_store: str = SOLAR_STORE,
) -> tuple[xr.Dataset, xr.DataArray]:
    ensure_runtime_dependencies()

    open_kwargs = {
        "consolidated": True,
        "storage_options": {"token": "anon"},
    }

    low_res_ds = xr.open_zarr(low_res_store, **open_kwargs)
    low_res_ds = normalize_lat_lon_names(low_res_ds)
    low_res_ds = low_res_ds[list(WB2_LOW_RES_VARS)].sel(
        time=abs_times,
        level=list(GRAPHCAST_13_LEVELS),
    )
    low_res_ds = low_res_ds.load()

    solar_ds = xr.open_zarr(solar_store, **open_kwargs)
    solar_ds = normalize_lat_lon_names(solar_ds)
    solar_da = solar_ds["toa_incident_solar_radiation"].sel(
        time=abs_times,
        lat=low_res_ds["lat"].values,
        lon=low_res_ds["lon"].values,
        method="nearest",
    )
    solar_da = solar_da.transpose("time", "lat", "lon").load()
    return low_res_ds, solar_da


def assemble_graphcast_low_res_dataset(
    low_res_ds: xr.Dataset,
    solar_da: xr.DataArray,
    *,
    abs_times: pd.DatetimeIndex,
    steps: int = 4,
) -> xr.Dataset:
    rel_times = build_relative_times(steps=steps)

    data_vars: dict[str, xr.Variable] = {}

    for var_name in STATIC_VARS:
        data = low_res_ds[var_name].transpose("lat", "lon").astype(np.float32).values
        data_vars[var_name] = xr.Variable(("lat", "lon"), data)

    for var_name in SURFACE_VARS:
        source = solar_da if var_name == "toa_incident_solar_radiation" else low_res_ds[var_name]
        data = source.transpose("time", "lat", "lon").astype(np.float32).values[None, ...]
        data_vars[var_name] = xr.Variable(("batch", "time", "lat", "lon"), data)

    for var_name in ATMOSPHERIC_VARS:
        data = low_res_ds[var_name].transpose("time", "level", "lat", "lon").astype(np.float32).values[None, ...]
        data_vars[var_name] = xr.Variable(("batch", "time", "level", "lat", "lon"), data)

    coords = {
        "lat": ("lat", low_res_ds["lat"].values.astype(np.float32)),
        "lon": ("lon", low_res_ds["lon"].values.astype(np.float32)),
        "level": ("level", np.asarray(GRAPHCAST_13_LEVELS, dtype=np.int32)),
        "time": ("time", rel_times),
        "datetime": xr.Variable(("batch", "time"), abs_times.values[None, :]),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def validate_graphcast_low_res_dataset(
    ds: xr.Dataset,
    *,
    abs_times: pd.DatetimeIndex,
    steps: int = 4,
) -> None:
    expected_sizes = {
        "batch": 1,
        "time": steps + 2,
        "lat": 181,
        "lon": 360,
        "level": len(GRAPHCAST_13_LEVELS),
    }
    for dim_name, expected_size in expected_sizes.items():
        actual_size = ds.sizes.get(dim_name)
        if actual_size != expected_size:
            raise ValueError(f"Unexpected {dim_name} size: expected {expected_size}, got {actual_size}")

    missing_vars = sorted(set(EXPECTED_DATA_VARS) - set(ds.data_vars))
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")

    rel_times = build_relative_times(steps=steps)
    if not np.array_equal(ds["time"].values, rel_times):
        raise ValueError("Relative time coordinate does not match GraphCast steps-04 layout")

    if not np.array_equal(ds["datetime"].values[0], abs_times.values):
        raise ValueError("Datetime coordinate does not match requested absolute times")

    if not np.array_equal(ds["level"].values, np.asarray(GRAPHCAST_13_LEVELS, dtype=np.int32)):
        raise ValueError("Pressure levels do not match GraphCast 13-level layout")

    if float(ds["lat"].values[0]) != -90.0 or float(ds["lat"].values[-1]) != 90.0:
        raise ValueError("Latitude coordinate must span [-90, 90] at 1.0 degree spacing")

    if float(ds["lon"].values[0]) != 0.0 or float(ds["lon"].values[-1]) != 359.0:
        raise ValueError("Longitude coordinate must span [0, 359] at 1.0 degree spacing")


def export_tauktae_graphcast_low_res(
    output_path: str | Path | None = None,
    *,
    steps: int = 4,
    force: bool = False,
    low_res_store: str = GRAPHCAST_LOW_RES_STORE,
    solar_store: str = SOLAR_STORE,
) -> Path:
    output_path = Path(output_path) if output_path is not None else default_output_path(steps=steps)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists. Pass force=True or use --force to overwrite it."
        )

    abs_times = build_track_window(CYCLONE_TAUKTAE_CENTERS, steps=steps)
    low_res_ds, solar_da = load_low_res_fields(
        abs_times,
        low_res_store=low_res_store,
        solar_store=solar_store,
    )
    export_ds = assemble_graphcast_low_res_dataset(
        low_res_ds,
        solar_da,
        abs_times=abs_times,
        steps=steps,
    )
    validate_graphcast_low_res_dataset(export_ds, abs_times=abs_times, steps=steps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {var_name: {"zlib": True, "complevel": 1} for var_name in export_ds.data_vars}
    export_ds.to_netcdf(output_path, engine="netcdf4", encoding=encoding)
    return output_path
