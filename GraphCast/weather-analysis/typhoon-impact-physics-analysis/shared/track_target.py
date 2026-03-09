from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from graphcast import xarray_jax

from cyclone_points import CYCLONE_CENTERS, pick_target_cyclone
from shared.patch_geometry import CenteredWindow


DirectionMode = Literal["along", "cross"]


@dataclass(frozen=True)
class TrackReference:
    init_lat: float
    init_lon: float
    target_lat: float
    target_lon: float
    along_hat: Tuple[float, float]
    cross_hat: Tuple[float, float]


@dataclass(frozen=True)
class TrackScalarDiagnostics:
    scalar: float
    axis_name: str
    predicted_center_lat: float
    predicted_center_lon: float
    predicted_dlon: float
    predicted_dlat: float


def _wrap_lon_delta(dlon: float) -> float:
    return float(((float(dlon) + 180.0) % 360.0) - 180.0)


def _normalize_vec(dx: float, dy: float) -> Tuple[float, float]:
    mag = math.hypot(float(dx), float(dy))
    if mag < 1e-12:
        return 0.0, 0.0
    return float(dx / mag), float(dy / mag)


def resolve_track_reference(target_time_idx: int) -> TrackReference:
    init_center = next(
        (row for row in CYCLONE_CENTERS if int(row.get("input_time_idx", -1)) == 1),
        None,
    )
    if init_center is None:
        raise ValueError("CYCLONE_CENTERS is missing the input_time_idx==1 reference center")

    target_center = pick_target_cyclone(target_time_idx)
    dlon = _wrap_lon_delta(float(target_center["lon"]) - float(init_center["lon"]))
    dlat = float(target_center["lat"]) - float(init_center["lat"])
    along_hat = _normalize_vec(dlon, dlat)
    cross_hat = (-along_hat[1], along_hat[0])
    return TrackReference(
        init_lat=float(init_center["lat"]),
        init_lon=float(init_center["lon"]),
        target_lat=float(target_center["lat"]),
        target_lon=float(target_center["lon"]),
        along_hat=along_hat,
        cross_hat=(float(cross_hat[0]), float(cross_hat[1])),
    )


def axis_from_reference(
    track_ref: TrackReference,
    direction_mode: DirectionMode,
) -> Tuple[float, float]:
    direction = str(direction_mode).lower().strip()
    if direction == "along":
        return track_ref.along_hat
    if direction == "cross":
        return track_ref.cross_hat
    raise ValueError(f"Unsupported direction_mode: {direction_mode}")


def _soft_center_from_field(
    field_2d: jax.Array,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    temperature: float,
) -> tuple[jax.Array, jax.Array]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    lat_grid = jnp.broadcast_to(jnp.asarray(lat_vals, dtype=jnp.float32)[:, None], field_2d.shape)
    lon_grid = jnp.broadcast_to(jnp.asarray(lon_vals, dtype=jnp.float32)[None, :], field_2d.shape)
    logits = -jnp.asarray(field_2d, dtype=jnp.float32) / float(temperature)
    weights = jax.nn.softmax(jnp.ravel(logits))
    pred_lat = jnp.sum(weights * jnp.ravel(lat_grid))
    pred_lon = jnp.sum(weights * jnp.ravel(lon_grid))
    return pred_lat, pred_lon


def _extract_center_field_window(
    outputs,
    *,
    center_field_name: str,
    target_time_idx: int,
    window: CenteredWindow,
):
    if center_field_name not in outputs:
        raise KeyError(
            f"{center_field_name!r} is not present in model outputs; cannot build track scalar"
        )

    field_da = outputs[center_field_name]
    if "batch" in field_da.dims:
        field_da = field_da.isel(batch=0)
    if "time" not in field_da.dims:
        raise ValueError(
            f"{center_field_name!r} must contain a time dimension for track-target analysis"
        )
    field_da = field_da.isel(time=target_time_idx)
    if "level" in field_da.dims:
        raise ValueError(
            f"{center_field_name!r} unexpectedly contains a level dimension; "
            "track center extraction expects a 2D lat/lon field"
        )
    field_da = field_da.isel(lat=window.lat_indices, lon=window.lon_indices).transpose("lat", "lon")
    return field_da


def track_scalar_from_outputs(
    outputs,
    *,
    center_field_name: str,
    target_time_idx: int,
    track_ref: TrackReference,
    window: CenteredWindow,
    direction_mode: DirectionMode,
    softmin_temperature: float,
) -> jax.Array:
    axis_u, axis_v = axis_from_reference(track_ref, direction_mode)
    field_da = _extract_center_field_window(
        outputs,
        center_field_name=center_field_name,
        target_time_idx=target_time_idx,
        window=window,
    )
    field_arr = xarray_jax.unwrap_data(field_da, require_jax=True)
    pred_lat, pred_lon = _soft_center_from_field(
        field_arr,
        lat_vals=window.lat_vals,
        lon_vals=window.lon_vals,
        temperature=softmin_temperature,
    )
    dlon = pred_lon - float(track_ref.init_lon)
    dlat = pred_lat - float(track_ref.init_lat)
    return dlon * float(axis_u) + dlat * float(axis_v)


def compute_track_scalar_diagnostics(
    outputs,
    *,
    center_field_name: str,
    target_time_idx: int,
    track_ref: TrackReference,
    window: CenteredWindow,
    direction_mode: DirectionMode,
    softmin_temperature: float,
) -> TrackScalarDiagnostics:
    field_da = _extract_center_field_window(
        outputs,
        center_field_name=center_field_name,
        target_time_idx=target_time_idx,
        window=window,
    )
    field_arr = np.asarray(field_da.values, dtype=np.float64)
    pred_lat, pred_lon = _soft_center_from_field(
        jnp.asarray(field_arr, dtype=jnp.float32),
        lat_vals=window.lat_vals,
        lon_vals=window.lon_vals,
        temperature=softmin_temperature,
    )
    pred_lat_f = float(pred_lat)
    pred_lon_f = float(pred_lon)
    dlon = _wrap_lon_delta(pred_lon_f - track_ref.init_lon)
    dlat = float(pred_lat_f - track_ref.init_lat)
    axis_u, axis_v = axis_from_reference(track_ref, direction_mode)
    scalar = dlon * axis_u + dlat * axis_v
    return TrackScalarDiagnostics(
        scalar=float(scalar),
        axis_name=str(direction_mode).lower().strip(),
        predicted_center_lat=pred_lat_f,
        predicted_center_lon=pred_lon_f,
        predicted_dlon=dlon,
        predicted_dlat=dlat,
    )
