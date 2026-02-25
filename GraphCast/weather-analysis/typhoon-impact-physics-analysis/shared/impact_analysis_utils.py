# -*- coding: utf-8 -*-
import numpy as np
import xarray


def build_indexer(da: xarray.DataArray, lat_slice, lon_slice, time_sel, level_sel):
    indexer = []
    for dim in da.dims:
        if dim == "time":
            indexer.append(time_sel)
        elif dim == "level":
            indexer.append(level_sel)
        elif dim == "lat":
            indexer.append(lat_slice)
        elif dim == "lon":
            indexer.append(lon_slice)
        else:
            indexer.append(slice(None))
    return tuple(indexer)


def resolve_level_sel(da: xarray.DataArray, perturb_levels):
    if "level" not in da.dims:
        return slice(None)
    if perturb_levels is None:
        return slice(None)
    levels = da.coords["level"].values
    level_idx = [int(np.where(levels == lvl)[0][0]) for lvl in perturb_levels if lvl in levels]
    if not level_idx:
        return slice(None)
    return level_idx
