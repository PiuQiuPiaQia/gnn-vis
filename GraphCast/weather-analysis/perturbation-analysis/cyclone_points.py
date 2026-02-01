# -*- coding: utf-8 -*-
"""Cyclone point data (copied from point folder)."""

CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.3138, "lon": 156.6947, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "input(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7054, "lon": 157.5024, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "input(0h)", "is_input": True, "input_time_idx": 1},
    {"time": "2022-01-01 12Z", "lat": -22.5048, "lon": 158.2994, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "target(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9030, "lon": 158.8031, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "target(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8032, "lon": 159.0031, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "target(+18h)", "is_input": False, "target_time_idx": 2},
]


def pick_target_cyclone(target_time_idx: int) -> dict:
    for c in CYCLONE_CENTERS:
        if not c.get("is_input", True) and c.get("target_time_idx") == target_time_idx:
            return c
    raise ValueError(f"no cyclone for target_time_idx={target_time_idx}")
