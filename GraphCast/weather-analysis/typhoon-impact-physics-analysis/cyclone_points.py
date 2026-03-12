# -*- coding: utf-8 -*-
"""台风中心点数据（从 point 文件夹复制而来）。"""

CYCLONE_CENTERS = [
    {"time": "2022-01-01 00Z", "lat": -21.3138, "lon": 156.6947, "pressure": 997.0, "wind_speed": 40, "category": "TS", "data_type": "input(-6h)", "is_input": True, "input_time_idx": 0},
    {"time": "2022-01-01 06Z", "lat": -21.7054, "lon": 157.5024, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "input(0h)", "is_input": True, "input_time_idx": 1},
    {"time": "2022-01-01 12Z", "lat": -22.5048, "lon": 158.2994, "pressure": 1000.0, "wind_speed": 35, "category": "TS", "data_type": "target(+6h)", "is_input": False, "target_time_idx": 0},
    {"time": "2022-01-01 18Z", "lat": -23.9030, "lon": 158.8031, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "target(+12h)", "is_input": False, "target_time_idx": 1},
    {"time": "2022-01-02 00Z", "lat": -25.8032, "lon": 159.0031, "pressure": 992.0, "wind_speed": 40, "category": "TS", "data_type": "target(+18h)", "is_input": False, "target_time_idx": 2},
]


CYCLONE_TAUKTAE_CENTERS = [
    {"time": "2021-05-16 00Z", "lat": 14.9979, "lon": 72.7998, "pressure": -1.0, "wind_speed": -1, "category": "Tauktae", "data_type": "tauktae", "is_input": True, "input_time_idx": 0},
    {"time": "2021-05-16 06Z", "lat": 15.7978, "lon": 72.7998, "pressure": -1.0, "wind_speed": -1, "category": "Tauktae", "data_type": "tauktae", "is_input": True, "input_time_idx": 1},
    {"time": "2021-05-16 12Z", "lat": 16.7974, "lon": 72.5001, "pressure": -1.0, "wind_speed": -1, "category": "Tauktae", "data_type": "tauktae", "is_input": False, "target_time_idx": 0},
    {"time": "2021-05-16 18Z", "lat": 17.6973, "lon": 71.9998, "pressure": -1.0, "wind_speed": -1, "category": "Tauktae", "data_type": "tauktae", "is_input": False, "target_time_idx": 1},
    {"time": "2021-05-17 00Z", "lat": 18.3979, "lon": 71.6989, "pressure": -1.0, "wind_speed": -1, "category": "Tauktae", "data_type": "tauktae", "is_input": False, "target_time_idx": 2},
]


CYCLONE_YAAS_CENTERS = [
    {"time": "2021-05-25 00Z", "lat": 18.0966, "lon": 88.5997, "pressure": -1.0, "wind_speed": -1, "category": "Yaas", "data_type": "yaas", "is_input": True, "input_time_idx": 0},
    {"time": "2021-05-25 06Z", "lat": 18.7964, "lon": 88.2007, "pressure": -1.0, "wind_speed": -1, "category": "Yaas", "data_type": "yaas", "is_input": True, "input_time_idx": 1},
    {"time": "2021-05-25 12Z", "lat": 19.4990, "lon": 88.2007, "pressure": -1.0, "wind_speed": -1, "category": "Yaas", "data_type": "yaas", "is_input": False, "target_time_idx": 0},
    {"time": "2021-05-25 18Z", "lat": 20.2970, "lon": 87.8081, "pressure": -1.0, "wind_speed": -1, "category": "Yaas", "data_type": "yaas", "is_input": False, "target_time_idx": 1},
    {"time": "2021-05-26 00Z", "lat": 20.5979, "lon": 87.5990, "pressure": -1.0, "wind_speed": -1, "category": "Yaas", "data_type": "yaas", "is_input": False, "target_time_idx": 2},
]


def pick_target_cyclone(target_time_idx: int) -> dict:
    for c in CYCLONE_CENTERS:
        if not c.get("is_input", True) and c.get("target_time_idx") == target_time_idx:
            return c
    raise ValueError(f"no cyclone for target_time_idx={target_time_idx}")
