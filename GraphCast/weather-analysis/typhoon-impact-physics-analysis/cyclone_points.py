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
    {"time": "2021-05-13 06Z", "lat": 10.0, "lon": 70.9, "pressure": 1007.0, "wind_speed": 20, "category": "TD", "data_type": "tauktae", "is_input": True, "input_time_idx": 0},
    {"time": "2021-05-13 12Z", "lat": 10.2, "lon": 71.3, "pressure": 1004.0, "wind_speed": 25, "category": "TD", "data_type": "tauktae", "is_input": True, "input_time_idx": 1},
    {"time": "2021-05-13 18Z", "lat": 10.5, "lon": 71.5, "pressure": 999.0, "wind_speed": 25, "category": "TD", "data_type": "tauktae", "is_input": False, "target_time_idx": 0},
    {"time": "2021-05-14 00Z", "lat": 10.9, "lon": 71.8, "pressure": 997.0, "wind_speed": 30, "category": "TD", "data_type": "tauktae", "is_input": False, "target_time_idx": 1},
    {"time": "2021-05-14 06Z", "lat": 11.2, "lon": 72.2, "pressure": 998.0, "wind_speed": 35, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 2},
    {"time": "2021-05-14 12Z", "lat": 11.7, "lon": 72.6, "pressure": 996.0, "wind_speed": 40, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 3},
    {"time": "2021-05-14 18Z", "lat": 12.3, "lon": 72.8, "pressure": 989.0, "wind_speed": 45, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 4},
    {"time": "2021-05-15 00Z", "lat": 12.8, "lon": 72.9, "pressure": 989.0, "wind_speed": 50, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 5},
    {"time": "2021-05-15 06Z", "lat": 13.3, "lon": 73.0, "pressure": 1002.0, "wind_speed": 55, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 6},
    {"time": "2021-05-15 12Z", "lat": 13.9, "lon": 73.0, "pressure": 975.0, "wind_speed": 65, "category": "H1", "data_type": "tauktae", "is_input": False, "target_time_idx": 7},
    {"time": "2021-05-15 18Z", "lat": 14.5, "lon": 72.8, "pressure": 975.0, "wind_speed": 70, "category": "H1", "data_type": "tauktae", "is_input": False, "target_time_idx": 8},
    {"time": "2021-05-16 00Z", "lat": 15.0, "lon": 72.8, "pressure": 965.0, "wind_speed": 80, "category": "H1", "data_type": "tauktae", "is_input": False, "target_time_idx": 9},
    {"time": "2021-05-16 06Z", "lat": 15.8, "lon": 72.8, "pressure": 962.0, "wind_speed": 90, "category": "H2", "data_type": "tauktae", "is_input": False, "target_time_idx": 10},
    {"time": "2021-05-16 12Z", "lat": 16.8, "lon": 72.5, "pressure": 952.0, "wind_speed": 100, "category": "H3", "data_type": "tauktae", "is_input": False, "target_time_idx": 11},
    {"time": "2021-05-16 18Z", "lat": 17.7, "lon": 72.0, "pressure": 942.0, "wind_speed": 115, "category": "H4", "data_type": "tauktae", "is_input": False, "target_time_idx": 12},
    {"time": "2021-05-17 00Z", "lat": 18.4, "lon": 71.7, "pressure": 931.0, "wind_speed": 120, "category": "H4", "data_type": "tauktae", "is_input": False, "target_time_idx": 13},
    {"time": "2021-05-17 06Z", "lat": 19.3, "lon": 71.5, "pressure": 934.0, "wind_speed": 120, "category": "H4", "data_type": "tauktae", "is_input": False, "target_time_idx": 14},
    {"time": "2021-05-17 12Z", "lat": 20.1, "lon": 71.4, "pressure": 949.0, "wind_speed": 100, "category": "H3", "data_type": "tauktae", "is_input": False, "target_time_idx": 15},
    {"time": "2021-05-17 18Z", "lat": 20.9, "lon": 71.2, "pressure": 942.0, "wind_speed": 110, "category": "H3", "data_type": "tauktae", "is_input": False, "target_time_idx": 16},
    {"time": "2021-05-18 00Z", "lat": 21.5, "lon": 71.2, "pressure": 959.0, "wind_speed": 90, "category": "H2", "data_type": "tauktae", "is_input": False, "target_time_idx": 17},
    {"time": "2021-05-18 06Z", "lat": 22.1, "lon": 71.6, "pressure": 972.0, "wind_speed": 70, "category": "H1", "data_type": "tauktae", "is_input": False, "target_time_idx": 18},
    {"time": "2021-05-18 12Z", "lat": 23.2, "lon": 72.1, "pressure": 984.0, "wind_speed": 55, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 19},
    {"time": "2021-05-18 18Z", "lat": 24.1, "lon": 73.1, "pressure": 989.0, "wind_speed": 45, "category": "TS", "data_type": "tauktae", "is_input": False, "target_time_idx": 20},
    {"time": "2021-05-19 00Z", "lat": 24.6, "lon": 73.6, "pressure": 998.0, "wind_speed": 30, "category": "TD", "data_type": "tauktae", "is_input": False, "target_time_idx": 21},
    {"time": "2021-05-19 06Z", "lat": 25.8, "lon": 74.8, "pressure": 997.0, "wind_speed": 20, "category": "TD", "data_type": "tauktae", "is_input": False, "target_time_idx": 22},
]


CYCLONE_YAAS_CENTERS = [
    {"time": "2021-05-23 00Z", "lat": 15.5, "lon": 90.0, "pressure": 1000.0, "wind_speed": 20, "category": "TD", "data_type": "yaas", "is_input": True, "input_time_idx": 0},
    {"time": "2021-05-23 06Z", "lat": 15.6, "lon": 89.8, "pressure": 998.0, "wind_speed": 25, "category": "TD", "data_type": "yaas", "is_input": True, "input_time_idx": 1},
    {"time": "2021-05-23 12Z", "lat": 15.7, "lon": 89.6, "pressure": 995.0, "wind_speed": 30, "category": "TD", "data_type": "yaas", "is_input": False, "target_time_idx": 0},
    {"time": "2021-05-23 18Z", "lat": 15.8, "lon": 89.5, "pressure": 994.0, "wind_speed": 35, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 1},
    {"time": "2021-05-24 00Z", "lat": 16.1, "lon": 89.8, "pressure": 991.0, "wind_speed": 35, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 2},
    {"time": "2021-05-24 06Z", "lat": 16.6, "lon": 89.6, "pressure": 991.0, "wind_speed": 35, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 3},
    {"time": "2021-05-24 12Z", "lat": 17.2, "lon": 89.4, "pressure": 988.0, "wind_speed": 40, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 4},
    {"time": "2021-05-24 18Z", "lat": 17.6, "lon": 89.0, "pressure": 983.0, "wind_speed": 50, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 5},
    {"time": "2021-05-25 00Z", "lat": 18.1, "lon": 88.6, "pressure": 981.0, "wind_speed": 55, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 6},
    {"time": "2021-05-25 06Z", "lat": 18.8, "lon": 88.2, "pressure": 981.0, "wind_speed": 55, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 7},
    {"time": "2021-05-25 12Z", "lat": 19.5, "lon": 88.2, "pressure": 975.0, "wind_speed": 65, "category": "H1", "data_type": "yaas", "is_input": False, "target_time_idx": 8},
    {"time": "2021-05-25 18Z", "lat": 20.3, "lon": 87.8, "pressure": 974.0, "wind_speed": 65, "category": "H1", "data_type": "yaas", "is_input": False, "target_time_idx": 9},
    {"time": "2021-05-26 00Z", "lat": 20.6, "lon": 87.6, "pressure": 977.0, "wind_speed": 60, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 10},
    {"time": "2021-05-26 06Z", "lat": 21.5, "lon": 86.9, "pressure": 979.0, "wind_speed": 55, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 11},
    {"time": "2021-05-26 12Z", "lat": 21.9, "lon": 86.5, "pressure": 985.0, "wind_speed": 45, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 12},
    {"time": "2021-05-26 18Z", "lat": 22.5, "lon": 85.8, "pressure": 990.0, "wind_speed": 35, "category": "TS", "data_type": "yaas", "is_input": False, "target_time_idx": 13},
    {"time": "2021-05-27 00Z", "lat": 23.1, "lon": 85.7, "pressure": 993.0, "wind_speed": 25, "category": "TD", "data_type": "yaas", "is_input": False, "target_time_idx": 14},
    {"time": "2021-05-27 06Z", "lat": 23.6, "lon": 85.6, "pressure": 997.0, "wind_speed": 20, "category": "TD", "data_type": "yaas", "is_input": False, "target_time_idx": 15},
    {"time": "2021-05-27 12Z", "lat": 24.3, "lon": 85.3, "pressure": 992.0, "wind_speed": 25, "category": "TD", "data_type": "yaas", "is_input": False, "target_time_idx": 16},
    {"time": "2021-05-27 18Z", "lat": 24.7, "lon": 84.8, "pressure": 992.0, "wind_speed": 25, "category": "TD", "data_type": "yaas", "is_input": False, "target_time_idx": 17},
]


def pick_target_cyclone(target_time_idx: int) -> dict:
    for c in CYCLONE_CENTERS:
        if not c.get("is_input", True) and c.get("target_time_idx") == target_time_idx:
            return c
    raise ValueError(f"no cyclone for target_time_idx={target_time_idx}")
