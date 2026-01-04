"""
区域数据提取工具

提供从全局气象数据中提取指定区域子集的实用函数。

路径: GraphCast/weather-analysis/region_utils.py
"""

from typing import Tuple
import xarray


def extract_region_data(
    data_array,
    center_lat: float,
    center_lon: float,
    radius: float = 15.0,
    resolution: float = 1.0
) -> Tuple:
    """
    从全局数据中提取指定区域的数据
    
    该函数根据中心坐标和半径裁剪 xarray DataArray，
    常用于台风分析、区域天气预报等场景。
    
    Args:
        data_array: xarray DataArray，包含 lat 和 lon 维度的气象数据
        center_lat: 中心纬度（度）
        center_lon: 中心经度（度）
        radius: 裁剪半径（度），默认 15.0
        resolution: 网格分辨率（度），默认 1.0
    
    Returns:
        tuple: (cropped_data, lat_range, lon_range)
            - cropped_data: 裁剪后的 xarray DataArray
            - lat_range: 纬度范围元组 (lat_min, lat_max)
            - lon_range: 经度范围元组 (lon_min, lon_max)
    
    Example:
        >>> import xarray as xr
        >>> # 假设有全球温度数据
        >>> data = xr.DataArray(...)
        >>> # 提取台风中心附近 ±15° 的区域
        >>> region_data, lat_range, lon_range = extract_region_data(
        ...     data, center_lat=-21.5, center_lon=157.0, radius=15.0
        ... )
        >>> print(f"区域范围: 纬度 {lat_range}, 经度 {lon_range}")
    """
    # 计算区域边界
    lat_min = center_lat - radius
    lat_max = center_lat + radius
    lon_min = center_lon - radius
    lon_max = center_lon + radius
    
    # 裁剪数据
    # 使用 xarray 的 sel 方法进行切片选择
    cropped = data_array.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    return cropped, (lat_min, lat_max), (lon_min, lon_max)


def extract_multiple_regions(
    data_array,
    centers: list,
    radius: float = 15.0,
    resolution: float = 1.0
) -> list:
    """
    批量提取多个区域的数据
    
    Args:
        data_array: xarray DataArray，包含 lat 和 lon 维度的气象数据
        centers: 中心坐标列表，每个元素为 (lat, lon) 元组
        radius: 裁剪半径（度），默认 15.0
        resolution: 网格分辨率（度），默认 1.0
    
    Returns:
        list: 每个元素为 (cropped_data, lat_range, lon_range) 元组
    
    Example:
        >>> centers = [(-21.5, 157.0), (-22.5, 158.0)]
        >>> regions = extract_multiple_regions(data, centers, radius=10.0)
        >>> for i, (region_data, lat_range, lon_range) in enumerate(regions):
        ...     print(f"区域 {i}: 纬度 {lat_range}, 经度 {lon_range}")
    """
    results = []
    for center_lat, center_lon in centers:
        result = extract_region_data(
            data_array, center_lat, center_lon, radius, resolution
        )
        results.append(result)
    return results
