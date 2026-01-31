"""
区域数据提取工具

提供从全局气象数据中提取指定区域子集的实用函数。

路径: GraphCast/weather-analysis/region_utils.py
"""

from typing import Tuple
import numpy as np
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


def compute_great_circle_distance(lat1, lon1, lat2, lon2):
    """
    计算球面距离（使用 Haversine 公式）

    该函数计算地球表面两点之间的大圆距离，考虑了地球的球形特性。
    适用于台风引导气流计算中的环形区域筛选。

    Args:
        lat1: 起点纬度（度）或纬度数组
        lon1: 起点经度（度）或经度数组
        lat2: 终点纬度（度）或纬度标量
        lon2: 终点经度（度）或经度标量

    Returns:
        float or ndarray: 球面距离（度）

    Notes:
        - 使用 Haversine 公式，精度高于简单欧几里得距离
        - 返回单位为度（°），方便与经纬度网格对比
        - 地球半径 R ≈ 6371 km，1° ≈ 111 km

    Example:
        >>> # 计算两点间距离
        >>> dist = compute_great_circle_distance(-21.5, 157.0, -22.5, 158.0)
        >>> print(f"距离: {dist:.2f}°")

        >>> # 计算网格点到台风中心的距离
        >>> lats, lons = np.meshgrid(lat_array, lon_array, indexing='ij')
        >>> distances = compute_great_circle_distance(lats, lons, -21.5, 157.0)
    """
    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)
    dlat_rad = np.radians(lat2 - lat1)

    # Haversine 公式
    a = np.sin(dlat_rad/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 转换为度（地球半径 R = 6371 km，1度 ≈ 111 km）
    distance_deg = np.degrees(c)
    return distance_deg


def extract_annulus_mean(
    data_array,
    center_lat: float,
    center_lon: float,
    inner_radius: float = 3.0,
    outer_radius: float = 7.0,
    use_great_circle: bool = True
) -> float:
    """
    计算环形区域（annulus）内的平均值

    该函数用于台风引导气流计算，遵循 JTWC/CMA 气象学标准：
    在台风中心外围环形区域内计算平均风场，排除台风自身环流影响。

    Args:
        data_array: xarray DataArray，2D 数据（必须包含 lat, lon 维度）
        center_lat: 中心纬度（度）
        center_lon: 中心经度（度）
        inner_radius: 内半径（度），默认 3.0°（排除台风环流核心）
        outer_radius: 外半径（度），默认 7.0°（环境风场边界）
        use_great_circle: 是否使用球面距离，默认 True（推荐）

    Returns:
        float: 环形区域内的平均值

    Notes:
        - 气象学标准（Holland 1984）：引导气流 = 环形区域平均风场
        - 内半径 3°：排除台风中心的环流影响
        - 外半径 7°：捕获环境引导气流
        - 对于风场 u/v 分量，分别调用此函数计算平均

    Example:
        >>> # 计算 850hPa u 分量的环形平均
        >>> u_850 = data['u_component_of_wind'].sel(level=850)
        >>> u_mean = extract_annulus_mean(u_850, -21.5, 157.0, 3.0, 7.0)
        >>> print(f"环形平均 u 分量: {u_mean:.2f} m/s")

    References:
        - Holland, G. J. (1984). Tropical cyclone motion.
        - JTWC Tropical Cyclone Best Track Database
    """
    # 步骤1：先用矩形裁剪，优化性能
    # 提取稍大于 outer_radius 的矩形区域
    outer_rect, _, _ = extract_region_data(
        data_array, center_lat, center_lon,
        radius=outer_radius + 2.0,  # 稍大一些确保覆盖完整
        resolution=1.0
    )

    # 步骤2：获取裁剪后的经纬度坐标
    lat_coords = outer_rect.lat.values
    lon_coords = outer_rect.lon.values

    # 步骤3：创建 2D 网格
    lat_2d, lon_2d = np.meshgrid(lat_coords, lon_coords, indexing='ij')

    # 步骤4：计算每个网格点到中心的距离
    if use_great_circle:
        # 使用球面距离（精确）
        distances = compute_great_circle_distance(lat_2d, lon_2d, center_lat, center_lon)
    else:
        # 使用欧几里得距离（近似，仅在小区域有效）
        distances = np.sqrt((lat_2d - center_lat)**2 + (lon_2d - center_lon)**2)

    # 步骤5：创建环形区域掩码
    # mask = True 表示在环形区域内，False 表示在区域外
    mask = (distances >= inner_radius) & (distances <= outer_radius)

    # 步骤6：应用掩码并计算平均值
    # xarray.where(condition, keep_value, drop_value)
    # 这里我们保留 mask=True 的值，其他设为 NaN
    data_masked = outer_rect.where(xarray.DataArray(mask, coords=outer_rect.coords, dims=outer_rect.dims))

    # 计算平均值（自动忽略 NaN）
    mean_value = float(data_masked.mean().values)

    # 如果没有有效值（例如环形区域完全超出数据范围），返回 NaN
    if np.isnan(mean_value):
        print(f"  警告: 环形区域 ({inner_radius}°-{outer_radius}°) 内无有效数据")
        return 0.0

    return mean_value
