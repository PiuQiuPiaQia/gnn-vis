#!/usr/bin/env python3
"""
经纬度与网格索引转换通用工具模块

提供经纬度坐标与网格索引之间的双向转换功能，
适用于各种分辨率的全球气象数据网格（如 ERA5、GraphCast 等）。

作者: gnn-vis team
日期: 2026-01-04
"""

import numpy as np
from typing import Tuple, Union, Optional


class LatLonGridConverter:
    """
    经纬度网格坐标转换器
    
    提供经纬度坐标与网格索引之间的双向转换功能。
    
    参数:
        resolution: 网格分辨率（度），默认为 1.0
        lat_min: 纬度最小值（度），默认为 -90.0（南极）
        lat_max: 纬度最大值（度），默认为 90.0（北极）
        lon_min: 经度最小值（度），默认为 0.0
        lon_max: 经度最大值（度），默认为 360.0
        
    示例:
        >>> # 使用默认的 1 度分辨率
        >>> converter = LatLonGridConverter(resolution=1.0)
        >>> lat_idx, lon_idx = converter.latlon_to_index(-21.7, 157.5)
        >>> print(f"索引: ({lat_idx}, {lon_idx})")
        索引: (68, 158)
        
        >>> # 使用 0.25 度分辨率
        >>> converter = LatLonGridConverter(resolution=0.25)
        >>> lat_idx, lon_idx = converter.latlon_to_index(40.5, 120.25)
        >>> print(f"索引: ({lat_idx}, {lon_idx})")
        索引: (522, 481)
    """
    
    def __init__(self, 
                 resolution: float = 1.0,
                 lat_min: float = -90.0,
                 lat_max: float = 90.0,
                 lon_min: float = 0.0,
                 lon_max: float = 360.0):
        """
        初始化网格转换器
        
        参数:
            resolution: 网格分辨率（度）
            lat_min: 纬度最小值（度）
            lat_max: 纬度最大值（度）
            lon_min: 经度最小值（度）
            lon_max: 经度最大值（度）
        """
        self.resolution = resolution
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        
        # 计算网格维度
        self.lat_size = int((lat_max - lat_min) / resolution) + 1
        self.lon_size = int((lon_max - lon_min) / resolution)
        
    def latlon_to_index(self, 
                       lat: Union[float, np.ndarray], 
                       lon: Union[float, np.ndarray]) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        将经纬度坐标转换为网格索引
        
        参数:
            lat: 纬度（度），南纬为负，北纬为正，范围 [-90, 90]
            lon: 经度（度），范围 [0, 360) 或 [-180, 180]
                 如果输入为 [-180, 180] 范围，会自动转换为 [0, 360)
        
        返回:
            (lat_idx, lon_idx): 网格索引元组
                - lat_idx: 纬度索引，从南向北递增 [0, lat_size-1]
                - lon_idx: 经度索引，从西向东递增 [0, lon_size-1]
        
        示例:
            >>> converter = LatLonGridConverter(resolution=1.0)
            >>> lat_idx, lon_idx = converter.latlon_to_index(-21.7, 157.5)
            >>> print(f"索引: ({lat_idx}, {lon_idx})")
            索引: (68, 158)
            
            >>> # 批量转换
            >>> lats = np.array([-21.7, 40.5, 0.0])
            >>> lons = np.array([157.5, 120.5, 180.0])
            >>> lat_idxs, lon_idxs = converter.latlon_to_index(lats, lons)
        """
        # 处理经度：确保在 [lon_min, lon_max) 范围内
        lon = np.asarray(lon)
        if np.any((lon >= -180) & (lon < 0)):
            # 如果经度在 [-180, 0) 范围，转换为 [180, 360)
            lon = np.where(lon < 0, lon + 360, lon)
        
        # 计算索引
        # 纬度索引：从南向北递增
        lat_idx = np.round((lat - self.lat_min) / self.resolution).astype(int)
        
        # 经度索引：从西向东递增
        lon_idx = np.round((lon - self.lon_min) / self.resolution).astype(int)
        
        # 边界检查
        if np.any(lat_idx < 0) or np.any(lat_idx >= self.lat_size):
            print(f"警告: 纬度索引超出范围 [0, {self.lat_size-1}]")
        if np.any(lon_idx < 0) or np.any(lon_idx >= self.lon_size):
            print(f"警告: 经度索引超出范围 [0, {self.lon_size-1}]")
        
        # 如果输入是标量，返回标量；如果是数组，返回数组
        if np.ndim(lat_idx) == 0:
            return int(lat_idx), int(lon_idx)
        else:
            return lat_idx, lon_idx
    
    def index_to_latlon(self,
                       lat_idx: Union[int, np.ndarray],
                       lon_idx: Union[int, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        将网格索引转换回经纬度坐标
        
        参数:
            lat_idx: 纬度索引
            lon_idx: 经度索引
        
        返回:
            (lat, lon): 经纬度坐标元组
                - lat: 纬度（度）
                - lon: 经度（度）
        
        示例:
            >>> converter = LatLonGridConverter(resolution=1.0)
            >>> lat, lon = converter.index_to_latlon(68, 158)
            >>> print(f"经纬度: ({lat}, {lon})")
            经纬度: (-22.0, 158.0)
        """
        lat = self.lat_min + np.asarray(lat_idx) * self.resolution
        lon = self.lon_min + np.asarray(lon_idx) * self.resolution
        
        # 如果输入是标量，返回标量；如果是数组，返回数组
        if np.ndim(lat) == 0:
            return float(lat), float(lon)
        else:
            return lat, lon
    
    def get_grid_info(self) -> dict:
        """
        获取网格配置信息
        
        返回:
            包含网格配置信息的字典
        """
        return {
            'resolution': self.resolution,
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
            'lon_min': self.lon_min,
            'lon_max': self.lon_max,
            'lat_size': self.lat_size,
            'lon_size': self.lon_size
        }
    
    def __repr__(self):
        return (f"LatLonGridConverter(resolution={self.resolution}, "
                f"lat=[{self.lat_min}, {self.lat_max}], "
                f"lon=[{self.lon_min}, {self.lon_max}], "
                f"grid_shape=({self.lat_size}, {self.lon_size}))")


# ==================== 便捷函数 ====================

def latlon_to_index(lat: Union[float, np.ndarray], 
                   lon: Union[float, np.ndarray],
                   resolution: float = 1.0,
                   lat_min: float = -90.0,
                   lon_min: float = 0.0) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    将经纬度坐标转换为网格索引（便捷函数）
    
    参数:
        lat: 纬度（度），南纬为负，北纬为正
        lon: 经度（度），东经为正（0-360 范围）
        resolution: 网格分辨率（度），默认 1.0
        lat_min: 纬度起始值，默认 -90.0
        lon_min: 经度起始值，默认 0.0
    
    返回:
        (lat_idx, lon_idx): 网格索引元组
    
    示例:
        >>> lat_idx, lon_idx = latlon_to_index(-21.7, 157.5, resolution=1.0)
        >>> print(f"索引: ({lat_idx}, {lon_idx})")
        索引: (68, 158)
        
        >>> # 使用 0.25 度分辨率
        >>> lat_idx, lon_idx = latlon_to_index(40.5, 120.25, resolution=0.25)
        >>> print(f"索引: ({lat_idx}, {lon_idx})")
        索引: (522, 481)
    """
    # 处理经度范围
    lon = np.asarray(lon)
    if np.any((lon >= -180) & (lon < 0)):
        lon = np.where(lon < 0, lon + 360, lon)
    
    # 计算索引
    lat_idx = np.round((lat - lat_min) / resolution).astype(int)
    lon_idx = np.round((lon - lon_min) / resolution).astype(int)
    
    # 如果输入是标量，返回标量
    if np.ndim(lat_idx) == 0:
        return int(lat_idx), int(lon_idx)
    else:
        return lat_idx, lon_idx


def index_to_latlon(lat_idx: Union[int, np.ndarray],
                   lon_idx: Union[int, np.ndarray],
                   resolution: float = 1.0,
                   lat_min: float = -90.0,
                   lon_min: float = 0.0) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    将网格索引转换回经纬度坐标（便捷函数）
    
    参数:
        lat_idx: 纬度索引
        lon_idx: 经度索引
        resolution: 网格分辨率（度），默认 1.0
        lat_min: 纬度起始值，默认 -90.0
        lon_min: 经度起始值，默认 0.0
    
    返回:
        (lat, lon): 经纬度坐标元组
    
    示例:
        >>> lat, lon = index_to_latlon(68, 158, resolution=1.0)
        >>> print(f"经纬度: ({lat}, {lon})")
        经纬度: (-22.0, 158.0)
    """
    lat = lat_min + np.asarray(lat_idx) * resolution
    lon = lon_min + np.asarray(lon_idx) * resolution
    
    # 如果输入是标量，返回标量
    if np.ndim(lat) == 0:
        return float(lat), float(lon)
    else:
        return lat, lon


# ==================== 使用示例 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("经纬度与网格索引转换工具 - 使用示例")
    print("=" * 70)
    
    # 示例 1: 使用类方式（推荐用于多次转换）
    print("\n【示例 1】使用 LatLonGridConverter 类（1 度分辨率）")
    print("-" * 70)
    converter = LatLonGridConverter(resolution=1.0)
    print(converter)
    
    # 台风 Seth 中心位置
    cyclone_lat, cyclone_lon = -21.7005, 157.5000
    lat_idx, lon_idx = converter.latlon_to_index(cyclone_lat, cyclone_lon)
    print(f"\n输入: 纬度 {cyclone_lat}°, 经度 {cyclone_lon}°")
    print(f"输出: lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    # 反向验证
    recovered_lat, recovered_lon = converter.index_to_latlon(lat_idx, lon_idx)
    print(f"反向: 纬度 {recovered_lat}°, 经度 {recovered_lon}°")
    print(f"误差: Δlat={abs(recovered_lat - cyclone_lat):.6f}°, "
          f"Δlon={abs(recovered_lon - cyclone_lon):.6f}°")
    
    # 示例 2: 使用 0.25 度分辨率
    print("\n【示例 2】使用 LatLonGridConverter 类（0.25 度分辨率）")
    print("-" * 70)
    converter_025 = LatLonGridConverter(resolution=0.25)
    print(converter_025)
    
    lat_idx, lon_idx = converter_025.latlon_to_index(cyclone_lat, cyclone_lon)
    print(f"\n输入: 纬度 {cyclone_lat}°, 经度 {cyclone_lon}°")
    print(f"输出: lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    recovered_lat, recovered_lon = converter_025.index_to_latlon(lat_idx, lon_idx)
    print(f"反向: 纬度 {recovered_lat}°, 经度 {recovered_lon}°")
    print(f"误差: Δlat={abs(recovered_lat - cyclone_lat):.6f}°, "
          f"Δlon={abs(recovered_lon - cyclone_lon):.6f}°")
    
    # 示例 3: 使用便捷函数
    print("\n【示例 3】使用便捷函数（快速单次转换）")
    print("-" * 70)
    lat_idx, lon_idx = latlon_to_index(cyclone_lat, cyclone_lon, resolution=1.0)
    print(f"latlon_to_index({cyclone_lat}, {cyclone_lon}, resolution=1.0)")
    print(f"  -> ({lat_idx}, {lon_idx})")
    
    lat, lon = index_to_latlon(lat_idx, lon_idx, resolution=1.0)
    print(f"\nindex_to_latlon({lat_idx}, {lon_idx}, resolution=1.0)")
    print(f"  -> ({lat}, {lon})")
    
    # 示例 4: 批量转换
    print("\n【示例 4】批量转换多个坐标")
    print("-" * 70)
    lats = np.array([-21.7, 40.5, 0.0, -45.0, 60.0])
    lons = np.array([157.5, 120.5, 180.0, 90.0, 270.0])
    
    print(f"输入纬度: {lats}")
    print(f"输入经度: {lons}")
    
    lat_idxs, lon_idxs = latlon_to_index(lats, lons, resolution=1.0)
    print(f"\n纬度索引: {lat_idxs}")
    print(f"经度索引: {lon_idxs}")
    
    recovered_lats, recovered_lons = index_to_latlon(lat_idxs, lon_idxs, resolution=1.0)
    print(f"\n恢复纬度: {recovered_lats}")
    print(f"恢复经度: {recovered_lons}")
    
    # 示例 5: 处理负经度（-180 到 180 范围）
    print("\n【示例 5】处理负经度（西经表示）")
    print("-" * 70)
    west_lon = -120.5  # 西经 120.5 度
    print(f"输入: 西经 {abs(west_lon)}° (即 {west_lon}°)")
    
    lat_idx, lon_idx = latlon_to_index(0.0, west_lon, resolution=1.0)
    print(f"索引: lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    lat, lon = index_to_latlon(lat_idx, lon_idx, resolution=1.0)
    print(f"转换为正经度: {lon}° (东经)")
    
    print("\n" + "=" * 70)
    print("示例结束")
    print("=" * 70)
