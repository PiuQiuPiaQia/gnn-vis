#!/usr/bin/env python3
"""
计算台风中心在全球网格张量中的索引位置

根据 ERA5/GraphCast 的全球分辨率（1度），
将真实经纬度坐标转换为网格索引 (lat_index, lon_index)

台风中心: (-21.7005°S, 157.5°E)
计算结果: (lat_idx, lon_idx) = (68, 158)

注意：本脚本已更新为使用通用的 latlon_utils 模块
"""

import xarray as xr
import numpy as np
from pathlib import Path
from latlon_utils import LatLonGridConverter, latlon_to_index, index_to_latlon

# ==================== 配置参数 ====================

# 台风中心真实坐标
CYCLONE_CENTER_LAT = -21.7005  # 南纬 21.7005度
CYCLONE_CENTER_LON = 157.5000  # 东经 157.5度

# ERA5/GraphCast 全球网格参数
RESOLUTION = 1.0  # 分辨率 (度)
LAT_MIN_GLOBAL = -90.0  # 全球纬度起始值 (南纬90度)
LAT_MAX_GLOBAL = 90.0   # 全球纬度结束值 (北纬90度)
LON_MIN_GLOBAL = 0.0    # 全球经度起始值
LON_MAX_GLOBAL = 360.0  # 全球经度结束值 (不包括360)

# ==================== 使用通用转换器 ====================

# 创建全局网格转换器（复用通用模块）
converter = LatLonGridConverter(
    resolution=RESOLUTION,
    lat_min=LAT_MIN_GLOBAL,
    lat_max=LAT_MAX_GLOBAL,
    lon_min=LON_MIN_GLOBAL,
    lon_max=LON_MAX_GLOBAL
)

# 为兼容性保留原有函数接口
def latlon_to_grid_index(lat, lon, resolution=RESOLUTION,
                         lat_min=LAT_MIN_GLOBAL, lon_min=LON_MIN_GLOBAL):
    """
    将经纬度坐标转换为网格索引（使用通用工具模块）

    参数:
        lat: 纬度 (度), 南纬为负, 北纬为正
        lon: 经度 (度), 东经为正 (0-360范围)
        resolution: 网格分辨率 (度)
        lat_min: 纬度起始值
        lon_min: 经度起始值

    返回:
        (lat_idx, lon_idx): 网格索引元组
    """
    return latlon_to_index(lat, lon, resolution, lat_min, lon_min)


def grid_index_to_latlon(lat_idx, lon_idx, resolution=RESOLUTION,
                         lat_min=LAT_MIN_GLOBAL, lon_min=LON_MIN_GLOBAL):
    """
    将网格索引转换回经纬度坐标（使用通用工具模块）

    参数:
        lat_idx: 纬度索引
        lon_idx: 经度索引
        resolution: 网格分辨率 (度)
        lat_min: 纬度起始值
        lon_min: 经度起始值

    返回:
        (lat, lon): 经纬度坐标元组
    """
    return index_to_latlon(lat_idx, lon_idx, resolution, lat_min, lon_min)


def verify_index_with_dataset(lat_idx, lon_idx, dataset_file):
    """
    使用实际数据集验证索引计算是否正确

    参数:
        lat_idx: 纬度索引
        lon_idx: 经度索引
        dataset_file: NetCDF 数据集文件路径
    """
    print("\n【验证索引计算】")
    print("-" * 60)
    print(f"使用数据集验证: {dataset_file.name}")

    ds = xr.open_dataset(dataset_file)

    # 获取数据集中的坐标
    lats = ds.coords['lat'].values
    lons = ds.coords['lon'].values

    # 验证纬度索引
    if 0 <= lat_idx < len(lats):
        actual_lat = lats[lat_idx]
        print(f"✓ 纬度索引 {lat_idx} -> 实际纬度: {actual_lat:.4f}°")
        print(f"  期望纬度: {CYCLONE_CENTER_LAT:.4f}°")
        print(f"  差异: {abs(actual_lat - CYCLONE_CENTER_LAT):.6f}°")
    else:
        print(f"✗ 纬度索引 {lat_idx} 超出范围 [0, {len(lats)-1}]")

    # 验证经度索引
    if 0 <= lon_idx < len(lons):
        actual_lon = lons[lon_idx]
        print(f"✓ 经度索引 {lon_idx} -> 实际经度: {actual_lon:.4f}°")
        print(f"  期望经度: {CYCLONE_CENTER_LON:.4f}°")
        print(f"  差异: {abs(actual_lon - CYCLONE_CENTER_LON):.6f}°")
    else:
        print(f"✗ 经度索引 {lon_idx} 超出范围 [0, {len(lons)-1}]")

    ds.close()


# ==================== 主程序 ====================

print("=" * 70)
print("Cyclone Seth 台风中心网格索引计算")
print("=" * 70)
print(f"\n台风中心真实坐标:")
print(f"  纬度: {CYCLONE_CENTER_LAT}°S (南纬 {abs(CYCLONE_CENTER_LAT)}°)")
print(f"  经度: {CYCLONE_CENTER_LON}°E (东经 {CYCLONE_CENTER_LON}°)")

print(f"\nERA5/GraphCast 全球网格参数:")
print(f"  分辨率: {RESOLUTION}°")
print(f"  纬度范围: [{LAT_MIN_GLOBAL}°, {LAT_MAX_GLOBAL}°]")
print(f"  经度范围: [{LON_MIN_GLOBAL}°, {LON_MAX_GLOBAL}°)")
print(f"  纬度格点数: {int((LAT_MAX_GLOBAL - LAT_MIN_GLOBAL) / RESOLUTION) + 1}")
print(f"  经度格点数: {int((LON_MAX_GLOBAL - LON_MIN_GLOBAL) / RESOLUTION)}")

# 计算网格索引（使用通用转换器）
print("\n" + "=" * 70)
print("计算结果（使用 latlon_utils 通用模块）")
print("=" * 70)

lat_idx, lon_idx = converter.latlon_to_index(CYCLONE_CENTER_LAT, CYCLONE_CENTER_LON)

print(f"\n【公式】")
print("-" * 60)
print(f"lat_idx = (lat - lat_min) / resolution")
print(f"        = ({CYCLONE_CENTER_LAT} - {LAT_MIN_GLOBAL}) / {RESOLUTION}")
print(f"        = {(CYCLONE_CENTER_LAT - LAT_MIN_GLOBAL) / RESOLUTION:.1f}")
print(f"        = {lat_idx}")

print(f"\nlon_idx = (lon - lon_min) / resolution")
print(f"        = ({CYCLONE_CENTER_LON} - {LON_MIN_GLOBAL}) / {RESOLUTION}")
print(f"        = {(CYCLONE_CENTER_LON - LON_MIN_GLOBAL) / RESOLUTION:.1f}")
print(f"        = {lon_idx}")

print(f"\n【最终结果】")
print("-" * 60)
print(f"台风中心在全球网格中的索引位置:")
print(f"  (lat_idx, lon_idx) = ({lat_idx}, {lon_idx})")

# 反向验证
recovered_lat, recovered_lon = converter.index_to_latlon(lat_idx, lon_idx)
print(f"\n【反向验证】")
print("-" * 60)
print(f"索引 ({lat_idx}, {lon_idx}) 对应的经纬度:")
print(f"  纬度: {recovered_lat:.4f}° (误差: {abs(recovered_lat - CYCLONE_CENTER_LAT):.6f}°)")
print(f"  经度: {recovered_lon:.4f}° (误差: {abs(recovered_lon - CYCLONE_CENTER_LON):.6f}°)")

# 使用裁剪后的数据集验证
PROJECT_ROOT = Path(__file__).parent.parent
data_dir = PROJECT_ROOT / "graphcast-data"
clipped_file = data_dir / "cyclone_seth_2022-01-01_clipped.nc"

if clipped_file.exists():
    # 对于裁剪后的数据集，需要调整索引
    print("\n" + "=" * 70)
    print("裁剪后数据集中的索引位置")
    print("=" * 70)

    ds = xr.open_dataset(clipped_file)
    clipped_lats = ds.coords['lat'].values
    clipped_lons = ds.coords['lon'].values

    # 在裁剪数据集中找到最接近的索引
    clipped_lat_idx = int(round((CYCLONE_CENTER_LAT - clipped_lats[0]) / RESOLUTION))
    clipped_lon_idx = int(round((CYCLONE_CENTER_LON - clipped_lons[0]) / RESOLUTION))

    print(f"\n裁剪范围:")
    print(f"  纬度: [{clipped_lats[0]:.2f}°, {clipped_lats[-1]:.2f}°]")
    print(f"  经度: [{clipped_lons[0]:.2f}°, {clipped_lons[-1]:.2f}°]")

    print(f"\n在裁剪数据集中的索引:")
    print(f"  (clipped_lat_idx, clipped_lon_idx) = ({clipped_lat_idx}, {clipped_lon_idx})")

    if 0 <= clipped_lat_idx < len(clipped_lats) and 0 <= clipped_lon_idx < len(clipped_lons):
        print(f"\n验证:")
        print(f"  实际纬度: {clipped_lats[clipped_lat_idx]:.4f}°")
        print(f"  实际经度: {clipped_lons[clipped_lon_idx]:.4f}°")

    ds.close()

# 使用原始数据集验证
original_file = data_dir / "graphcast_dataset_source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc"
if original_file.exists():
    verify_index_with_dataset(lat_idx, lon_idx, original_file)

# ==================== 代码示例 ====================

print("\n" + "=" * 70)
print("Python 代码使用示例")
print("=" * 70)

print("""
# 获取台风中心点的数据值
import xarray as xr

ds = xr.open_dataset('cyclone_seth_2022-01-01_clipped.nc')

# 方法1: 使用裁剪数据集中的局部索引
clipped_lat_idx = {clipped_lat_idx}
clipped_lon_idx = {clipped_lon_idx}

# 获取地表温度
temp_at_center = ds['2m_temperature'][:, :, clipped_lat_idx, clipped_lon_idx]

# 获取多层温度
temp_3d_at_center = ds['temperature'][:, :, :, clipped_lat_idx, clipped_lon_idx]

# 方法2: 直接使用经纬度选择 (推荐)
temp_at_center = ds['2m_temperature'].sel(lat={CYCLONE_CENTER_LAT}, lon={CYCLONE_CENTER_LON}, method='nearest')

print(f"台风中心温度: {{temp_at_center.values}} K")
""".format(
    CYCLONE_CENTER_LAT=CYCLONE_CENTER_LAT,
    CYCLONE_CENTER_LON=CYCLONE_CENTER_LON,
    clipped_lat_idx=clipped_lat_idx if 'clipped_lat_idx' in locals() else 'N/A',
    clipped_lon_idx=clipped_lon_idx if 'clipped_lon_idx' in locals() else 'N/A'
))

print("\n" + "=" * 70)
print("计算完成")
print("=" * 70)
