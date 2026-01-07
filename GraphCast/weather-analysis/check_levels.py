#!/usr/bin/env python3
"""检查 ERA5 数据的气压层"""

import xarray

# 加载数据
# 旧路径 (1.0° 分辨率, 13 层): "/root/data/dataset/dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
dataset_file = "/root/autodl-tmp/data/dataset/dataset-source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc"

with open(dataset_file, "rb") as f:
    data = xarray.load_dataset(f).compute()

# 检查 u_component_of_wind 的 level 坐标
if 'u_component_of_wind' in data.data_vars:
    u_wind = data['u_component_of_wind']
    
    if 'level' in u_wind.coords:
        levels = u_wind.level.values
        print("=" * 60)
        print("气压层信息 (level coordinate)")
        print("=" * 60)
        print(f"总共有 {len(levels)} 个气压层:")
        print()
        for idx, level in enumerate(levels):
            print(f"  index {idx:2d} → {level:6.0f} hPa")
        print()
        print("=" * 60)
        print("引导层建议:")
        print("  - 500hPa (对流层中层,标准引导层)")
        print("  - 700hPa (较低的引导层)")
        print("  - 850hPa (边界层顶部)")
        print("=" * 60)
        
        # 找到最接近500hPa的索引
        target_level = 500
        diff = abs(levels - target_level)
        best_idx = diff.argmin()
        print(f"\n✓ 最接近 {target_level}hPa 的层: index {best_idx} ({levels[best_idx]:.0f} hPa)")
    else:
        print("警告: u_component_of_wind 没有 level 维度!")
        print("可用的维度:", u_wind.dims)
else:
    print("错误: 数据中没有 u_component_of_wind 变量!")
    print("可用的变量:", list(data.data_vars.keys()))
