import xarray as xr
import numpy as np
import packaging
import pandas
import zarr
import gcsfs
import matplotlib.pyplot as plt


# 数据集地址：https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
climatology = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr')
temperature = climatology['2m_temperature']
# 选择特定的时间范围
subset = temperature.sel(time=slice('2020-01-01', '2020-01-01'))

# 计算平均温度
mean_temp = subset.mean(dim='time')

mean_temp.to_netcdf('mean_temperature.nc')

# 绘制温度数据
mean_temp.plot()
plt.show()
plt.savefig('mean_temperature.png')



print("------------")
print(climatology)
print("------------")
print(subset)
print("------------")
# 经度以1.5度为步长，共有240个数据点
# 维度以0.5度为步长，共有121个数据点
print(mean_temp)
