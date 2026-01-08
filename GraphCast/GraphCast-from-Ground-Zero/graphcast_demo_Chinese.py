#!/usr/bin/env python
# coding: utf-8

# # 从零开始运行 GraphCast （AutoDL 或者其他新的环境）
# -------------------------------------------------------------------
# **这是从 https://google-deepmind/graphcast 复现的项目。由 https://github.com/sfsun67 改写和调试。**
# 
# **AutoDL 是国内的一家云计算平台，网址是https://www.autodl.com**
# 
# 你应该有类似的文件结构，这里的数据由 Google Cloud Bucket (https://console.cloud.google.com/storage/browser/dm_graphcast 提供。模型权重、标准化统计和示例输入可在Google Cloud Bucket上找到。完整的模型训练需要下载ERA5数据集，该数据集可从ECMWF获得。
# ```
# .
# ├── code
# │   ├── GraphCast-from-Ground-Zero
# │       ├──graphcast
# │       ├──tree
# │       ├──wrapt
# │       ├──graphcast_demo.ipynb
# │       ├──README.md
# │       ├──setup.py
# │       ├──...
# ├── data
# │   ├── dataset
# │       ├──dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc
# │       ├──dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc
# │       ├──dataset-source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc
# │       ├──...
# │   ├── params
# │       ├──params-GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz
# │       ├──params-GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz
# │       ├──...
# │   ├── stats
# │       ├──stats-mean_by_level.nc
# │       ├──...
# └────── 
# ```
# 
# PS: 
# 1. Python 要使用3.10版本。老版本会出现函数调用失效的问题。
# 2. 你需要仔细核对包的版本，防止出现意外的错误。例如， xarray 只能使用 2023.7.0 版本，其他版本会出现错误。
# 3. 你需要仔细核对所有包是否安装正确。未安装的包会导致意外错误。例如，tree 和 wrapt 是两个 GraphCast 所必需的包，但是并不在源文件中。例如，tree 和 wrapt 中的 .os 文件未导入，会引发循环调用。他们的原始文件可以在 Colaboratory(https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb) 的环境中找到。
# 
# 
# 
# *代码在如下机器上测试*
# 1. GPU: TITAN Xp 12GB; CPU: Xeon(R) E5-2680 v4;  JAX / 0.3.10 / 3.8(ubuntu18.04) / 11.1
# 2. GPU: V100-SXM2-32GB 32GB; CPU: Xeon(R) Platinum 8255C; JAX / 0.3.10 / 3.8(ubuntu18.04) / 11.1
# 3. GPU: RTX 2080 Ti(11GB); CPU: Xeon(R) Platinum 8255C; JAX / 0.3.10 / 3.8(ubuntu18.04) / 11.1
# -------------------------------------------------------------------
# 

# <p><small><small>版权所有 2023 年 DeepMind Technologies Limited。</small></small></p>
# <p><small><small>根据 Apache 许可证第 2.0 版（"许可证"）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在 <a href="http://www.apache.org/licenses/LICENSE-2.0">http://www.apache.org/licenses/LICENSE-2.0</a> 获取许可证的副本。</small></small></p>
# <p><small><small>除非适用法律要求或书面同意，根据许可证分发的软件是基于 "按原样" 分发的，没有任何明示或暗示的担保或条件。有关许可证下的具体语言，请参见许可证中的权限和限制。</small></small></p>
# 

# # 将 Python 版本更新到 3.10.
# 
# GraphCast 需要 Python >= 3.10 。推荐 Python 3.10。
# 
# 在终端中，新建一个名为 GraphCast 的环境。
# 
# 参考代码如下：
# ```
# 
# # 更新 conda （可选）
# conda update -n base -c defaults conda
# 
# # 在新环境 GraphCast 中安装 python=3.10  
# conda create -n GraphCast python=3.10    
# 
# # 更新bashrc中的环境变量
# conda init bash && source /root/.bashrc
# 
# # 激活新的环境
# conda activate GraphCast
# 
# # 验证版本
# python --version
# 
# # 在 Jupyter 中注册 Python 3.10 环境
# # 安装 ipykernel 包
# conda install ipykernel
# 
# # 注册的 Python 3.10 环境的内核名称
# python -m ipykernel install --user --name=GraphCast-python3.10
# ```
# 
# 注意：Jupyter 注册 Python 3.10 环境后，重启jupyter，使用新的内核 GraphCast-python3.10。

# # 安装和初始化
# 

# In[3]:


from jax.lib import xla_bridge
import jax
print("JAX devices:", jax.devices())


# In[4]:


# 学术资源加速 https://www.autodl.com/docs/network_turbo/  .

# import subprocess
# import os

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value


# In[5]:


# 这一步将使用 shapely 安装环境。为了避免出现ERROR： 无法为 shapely 构建轮子，而安装基于 pyproject.toml 的项目需要轮子。

# !pip uninstall -y shapely
# !conda install -y shapely
# !pip uninstall -y shapely


# In[6]:


# @title Pip 安装 graphcast 和其他依赖项


# %pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip


# In[7]:


# @title cartopy 崩溃的解决方法

# !pip uninstall -y shapely
# !pip install shapely --no-binary shapely


# In[8]:


# @title 安装其他依赖项，并解决 xarray 的版本问题。

# 这里需要将xarray的版本从2023.12.0(2023年12月30日安装)降低到2023.7.0，否则会报错。

# !conda install -y -c conda-forge ipywidgets
# !pip uninstall -y xarray
# !pip install xarray==2023.7.0


# In[9]:


# @title 导入库


import dataclasses
import datetime
import functools
import math
import re
from typing import Optional

import cartopy.crs as ccrs
#from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray




def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


# In[10]:


# @title 载入绘图函数


def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())


# # 加载数据并初始化模型

# ## 载入模型参数
# 
# 选择两种获取模型参数的方式之一：
# - **random**：您将获得随机预测，但您可以更改模型架构，这可能会使其运行更快或适应您的设备。
# - **checkpoint**：您将获得明智的预测，但受限于模型训练时使用的架构，这可能不适合您的设备。特别是生成梯度会使用大量内存，因此您至少需要25GB的内存（TPUv4或A100）。
# 
# 检查点在一些方面有所不同：
# - 网格大小指定了地球的内部图形表示。较小的网格将运行更快，但输出将更差。网格大小不影响模型的参数数量。
# - 分辨率和压力级别的数量必须匹配数据。较低的分辨率和较少的级别会运行得更快。数据分辨率仅影响编码器/解码器。
# - 我们的所有模型都预测降水。然而，ERA5包含降水，而HRES不包含。我们标记为 "ERA5" 的模型将降水作为输入，并期望以ERA5数据作为输入，而标记为 "ERA5-HRES" 的模型不以降水作为输入，并专门训练以HRES-fc0作为输入（请参阅下面的数据部分）。
# 
# 我们提供三个预训练模型：
# 1. `GraphCast`，用于GraphCast论文的高分辨率模型（0.25度分辨率，37个压力级别），在1979年至2017年间使用ERA5数据进行训练，
# 
# 2. `GraphCast_small`，GraphCast的较小低分辨率版本（1度分辨率，13个压力级别和较小的网格），在1979年至2015年间使用ERA5数据进行训练，适用于具有较低内存和计算约束的模型运行，
# 
# 3. `GraphCast_operational`，一个高分辨率模型（0.25度分辨率，13个压力级别），在1979年至2017年使用ERA5数据进行预训练，并在2016年至2021年间使用HRES数据进行微调。此模型可以从HRES数据初始化（不需要降水输入）。
# 

# In[11]:


# @title 选择模型
# Rewrite by S.F. Sune, https://github.com/sfsun67.
'''
    我们有三种训练好的模型可供选择, 需要从https://console.cloud.google.com/storage/browser/dm_graphcast准备：
    GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz
    GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz
    GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz
'''
# 在此路径 /root/data/params 中查找结果，并列出 "params/"中所有文件的名称，去掉名称中的 "params/"perfix。

import os
import glob

# 定义数据目录，请替换成你自己的目录。
dir_path_params = "/root/data/params"

# Use glob to get all file paths in the directory
file_paths_params = glob.glob(os.path.join(dir_path_params, "*"))

# Remove the directory path and the ".../params/" prefix from each file name
params_file_options = [os.path.basename(path) for path in file_paths_params]


random_mesh_size = widgets.IntSlider(
    value=4, min=4, max=6, description="Mesh size:")
random_gnn_msg_steps = widgets.IntSlider(
    value=4, min=1, max=32, description="GNN message steps:")
random_latent_size = widgets.Dropdown(
    options=[int(2**i) for i in range(4, 10)], value=32,description="Latent size:")
random_levels = widgets.Dropdown(
    options=[13, 37], value=13, description="Pressure levels:")


params_file = widgets.Dropdown(
    options=params_file_options,
    description="Params file:",
    layout={"width": "max-content"})

source_tab = widgets.Tab([
    widgets.VBox([
        random_mesh_size,
        random_gnn_msg_steps,
        random_latent_size,
        random_levels,
    ]),
    params_file,
])
# 设置默认显示的标签索引
source_tab.selected_index = 1

source_tab.set_title(0, "随机参数权重（Random）")
source_tab.set_title(1, "预训练权重（Checkpoint）")
widgets.VBox([
    source_tab,
    widgets.Label(value="运行下一个单元格以加载模型。重新运行该单元格将清除您的选择。")
])


# In[12]:


# @title 加载模型

source = source_tab.get_title(source_tab.selected_index)

if source == "随机参数权重（Random）":
  params = None  # Filled in below
  state = {}
  model_config = graphcast.ModelConfig(
      resolution=0,
      mesh_size=random_mesh_size.value,
      latent_size=random_latent_size.value,
      gnn_msg_steps=random_gnn_msg_steps.value,
      hidden_layers=1,
      radius_query_fraction_edge_length=0.6)
  task_config = graphcast.TaskConfig(
      input_variables=graphcast.TASK.input_variables,
      target_variables=graphcast.TASK.target_variables,
      forcing_variables=graphcast.TASK.forcing_variables,
      pressure_levels=graphcast.PRESSURE_LEVELS[random_levels.value],
      input_duration=graphcast.TASK.input_duration,
  )
else:
  assert source == "预训练权重（Checkpoint）"
  '''with gcs_bucket.blob(f"params/{params_file.value}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)'''

  with open(f"{dir_path_params}/{params_file.value}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

  params = ckpt.params
  state = {}

  model_config = ckpt.model_config
  task_config = ckpt.task_config
  print("模型描述:\n", ckpt.description, "\n")
  print("模型许可信息:\n", ckpt.license, "\n")

model_config


# ## 载入示例数据
# 
# 有几个示例数据集可用，在几个坐标轴上各不相同：
# - **来源**：fake、era5、hres
# - **分辨率**：0.25度、1度、6度
# - **级别**：13, 37
# - **步数**：包含多少个时间步
# 
# 并非所有组合都可用。
# - 由于加载内存的要求，较高分辨率只适用于较少的步数。
# - HRES 只有 0.25 度，13 个压力等级。
# 
# 数据分辨率必须与加载的模型相匹配。
# 
# 对基础数据集进行了一些转换：
# - 我们累积了 6 个小时的降水量，而不是默认的 1 个小时。
# - 对于 HRES 数据，每个时间步对应 HRES 在前导时间 0 的预报，实际上提供了 HRES 的 "初始化"。有关详细描述，请参见 GraphCast 论文中的 HRES-fc0。请注意，HRES 无法提供 6 小时的累积降水量，因此我们的模型以 HRES 输入不依赖于降水。但由于我们的模型可以预测降水，因此在示例数据中包含了 ERA5 降水量，以作为地面真实情况的示例。
# - 我们在数据中加入了 ERA5 的 "toa_incident_solar_radiation"。我们的模型使用 -6h、0h 和 +6h 辐射作为每 1 步预测的强迫项。在运行中，如果没有现成的 +6h 辐射，可以使用诸如 `pysolar` 等软件包计算辐射。
# 

# In[13]:


# @title 获取和筛选可用示例数据的列表

# Rewrite by S.F. Sune, https://github.com/sfsun67.
# 在"/root/data/dataset"路径下查找结果，并列出"dataset/"中所有文件的名称列表，去掉"dataset/"前缀。

# 定义数据目录，请替换成你自己的目录。
dir_path_dataset = "/root/data/dataset"

# Use glob to get all file paths in the directory
file_paths_dataset = glob.glob(os.path.join(dir_path_dataset, "*"))

# Remove the directory path and the ".../params/" prefix from each file name
dataset_file_options = [os.path.basename(path) for path in file_paths_dataset]
#print("dataset_file_options: ", dataset_file_options)

# Remove "dataset-" prefix from each file name
dataset_file_options = [name.removeprefix("dataset-") for name in dataset_file_options]


def data_valid_for_model(
    file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
  file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  #print("file_parts: ", file_parts)
  return (
      model_config.resolution in (0, float(file_parts["res"])) and
      len(task_config.pressure_levels) == int(file_parts["levels"]) and
      (
          ("total_precipitation_6hr" in task_config.input_variables and
           file_parts["source"] in ("era5", "fake")) or
          ("total_precipitation_6hr" not in task_config.input_variables and
           file_parts["source"] in ("hres", "fake"))
      )
  )


dataset_file = widgets.Dropdown(
    options=[
        (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
        for option in dataset_file_options
        if data_valid_for_model(option, model_config, task_config)
    ],
    description="数据文件:",
    layout={"width": "max-content"})
widgets.VBox([
    dataset_file,
    widgets.Label(value="运行下一个单元格以加载数据集。重新运行此单元格将清除您的选择，并重新筛选与您的模型匹配的数据集。")
])


# In[14]:


# @title 加载气象数据


if not data_valid_for_model(dataset_file.value, model_config, task_config):
  raise ValueError(
      "Invalid dataset file, rerun the cell above and choose a valid dataset file.")

'''with gcs_bucket.blob(f"dataset/{dataset_file.value}").open("rb") as f:
  example_batch = xarray.load_dataset(f).compute()'''

with open(f"{dir_path_dataset}/dataset-{dataset_file.value}", "rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.value.removesuffix(".nc")).items()]))

# ============================================================================
# 打印数据集的时间点信息
# ============================================================================
print("\n" + "="*80)
print("数据集时间点详细信息")
print("="*80)

if 'time' in example_batch.coords:
    time_coord = example_batch.coords['time']
    num_times = len(time_coord)
    
    print(f"\n⏰ 时间点总数: {num_times}")
    print(f"\n时间坐标类型: {type(time_coord.values[0])}")
    print(f"时间坐标dtype: {time_coord.values.dtype}")
    
    import pandas as pd
    
    # 检查时间坐标的类型
    if np.issubdtype(time_coord.values.dtype, np.timedelta64):
        # 时间坐标是 timedelta 类型（相对时间）
        print(f"\n⏱️  时间坐标存储为相对时间偏移量（timedelta64）")
        print(f"\n所有时间点（相对偏移）:")
        
        # 尝试从数据集属性中获取参考时间，如果没有则使用文件名中的日期
        reference_time = None
        if hasattr(example_batch, 'attrs') and 'reference_time' in example_batch.attrs:
            reference_time = pd.Timestamp(example_batch.attrs['reference_time'])
        else:
            # 从文件名中提取日期 (dataset_file.value 应该包含日期信息)
            file_parts = parse_file_parts(dataset_file.value.removesuffix(".nc"))
            if 'date' in file_parts:
                reference_time = pd.Timestamp(file_parts['date'])
        
        for i, t in enumerate(time_coord.values):
            # 转换 timedelta64 为小时数
            hours_offset = t / np.timedelta64(1, 'h')
            
            if reference_time is not None:
                # 如果有参考时间，计算绝对时间
                abs_time = reference_time + pd.Timedelta(t)
                print(f"  [{i}] +{hours_offset:6.1f}h -> {abs_time.strftime('%Y-%m-%d %H:%M:%S')} UTC (星期{['一','二','三','四','五','六','日'][abs_time.weekday()]})")
            else:
                print(f"  [{i}] +{hours_offset:6.1f}h")
        
        # 计算时间间隔
        if num_times > 1:
            time_diff = time_coord.values[1] - time_coord.values[0]
            hours_diff = time_diff / np.timedelta64(1, 'h')
            print(f"\n⏱️  时间间隔: {hours_diff:.1f} 小时")
        
        if reference_time is not None:
            print(f"\n📅 参考时间: {reference_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # 检查绝对时间是否包含18:00
            print(f"\n🔍 检查是否包含18:00的时间点:")
            has_18_00 = False
            found_18_00_times = []
            
            for i, t in enumerate(time_coord.values):
                abs_time = reference_time + pd.Timedelta(t)
                if abs_time.hour == 18:
                    has_18_00 = True
                    found_18_00_times.append((i, abs_time))
            
            if has_18_00:
                print(f"  ✅ 是的，数据集包含18:00的时间点:")
                for idx, dt in found_18_00_times:
                    print(f"     索引[{idx}]: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    if dt.date() < reference_time.date():
                        print(f"     -> 这是前一天的数据")
                    elif dt.date() == reference_time.date():
                        print(f"     -> 这是当天的数据")
            else:
                print(f"  ❌ 否，数据集不包含18:00的时间点")
    
    else:
        # 时间坐标是 datetime64 类型（绝对时间）
        print(f"\n📅 时间坐标存储为绝对时间（datetime64）")
        print(f"\n所有时间点:")
        
        for i, t in enumerate(time_coord.values):
            dt = pd.Timestamp(t).to_pydatetime()
            print(f"  [{i}] {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC (星期{['一','二','三','四','五','六','日'][dt.weekday()]})")
        
        # 计算时间间隔
        if num_times > 1:
            time_diff = time_coord.values[1] - time_coord.values[0]
            hours_diff = time_diff / np.timedelta64(1, 'h')
            print(f"\n⏱️  时间间隔: {hours_diff:.1f} 小时")
        
        # 检查是否包含18:00的数据
        print(f"\n🔍 检查是否包含18:00的时间点:")
        has_18_00 = False
        found_18_00_times = []
        
        for i, t in enumerate(time_coord.values):
            dt = pd.Timestamp(t).to_pydatetime()
            if dt.hour == 18:
                has_18_00 = True
                found_18_00_times.append((i, dt))
        
        if has_18_00:
            print(f"  ✅ 是的，数据集包含18:00的时间点:")
            for idx, dt in found_18_00_times:
                print(f"     索引[{idx}]: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                first_dt = pd.Timestamp(time_coord.values[0]).to_pydatetime()
                if dt.date() < first_dt.date():
                    print(f"     -> 这是前一天的数据")
                elif dt.date() == first_dt.date():
                    print(f"     -> 这是当天的数据")
        else:
            print(f"  ❌ 否，数据集不包含18:00的时间点")
    
    # 补充说明
    print(f"\n💡 说明:")
    print(f"  - GraphCast 使用相邻两个时间点作为输入 (例如: 00:00 和 06:00)")
    print(f"  - 然后预测未来的时间点 (例如: 12:00, 18:00, ...)")
    print(f"  - ERA5 标准数据包含: 00:00, 06:00, 12:00, 18:00 四个时间点")

print("="*80 + "\n")

example_batch


# In[15]:


# @title 选择绘图数据

plot_example_variable = widgets.Dropdown(
    options=example_batch.data_vars.keys(),
    value="2m_temperature",
    description="变量")
plot_example_level = widgets.Dropdown(
    options=example_batch.coords["level"].values,
    value=500,
    description="级别")
plot_example_robust = widgets.Checkbox(value=True, description="鲁棒性")
plot_example_max_steps = widgets.IntSlider(
    min=1, max=example_batch.dims["time"], value=example_batch.dims["time"],
    description="最大步")

widgets.VBox([
    plot_example_variable,
    plot_example_level,
    plot_example_robust,
    plot_example_max_steps,
    widgets.Label(value="运行下一个单元格以绘制数据。重新运行此单元格将清除您的选择。")
])


# In[16]:


# @title 绘制示例数据


plot_size = 7

data = {
    " ": scale(select(example_batch, plot_example_variable.value, plot_example_level.value, plot_example_max_steps.value),
              robust=plot_example_robust.value),
}
fig_title = plot_example_variable.value
if "等级" in example_batch[plot_example_variable.value].coords:
  fig_title += f" at {plot_example_level.value} hPa"

plot_data(data, fig_title, plot_size, plot_example_robust.value)


# In[17]:


# @title 选择要提取的训练和评估数据

train_steps = widgets.IntSlider(
    value=1, min=1, max=example_batch.sizes["time"]-2, description="训练步数")
eval_steps = widgets.IntSlider(
    value=example_batch.sizes["time"]-2, min=1, max=example_batch.sizes["time"]-2, description="评估步数")

widgets.VBox([
    train_steps,
    eval_steps,
    widgets.Label(value="运行下一个单元格以提取数据。重新运行此单元格将清除您的选择。")
])


# In[18]:


# @title 提取训练和评估数据

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps.value*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps.value*6}h"),
    **dataclasses.asdict(task_config))

print("所有示例：  ", example_batch.dims.mapping)
print("训练输入：  ", train_inputs.dims.mapping)
print("训练目标： ", train_targets.dims.mapping)
print("训练强迫：", train_forcings.dims.mapping)
print("评估输入：   ", eval_inputs.dims.mapping)
print("评估目标：  ", eval_targets.dims.mapping)
print("评估强迫项: ", eval_forcings.dims.mapping)

# ============================================================================
# 打印 train_inputs 和 train_targets 的时间信息
# ============================================================================
print("\n" + "="*80)
print("训练数据时间信息")
print("="*80)

import pandas as pd

# 从文件名中提取参考日期
file_parts = parse_file_parts(dataset_file.value.removesuffix(".nc"))
reference_time = None
if 'date' in file_parts:
    reference_time = pd.Timestamp(file_parts['date'])
    print(f"\n📅 数据集参考日期: {reference_time.strftime('%Y-%m-%d')}")

# 打印 train_inputs 的时间
print("\n🔹 train_inputs 包含的时间点:")
if 'time' in train_inputs.coords:
    train_input_times = train_inputs.coords['time'].values
    print(f"   时间点数量: {len(train_input_times)}")
    
    for i, t in enumerate(train_input_times):
        if np.issubdtype(train_input_times.dtype, np.timedelta64):
            hours_offset = t / np.timedelta64(1, 'h')
            if reference_time is not None:
                abs_time = reference_time + pd.Timedelta(t)
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h -> 绝对时间: {abs_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h")
        else:
            dt = pd.Timestamp(t).to_pydatetime()
            print(f"   [{i}] {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    print("   ⚠️ train_inputs 没有 time 坐标")

# 打印 train_targets 的时间
print("\n🔹 train_targets 包含的时间点:")
if 'time' in train_targets.coords:
    train_target_times = train_targets.coords['time'].values
    print(f"   时间点数量: {len(train_target_times)}")
    
    for i, t in enumerate(train_target_times):
        if np.issubdtype(train_target_times.dtype, np.timedelta64):
            hours_offset = t / np.timedelta64(1, 'h')
            if reference_time is not None:
                abs_time = reference_time + pd.Timedelta(t)
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h -> 绝对时间: {abs_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h")
        else:
            dt = pd.Timestamp(t).to_pydatetime()
            print(f"   [{i}] {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    print("   ⚠️ train_targets 没有 time 坐标")

# ============================================================================
# 打印 eval_inputs 和 eval_targets 的时间信息
# ============================================================================
print("\n" + "="*80)
print("评估数据时间信息")
print("="*80)

# 打印 eval_inputs 的时间
print("\n🔹 eval_inputs 包含的时间点:")
if 'time' in eval_inputs.coords:
    eval_input_times = eval_inputs.coords['time'].values
    print(f"   时间点数量: {len(eval_input_times)}")
    
    for i, t in enumerate(eval_input_times):
        if np.issubdtype(eval_input_times.dtype, np.timedelta64):
            hours_offset = t / np.timedelta64(1, 'h')
            if reference_time is not None:
                abs_time = reference_time + pd.Timedelta(t)
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h -> 绝对时间: {abs_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h")
        else:
            dt = pd.Timestamp(t).to_pydatetime()
            print(f"   [{i}] {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    print("   ⚠️ eval_inputs 没有 time 坐标")

# 打印 eval_targets 的时间
print("\n🔹 eval_targets 包含的时间点:")
if 'time' in eval_targets.coords:
    eval_target_times = eval_targets.coords['time'].values
    print(f"   时间点数量: {len(eval_target_times)}")
    
    for i, t in enumerate(eval_target_times):
        if np.issubdtype(eval_target_times.dtype, np.timedelta64):
            hours_offset = t / np.timedelta64(1, 'h')
            if reference_time is not None:
                abs_time = reference_time + pd.Timedelta(t)
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h -> 绝对时间: {abs_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                print(f"   [{i}] 相对时间: +{hours_offset:6.1f}h")
        else:
            dt = pd.Timestamp(t).to_pydatetime()
            print(f"   [{i}] {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
else:
    print("   ⚠️ eval_targets 没有 time 坐标")

# 补充说明
print("\n💡 说明:")
print("   - eval_inputs 固定包含2个连续时间点 (用于预测的输入)")
print("   - eval_targets 包含多个预测目标时间点 (由 eval_steps 参数决定)")
print("   - 时间间隔为6小时")
print("="*80 + "\n")


# In[19]:


# @title 加载规范化数据
# Rewrite by S.F. Sune, https://github.com/sfsun67.
dir_path_stats = "/root/data/stats"

with open(f"{dir_path_stats}/stats-diffs_stddev_by_level.nc", "rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(f"{dir_path_stats}/stats-mean_by_level.nc", "rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with open(f"{dir_path_stats}/stats-stddev_by_level.nc", "rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()


# In[20]:


# @title 构建 jitted 函数，并可能初始化随机权重
# 构建模型并初始化权重

# 模型组网
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor

# 前向运算
@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

# 计算损失函数
@hk.transform_with_state    # used to convert a pure function into a stateful function
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)    # constructs and wraps a GraphCast Predictor, which is a model used for making predictions in a graph-based machine learning task.
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

# 计算梯度
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


# # 运行模型
# 
# 请注意，第一次运行下面的单元格可能需要一段时间（可能几分钟），因为这包括代码编译的时间。第二次运行时速度会明显加快。
# 
# 这将使用 python 循环迭代预测步骤，其中 1 步的预测是固定的。这比下面的训练步骤对内存的要求要低，应该可以使用小型 GraphCast 模型对 1 度分辨率数据进行 4 步预测。

# In[21]:


# @标题 递归计算（在 python 中的循环）
import pickle
assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

# with open('eval_inputs.pkl', 'wb') as file:
#     pickle.dump(eval_inputs, file)
# with open('eval_targets.pkl', 'wb') as file:
#     pickle.dump(eval_targets, file)
# with open('eval_forcings.pkl', 'wb') as file:
#     pickle.dump(eval_forcings, file)


# with jax.disable_jit():
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
predictions


with open('predictions.pkl', 'wb') as file:
    pickle.dump(predictions, file)


# In[22]:


# @title 选择要绘制的预测结果

plot_pred_variable = widgets.Dropdown(
    options=predictions.data_vars.keys(),
    value="2m_temperature",
    description="变量")
plot_pred_level = widgets.Dropdown(
    options=predictions.coords["level"].values,
    value=500,
    description="级别")
plot_pred_robust = widgets.Checkbox(value=True, description="鲁棒性")
plot_pred_max_steps = widgets.IntSlider(
    min=1,
    max=predictions.dims["time"],
    value=predictions.dims["time"],
    description="最大步")

widgets.VBox([
    plot_pred_variable,
    plot_pred_level,
    plot_pred_robust,
    plot_pred_max_steps,
    widgets.Label(value="运行下一个单元格，绘制预测结果。重新运行该单元格将清除您的选择。")
])


# In[23]:
# @title 打印输入数据集的格式

print("=== 输入数据集格式信息 ===")
print("\n1. eval_inputs 格式:")
print(f"   类型: {type(eval_inputs)}")
print(f"   维度: {eval_inputs.dims}")
print(f"   坐标: {list(eval_inputs.coords.keys())}")
print(f"   数据变量: {list(eval_inputs.data_vars.keys())}")
print(f"   形状: {eval_inputs.sizes}")

print("\n2. eval_targets 格式:")
print(f"   类型: {type(eval_targets)}")
print(f"   维度: {eval_targets.dims}")
print(f"   坐标: {list(eval_targets.coords.keys())}")
print(f"   数据变量: {list(eval_targets.data_vars.keys())}")
print(f"   形状: {eval_targets.sizes}")

print("\n3. eval_forcings 格式:")
print(f"   类型: {type(eval_forcings)}")
print(f"   维度: {eval_forcings.dims}")
print(f"   坐标: {list(eval_forcings.coords.keys())}")
print(f"   数据变量: {list(eval_forcings.data_vars.keys())}")
print(f"   形状: {eval_forcings.sizes}")


# === 输入数据集格式信息 ===

# 1. eval_inputs 格式:
#    类型: <class 'xarray.core.dataset.Dataset'>
#    维度: Frozen({'batch': 1, 'time': 2, 'lat': 721, 'lon': 1440, 'level': 37})
#    坐标: ['lon', 'lat', 'level', 'time']
#    数据变量: ['2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity', 'toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask']
#    形状: Frozen({'batch': 1, 'time': 2, 'lat': 721, 'lon': 1440, 'level': 37})

# 2. eval_targets 格式:
#    类型: <class 'xarray.core.dataset.Dataset'>
#    维度: Frozen({'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440, 'level': 37})
#    坐标: ['lon', 'lat', 'level', 'time']
#    数据变量: ['2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity']
#    形状: Frozen({'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440, 'level': 37})

# 3. eval_forcings 格式:
#    类型: <class 'xarray.core.dataset.Dataset'>
#    维度: Frozen({'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440})
#    坐标: ['lon', 'lat', 'time']
#    数据变量: ['toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos']
#    形状: Frozen({'batch': 1, 'time': 1, 'lat': 721, 'lon': 1440})








# # 计算权重

# In[ ]:


print(train_inputs.dims)  # 查看 train_inputs 的维度
print(train_forcings.dims)  # 查看 train_forcings 的维度


# In[ ]:


# import xarray as xr
# import jax
# import numpy as np

# # 函数：逐通道置零并计算所有预测结果的变化
# def zero_out_channel(inputs, forcings, channel_name, variable='inputs'):
#     """将指定通道的输入变量或强迫项置零，并计算所有预测结果的变化"""
#     if variable == 'inputs':
#         inputs_copy = inputs.copy()  # 复制inputs
#         inputs_copy[channel_name].data = np.zeros_like(inputs_copy[channel_name].data)  # 将指定通道的输入置为零
#         forcings_copy = forcings  # forcings不变
#     elif variable == 'forcings':
#         forcings_copy = forcings.copy()  # 复制forcings
#         forcings_copy[channel_name].data = np.zeros_like(forcings_copy[channel_name].data)  # 将指定通道的强迫项置为零
#         inputs_copy = inputs  # inputs不变

#     # 计算新的所有预测结果
#     predictions = rollout.chunked_prediction(
#         run_forward_jitted,
#         rng=jax.random.PRNGKey(0),
#         inputs=inputs_copy,
#         targets_template=train_targets * np.nan,
#         forcings=forcings_copy
#     )

#     return predictions

# # 计算原始预测结果（遮挡前）
# def get_original_predictions(inputs, forcings):
#     """计算不做任何遮挡的原始预测结果"""
#     return rollout.chunked_prediction(
#         run_forward_jitted,
#         rng=jax.random.PRNGKey(0),
#         inputs=inputs,
#         targets_template=train_targets * np.nan,
#         forcings=forcings
#     )

# # 评估每个变量对所有预测结果的影响
# def evaluate_variable_importance(train_inputs, train_forcings, variable_type='inputs'):
#     channel_importance = {}  # 用来存储每个变量对所有预测的影响值

#     # 获取原始预测结果
#     original_predictions = get_original_predictions(train_inputs, train_forcings)

#     # 计算每个变量的影响
#     if variable_type == 'inputs':
#         channels = list(train_inputs.data_vars)  # 获取inputs中的所有变量名
#     elif variable_type == 'forcings':
#         channels = list(train_forcings.data_vars)  # 获取forcings中的所有变量名

#     # 对每个通道进行置零操作，并计算预测结果变化
#     for channel_name in channels:
#         # 逐通道置零并计算所有预测结果
#         predictions_with_zero = zero_out_channel(train_inputs, train_forcings, channel_name, variable=variable_type)

#         # 计算预测结果的变化（对于所有预测变量）
#         prediction_diff = {}
#         for var_name in original_predictions:
#             # 计算该预测变量的变化
#             prediction_diff[var_name] = np.abs(original_predictions[var_name] - predictions_with_zero[var_name])

#         # 计算所有输出变量的变化平均值，作为该通道的重要性度量
#         total_diff = np.mean([np.mean(diff) for diff in prediction_diff.values()])  # 对所有变量的差异取平均
#         channel_importance[channel_name] = total_diff  # 存储每个通道的总影响值

#     return channel_importance

# # 评估不同大气变量的影响
# inputs_importance = evaluate_variable_importance(train_inputs, train_forcings, variable_type='inputs')
# forcings_importance = evaluate_variable_importance(train_inputs, train_forcings, variable_type='forcings')

# 输出每个变量对所有预测结果的影响
# print(inputs_importance)
# print(forcings_importance)
# inputs_importance
# forcings_importance


# 2m_temperature
# mean_sea_level_pressure
# 10m_v_component_of_wind
# 10m_u_component_of_wind
# total_precipitation_6hr
# temperature
# geopotential
# u_component_of_wind
# v_component_of_wind
# vertical_velocity
# specific_humidity
# toa_incident_solar_radiation
# year_progress_sin
# year_progress_cos
# day_progress_sin
# day_progress_cos
# geopotential_at_surface
# land_sea_mask


# # 训练模型
# 
# 以下操作需要大量内存，而且根据所使用的加速器，只能在低分辨率数据上拟合很小的 "随机 "模型。它使用上面选择的训练步数。
# 
# 第一次执行单元需要更多时间，因为其中包括函数的 jit 时间。

# In[ ]:


# # @title 损失计算（多步骤递归（自回归）损失）
# loss, diagnostics = loss_fn_jitted(
#     rng=jax.random.PRNGKey(0),
#     inputs=train_inputs,
#     targets=train_targets,
#     forcings=train_forcings)

# print("Loss:", float(loss))


# In[ ]:


# # @title 梯度计算（通过时间进行反推）
# loss, diagnostics, next_state, grads = grads_fn_jitted(
#     inputs=train_inputs,
#     targets=train_targets,
#     forcings=train_forcings)
# mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
# print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")


# In[ ]:


# # @title 递归（自回归）推出（在 JAX 中保持循环）
# print("Inputs:  ", train_inputs.dims.mapping)
# print("Targets: ", train_targets.dims.mapping)
# print("Forcings:", train_forcings.dims.mapping)

# predictions = run_forward_jitted(
#     rng=jax.random.PRNGKey(0),
#     inputs=train_inputs,
#     targets_template=train_targets * np.nan,
#     forcings=train_forcings)
# predictions



# In[ ]:

# # 使用平流方程计算6小时预报

# @title 平流方程实现和测试

# 导入平流计算模块
import sys
import os
import importlib

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 清除可能的模块缓存
if 'advection_calculation' in sys.modules:
    del sys.modules['advection_calculation']

try:
    # 导入模块
    import advection_calculation
    
    # 导入所需函数
    from advection_calculation import (
        calculate_advection_forecast, 
        calculate_enhanced_advection_forecast,
        calculate_correlation as advection_calculate_correlation, 
        print_correlation_results as advection_print_correlation_results
    )
    
    # 导入地表能量平衡模块
    import surface_energy_balance
    from surface_energy_balance import (
        calculate_surface_energy_balance_forecast,
        calculate_correlation as seb_calculate_correlation,
        print_correlation_results as seb_print_correlation_results
    )
    
    print("平流方程模块导入成功！")
    print("地表能量平衡模块导入成功！")
    
    # 演示公式说明
    print("\n" + "="*80)
    print("可用的物理模型公式说明")
    print("="*80)
    
    print("\n1. 平流方程 (Advection Equation):")
    print("使用的公式: T(t+Δt)(x,y,z) = T(t)(x,y,z) - Δt[u∂T/∂x + v∂T/∂y + w∂T/∂z]")
    print("其中:")
    print("  T(t+Δt) - 6小时后的温度场")
    print("  T(t)    - 当前时刻的温度场") 
    print("  Δt      - 时间步长 (6小时 = 21600秒)")
    print("  u       - x方向(经度)风速分量 (u_component_of_wind)")
    print("  v       - y方向(纬度)风速分量 (v_component_of_wind)")
    print("  w       - z方向(垂直)风速分量 (vertical_velocity)")
    print("  ∂T/∂x   - 温度在经度方向的梯度")
    print("  ∂T/∂y   - 温度在纬度方向的梯度")
    print("  ∂T/∂z   - 温度在垂直方向的梯度")
    
    print("\n2. 地表能量平衡 (Surface Energy Balance):")
    print("使用的公式: ΔT = (R_n - H - LE - G) · dt / (ρ · c_p · z_heat)")
    print("其中:")
    print("  R_n     - 净辐射 (W m⁻²)")
    print("  H       - 感热通量 (W m⁻²)")
    print("  LE      - 潜热通量 (W m⁻²)")
    print("  G       - 地热通量 (W m⁻²)")
    print("  ρ       - 空气密度 (kg m⁻³)")
    print("  c_p     - 定压比热 (J kg⁻¹ K⁻¹)")
    print("  z_heat  - 有效热容量深度 (m)")
    print("  dt      - 时间步长 (6小时 = 21600秒)")
    
except Exception as e:
    print(f"导入失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径
    print(f"advection_calculation.py 是否存在: {os.path.exists('advection_calculation.py')}")
    
    # 尝试列出当前目录的文件
    try:
        files = [f for f in os.listdir('.') if f.endswith('.py')]
        print(f"当前目录的Python文件: {files}")
    except:
        pass


# In[ ]:


# @title 使用物理公式计算6小时预报并比较相关性

try:
    # 选择使用的方法 - 目前使用地表能量平衡方法
    print("="*80)
    print("使用地表能量平衡计算6小时预报")
    print("="*80)
    
    # 使用地表能量平衡方法
    seb_results = calculate_surface_energy_balance_forecast(eval_inputs)
    
    print(f"地表能量平衡计算完成，得到 {len(seb_results)} 个预报变量:")
    for var_name, data in seb_results.items():
        print(f"  {var_name}: {data.shape}")
    
    # 计算与GraphCast结果的相关性
    print("\n" + "="*80)
    print("计算与GraphCast预测结果的相关性")
    print("="*80)
    
    correlations = seb_calculate_correlation(seb_results, predictions)
    seb_print_correlation_results(correlations)
    
    # 将结果保存到变量中供后续绘图使用
    physics_results = seb_results
    physics_method_name = "Surface Energy Balance"
    
    # =========== 平流方程方法（已注释） ===========
    # # 使用平流方程计算6小时预报
    # print("="*80)
    # print("使用平流方程计算6小时预报")
    # print("="*80)
    # 
    # advection_results = calculate_enhanced_advection_forecast(eval_inputs)
    # 
    # print(f"平流方程计算完成，得到 {len(advection_results)} 个预报变量:")
    # for var_name, data in advection_results.items():
    #     print(f"  {var_name}: {data.shape}")
    # 
    # # 计算与GraphCast结果的相关性
    # print("\n" + "="*80)
    # print("计算与GraphCast预测结果的相关性")
    # print("="*80)
    # 
    # correlations = advection_calculate_correlation(advection_results, predictions)
    # advection_print_correlation_results(correlations)
    # 
    # # 将结果保存到变量中供后续绘图使用
    # physics_results = advection_results
    # physics_method_name = "Advection"
    
except Exception as e:
    print(f"计算过程中出现错误: {e}")
    import traceback
    traceback.print_exc()


# In[ ]:
# @title 绘制输入、预测、物理公式计算结果及差异对比图

try:
    # 选择要可视化的变量
    variables_to_plot = ['2m_temperature', 'temperature']
    
    for var_name in variables_to_plot:
        if var_name in physics_results and var_name in predictions and var_name in eval_targets:
            print(f"\nPlotting comprehensive comparison for {var_name} ...")
            
            # 获取数据
            input_data = eval_inputs[var_name].isel(time=-1, batch=0)  # 输入数据（最后时刻）
            target_data = eval_targets[var_name].isel(time=0, batch=0)  # 真实目标值（6小时后）
            gc_prediction = predictions[var_name].isel(time=0, batch=0)  # GraphCast预测
            physics_prediction = physics_results[var_name]  # 物理公式预测
            
            # 计算差异（使用真实目标值）
            diff_target_gc = target_data - gc_prediction  # 真实值 - GraphCast预测
            diff_target_physics = target_data - physics_prediction  # 真实值 - 物理公式预测
            diff_gc_physics = gc_prediction - physics_prediction  # GraphCast - 物理公式
            
            # 如果是3D数据，选择一个层次进行可视化
            if 'level' in input_data.dims:
                level_idx = len(input_data.level) // 2  # 选择中间层
                input_data = input_data.isel(level=level_idx)
                target_data = target_data.isel(level=level_idx)
                gc_prediction = gc_prediction.isel(level=level_idx)
                physics_prediction = physics_prediction.isel(level=level_idx)
                diff_target_gc = diff_target_gc.isel(level=level_idx)
                diff_target_physics = diff_target_physics.isel(level=level_idx)
                diff_gc_physics = diff_gc_physics.isel(level=level_idx)
                level_info = f" (Level: {input_data.level.values:.0f} hPa)"
            else:
                level_info = ""
            
            # 创建2x4的子图布局（增加一列显示真实目标值）
            fig, axes = plt.subplots(2, 4, figsize=(26, 12))
            fig.suptitle(f'{var_name}{level_info} - Comprehensive Comparison ({physics_method_name})', fontsize=18, fontweight='bold')
            
            # 确定颜色范围
            vmin = min(input_data.min().values, target_data.min().values, 
                      gc_prediction.min().values, physics_prediction.min().values)
            vmax = max(input_data.max().values, target_data.max().values,
                      gc_prediction.max().values, physics_prediction.max().values)
            
            # 1. Input (Current Time)
            im1 = axes[0,0].imshow(input_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                  extent=[input_data.lon.min(), input_data.lon.max(), 
                                         input_data.lat.min(), input_data.lat.max()],
                                  origin='lower', aspect='auto')
            axes[0,0].set_title('Input (Current Time)', fontweight='bold', fontsize=14)
            axes[0,0].set_xlabel('Longitude')
            axes[0,0].set_ylabel('Latitude')
            plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
            
            # 2. Target (Ground Truth, 6h Later)
            im2 = axes[0,1].imshow(target_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                  extent=[target_data.lon.min(), target_data.lon.max(), 
                                         target_data.lat.min(), target_data.lat.max()],
                                  origin='lower', aspect='auto')
            axes[0,1].set_title('Target (Ground Truth, 6h Later)', fontweight='bold', fontsize=14)
            axes[0,1].set_xlabel('Longitude')
            axes[0,1].set_ylabel('Latitude')
            plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
            
            # 3. GraphCast Prediction
            im3 = axes[0,2].imshow(gc_prediction, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                  extent=[gc_prediction.lon.min(), gc_prediction.lon.max(), 
                                         gc_prediction.lat.min(), gc_prediction.lat.max()],
                                  origin='lower', aspect='auto')
            axes[0,2].set_title('GraphCast Prediction (6h Later)', fontweight='bold', fontsize=14)
            axes[0,2].set_xlabel('Longitude')
            axes[0,2].set_ylabel('Latitude')
            plt.colorbar(im3, ax=axes[0,2], shrink=0.8)
            
            # 4. Physics Prediction
            im4 = axes[0,3].imshow(physics_prediction, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                                  extent=[physics_prediction.lon.min(), physics_prediction.lon.max(), 
                                         physics_prediction.lat.min(), physics_prediction.lat.max()],
                                  origin='lower', aspect='auto')
            axes[0,3].set_title(f'{physics_method_name} Prediction (6h Later)', fontweight='bold', fontsize=14)
            axes[0,3].set_xlabel('Longitude')
            axes[0,3].set_ylabel('Latitude')
            plt.colorbar(im4, ax=axes[0,3], shrink=0.8)
            
            # 计算所有diff图的统一颜色范围
            diff_max_unified = max(
                abs(diff_target_gc.min().values), abs(diff_target_gc.max().values),
                abs(diff_target_physics.min().values), abs(diff_target_physics.max().values),
                abs(diff_gc_physics.min().values), abs(diff_gc_physics.max().values)
            )
            
            # 5. Empty subplot
            axes[1,0].axis('off')
            
            # 6. Difference: Target - GraphCast
            im6 = axes[1,1].imshow(diff_target_gc, cmap='RdBu_r', vmin=-diff_max_unified, vmax=diff_max_unified,
                                  extent=[diff_target_gc.lon.min(), diff_target_gc.lon.max(), 
                                         diff_target_gc.lat.min(), diff_target_gc.lat.max()],
                                  origin='lower', aspect='auto')
            axes[1,1].set_title('Diff: Target - GraphCast', fontweight='bold', fontsize=14)
            axes[1,1].set_xlabel('Longitude')
            axes[1,1].set_ylabel('Latitude')
            plt.colorbar(im6, ax=axes[1,1], shrink=0.8)
            
            # 7. Difference: Target - Physics
            im7 = axes[1,2].imshow(diff_target_physics, cmap='RdBu_r', vmin=-diff_max_unified, vmax=diff_max_unified,
                                  extent=[diff_target_physics.lon.min(), diff_target_physics.lon.max(), 
                                         diff_target_physics.lat.min(), diff_target_physics.lat.max()],
                                  origin='lower', aspect='auto')
            axes[1,2].set_title(f'Diff: Target - {physics_method_name}', fontweight='bold', fontsize=14)
            axes[1,2].set_xlabel('Longitude')
            axes[1,2].set_ylabel('Latitude')
            plt.colorbar(im7, ax=axes[1,2], shrink=0.8)
            
            # 8. Difference: GraphCast - Physics
            im8 = axes[1,3].imshow(diff_gc_physics, cmap='RdBu_r', vmin=-diff_max_unified, vmax=diff_max_unified,
                                  extent=[diff_gc_physics.lon.min(), diff_gc_physics.lon.max(), 
                                         diff_gc_physics.lat.min(), diff_gc_physics.lat.max()],
                                  origin='lower', aspect='auto')
            axes[1,3].set_title(f'Diff: GraphCast - {physics_method_name}', fontweight='bold', fontsize=14)
            axes[1,3].set_xlabel('Longitude')
            axes[1,3].set_ylabel('Latitude')
            plt.colorbar(im8, ax=axes[1,3], shrink=0.8)
            
            # 计算RMSE（使用真实目标值）
            rmse_gc = float(np.sqrt(((diff_target_gc)**2).mean()))  # Target vs GraphCast
            rmse_physics = float(np.sqrt(((diff_target_physics)**2).mean()))  # Target vs Physics
            rmse_gc_physics = float(np.sqrt(((diff_gc_physics)**2).mean()))  # GraphCast vs Physics
            
            # 添加统计信息到各个差异图
            axes[1,1].text(0.02, 0.98, f'RMSE: {rmse_gc:.4f}', transform=axes[1,1].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[1,2].text(0.02, 0.98, f'RMSE: {rmse_physics:.4f}', transform=axes[1,2].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[1,3].text(0.02, 0.98, f'RMSE: {rmse_gc_physics:.4f}', transform=axes[1,3].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # 打印统计信息
            print(f"\n{var_name}{level_info} Statistics:")
            print(f"  Input Range: [{input_data.min().values:.2f}, {input_data.max().values:.2f}]")
            print(f"  Target (Ground Truth) Range: [{target_data.min().values:.2f}, {target_data.max().values:.2f}]")
            print(f"  GraphCast Prediction Range: [{gc_prediction.min().values:.2f}, {gc_prediction.max().values:.2f}]")
            print(f"  {physics_method_name} Prediction Range: [{physics_prediction.min().values:.2f}, {physics_prediction.max().values:.2f}]")
            print(f"\n  Prediction Accuracy:")
            print(f"  RMSE (Target vs GraphCast): {rmse_gc:.4f}")
            print(f"  RMSE (Target vs {physics_method_name}): {rmse_physics:.4f}")
            print(f"  RMSE (GraphCast vs {physics_method_name}): {rmse_gc_physics:.4f}")
            
            # 比较哪个模型更准确
            if rmse_gc < rmse_physics:
                print(f"  -> GraphCast is more accurate (lower RMSE by {rmse_physics - rmse_gc:.4f})")
            else:
                print(f"  -> {physics_method_name} is more accurate (lower RMSE by {rmse_gc - rmse_physics:.4f})")
    
    # 绘制相关性总结图
    if correlations:
        print(f"\nPlotting correlation summary...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        var_names = list(correlations.keys())
        correlation_values = list(correlations.values())
        
        # 根据相关性值设置颜色
        colors_list = ['green' if c > 0.8 else 'orange' if c > 0.6 else 'red' for c in correlation_values]
        
        bars = ax.bar(var_names, correlation_values, color=colors_list, alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, correlation_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Correlation', fontweight='bold')
        ax.set_title(f'{physics_method_name} vs GraphCast Correlation', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 添加参考线
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Fair (>0.4)')
        
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # 计算并显示平均相关性
        avg_correlation = np.mean(correlation_values)
        print(f"\nAverage Correlation: {avg_correlation:.4f}")
        
        if avg_correlation > 0.8:
            print(f"Overall: {physics_method_name} and GraphCast predictions are highly consistent!")
        elif avg_correlation > 0.6:
            print(f"Overall: {physics_method_name} and GraphCast predictions are fairly consistent.")
        elif avg_correlation > 0.4:
            print(f"Overall: There are some differences between {physics_method_name} and GraphCast predictions.")
        else:
            print(f"Overall: {physics_method_name} and GraphCast predictions differ significantly.")

except Exception as e:
    print(f"Error occurred during plotting: {e}")
    import traceback
    traceback.print_exc()


# %%
