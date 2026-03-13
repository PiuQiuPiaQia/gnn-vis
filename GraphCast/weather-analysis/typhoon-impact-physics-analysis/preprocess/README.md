# Preprocess

用于从 WeatherBench2 拉取台风时间窗的低分辨率 ERA5 数据，并导出为 GraphCast 可用的 `.nc` 文件。

## 导出脚本

### Tauktae

按 `CYCLONE_TAUKTAE_CENTERS` 的时间范围抓取数据：

```bash
python preprocess/export_tauktae_graphcast_lowres.py \
  --start '05/13/2021 06Z' \
  --end '05/19/2021 06Z' \
  --force
```

默认输出示例：

```text
/root/autodl-tmp/dataset/dataset-source-era5_date-2021-05-13_res-1.0_levels-13_steps-23.nc
```

### Yaas

按 `CYCLONE_YAAS_CENTERS` 的时间范围抓取数据：

```bash
python preprocess/export_yaas_graphcast_lowres.py --force
```

也可以显式指定时间范围：

```bash
python preprocess/export_yaas_graphcast_lowres.py \
  --start '05/23/2021 00Z' \
  --end '05/27/2021 18Z' \
  --force
```

## 数据集校验 Notebook

`preprocess/validate_graphcast_lowres_dataset.ipynb` 用于校验导出的 `.nc` 数据集是否符合 GraphCast 低分辨率样本的基本规范。

这个 notebook 的基本能力：

- 从 `/root/data/dataset` 读取候选数据集，并与参考 GraphCast 样本做结构对比。
- 检查坐标、变量布局，以及 `time` / `datetime` 是否保持 6 小时步长。
- 允许候选数据集的绝对时间范围和总步数与参考样本不同。
- 调用 `GraphCast_small` 对候选数据集做 3 步预测。
- 导出预测值、真实值、差值，并绘制前 3 个目标时次的热力图用于快速检查。

建议在 `GraphCast` conda 环境中运行该 notebook。
