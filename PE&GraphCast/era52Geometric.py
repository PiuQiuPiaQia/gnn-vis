import xarray as xr
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def loadEra5Dateset():
    # 假设 mean_temp 是一个 xarray DataArray
    mean_temp = xr.open_dataarray("./mean_temperature.nc")

    # print(mean_temp)

    # 提取节点特征（例如，温度）
    node_features = mean_temp.values.flatten()

    # 定义节点坐标（这里我们使用网格点的索引作为坐标）
    num_nodes = mean_temp.shape[0] * mean_temp.shape[1]
    node_coords = torch.tensor(np.indices((mean_temp.shape[0], mean_temp.shape[1])).reshape(2, -1).T)
    # print(node_coords)

    # 创建边索引（这里我们使用网格点之间的连接作为边）
    edge_index = []
    for i in range(mean_temp.shape[0] - 1):
        for j in range(mean_temp.shape[1]):
            edge_index.append([i * mean_temp.shape[1] + j, (i + 1) * mean_temp.shape[1] + j])
    for i in range(mean_temp.shape[0]):
        for j in range(mean_temp.shape[1] - 1):
            edge_index.append([i * mean_temp.shape[1] + j, i * mean_temp.shape[1] + j + 1])
    edge_index = torch.tensor(edge_index).t().contiguous()

    # 创建 PyTorch Geometric 的 Data 对象
    data = Data(x=torch.tensor(node_features, dtype=torch.float), pos=node_coords, edge_index=edge_index)
    # 创建图
    G = to_networkx(data)

    # print(data)

    return G, data
